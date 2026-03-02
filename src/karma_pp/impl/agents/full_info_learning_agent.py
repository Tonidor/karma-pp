"""Full-information mean-field learning agent for the karma mechanism.

Implements the Stationary Nash Equilibrium (SNE) computation algorithm from:
  Elokda et al. (2023). "A Self-Contained Karma Economy for the Dynamic 
  Allocation of Common Resources". Dynamic Games and Applications, 2023.

The agent model maintains a policy π[b|u,k] and distribution d[u,k]
shared across all agents using the same model instance. At each adaptation
step it computes the mean-field best-response policy via value iteration and
nudges π towards the softmax-perturbed best response (evolutionary dynamics).

Individual agents evolve their urgency independently according to the shared
Markov transition matrix, and their karma balances are updated by the mechanism
as usual. Only the policy π and the distribution d are shared.

Implementation details
------------------
- Bids: agents always bid 0 for the "no resource" outcome and a scalar
  b ∈ {0, …, k} for the "resource" outcome.
- Shared policy: π and d live on the model instance (self), not per-agent
  PolicyState. adapt() updates them exactly once per timestep (the first
  agent to call adapt in that timestep does the work; subsequent calls are no-ops).
- d estimation: re-estimated from the actual simulation population state every
  step via FullInfoObservation.population_privates and .population_balances.
- γ discount: only γ < 1 (discounted value function) is currentlysupported.
"""

from dataclasses import dataclass
from typing import cast

import numpy as np
import structlog

from karma_pp.core.types import AgentState, PopulationState
from karma_pp.impl.agents.resource_agent import ResourceAgent, ResourceAgentObservation
from karma_pp.impl.mechanisms.karma.karma_mechanism import (
    KarmaDynamics,
    KarmaResolution,
    KarmaState,
)
from karma_pp.impl.worlds.resource_world.resource_world import (
    ResourceWorldDynamics,
    ResourceWorldState,
)

log = structlog.get_logger(__name__)

Outcome = tuple[bool, ...]


@dataclass(frozen=True)
class FullInfoObservation(ResourceAgentObservation):
    """Observation that contains all agent private states and karma balances."""

    agent_balance: int
    population_privates: tuple[int, ...]
    population_balances: tuple[int, ...]


@dataclass
class FullInfoPolicyState:
    """Per-agent policy state.

    The learning state (π, d, V) lives on the shared model instance.
    This dataclass is intentionally empty; it acts as a typed marker so the
    framework's generic machinery works correctly.
    """


class FullInfoLearningAgent(
    ResourceAgent[
        ResourceWorldState,
        KarmaState,
        FullInfoPolicyState,
        KarmaResolution,
    ]
):
    """Mean-field best-response agent (Algorithm 1, Elokda et al. 2023).

    Hyperparameters
    ---------------
    gamma       : float  Discount factor in (0, 1). γ=1 is not supported yet.
    eta         : float  Policy learning rate η in Algorithm 1.
    lam         : float  Softmax temperature λ; higher → sharper best response.
    dt          : float  Evolutionary step size dt in Algorithm 1.
    vi_tol      : float  Value-iteration convergence tolerance.
    vi_max_iter : int    Maximum value-iteration sweeps per adaptation step.
    """

    def __init__(
        self,
        transition_matrix: list[list[float]],
        initial_urgency: int,
        urgency_levels: list[int],
        reward_per_resource: list[float],
        no_resource_penalty: list[float],
        gamma: float = 0.99,
        eta: float = 0.01,
        lam: float = 10.0,
        dt: float = 0.1,
        vi_tol: float = 1e-6,
        vi_max_iter: int = 1000,
    ) -> None:
        super().__init__(
            transition_matrix=transition_matrix,
            initial_urgency=initial_urgency,
            urgency_levels=urgency_levels,
            reward_per_resource=reward_per_resource,
            no_resource_penalty=no_resource_penalty,
        )
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1); got {gamma}.")
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.lam = float(lam)
        self.dt = float(dt)
        self.vi_tol = float(vi_tol)
        self.vi_max_iter = int(vi_max_iter)

        # Shared learning state
        self.pi: np.ndarray | None = None          # (nu, nk, nk): π[u, k, b]
        self.d: np.ndarray | None = None           # (nu, nk):     d[u, k]
        self._V: np.ndarray | None = None          # (nu, nk):     V[u, k]
        self._mechanism_dynamics: KarmaDynamics | None = None
        self._last_update_step: int = -1           # timestep of last adapt run

    def _initialize_policy(
        self,
        world_dynamics: ResourceWorldDynamics,
        mechanism_dynamics: KarmaDynamics,
        rng: np.random.Generator,
    ) -> FullInfoPolicyState:
        nu = len(self.urgency_levels)
        nk = mechanism_dynamics.max_balance + 1  # karma states 0 … max_balance

        # π: uniform over valid bids {0, …, k} for each state (u, k).
        pi = np.zeros((nu, nk, nk), dtype=float)
        for k in range(nk):
            pi[:, k, : k + 1] = 1.0 / (k + 1)

        # d: uniform; replaced by empirical estimate on first adapt call.
        d = np.ones((nu, nk), dtype=float) / (nu * nk)

        self.pi = pi
        self.d = d
        self._V = np.zeros((nu, nk), dtype=float)
        self._mechanism_dynamics = mechanism_dynamics
        self._world_dynamics = world_dynamics
        return FullInfoPolicyState()

    def get_observation(
        self,
        agent_id: int,
        agent_state: AgentState[int, FullInfoPolicyState],
        world_state: ResourceWorldState,
        mechanism_state: KarmaState,
        population_state: PopulationState[int, FullInfoPolicyState],
        membership: tuple[int, int],
        rng: np.random.Generator,
    ) -> FullInfoObservation:
        balance = mechanism_state.agent_balances[agent_id]
        return FullInfoObservation(
            resource_capacities=world_state.resource_capacities,
            agent_balance=balance,
            population_privates=tuple(
                s.private for s in population_state.agent_states.values()
            ),
            population_balances=tuple(
                mechanism_state.agent_balances[aid]
                for aid in population_state.agent_states
            ),
        )

    def _get_action(
        self,
        agent_state: AgentState[int, FullInfoPolicyState],
        outcomes: list[Outcome],
        observation: ResourceAgentObservation,
        rng: np.random.Generator,
    ) -> list[int]:
        obs = cast(FullInfoObservation, observation)
        md = self._mechanism_dynamics
        assert md is not None, "_initialize_policy must be called before _get_action."
        nk = md.max_balance + 1

        u_idx = self._state_idx_from_urgency(agent_state.private)
        k = int(np.clip(obs.agent_balance, 0, nk - 1))

        probs = self.pi[u_idx, k, : k + 1].copy()  # type: ignore[index]
        prob_sum = probs.sum()
        if prob_sum < 1e-15:
            probs = np.ones(k + 1) / (k + 1)
        else:
            probs /= prob_sum

        b = int(rng.choice(k + 1, p=probs))
        # Bid 0 for outcome index 0 = (False,) = no resource.
        # Bid b for outcome index 1 = (True,)  = resource.
        return [0, b]

    def adapt(
        self,
        previous: AgentState[int, FullInfoPolicyState],
        observation: ResourceAgentObservation,
        resolution: KarmaResolution,
        reward: float,
        timestep: int,
        rng: np.random.Generator,
    ) -> AgentState[int, FullInfoPolicyState]:
        del resolution, reward, rng
        # Shared update: only the first agent to arrive this timestep runs it.
        if timestep == self._last_update_step:
            return AgentState(private=previous.private, policy=previous.policy)
        self._last_update_step = timestep

        obs = cast(FullInfoObservation, observation)
        md = self._mechanism_dynamics
        assert md is not None

        nu = len(self.urgency_levels)
        nk = md.max_balance + 1
        collective_dist = self._world_dynamics.collective_size_distribution

        # ----------------------------------------------------------
        # Step 1: Re-estimate d from simulation population state.
        # observation.population_privates/balances contain the post-transition
        # (urgency, karma) pairs for all agents.
        # ----------------------------------------------------------
        d_new = np.zeros((nu, nk), dtype=float)
        for urgency, karma in zip(obs.population_privates, obs.population_balances):
            try:
                u_idx = self._state_idx_from_urgency(urgency)
            except ValueError:
                raise ValueError(f"Invalid urgency: {urgency}, urgency levels need to be homogeneous.")
            if karma >= nk:
                raise ValueError(f"Invalid karma: {karma}, karma must be less than or equal to {nk}.")
            k_idx = int(karma)
            d_new[u_idx, k_idx] += 1.0
        total = d_new.sum()
        if total == 0:
            raise ValueError("Total number of agents is 0, cannot estimate d.")
        self.d = d_new / total

        # ----------------------------------------------------------
        # Step 2: Mean-field bid distribution  v_b[b] = Σ_{u,k} d[u,k] π[u,k,b]
        # ----------------------------------------------------------
        v_b_arr = np.einsum("ukb,uk->b", self.pi, self.d)   # shape (nk,)
        v_b_arr = np.clip(v_b_arr, 0.0, None)  # safe guard against negative values
        s = v_b_arr.sum()
        if s > 0:
            v_b_arr /= s
        v_b: dict[int, float] = {b: float(v_b_arr[b]) for b in range(nk)}

        # ----------------------------------------------------------
        # Step 3: Outcome probabilities  γ(b) = P(ego wins | bid b)
        # Uses collective_size_distribution: γ(b) = Σ_n p(n) γ_n(b).
        # ----------------------------------------------------------
        sel = md.selection_rule
        gamma_b = np.zeros(nk, dtype=float)  # γ(b) = Σ_n p(n) γ_n(b), marginal over sizes
        gamma_b_by_n: dict[int, np.ndarray] = {}  # n -> γ_n(b) for each bid b
        for n_size, p_n in collective_dist.items():
            # γ_n(b) = P(ego wins | bid b, collective size n)
            gamma_b_by_n[n_size] = np.array(
                [sel.compute_gamma(b, v_b, n_size) for b in range(nk)], dtype=float
            )
            gamma_b += p_n * gamma_b_by_n[n_size]

        # ----------------------------------------------------------
        # Step 4: Karma transition kernels  κ[k, b, k']
        # kappa_win[k, b, k'] = P(k' | k, b, won)
        # kappa_lose[k, b, k'] = P(k' | k, b, lost)
        # κ_win, κ_lose marginalize over collective_size_distribution: κ = Σ_n p(n) κ_n.
        # Only filled for valid bids b ≤ k; rest stays zero (masked later).
        # ----------------------------------------------------------
        red = md.redistribution_rule
        kappa_win = np.zeros((nk, nk, nk), dtype=float)
        kappa_lose = np.zeros((nk, nk, nk), dtype=float)

        for n_size, p_n in collective_dist.items():
            gamma_b_n = gamma_b_by_n[n_size]
            for k_state in range(nk):
                for b in range(k_state + 1):
                    g_b = float(gamma_b_n[b])
                    kappa_won_dist, kappa_lose_dist = red.compute_kappa(
                        k_state, b, v_b, n_size, gamma_b=g_b
                    )
                    # Check for out-of-bounds karma states that will be clipped.
                    won_oob = {k: p for k, p in kappa_won_dist.items() if (k < 0 or k > nk - 1) and p > 0}
                    lose_oob = {k: p for k, p in kappa_lose_dist.items() if (k < 0 or k > nk - 1) and p > 0}
                    if won_oob or lose_oob:
                        log.warning(
                            "karma_transition_clipped",
                            timestep=timestep,
                            k_state=k_state,
                            bid=b,
                            n_size=n_size,
                            won_oob=won_oob,
                            lose_oob=lose_oob,
                        )
                    for k_next, prob in kappa_won_dist.items():
                        kc = int(np.clip(k_next, 0, nk - 1))
                        kappa_win[k_state, b, kc] += p_n * prob
                    for k_next, prob in kappa_lose_dist.items():
                        kc = int(np.clip(k_next, 0, nk - 1))
                        kappa_lose[k_state, b, kc] += p_n * prob

        # ----------------------------------------------------------
        # Step 5: Immediate expected reward  ζ[u, k, b]
        # ζ[u, b] = γ(b)·r_win[u] + (1−γ(b))·r_lose[u]
        # (independent of k; shape (nu, nk, nk) for uniform indexing)
        # ----------------------------------------------------------
        r_win = np.array(
            [self._outcome_reward(u, (True,)) for u in self.urgency_levels],
            dtype=float,
        )   # (nu,)
        r_lose = np.array(
            [self._outcome_reward(u, (False,)) for u in self.urgency_levels],
            dtype=float,
        )  # (nu,)

        # Build (nu, nk) then broadcast to (nu, nk, nk).
        # ζ (zeta) does not depend on k, so we can compute it once and broadcast.
        # zeta: immediate expected reward
        zeta_2d = (
            gamma_b[np.newaxis, :] * r_win[:, np.newaxis]
            + (1.0 - gamma_b[np.newaxis, :]) * r_lose[:, np.newaxis]
        )  # (nu, nk)
        zeta = np.broadcast_to(zeta_2d[:, np.newaxis, :], (nu, nk, nk)).copy()

        # Mask invalid bids (b > k) with −∞.
        bid_mask = np.full((nk, nk), -np.inf, dtype=float)
        for k_state in range(nk):
            bid_mask[k_state, : k_state + 1] = 0.0
        zeta += bid_mask[np.newaxis, :, :]

        # ----------------------------------------------------------
        # Step 6: Full state transition  P[u, k, b, u', k']
        # = tm[u, u'] · (γ(b)·κ_win[k,b,k'] + (1−γ(b))·κ_lose[k,b,k'])
        # ----------------------------------------------------------
        kappa_mixed = (
            gamma_b[np.newaxis, :, np.newaxis] * kappa_win
            + (1.0 - gamma_b[np.newaxis, :, np.newaxis]) * kappa_lose
        )  # (nk, nk, nk)

        tm = np.array(self.transition_matrix, dtype=float)  # (nu, nu)
        # Shapes: tm → (nu,1,1,nu,1), kappa_mixed → (1,nk,nk,1,nk)
        P_trans = (
            tm[:, np.newaxis, np.newaxis, :, np.newaxis]
            * kappa_mixed[np.newaxis, :, :, np.newaxis, :]
        )  # (nu, nk, nk, nu, nk)

        # ----------------------------------------------------------
        # Step 7: Value iteration  V[u,k] = max_b Q[u,k,b]
        # Q[u,k,b] = ζ[u,k,b] + γ · Σ_{u',k'} P[u,k,b,u',k'] · V[u',k']
        # ----------------------------------------------------------
        V = self._V.copy()
        V_new = V  # will be overwritten in loop
        for _ in range(self.vi_max_iter):
            # Bootstrap: Expected future value
            bootstrap = np.einsum("ukbpq,pq->ukb", P_trans, V)   # (nu, nk, nk)
            Q = zeta + self.gamma * bootstrap
            V_new = np.max(Q, axis=2)   # max over b; −∞ entries are ignored
            if np.max(np.abs(V_new - V)) < self.vi_tol:
                V = V_new
                break
            V = V_new
        self._V = V

        # ----------------------------------------------------------
        # Step 8: Perturbed best response  π̃[u,k,b] = softmax_λ(Q[u,k,b])
        # ----------------------------------------------------------
        Q_lam = self.lam * Q                                    # (nu, nk, nk)
        Q_lam -= np.max(Q_lam, axis=2, keepdims=True)          # numerical stability
        exp_Q = np.exp(Q_lam)
        # Entries where Q was −∞: exp(−∞) = 0 in numpy, but (−∞)−(−∞) = nan.
        exp_Q[~np.isfinite(Q_lam)] = 0.0
        row_sum = exp_Q.sum(axis=2, keepdims=True)
        row_sum = np.where(row_sum < 1e-15, 1.0, row_sum)
        pi_tilde = exp_Q / row_sum                              # (nu, nk, nk)

        # ----------------------------------------------------------
        # Step 9: Policy update  π ← (1 − η dt) π + η dt π̃
        # ----------------------------------------------------------
        self.pi = (1.0 - self.eta * self.dt) * self.pi + self.eta * self.dt * pi_tilde

        # Re-enforce validity: zero b > k and renormalise rows.
        for k_state in range(nk):
            self.pi[:, k_state, k_state + 1 :] = 0.0
            rs = self.pi[:, k_state, : k_state + 1].sum(axis=1, keepdims=True)
            rs = np.where(rs < 1e-15, 1.0, rs)
            self.pi[:, k_state, : k_state + 1] /= rs

        log.debug(
            "full_info_adapt",
            timestep=timestep,
            mean_V=float(np.mean(self._V)),
            vi_delta=float(np.max(np.abs(V_new - self._V))),
        )

        return AgentState(private=previous.private, policy=previous.policy)
