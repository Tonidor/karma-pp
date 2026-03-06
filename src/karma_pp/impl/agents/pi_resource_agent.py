from dataclasses import dataclass
from pathlib import Path

import numpy as np

from karma_pp.impl.agents.resource_agent import ResourceAgentObservation, ResourceAgent
from karma_pp.impl.mechanisms.karma.karma_mechanism import (
    KarmaDynamics,
    KarmaResolution,
    KarmaState,
)
from karma_pp.impl.worlds.resource_world.resource_world import (
    ResourceWorldDynamics,
    ResourceWorldState,
)
from karma_pp.core.types import AgentState, PopulationState


PolicyState = type(None)  # fixed policy; no per-agent learning state
Outcome = tuple[bool]
Bid = int


@dataclass(frozen=True)
class PiObservation(ResourceAgentObservation):
    """Observation for fixed-π bidding agent; extends base with agent_balance."""

    agent_balance: int


class PiResourceAgent(
    ResourceAgent[
        ResourceWorldState,
        KarmaState,
        PolicyState,
        KarmaResolution,
    ]
):
    """
    Resource agent that acts according to a pre-trained policy π loaded from a .npy file.

    The loaded policy is expected to have shape (nu, nk, nk) where:
        - nu = number of urgency levels (len(urgency_levels)),
        - nk = max_balance + 1 (karma states 0 … max_balance),
        - π[u, k, b] = P(bid b | urgency-index u, karma k), for 0 ≤ b ≤ k.

    The bidding rule matches the full-information learning agent:
        - outcome index 0 = (False,)  → bid 0
        - outcome index 1 = (True,)   → bid b ~ π[private_u_idx, karma_k, ·]
    """

    def __init__(
        self,
        transition_matrix: list[list[float]],
        initial_urgency: int,
        urgency_levels: list[int],
        reward_per_resource: list[float],
        no_resource_penalty: list[float],
        policy_path: str | Path,
    ):
        super().__init__(
            transition_matrix=transition_matrix,
            initial_urgency=initial_urgency,
            urgency_levels=urgency_levels,
            reward_per_resource=reward_per_resource,
            no_resource_penalty=no_resource_penalty,
        )
        self._policy_path = Path(policy_path)
        if not self._policy_path.is_file():
            raise FileNotFoundError(f"Policy file not found: {self._policy_path}")

        # Load raw π; we validate its shape once mechanism_dynamics is known.
        pi_arr = np.load(self._policy_path)
        if pi_arr.ndim != 3:
            raise ValueError(
                f"Loaded policy from {self._policy_path} must be 3D (nu, nk, nk); "
                f"got shape {pi_arr.shape}."
            )
        self._pi_loaded: np.ndarray = np.asarray(pi_arr, dtype=float)
        self._pi: np.ndarray | None = None
        self._mechanism_dynamics: KarmaDynamics | None = None

    def _initialize_policy(
        self,
        agent_id: int,
        world_dynamics: ResourceWorldDynamics,
        mechanism_dynamics: KarmaDynamics,
        rng: np.random.Generator,
    ) -> PolicyState:
        del agent_id, world_dynamics, rng

        # Validate and normalise the loaded π against the actual dynamics.
        nu_expected = len(self.urgency_levels)
        nk_expected = mechanism_dynamics.max_balance + 1

        if self._pi_loaded.shape[0] != nu_expected:
            raise ValueError(
                f"Loaded policy nu={self._pi_loaded.shape[0]} does not match "
                f"len(urgency_levels)={nu_expected}."
            )
        if self._pi_loaded.shape[1] < nk_expected or self._pi_loaded.shape[2] < nk_expected:
            raise ValueError(
                f"Loaded policy nk={self._pi_loaded.shape[1]} is incompatible with "
                f"max_balance+1={nk_expected}."
            )

        # Restrict to the required karma range and ensure probabilities are valid.
        pi = np.array(self._pi_loaded[:, :nk_expected, :nk_expected], dtype=float)
        # Enforce zero probability for invalid bids b > k and renormalise each row.
        for k_state in range(nk_expected):
            pi[:, k_state, k_state + 1 :] = 0.0
            row_sums = pi[:, k_state, : k_state + 1].sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums < 1e-15, 1.0, row_sums)
            pi[:, k_state, : k_state + 1] /= row_sums

        self._pi = pi
        self._mechanism_dynamics = mechanism_dynamics
        return None

    def get_observation(
        self,
        agent_id: int,
        agent_state: AgentState[int, PolicyState],
        world_state: ResourceWorldState,
        mechanism_state: KarmaState,
        population_state: PopulationState[int, PolicyState],
        membership: tuple[int, int],
        rng: np.random.Generator,
    ) -> PiObservation:
        del agent_state, population_state, membership, rng
        balance = mechanism_state.agent_balances[agent_id]
        return PiObservation(
            resource_capacities=world_state.resource_capacities,
            agent_balance=balance,
        )

    def _get_action(
        self,
        agent_state: AgentState[int, PolicyState],
        outcomes: list[Outcome],
        observation: PiObservation,
        rng: np.random.Generator,
    ) -> list[Bid]:
        if self._mechanism_dynamics is None or self._pi is None:
            raise RuntimeError("_initialize_policy must be called before _get_action.")

        nk = self._mechanism_dynamics.max_balance + 1

        u_idx = int(agent_state.private)
        if u_idx < 0 or u_idx >= len(self.urgency_levels):
            raise ValueError(
                f"Urgency index {u_idx} out of bounds for configured urgency_levels."
            )
        k = int(np.clip(observation.agent_balance, 0, nk - 1))

        probs = self._pi[u_idx, k, : k + 1].astype(float)
        prob_sum = probs.sum()
        if prob_sum < 1e-15:
            probs = np.ones(k + 1, dtype=float) / (k + 1)
        else:
            probs /= prob_sum

        b = int(rng.choice(k + 1, p=probs))

        # Bid 0 for outcome index 0 = (False,) = no resource,
        # bid b for outcome index 1 = (True,)  = resource.
        # For multiple resources, this mirrors the full-info agent's convention.
        bids: list[Bid] = [0 for _ in outcomes]
        if len(outcomes) >= 2:
            bids[1] = b
        else:
            # Degenerate case: only one outcome; treat it as the "resource" outcome.
            bids[0] = b
        return bids

    def adapt(
        self,
        agent_id: int,
        previous: AgentState[int, PolicyState],
        observation: PiObservation,
        resolution: KarmaResolution,
        reward: float,
        timestep: int,
        rng: np.random.Generator,
    ) -> tuple[AgentState[int, PolicyState], bool]:
        """Fixed policy: no learning, so the state does not change and never converges early."""
        del agent_id, observation, resolution, reward, timestep, rng
        return previous, False

