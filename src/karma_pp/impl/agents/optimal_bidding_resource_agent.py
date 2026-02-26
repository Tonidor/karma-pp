from dataclasses import dataclass
import math

import numpy as np

from karma_pp.impl.agents.resource_agent import ResourceAgentObservation, ResourceAgent
from karma_pp.impl.mechanisms.karma.karma_mechanism import KarmaDynamics, KarmaResolution, KarmaState
from karma_pp.impl.worlds.resource_world.resource_world import ResourceWorldDynamics, ResourceWorldState
from karma_pp.src.types import AgentState, PopulationState

Outcome = tuple[bool]
Commit = int


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(frozen=True)
class OptimalBiddingObservation(ResourceAgentObservation):
    """Observation for optimal bidding agent; extends base with agent_balance."""

    agent_balance: int


@dataclass
class OptimalBiddingPolicyState:
    support_mu: list[float]
    support_var: list[float]
    threshold_mu: float
    threshold_var: float
    n_agents_mu: float
    value_weight: float
    value_bias: float
    last_balance: int | None


class OptimalBiddingResourceAgent(
    ResourceAgent[
        ResourceWorldState,
        KarmaState,
        OptimalBiddingPolicyState,
        KarmaResolution,
    ]
):
    """Optimal bidding agent for karma mechanism."""

    def __init__(
        self,
        transition_matrix: list[list[float]],
        initial_urgency: int,
        urgency_levels: list[int],
        reward_per_resource: list[float],
        no_resource_penalty: list[float],
        gamma: float = 0.99,
        belief_alpha: float = 0.05,
        value_alpha: float = 0.02,
        init_support_std: float = 1.0,
        init_threshold_std: float = 1.0,
        init_value_weight: float = 1.0,
        init_value_bias: float = 0.0,
        min_std: float = 0.5,
        min_lambda: float = 1e-3,
    ):
        super().__init__(
            transition_matrix=transition_matrix,
            initial_urgency=initial_urgency,
            urgency_levels=urgency_levels,
            reward_per_resource=reward_per_resource,
            no_resource_penalty=no_resource_penalty,
        )
        self.gamma = float(gamma)
        self.belief_alpha = float(belief_alpha)
        self.value_alpha = float(value_alpha)
        self.init_support_var = float(init_support_std) ** 2
        self.init_threshold_var = float(init_threshold_std) ** 2
        self.init_value_weight = float(init_value_weight)
        self.init_value_bias = float(init_value_bias)
        self.min_std = float(min_std)
        self.min_lambda = float(min_lambda)
        self._n_outcomes = 0
        self._max_balance = 0

    def _initialize_policy(
        self,
        world_dynamics: ResourceWorldDynamics,
        mechanism_dynamics: KarmaDynamics,
        rng: np.random.Generator,
    ) -> OptimalBiddingPolicyState:
        del rng
        self._n_outcomes = len(world_dynamics.resource_capacities) + 1
        self._max_balance = mechanism_dynamics.max_balance
        return OptimalBiddingPolicyState(
            support_mu=[0.0 for _ in range(self._n_outcomes)],
            support_var=[self.init_support_var for _ in range(self._n_outcomes)],
            threshold_mu=0.0,
            threshold_var=self.init_threshold_var,
            n_agents_mu=2.0,
            value_weight=max(self.init_value_weight, self.min_lambda),
            value_bias=self.init_value_bias,
            last_balance=None,
        )

    def get_observation(
        self,
        agent_id: int,
        agent_state: AgentState[int, OptimalBiddingPolicyState],
        world_state: ResourceWorldState,
        mechanism_state: KarmaState,
        population_state: PopulationState[int, OptimalBiddingPolicyState],
        rng: np.random.Generator,
    ) -> OptimalBiddingObservation:
        balance = mechanism_state.agent_balances[agent_id]
        return OptimalBiddingObservation(
            resource_capacities=world_state.resource_capacities,
            agent_balance=balance,
        )

    def _get_action(
        self,
        agent_state: AgentState[int, OptimalBiddingPolicyState],
        outcomes: list[Outcome],
        observation: OptimalBiddingObservation,
        rng: np.random.Generator,
    ) -> list[Commit]:
        del rng
        balance = int(np.clip(observation.agent_balance, 0, self._max_balance))
        policy = agent_state.policy
        lambda_k = self._shadow_price(balance=balance, policy=policy)
        urgency = int(agent_state.private)
        n_resources = self._n_outcomes - 1
        utility_gain = [0.0] + [urgency for _ in range(n_resources)]
        signals = [0 for _ in range(len(outcomes))]
        for outcome_idx in range(len(outcomes)):
            best_bid = 0
            best_value = -float("inf")
            for bid in range(balance + 1):
                win_prob = self._win_probability(bid=bid, outcome_idx=outcome_idx, policy=policy)
                surplus = win_prob * (utility_gain[outcome_idx] - lambda_k * bid)
                if surplus > best_value:
                    best_value = surplus
                    best_bid = bid
            signals[outcome_idx] = int(best_bid)
        return signals

    def adapt(
        self,
        previous: AgentState[int, OptimalBiddingPolicyState],
        observation: OptimalBiddingObservation,
        resolution: KarmaResolution,
        reward: float,
        timestep: int,
        rng: np.random.Generator,
    ) -> AgentState[int, OptimalBiddingPolicyState]:
        del timestep, rng
        old = previous.policy
        balance_next = int(np.clip(observation.agent_balance, 0, self._max_balance))
        last_balance = old.last_balance if old.last_balance is not None else balance_next
        value_curr = self._value(balance=last_balance, policy=old)
        value_next = self._value(balance=balance_next, policy=old)
        td_error = float(reward) + self.gamma * value_next - value_curr
        phi = math.log1p(max(0, last_balance))
        new_weight = max(old.value_weight + self.value_alpha * td_error * phi, self.min_lambda)
        new_bias = old.value_bias + self.value_alpha * td_error

        scores = [float(x) for x in getattr(resolution, "outcome_scores", [])]
        finite_scores = [
            (idx, score) for idx, score in enumerate(scores) if idx < self._n_outcomes and np.isfinite(score)
        ]
        support_mu = list(old.support_mu)
        support_var = list(old.support_var)
        if finite_scores:
            threshold_sample = max(score for _, score in finite_scores)
            threshold_mu, threshold_var = self._ema_gaussian_update(
                old.threshold_mu,
                old.threshold_var,
                threshold_sample,
            )
            for idx, sample in finite_scores:
                mu, var = self._ema_gaussian_update(support_mu[idx], support_var[idx], sample)
                support_mu[idx] = mu
                support_var[idx] = var
        else:
            threshold_mu = old.threshold_mu
            threshold_var = old.threshold_var

        n_agents_obs = float(getattr(resolution, "n_agents", old.n_agents_mu))
        n_agents_mu = (1.0 - self.belief_alpha) * old.n_agents_mu + self.belief_alpha * n_agents_obs
        new_policy = OptimalBiddingPolicyState(
            support_mu=support_mu,
            support_var=support_var,
            threshold_mu=float(threshold_mu),
            threshold_var=float(threshold_var),
            n_agents_mu=float(n_agents_mu),
            value_weight=float(new_weight),
            value_bias=float(new_bias),
            last_balance=balance_next,
        )
        return AgentState(private=previous.private, policy=new_policy)

    def _value(self, balance: int, policy: OptimalBiddingPolicyState) -> float:
        return policy.value_weight * math.log1p(max(0, balance)) + policy.value_bias

    def _shadow_price(self, balance: int, policy: OptimalBiddingPolicyState) -> float:
        denom = max(1.0, float(balance) + 1.0)
        return max(self.min_lambda, policy.value_weight / denom)

    def _win_probability(self, bid: int, outcome_idx: int, policy: OptimalBiddingPolicyState) -> float:
        if outcome_idx >= len(policy.support_mu):
            return 0.0
        gap_mu = policy.threshold_mu - policy.support_mu[outcome_idx]
        gap_var = max(0.0, policy.threshold_var + policy.support_var[outcome_idx])
        gap_std = max(self.min_std, math.sqrt(gap_var))
        z = (float(bid) - gap_mu) / gap_std
        return float(np.clip(_normal_cdf(z), 0.0, 1.0))

    def _ema_gaussian_update(self, mu: float, var: float, sample: float) -> tuple[float, float]:
        alpha = self.belief_alpha
        new_mu = (1.0 - alpha) * mu + alpha * sample
        centered = sample - new_mu
        new_var = (1.0 - alpha) * var + alpha * centered * centered
        new_var = max(new_var, self.min_std * self.min_std)
        return float(new_mu), float(new_var)
