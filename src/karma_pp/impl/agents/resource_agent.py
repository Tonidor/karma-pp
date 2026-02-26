from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from karma_pp.impl.worlds.resource_world.resource_world import ResourceWorldDynamics, ResourceWorldState
from karma_pp.src.agent import AgentModel
from karma_pp.src.types import AgentState, PopulationState

Urgency = int  # Private state
Outcome = tuple[bool]  # One outcome: which resource (if any)
Signal = list[int]  # Per-outcome signal (e.g. bid, reward, etc.)


@dataclass(frozen=True)
class ResourceAgentObservation:
    resource_capacities: list[int]

class ResourceAgent[
    WORLD_STATE,
    MECHANISM_STATE,
    POLICY_STATE,
    RESOLUTION,
](
    AgentModel[
        WORLD_STATE,
        MECHANISM_STATE,
        Urgency,
        POLICY_STATE,
        ResourceAgentObservation,
        RESOLUTION,
        Outcome,
        Signal,
    ]
):
    """Abstract resource-domain agent with shared domain logic."""

    def __init__(
        self,
        transition_matrix: list[list[float]],
        initial_urgency: int,
        urgency_levels: list[int],
        reward_per_resource: list[float],
        no_resource_penalty: list[float],
    ):
        if len(urgency_levels) != len(transition_matrix):
            raise ValueError("urgency_levels length must match transition_matrix state count.")
        tm = np.asarray(transition_matrix, dtype=float)
        if np.any(tm < 0):
            raise ValueError("transition_matrix entries must be non-negative.")
        if not np.allclose(tm.sum(axis=1), 1.0):
            raise ValueError("transition_matrix rows must sum to 1.")
        self.transition_matrix = transition_matrix
        self.urgency_levels = urgency_levels
        self.initial_state = self._state_idx_from_urgency(initial_urgency)
        self.reward_per_resource = np.array(reward_per_resource)
        self.no_resource_penalty = np.array(no_resource_penalty)
        self.n_resources = len(reward_per_resource)

    def initialize(
        self,
        world_dynamics: ResourceWorldDynamics,
        mechanism_dynamics: object,
        rng: np.random.Generator,
    ) -> AgentState[int, POLICY_STATE]:
        if self.n_resources != len(world_dynamics.resource_capacities):
            raise ValueError("reward_per_resource length must match world resource_capacities.")
        initial_private = int(self.urgency_levels[self.initial_state])
        policy_state = self._initialize_policy(world_dynamics, mechanism_dynamics, rng)
        return AgentState(private=initial_private, policy=policy_state)

    def compute_reward(
        self,
        agent_state: AgentState[Urgency, POLICY_STATE],
        observation: ResourceAgentObservation,
        resolution: RESOLUTION,
    ) -> float:
        selected_outcome = np.array(resolution.selected_outcome)
        urgency = agent_state.private
        resource_rewards = self.reward_per_resource * selected_outcome
        resource_penalties = self.no_resource_penalty * (1 - selected_outcome)
        reward = urgency * sum(resource_rewards + resource_penalties)
        return reward

    def get_action(
        self,
        agent_state: AgentState[int, POLICY_STATE],
        observation: ResourceAgentObservation,
        rng: np.random.Generator,
    ) -> list[tuple[Outcome, Signal]]:
        outcomes = self._get_outcomes(observation=observation)
        signals = self._get_action(
            agent_state=agent_state,
            outcomes=outcomes,
            observation=observation,
            rng=rng,
        )
        if len(signals) != len(outcomes):
            raise ValueError("Policy signal vector must match number of outcomes.")
        return list(zip(outcomes, signals))

    def update_state(
        self,
        previous: AgentState[Urgency, POLICY_STATE],
        observation: ResourceAgentObservation,
        resolution: RESOLUTION,
        rng: np.random.Generator,
    ) -> AgentState[Urgency, POLICY_STATE]:
        current_state_idx = self._state_idx_from_urgency(previous.private)
        next_private_idx = rng.choice(
            len(self.transition_matrix[current_state_idx]),
            p=self.transition_matrix[current_state_idx],
        )
        return AgentState(
            private=self.urgency_levels[next_private_idx],
            policy=previous.policy,
        )

    def _state_idx_from_urgency(self, urgency: Urgency) -> int:
        for idx, level in enumerate(self.urgency_levels):
            if level == urgency:
                return idx
        raise ValueError(f"Urgency value {urgency} not found in configured urgency_levels.")

    def _get_outcomes(self, observation: ResourceAgentObservation) -> list[Outcome]:
        n = len(observation.resource_capacities)
        outcomes: list[Outcome] = [tuple(False for _ in range(n))]
        outcomes.extend(tuple(i == j for j in range(n)) for i in range(n))
        return outcomes

    @abstractmethod
    def _initialize_policy(
        self,
        world_dynamics: ResourceWorldDynamics,
        mechanism_dynamics: object,
        rng: np.random.Generator,
    ) -> POLICY_STATE:
        ...

    @abstractmethod
    def get_observation(
        self,
        agent_id: int,
        agent_state: AgentState[int, POLICY_STATE],
        world_state: ResourceWorldState,
        mechanism_state: MECHANISM_STATE,
        population_state: PopulationState[int, POLICY_STATE],
        rng: np.random.Generator,
    ) -> ResourceAgentObservation:
        """Build the observation for one agent."""
        ...

    @abstractmethod
    def _get_action(
        self,
        agent_state: AgentState[int, POLICY_STATE],
        observation: ResourceAgentObservation,
        rng: np.random.Generator,
    ) -> list[Signal]:
        ...

    @abstractmethod
    def adapt(
        self,
        previous: AgentState[Urgency, POLICY_STATE],
        observation: ResourceAgentObservation,
        resolution: RESOLUTION,
        reward: float,
        timestep: int,
        rng: np.random.Generator,
    ) -> AgentState[Urgency, POLICY_STATE]:
        """Update learning/adaptation state."""
        ...
