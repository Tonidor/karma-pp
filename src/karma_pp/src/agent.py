from abc import ABC, abstractmethod

import numpy as np

from karma_pp.src.types import AgentState, PopulationState


class AgentModel[
    WORLD_STATE,
    MECHANISM_STATE,
    PRIVATE_STATE,
    POLICY_STATE,
    OBSERVATION,
    RESOLUTION,
    OUTCOME,
    SIGNAL,
](ABC):
    """Single behavior model for one agent type."""

    @abstractmethod
    def initialize(
        self,
        world_dynamics: object,
        mechanism_dynamics: object,
        rng: np.random.Generator,
    ) -> AgentState[PRIVATE_STATE, POLICY_STATE]:
        """Create initial agent state."""
        ...

    @abstractmethod
    def get_observation(
        self,
        agent_id: int,
        agent_state: AgentState[PRIVATE_STATE, POLICY_STATE],
        world_state: WORLD_STATE,
        mechanism_state: MECHANISM_STATE,
        population_state: PopulationState[PRIVATE_STATE, POLICY_STATE],
        rng: np.random.Generator,
    ) -> OBSERVATION:
        """Build the observation for one agent."""
        ...

    @abstractmethod
    def get_action(
        self,
        agent_state: AgentState[PRIVATE_STATE, POLICY_STATE],
        observation: OBSERVATION,
        rng: np.random.Generator,
    ) -> list[tuple[OUTCOME, SIGNAL]]:
        """Return list of actions, each action is (outcome, signal)."""
        ...

    @abstractmethod
    def compute_reward(
        self,
        agent_state: AgentState[PRIVATE_STATE, POLICY_STATE],
        observation: OBSERVATION,
        resolution: RESOLUTION,
    ) -> float:
        """Compute scalar reward for one agent."""
        ...

    @abstractmethod
    def update_state(
        self,
        previous: AgentState[PRIVATE_STATE, POLICY_STATE],
        observation: OBSERVATION,
        resolution: RESOLUTION,
        rng: np.random.Generator,
    ) -> AgentState[PRIVATE_STATE, POLICY_STATE]:
        """Update environment-driven agent state."""
        ...

    @abstractmethod
    def adapt(
        self,
        previous: AgentState[PRIVATE_STATE, POLICY_STATE],
        observation: OBSERVATION,
        resolution: RESOLUTION,
        reward: float,
        timestep: int,
        rng: np.random.Generator,
    ) -> AgentState[PRIVATE_STATE, POLICY_STATE]:
        """Update learning/adaptation state."""
        ...
