"""SilentResourceAgent: reports zero signal for each outcome."""

import numpy as np

from karma_pp.impl.agents.resource_agent import ResourceAgent, ResourceAgentObservation
from karma_pp.impl.worlds.resource_world.resource_world import ResourceWorldDynamics, ResourceWorldState
from karma_pp.core.types import AgentState, PopulationState, Resolution

PolicyState = None
Outcome = tuple[bool]
Signal = list[float]


class SilentResourceAgent[MECHANISM_STATE](
    ResourceAgent[
        ResourceWorldState,
        MECHANISM_STATE,
        PolicyState,
        Resolution[Outcome],
    ]
):
    """Agent that reports no signal (zero) for each outcome."""

    def _initialize_policy(
        self,
        world_dynamics: ResourceWorldDynamics,
        mechanism_dynamics: object,
        rng: np.random.Generator,
    ) -> PolicyState:
        return None

    def get_observation(
        self,
        agent_id: int,
        agent_state: AgentState[int, PolicyState],
        world_state: ResourceWorldState,
        mechanism_state: MECHANISM_STATE,
        population_state: PopulationState[int, PolicyState],
        rng: np.random.Generator,
    ) -> ResourceAgentObservation:
        return ResourceAgentObservation(
            resource_capacities=world_state.resource_capacities,
        )

    def _get_action(
        self,
        agent_state: AgentState[int, PolicyState],
        outcomes: list[Outcome],
        observation: ResourceAgentObservation,
        rng: np.random.Generator,
    ) -> list[float]:
        return [0.0] * len(outcomes)

    def adapt(
        self,
        previous: AgentState[int, PolicyState],
        observation: ResourceAgentObservation,
        resolution: Resolution[Outcome],
        reward: float,
        timestep: int,
        rng: np.random.Generator,
    ) -> AgentState[int, PolicyState]:
        return previous
