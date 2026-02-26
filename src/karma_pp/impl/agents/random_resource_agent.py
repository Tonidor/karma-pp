from dataclasses import dataclass
from types import NoneType

import numpy as np

from karma_pp.impl.agents.resource_agent import ResourceAgentObservation, ResourceAgent
from karma_pp.impl.mechanisms.karma.karma_mechanism import KarmaDynamics, KarmaResolution, KarmaState
from karma_pp.impl.worlds.resource_world.resource_world import ResourceWorldDynamics, ResourceWorldState
from karma_pp.src.types import AgentState, PopulationState

PolicyState = NoneType
Outcome = tuple[bool]
Commit = int

@dataclass(frozen=True)
class RandomObservation(ResourceAgentObservation):
    """Observation for random bidding agent; extends base with agent_balance."""

    agent_balance: int


class RandomResourceAgent(
    ResourceAgent[
        ResourceWorldState,
        KarmaState,
        PolicyState,
        KarmaResolution,
    ]
):
    """Random agent that bids uniformly in [0, balance] for karma mechanism."""

    def _initialize_policy(
        self,
        world_dynamics: ResourceWorldDynamics,
        mechanism_dynamics: KarmaDynamics,
        rng: np.random.Generator,
    ) -> PolicyState:
        return None

    def get_observation(
        self,
        agent_id: int,
        agent_state: AgentState[int, PolicyState],
        world_state: ResourceWorldState,
        mechanism_state: KarmaState,
        population_state: PopulationState[int, PolicyState],
        rng: np.random.Generator,
    ) -> RandomObservation:
        balance = mechanism_state.agent_balances[agent_id]
        return RandomObservation(
            resource_capacities=world_state.resource_capacities,
            agent_balance=balance,
        )

    def _get_action(
        self,
        agent_state: AgentState[int, PolicyState],
        outcomes: list[Outcome],
        observation: RandomObservation,
        rng: np.random.Generator,
    ) -> list[Commit]:
        return [int(rng.integers(0, observation.agent_balance + 1)) for _ in self._get_outcomes(observation)]

    def adapt(
        self,
        previous: AgentState[int, PolicyState],
        observation: RandomObservation,
        resolution: KarmaResolution,
        reward: float,
        timestep: int,
        rng: np.random.Generator,
    ) -> AgentState[int, PolicyState]:
        return previous
