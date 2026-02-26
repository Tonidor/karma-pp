from dataclasses import dataclass

import numpy as np

from karma_pp.impl.agents.resource_agent import ResourceAgentObservation, ResourceAgent
from karma_pp.impl.mechanisms.benevolent_dictator import DictatorResolution
from karma_pp.impl.worlds.resource_world.resource_world import ResourceWorldDynamics, ResourceWorldState
from karma_pp.src.types import AgentState, PopulationState

PolicyState = None
Outcome = tuple[bool]
Reward = float  # Signal: rewards per outcome

class TruthfulResourceAgent[MECHANISM_STATE](
    ResourceAgent[
        ResourceWorldState,
        MECHANISM_STATE,
        PolicyState,
        DictatorResolution,
    ]
):
    """Truthful agent that reports rewards for benevolent dictator."""

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
    ) -> list[Reward]:
        urgency = float(agent_state.private)
        rewards: list[Reward] = []
        for outcome in outcomes:
            outcome_arr = np.array(outcome)
            resource_rewards = self.reward_per_resource * outcome_arr
            resource_penalties = self.no_resource_penalty * (1 - outcome_arr)
            reward = urgency * sum(resource_rewards + resource_penalties)
            rewards.append(reward)
        return rewards

    def adapt(
        self,
        previous: AgentState[int, PolicyState],
        observation: ResourceAgentObservation,
        resolution: DictatorResolution,
        reward: float,
        timestep: int,
        rng: np.random.Generator,
    ) -> AgentState[int, PolicyState]:
        return previous
