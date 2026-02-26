from typing import Any

import numpy as np
import structlog

from karma_pp.src.agent import AgentModel
from karma_pp.src.types import AgentState, PopulationState
from karma_pp.utils.loading_utils import instantiate

log = structlog.get_logger(__name__)


class Population[
    AGENT_PRIVATE,
    POLICY_STATE,
    WORLD_STATE,
    MECHANISM_STATE,
    OBSERVATION,
    RESOLUTION,
    OUTCOME,
    SIGNAL,
]:
    """Manages the agent states and their shared behavior logic."""

    def __init__(self, agent_type_cfgs: dict[str, dict[str, Any]]):
        self.agent_type_cfgs = agent_type_cfgs

        self.agent_ids: list[int] = []
        self.agent_weights: dict[int, int] = {}
        self.model_registry: dict[str, AgentModel] = {}
        self.agent_model_map: dict[int, str] = {}

        agent_id_counter = 0
        for model_id, cfg in agent_type_cfgs.items():
            self.agent_ids.extend([agent_id_counter + i for i in range(cfg["n_agents"])])
            self.agent_weights.update(
                {agent_id_counter + i: int(cfg["weight"]) for i in range(cfg["n_agents"])}
            )
            model_cfg = cfg["agent_model"]
            shared_model = instantiate(
                model_cfg["code"],
                model_cfg["parameters"],
            )
            self.model_registry[model_id] = shared_model
            self.agent_model_map.update({agent_id_counter + i: model_id for i in range(cfg["n_agents"])})
            agent_id_counter += cfg["n_agents"]
        log.debug("population_initialized", n_agents=len(self.agent_ids), agent_models=self.model_registry.keys())

    @property
    def n_agents(self) -> int:
        """Number of agents in the population."""
        return len(self.agent_ids)

    def initialize(
        self,
        world_dynamics: object,
        mechanism_dynamics: object,
        rng: np.random.Generator,
    ) -> PopulationState[AGENT_PRIVATE, POLICY_STATE]:
        """Create initial population state."""
        log.debug("population_state_initializing")
        agent_states: dict[int, AgentState[AGENT_PRIVATE, POLICY_STATE]] = {}

        for agent_id, model_id in self.agent_model_map.items():
            model = self.model_registry[model_id]
            agent_state = model.initialize(
                world_dynamics=world_dynamics,
                mechanism_dynamics=mechanism_dynamics,
                rng=rng,
            )
            agent_states[agent_id] = agent_state
        log.debug("population_state_initialized")
        return PopulationState[AGENT_PRIVATE, POLICY_STATE](agent_states)

    def get_observations(
        self,
        population_state: PopulationState[AGENT_PRIVATE, POLICY_STATE],
        world_state: WORLD_STATE,
        mechanism_state: MECHANISM_STATE,
        rng: np.random.Generator,
    ) -> dict[int, OBSERVATION]:
        """Get the observations for the population."""
        observations: dict[int, OBSERVATION] = {}
        for agent_id, state in population_state.agent_states.items():
            model = self.model_registry[self.agent_model_map[agent_id]]
            observations[agent_id] = model.get_observation(
                agent_id=agent_id,
                agent_state=state,
                world_state=world_state,
                mechanism_state=mechanism_state,
                population_state=population_state,
                rng=rng,
            )
        return observations

    def get_actions(
        self,
        population_state: PopulationState[AGENT_PRIVATE, POLICY_STATE],
        observations: dict[int, OBSERVATION],
        rng: np.random.Generator,
    ) -> list[list[tuple[OUTCOME, SIGNAL]]]:
        """Get per-agent list of actions; each action is (outcome, signal)."""
        actions: list[list[tuple[OUTCOME, SIGNAL]]] = []
        for agent_id, state in population_state.agent_states.items():
            model = self.model_registry[self.agent_model_map[agent_id]]
            actions.append(
                model.get_action(
                    agent_state=state,
                    observation=observations[agent_id],
                    rng=rng,
                )
            )
        for i, (agent_id, state) in enumerate(population_state.agent_states.items()):
            log.info(
                "get_actions",
                agent_id=agent_id,
                actions=actions[i],
            )
        return actions

    def get_rewards(
        self,
        population_state: PopulationState[AGENT_PRIVATE, POLICY_STATE],
        observations: dict[int, OBSERVATION],
        resolutions: dict[int, RESOLUTION],
    ) -> dict[int, float]:
        """Get the rewards for the population."""
        rewards: dict[int, float] = {}
        for agent_id, state in population_state.agent_states.items():
            model = self.model_registry[self.agent_model_map[agent_id]]
            rewards[agent_id] = model.compute_reward(
                agent_state=state,
                observation=observations[agent_id],
                resolution=resolutions[agent_id],
            )
        log.info("get_rewards", rewards=rewards)
        return rewards
    
    def update_state(
        self,
        previous: PopulationState[AGENT_PRIVATE, POLICY_STATE],
        observations: dict[int, OBSERVATION],
        resolutions: dict[int, RESOLUTION],
        rng: np.random.Generator,
    ) -> PopulationState[AGENT_PRIVATE, POLICY_STATE]:
        """Update the population state."""
        new_agent_states: dict[int, AgentState[AGENT_PRIVATE, POLICY_STATE]] = {}
        for agent_id, state in previous.agent_states.items():
            model = self.model_registry[self.agent_model_map[agent_id]]
            new_state = model.update_state(
                previous=state,
                observation=observations[agent_id],
                resolution=resolutions[agent_id],
                rng=rng,
            )
            new_agent_states[agent_id] = new_state
        private_agent_states = [s.private for s in new_agent_states.values()]
        log.info(
            "population_state_updated",
            private_agent_states=private_agent_states,
        )
        return PopulationState(new_agent_states)

    def adapt(
        self,
        population_state: PopulationState[AGENT_PRIVATE, POLICY_STATE],
        observations: dict[int, OBSERVATION],
        resolutions: dict[int, RESOLUTION],
        rewards: dict[int, float],
        timestep: int,
        rng: np.random.Generator,
    ) -> PopulationState[AGENT_PRIVATE, POLICY_STATE]:
        """Adapt the population to the observation."""
        new_agent_states: dict[int, AgentState[AGENT_PRIVATE, POLICY_STATE]] = {}
        for agent_id, state in population_state.agent_states.items():
            model = self.model_registry[self.agent_model_map[agent_id]]
            resolution = resolutions[agent_id]
            new_state = model.adapt(
                previous=state,
                observation=observations[agent_id],
                resolution=resolution,
                reward=rewards[agent_id],
                timestep=timestep,
                rng=rng,
            )
            new_agent_states[agent_id] = new_state
        return PopulationState[AGENT_PRIVATE, POLICY_STATE](new_agent_states)
