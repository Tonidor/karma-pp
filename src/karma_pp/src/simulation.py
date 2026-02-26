from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from karma_pp.src.population import Population
from karma_pp.src.types import PopulationState
from karma_pp.src.world import World
from karma_pp.src.mechanism import Mechanism
from karma_pp.utils.loading_utils import Config, instantiate

log = structlog.get_logger(__name__)


@dataclass
class Result[REPORT, WORLD_STATE, PRIVATE_STATE, POLICY_STATE, MECHANISM_STATE]:
    report: REPORT | None
    rewards: dict[int, float | None]
    world_state: WORLD_STATE
    population_state: PopulationState[PRIVATE_STATE, POLICY_STATE]
    mechanism_state: MECHANISM_STATE


def create_components(
    world_cfg: Config,
    mechanism_cfg: Config,
    population_params: dict,
) -> tuple[World, Mechanism, Population]:
    log.debug("creating_components", world_code=world_cfg["code"], mechanism_code=mechanism_cfg["code"])
    world = instantiate(world_cfg["code"], world_cfg["parameters"])
    mechanism = instantiate(mechanism_cfg["code"], mechanism_cfg["parameters"])
    population = Population(population_params)
    log.debug("components_instantiated")
    return world, mechanism, population


def run_simulation(
    world: World,
    population: Population,
    mechanism: Mechanism,
    steps: int,
    seed: int,
) -> list[Result[Any, Any, Any, Any, Any]]:
    log.info("simulation_starting", steps=steps, seed=seed)
    rng = np.random.default_rng(seed)

    log.debug("initializing_states")
    world_state, world_dynamics = world.initialize(
        n_agents=population.n_agents,
        rng=rng,
    )
    mechanism_state, mechanism_dynamics = mechanism.initialize(
        agent_weights=population.agent_weights,
        rng=rng,
    )
    population_state = population.initialize(
        world_dynamics=world_dynamics, 
        mechanism_dynamics=mechanism_dynamics, 
        rng=rng,
    )
    results = [Result(
        report=None,  # No report at initial state
        rewards={agent_id: None for agent_id in population.agent_ids},
        world_state=world_state,
        population_state=population_state,
        mechanism_state=mechanism_state,
    )]
    log.info("states_initialized")

    observations = population.get_observations(
        population_state=population_state,
        world_state=world_state,
        mechanism_state=mechanism_state,
        rng=rng,
    )
    for t in range(1, steps + 1):
        log.info("timestep", timestep=t, private_states=[s.private for s in population_state.agent_states.values()])

        # Act
        agent_ids = list(population_state.agent_states.keys())
        agent_actions = population.get_actions(
            population_state=population_state,
            observations=observations,
            rng=rng,
        )
        collective_action = world.filter_actions(
            world_state=world_state,
            agent_actions=agent_actions,
            agent_ids=agent_ids,
        )
        report = mechanism.run(
            mechanism_state=mechanism_state,
            collective_action=collective_action,
            rng=rng,
        )

        # Update mechanism and world states
        mechanism_state = mechanism.update_state(
            previous=mechanism_state,
            report=report,
            rng=rng,
        )
        world_state = world.update_state(
            previous=world_state,
            report=report,
            rng=rng,
        )

        # Reward and update population state
        resolutions = mechanism.get_resolutions(
            mechanism_state=mechanism_state,
            collective_action=collective_action,
            report=report,
        )
        rewards = population.get_rewards(
            population_state=population_state,
            observations=observations,
            resolutions=resolutions,
        )
        population_state = population.update_state(
            previous=population_state,
            observations=observations,
            resolutions=resolutions,
            rng=rng,
        )

        # Observe and adapt
        observations = population.get_observations(
            population_state=population_state,
            world_state=world_state,
            mechanism_state=mechanism_state,
            rng=rng,
        )
        population_state = population.adapt(
            population_state=population_state,
            observations=observations,
            resolutions=resolutions,
            rewards=rewards,
            timestep=t,
            rng=rng,
        )

        result = Result(
            report=report,
            rewards=rewards,
            world_state=world_state,
            population_state=population_state,
            mechanism_state=mechanism_state,
        )
        results.append(result)
        log.debug("timestep_complete")

    log.info("simulation_finished", total_steps=len(results) - 1)
    return results
