from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from karma_pp.core.population import Population
from karma_pp.core.types import PopulationState
from karma_pp.core.world import World
from karma_pp.core.mechanism import Mechanism
from karma_pp.utils.loading_utils import Config, instantiate

log = structlog.get_logger(__name__)


@dataclass
class Result[REPORT, WORLD_STATE, PRIVATE_STATE, POLICY_STATE, MECHANISM_STATE]:
    collectives: dict[int, list[int]]  # collective_id -> agent_ids
    reports: dict[int, REPORT | None]  # collective_id -> report
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
    log.info("states_initialized")

    prev_resolutions = {agent_id: None for agent_id in population.agent_ids}
    prev_rewards = {agent_id: None for agent_id in population.agent_ids}
    results = [Result(
        collectives={0: population.agent_ids},
        reports={0: None},
        rewards=prev_rewards,
        world_state=world_state,
        population_state=population_state,
        mechanism_state=mechanism_state,
    )]

    for t in range(1, steps + 1):
        log.info("timestep", timestep=t, private_states=[s.private for s in population_state.agent_states.values()])

        # Form collectives
        collectives: dict[int, list[int]] = world.get_collectives(
            world_state=world_state,
            agent_ids=population_state.agent_ids,
        )

        # Observe
        observations = population.get_observations(
            population_state=population_state,
            world_state=world_state,
            mechanism_state=mechanism_state,
            collectives=collectives,
            rng=rng,
        )

        # Adapt
        population_state = population.adapt(
            population_state=population_state,
            observations=observations,
            resolutions=prev_resolutions,
            rewards=prev_rewards,
            timestep=t,
            rng=rng,
        )

        # Plan
        agent_actions = population.get_actions(
            population_state=population_state,
            observations=observations,
            rng=rng,
        )

        # Coordinate
        reports = {}
        resolutions = {}
        for collective_id, collective in collectives.items():

            collective_action = world.filter_actions(
                world_state=world_state,
                agent_actions=agent_actions,
                collective=collective,
            )
            report = mechanism.run(
                mechanism_state=mechanism_state,
                collective_action=collective_action,
                rng=rng,
            )
            agent_resolutions = mechanism.get_resolutions(
                mechanism_state=mechanism_state,
                collective_action=collective_action,
                report=report,
            )
            reports[collective_id] = report
            resolutions.update(agent_resolutions)

        # Update
        mechanism_state = mechanism.update_state(
            previous=mechanism_state,
            reports=reports,
            rng=rng,
        )
        world_state = world.update_state(
            previous=world_state,
            reports=reports,
            rng=rng,
        )

        # Act
        rewards: dict[int, float] = population.get_rewards(
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

        # Store results
        prev_resolutions = resolutions
        prev_rewards = rewards
        result = Result(
            collectives=collectives,
            reports=reports,
            rewards=rewards,
            world_state=world_state,
            population_state=population_state,
            mechanism_state=mechanism_state,
        )
        results.append(result)

        log.debug("timestep_complete")

    log.info("simulation_finished", total_steps=len(results) - 1)
    return results
