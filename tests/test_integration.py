"""End-to-end integration tests: run full simulations via create_components."""

import pathlib

import numpy as np
import pytest
import yaml

from karma_pp.core.simulation import create_components, run_simulation


DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


def _load_scenario(name: str):
    scenario_path = DATA_DIR / "scenarios" / f"{name}.yaml"
    with open(scenario_path) as f:
        cfg = yaml.safe_load(f)
    scenario_dir = scenario_path.parent
    with open(scenario_dir / cfg["world_file"]) as f:
        world_cfg = yaml.safe_load(f)
    with open(scenario_dir / cfg["mechanism_file"]) as f:
        mech_cfg = yaml.safe_load(f)
    return world_cfg, mech_cfg, cfg["population"]


@pytest.mark.parametrize("scenario_name", [
    "2x2_benevolent_dictator",
    "2x2_randoms",
    "2x2_q_learners",
    "2x2_optimal_bidders",
    "5x3_benevolent_dictator",
    "5x3_randoms",
    "5x3_q_learners",
    "5x3_optimal_bidders",
])
def test_scenario_runs_without_error(scenario_name):
    world_cfg, mech_cfg, pop_params = _load_scenario(scenario_name)
    world, mechanism, population = create_components(world_cfg, mech_cfg, pop_params)
    results = run_simulation(world, population, mechanism, steps=10, seed=42)
    assert len(results) == 11  # initial state + 10 steps


@pytest.mark.parametrize("scenario_name", [
    "2x2_benevolent_dictator",
    "2x2_randoms",
    "2x2_q_learners",
    "2x2_optimal_bidders",
])
def test_results_are_deterministic(scenario_name):
    """Same seed must produce identical rewards."""
    world_cfg, mech_cfg, pop_params = _load_scenario(scenario_name)

    def _run(seed):
        world, mechanism, population = create_components(world_cfg, mech_cfg, pop_params)
        return run_simulation(world, population, mechanism, steps=5, seed=seed)

    r1 = _run(0)
    r2 = _run(0)

    for step_a, step_b in zip(r1[1:], r2[1:]):
        for agent_id in step_a.rewards:
            assert step_a.rewards[agent_id] == pytest.approx(step_b.rewards[agent_id])


@pytest.mark.parametrize("scenario_name", ["2x2_randoms", "2x2_q_learners"])
def test_karma_is_conserved_throughout_run(scenario_name):
    """Total karma must be identical at every step of a karma-mechanism run."""
    world_cfg, mech_cfg, pop_params = _load_scenario(scenario_name)
    world, mechanism, population = create_components(world_cfg, mech_cfg, pop_params)
    results = run_simulation(world, population, mechanism, steps=20, seed=7)

    initial_total = sum(results[0].mechanism_state.agent_balances.values())
    for result in results[1:]:
        total = sum(result.mechanism_state.agent_balances.values())
        assert total == initial_total, f"Karma not conserved: {initial_total} → {total}"


def test_initial_result_has_no_report():
    world_cfg, mech_cfg, pop_params = _load_scenario("2x2_randoms")
    world, mechanism, population = create_components(world_cfg, mech_cfg, pop_params)
    results = run_simulation(world, population, mechanism, steps=3, seed=0)
    assert results[0].report is None


def test_all_agents_receive_rewards_each_step():
    world_cfg, mech_cfg, pop_params = _load_scenario("2x2_randoms")
    world, mechanism, population = create_components(world_cfg, mech_cfg, pop_params)
    results = run_simulation(world, population, mechanism, steps=5, seed=1)
    expected_ids = set(population.agent_ids)
    for result in results[1:]:
        assert set(result.rewards.keys()) == expected_ids
