import copy
from pathlib import Path

import click
import numpy as np
import structlog
import yaml

from karma_pp.logging_config import configure_logging
from karma_pp.src.simulation import create_components, run_simulation
from karma_pp.utils.agent_measures import (
    get_markov_lambda_star,
    get_markov_spike_index,
    get_markov_stationary_distributions,
    get_markov_surplus_efficiency,
    get_markov_threshold,
)
from karma_pp.utils.system_measures import (
    get_access_fairness,
    get_efficiency,
    nash_welfare,
)
from karma_pp.utils.plots import (
    plot_access_fairness_vs_efficiency,
    plot_efficiency_fairness_comparison,
    plot_markov_stationary_violin,
    plot_metrics_table,
    plot_nash_welfare,
    plot_smoothed_reward_over_time,
)

log = structlog.get_logger(__name__)


def _default_labels(scenarios: tuple[str, ...]) -> list[str]:
    return [Path(scenario).stem for scenario in scenarios]


def _extract_markov_urgency_inputs(
    population_cfg: dict,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """Expand agent type configs into per-agent transition matrices, weights, and urgency levels."""
    try:
        agent_type_cfgs = population_cfg["parameters"]["agent_type_cfgs"]
    except KeyError as e:
        raise click.ClickException(
            f"Population config missing expected key for Markov metrics: {e}."
        ) from e

    transition_matrices: list[np.ndarray] = []
    weights: list[float] = []
    urgency_levels: list[np.ndarray] = []
    for type_name, type_cfg in agent_type_cfgs.items():
        try:
            n_agents = int(type_cfg["n_agents"])
            weight = float(type_cfg["weight"])
            transition_matrix = np.asarray(
                type_cfg["agent_model"]["parameters"]["transition_matrix"],
                dtype=float,
            )
        except KeyError as e:
            raise click.ClickException(
                f"Agent type {type_name!r} missing transition matrix config key: {e}. "
                "Expected agent_model.parameters.transition_matrix."
            ) from e
        levels = type_cfg["agent_model"]["parameters"].get("urgency_levels")
        if levels is None:
            raise click.ClickException(
                f"Agent type {type_name!r} missing required key "
                "'agent_model.parameters.urgency_levels'."
            )
        levels_array = np.asarray(levels, dtype=float)
        if levels_array.shape[0] != transition_matrix.shape[0]:
            raise click.ClickException(
                f"Agent type {type_name!r} has urgency_levels of length {levels_array.shape[0]} "
                f"but transition matrix has {transition_matrix.shape[0]} states."
            )
        for _ in range(n_agents):
            transition_matrices.append(transition_matrix)
            weights.append(weight)
            urgency_levels.append(levels_array)

    if not transition_matrices:
        raise click.ClickException("No agent transition matrices found in population config.")
    return transition_matrices, np.asarray(weights, dtype=float), urgency_levels


def _extract_step_arrays(
    results: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract access, reward, urgency, and allocated-agent arrays from results.

    Rewards are taken from result.rewards (agent compute_reward), not recomputed.
    Access and urgencies are derived from report selected_outcomes and population_state.
    Returns allocation_row_indices so urgencies can be correctly sliced for nash_welfare.
    """
    if not results:
        raise ValueError("No simulation results available.")

    n_agents = results[0].population_state.n_agents
    access_by_step: list[list[float]] = []
    rewards_by_step: list[list[float]] = []
    urgencies_by_step: list[list[float]] = []
    allocation: list[int] = []
    allocation_row_indices: list[int] = []

    for result in results[1:]:
        report = result.report
        if report is None:
            continue

        selected_outcomes = getattr(report, "selected_outcomes", None)
        if not isinstance(selected_outcomes, dict) or len(selected_outcomes) != n_agents:
            continue

        # Rewards from results (agent compute_reward), in stable agent id order
        agent_ids = sorted(result.population_state.agent_states.keys())
        if None in (result.rewards.get(aid) for aid in agent_ids):
            continue
        rewards_row = [float(result.rewards[aid]) for aid in agent_ids]
        urgency_values = [float(result.population_state.agent_states[aid].private) for aid in agent_ids]
        access_row: list[float] = []
        urgency_row: list[float] = []
        allocated_agent = None
        outcomes_list = [selected_outcomes.get(aid) for aid in agent_ids]
        if None in outcomes_list:
            continue
        for idx, (urgency_value, outcome) in enumerate(zip(urgency_values, outcomes_list)):
            outcome_array = np.asarray(outcome, dtype=float)
            has_access = bool(np.any(outcome_array > 0))
            access_row.append(1.0 if has_access else 0.0)
            if has_access and allocated_agent is None:
                allocated_agent = idx
            urgency_row.append(float(urgency_value))

        row_idx = len(access_by_step)
        access_by_step.append(access_row)
        rewards_by_step.append(rewards_row)
        urgencies_by_step.append(urgency_row)
        if allocated_agent is not None:
            allocation.append(allocated_agent)
            allocation_row_indices.append(row_idx)

    if not access_by_step or not rewards_by_step:
        raise ValueError("Could not extract plot data from simulation results.")

    urgencies_arr = np.asarray(urgencies_by_step, dtype=float)
    allocation_arr = np.asarray(allocation, dtype=int)
    allocation_indices_arr = np.asarray(allocation_row_indices, dtype=int)
    allocated_urgencies = (
        urgencies_arr[allocation_indices_arr, :] if allocation_indices_arr.size > 0 else np.empty((0, n_agents))
    )
    return (
        np.asarray(access_by_step, dtype=float),
        np.asarray(rewards_by_step, dtype=float),
        urgencies_arr,
        allocation_arr,
        allocated_urgencies,
    )


def _extract_rewards_from_results(results: list) -> np.ndarray:
    """Extract per-agent rewards per timestep from simulation results (result.rewards)."""
    if not results or len(results) < 2:
        raise ValueError("Need at least two results (initial + one step).")
    agent_ids = sorted(results[1].rewards.keys())
    rows = []
    for result in results[1:]:
        row = [result.rewards[aid] for aid in agent_ids]
        rows.append(row)
    return np.asarray(rows, dtype=float)


def _compute_markov_metrics(
    world_cfg: dict,
    population_cfg: dict,
    lambda_star: float | None,
    delta_r: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[float], list[float], float]:
    """Compute Markov metrics: stationary, urgency_levels, thresholds, spike_indices, surplus_efficiency, lambda_star."""
    try:
        resource_count = float(len(world_cfg["parameters"]["resource_capacities"]))
    except KeyError as e:
        raise click.ClickException(
            f"World config missing expected key for resource_capacities: {e}."
        ) from e

    transition_matrices, weights, urgency_levels = _extract_markov_urgency_inputs(population_cfg)
    stationary = get_markov_stationary_distributions(transition_matrices)
    lambda_star_used = (
        float(lambda_star)
        if lambda_star is not None
        else get_markov_lambda_star(
            transition_matrices=transition_matrices,
            weights=weights,
            resource_count=resource_count,
            urgency_levels=urgency_levels,
        )
    )
    thresholds = [
        get_markov_threshold(weight=float(w_i), lambda_star=lambda_star_used)
        for w_i in weights
    ]
    spike_indices = [
        get_markov_spike_index(
            stationary_distribution=pi_i,
            urgency_levels=u_i,
            threshold=t_i,
        )
        for pi_i, u_i, t_i in zip(stationary, urgency_levels, thresholds)
    ]
    surplus_efficiency = [
        get_markov_surplus_efficiency(
            stationary_distribution=pi_i,
            urgency_levels=u_i,
            threshold=t_i,
            delta_r=delta_r,
        )
        for pi_i, u_i, t_i in zip(stationary, urgency_levels, thresholds)
    ]
    return stationary, urgency_levels, thresholds, spike_indices, surplus_efficiency, lambda_star_used


def _load_scenario_cfgs(scenario: str) -> tuple[dict, dict, dict]:
    """Load world, population, and mechanism configs directly from a scenario YAML."""
    with open(scenario, "r") as f:
        scenario_cfg = yaml.safe_load(f)

    with open(scenario_cfg["world_file"], "r") as f:
        world_cfg = yaml.safe_load(f)
    with open(scenario_cfg["mechanism_file"], "r") as f:
        mechanism_cfg = yaml.safe_load(f)

    population_cfg = {
        "code": "karma_pp.src.population.Population",
        "parameters": {
            "agent_type_cfgs": scenario_cfg["population"],
        },
    }
    return world_cfg, population_cfg, mechanism_cfg


def _mean_efficiency_and_fairness(
    world_cfg: dict,
    population_cfg: dict,
    mechanism_cfg: dict,
    steps: int,
    n_runs: int,
    seed_offset: int,
    seed_base: int,
) -> tuple[float, float]:
    effs: list[float] = []
    fairs: list[float] = []
    for r in range(n_runs):
        eff, fair = _run_scenario_and_metrics(
            world_cfg,
            population_cfg,
            mechanism_cfg,
            steps,
            seed_base + seed_offset + r,
        )
        effs.append(eff)
        fairs.append(fair)
    return float(np.mean(effs)), float(np.mean(fairs))


def _run_efficiency_fairness_comparison(
    random_scenario: str,
    q_learner_scenario: str,
    optimal_scenario: str,
    dictator_scenario: str,
    steps: int,
    n_runs: int,
    seed_base: int,
    save_path: str,
) -> None:
    """Run random, optimal, dictator, and Q variants; plot mean efficiency vs fairness."""
    world_cfg, random_pop_cfg, mechanism_cfg = _load_scenario_cfgs(random_scenario)

    eff_list: list[float] = []
    fair_list: list[float] = []
    labels_list: list[str] = []

    log.info("compare_runs", variant="Random", n_runs=n_runs, steps=steps)
    effs, fairs = [], []
    for r in range(n_runs):
        eff, fair = _run_scenario_and_metrics(
            world_cfg, random_pop_cfg, mechanism_cfg, steps, seed_base + r
        )
        effs.append(eff)
        fairs.append(fair)
    eff_list.append(float(np.mean(effs)))
    fair_list.append(float(np.mean(fairs)))
    labels_list.append("Random")

    optimal_world_cfg, optimal_pop_cfg, optimal_mechanism_cfg = _load_scenario_cfgs(optimal_scenario)

    log.info("compare_runs", variant="Optimal", n_runs=n_runs, steps=steps)
    effs, fairs = [], []
    for r in range(n_runs):
        seed = seed_base + 1_000 + r
        eff, fair = _run_scenario_and_metrics(
            optimal_world_cfg, optimal_pop_cfg, optimal_mechanism_cfg, steps, seed
        )
        effs.append(eff)
        fairs.append(fair)
    eff_list.append(float(np.mean(effs)))
    fair_list.append(float(np.mean(fairs)))
    labels_list.append("Optimal")

    dictator_world_cfg, dictator_pop_cfg, dictator_mechanism_cfg = _load_scenario_cfgs(dictator_scenario)

    log.info("compare_runs", variant="Benevolent Dictator", n_runs=n_runs, steps=steps)
    effs, fairs = [], []
    for r in range(n_runs):
        seed = seed_base + 5_000 + r
        eff, fair = _run_scenario_and_metrics(
            dictator_world_cfg, dictator_pop_cfg, dictator_mechanism_cfg, steps, seed
        )
        effs.append(eff)
        fairs.append(fair)
    eff_list.append(float(np.mean(effs)))
    fair_list.append(float(np.mean(fairs)))
    labels_list.append("Benevolent Dictator")

    q_world_cfg, q_pop_base, q_mechanism_cfg = _load_scenario_cfgs(q_learner_scenario)

    for gamma_idx, gamma in enumerate(_COMPARE_GAMMAS):
        population_cfg = copy.deepcopy(q_pop_base)
        try:
            population_cfg["parameters"]["agent_type_cfgs"]["q_learner"]["agent_model"]["parameters"]["gamma"] = gamma
        except KeyError as e:
            raise click.ClickException(
                f"Q-learner population config missing expected key for gamma: {e}. "
                "Expected agent_type_cfgs.q_learner.agent_model.parameters.gamma"
            ) from e
        log.info("compare_runs", variant=f"Q γ={gamma}", n_runs=n_runs, steps=steps)
        effs, fairs = [], []
        for r in range(n_runs):
            seed = seed_base + 10_000 + gamma_idx * 100 + r
            eff, fair = _run_scenario_and_metrics(
                q_world_cfg, population_cfg, q_mechanism_cfg, steps, seed
            )
            effs.append(eff)
            fairs.append(fair)
        eff_list.append(float(np.mean(effs)))
        fair_list.append(float(np.mean(fairs)))
        labels_list.append(f"Q γ={gamma:.2f}")

    plot_efficiency_fairness_comparison(
        access_fairness=fair_list,
        efficiency=eff_list,
        labels=labels_list,
        save_path=save_path,
    )
    log.info("compare_plot_saved", path=save_path)


@click.command()
@click.argument(
    "plot_name",
    type=click.Choice([
        "access_fairness_vs_efficiency",
        "nash_welfare",
        "reward_over_time",
        "efficiency_fairness_comparison",
        "markov_urgency_metrics",
        "markov_urgency_violin",
        "metrics_table",
    ]),
)
@click.option("--scenario", "scenarios", multiple=True, type=click.Path(exists=True))
@click.option("--label", "labels", multiple=True, type=str)
@click.option("--steps", type=int, default=1000)
@click.option("--seed", type=int, default=42)
@click.option("--window", type=int, default=50, help="Rolling window for reward_over_time smoothing.")
@click.option(
    "--n-runs",
    type=int,
    default=3,
    help="For efficiency_fairness_comparison/metrics_table: runs per variant.",
)
@click.option(
    "--seed-base",
    type=int,
    default=42,
    help="For efficiency_fairness_comparison/metrics_table: base seed for runs.",
)
@click.option(
    "--delta-r",
    type=float,
    default=1.0,
    help="For markov_urgency_metrics: reward gain factor Delta r.",
)
@click.option(
    "--lambda-star",
    type=float,
    default=None,
    help="For markov_urgency_metrics/markov_urgency_violin: optional lambda* override.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(),
    default=None,
    help="Output path for plot (default depends on plot type).",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
)
def vis(
    plot_name: str,
    scenarios: tuple[str, ...],
    labels: tuple[str, ...],
    steps: int,
    seed: int,
    window: int,
    n_runs: int,
    seed_base: int,
    delta_r: float,
    lambda_star: float | None,
    out_path: str | None,
    log_level: str,
):
    """Create visualizations from simulation outputs."""
    configure_logging(level=log_level)

    if plot_name == "efficiency_fairness_comparison":
        if len(scenarios) < 4:
            raise click.ClickException(
                f"{plot_name} requires 4 --scenario values (random, optimal, dictator, q-learner in that order)"
            )
        random_scenario, optimal_scenario, dictator_scenario, q_learner_scenario = scenarios[0], scenarios[1], scenarios[2], scenarios[3]
        save_path = out_path or "data/plots/efficiency_fairness_comparison.png"
        _run_efficiency_fairness_comparison(
            random_scenario=random_scenario,
            q_learner_scenario=q_learner_scenario,
            optimal_scenario=optimal_scenario,
            dictator_scenario=dictator_scenario,
            steps=steps,
            n_runs=n_runs,
            seed_base=seed_base,
            save_path=save_path,
        )
        return

    if plot_name == "metrics_table":
        if len(scenarios) < 3:
            raise click.ClickException(
                f"{plot_name} requires 3 --scenario values (random, q-learner, optimal in that order)"
            )
        random_scenario, q_learner_scenario, optimal_scenario = scenarios[0], scenarios[1], scenarios[2]
        random_world_cfg, random_pop_cfg, random_mech_cfg = _load_scenario_cfgs(random_scenario)
        q_world_cfg, q_pop_cfg, q_mech_cfg = _load_scenario_cfgs(q_learner_scenario)
        optimal_world_cfg, optimal_pop_cfg, optimal_mech_cfg = _load_scenario_cfgs(optimal_scenario)

        eff_random, fair_random = _mean_efficiency_and_fairness(
            random_world_cfg,
            random_pop_cfg,
            random_mech_cfg,
            steps,
            n_runs,
            0,
            seed_base,
        )
        eff_q, fair_q = _mean_efficiency_and_fairness(
            q_world_cfg,
            q_pop_cfg,
            q_mech_cfg,
            steps,
            n_runs,
            1_000,
            seed_base,
        )
        eff_opt, fair_opt = _mean_efficiency_and_fairness(
            optimal_world_cfg,
            optimal_pop_cfg,
            optimal_mech_cfg,
            steps,
            n_runs,
            2_000,
            seed_base,
        )

        def _markov_means(world_cfg: dict, population_cfg: dict) -> tuple[float, float, float]:
            _, _, _, spike, surplus, lambda_used = _compute_markov_metrics(
                world_cfg, population_cfg, lambda_star, delta_r
            )
            return float(np.mean(spike)), float(np.mean(surplus)), float(lambda_used)

        spike_random, surplus_random, lambda_random = _markov_means(random_world_cfg, random_pop_cfg)
        spike_q, surplus_q, lambda_q = _markov_means(q_world_cfg, q_pop_cfg)
        spike_opt, surplus_opt, lambda_opt = _markov_means(optimal_world_cfg, optimal_pop_cfg)

        row_labels = [
            "Efficiency",
            "Fairness",
            "Spike index",
            "Surplus efficiency",
            "Lambda*",
        ]
        column_labels = ["Random", "Q-learning", "Optimal"]
        cell_text = [
            [f"{eff_random:.4f}", f"{eff_q:.4f}", f"{eff_opt:.4f}"],
            [f"{fair_random:.4f}", f"{fair_q:.4f}", f"{fair_opt:.4f}"],
            [f"{spike_random:.4f}", f"{spike_q:.4f}", f"{spike_opt:.4f}"],
            [f"{surplus_random:.4f}", f"{surplus_q:.4f}", f"{surplus_opt:.4f}"],
            [f"{lambda_random:.4f}", f"{lambda_q:.4f}", f"{lambda_opt:.4f}"],
        ]
        save_path = out_path or "data/plots/metrics_table.png"
        plot_metrics_table(
            row_labels=row_labels,
            column_labels=column_labels,
            cell_text=cell_text,
            title="Random vs Q-learning vs Optimal",
            save_path=save_path,
        )
        log.info("metrics_table_saved", path=save_path)
        return

    if plot_name in ("access_fairness_vs_efficiency", "nash_welfare", "reward_over_time", "markov_urgency_metrics", "markov_urgency_violin"):
        if not scenarios:
            raise click.ClickException(
                f"{plot_name} requires at least one --scenario"
            )

    if labels and len(labels) != len(scenarios):
        raise click.ClickException("--label must be provided once per --scenario.")
    scenario_labels = list(labels) if labels else _default_labels(scenarios)

    Path("data/plots").mkdir(parents=True, exist_ok=True)

    if plot_name == "markov_urgency_metrics":
        scenario_outputs: dict[str, dict] = {}
        for scenario in scenarios:
            world_cfg, population_cfg, _ = _load_scenario_cfgs(scenario)
            stationary, urgency_levels, thresholds, spike_indices, surplus_efficiency, lambda_star_used = (
                _compute_markov_metrics(world_cfg, population_cfg, lambda_star, delta_r)
            )
            resource_count = float(len(world_cfg["parameters"]["resource_capacities"]))
            n_agents = len(stationary)
            metrics = {
                "lambda_star": float(lambda_star_used),
                "lambda_source": "user" if lambda_star is not None else "computed",
                "effective_capacity": float(min(resource_count, n_agents)),
                "thresholds": [float(x) for x in thresholds],
                "spike_indices": [float(x) for x in spike_indices],
                "surplus_efficiency": [float(x) for x in surplus_efficiency],
                "stationary_distributions": [pi.tolist() for pi in stationary],
            }
            scenario_outputs[Path(scenario).stem] = metrics

        yaml_output = yaml.safe_dump(scenario_outputs, sort_keys=False)
        if out_path is not None:
            with open(out_path, "w") as f:
                f.write(yaml_output)
            log.info("markov_metrics_saved", path=out_path)
        click.echo(yaml_output)
        return

    if plot_name == "markov_urgency_violin":
        scenario = scenarios[0]
        world_cfg, population_cfg, _ = _load_scenario_cfgs(scenario)
        stationary, urgency_levels, thresholds, spike_indices, surplus_efficiency, lambda_star_used = (
            _compute_markov_metrics(world_cfg, population_cfg, lambda_star, delta_r)
        )
        save_path = out_path or "data/plots/markov_urgency_violin.png"
        plot_markov_stationary_violin(
            urgency_levels=urgency_levels,
            stationary_distributions=stationary,
            spike_indices=spike_indices,
            surplus_efficiency=surplus_efficiency,
            thresholds=thresholds,
            save_path=save_path,
        )
        log.info("markov_violin_saved", path=save_path, lambda_star=lambda_star_used)
        return

    if plot_name == "reward_over_time":
        scenario = scenarios[0]
        log.info("visualization_scenario_start", scenario=scenario, steps=steps, seed=seed)
        world_cfg, population_cfg, mechanism_cfg = _load_scenario_cfgs(scenario)
        population_params = population_cfg["parameters"]["agent_type_cfgs"]
        world, mechanism, population = create_components(world_cfg, mechanism_cfg, population_params)
        results = run_simulation(world, population, mechanism, steps, seed)
        rewards_by_step = _extract_rewards_from_results(results)
        plot_smoothed_reward_over_time(
            rewards_by_step=rewards_by_step,
            window=window,
            save_path=out_path or "data/plots/reward_over_time.png",
        )
        log.info("visualization_scenario_complete", scenario=scenario)
        return

    access_fairness_values: list[float] = []
    efficiency_values: list[float] = []
    nash_welfare_values: list[float] = []

    for scenario in scenarios:
        log.info("visualization_scenario_start", scenario=scenario, steps=steps, seed=seed)
        world_cfg, population_cfg, mechanism_cfg = _load_scenario_cfgs(scenario)
        population_params = population_cfg["parameters"]["agent_type_cfgs"]
        world, mechanism, population = create_components(world_cfg, mechanism_cfg, population_params)
        results = run_simulation(world, population, mechanism, steps, seed)

        access_by_step, rewards_by_step, _, allocation, allocated_urgencies = _extract_step_arrays(results)
        access_fairness_values.append(get_access_fairness(access_by_step))
        efficiency_values.append(get_efficiency(rewards_by_step))

        if allocation.size == 0:
            raise click.ClickException(
                f"No allocated outcomes found for scenario {scenario!r}; cannot compute nash_welfare."
            )

        agent_ids = sorted(results[0].population_state.agent_states.keys())
        weights = np.asarray([population.agent_weights[aid] for aid in agent_ids])
        nash_value, _ = nash_welfare(
            allocation=allocation,
            urgencies=allocated_urgencies,
            social_weights=weights,
        )
        nash_welfare_values.append(float(nash_value))
        log.info("visualization_scenario_complete", scenario=scenario)

    if plot_name == "access_fairness_vs_efficiency":
        save_path = out_path or "data/plots/access_fairness_vs_efficiency.png"
        plot_access_fairness_vs_efficiency(
            access_fairness=access_fairness_values,
            efficiency=efficiency_values,
            labels=scenario_labels,
            save_path=save_path,
        )
        log.info("plot_saved", path=save_path)
    elif plot_name == "nash_welfare":
        save_path = out_path or "data/plots/nash_welfare.png"
        plot_nash_welfare(
            nash_welfare_values=nash_welfare_values,
            labels=scenario_labels,
            save_path=save_path,
        )
        log.info("plot_saved", path=save_path)
    else:
        raise click.ClickException(f"Unsupported plot_name {plot_name!r}.")


def _run_scenario_and_metrics(
    world_cfg: dict,
    population_cfg: dict,
    mechanism_cfg: dict,
    steps: int,
    seed: int,
) -> tuple[float, float]:
    """Run one simulation and return (efficiency, access_fairness)."""
    population_params = population_cfg["parameters"]["agent_type_cfgs"]
    world, mechanism, population = create_components(world_cfg, mechanism_cfg, population_params)
    results = run_simulation(world, population, mechanism, steps, seed)
    access_by_step, rewards_by_step, _, _, _ = _extract_step_arrays(results)
    return get_efficiency(rewards_by_step), get_access_fairness(access_by_step)


# Discount factors (gamma) for Q-learner comparison
_COMPARE_GAMMAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98, 1.0]


if __name__ == "__main__":
    vis()
