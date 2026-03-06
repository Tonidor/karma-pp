"""Visualization commands for database-backed experiments.

All plots use data from the database only (experiment IDs or group IDs).
No simulation execution is performed in this module.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from karma_pp.db.base import Database
from karma_pp.logging_config import configure_logging
from karma_pp.utils.agent_measures import (
    get_markov_lambda_star,
    get_markov_spike_index,
    get_markov_stationary_distributions,
    get_markov_surplus_efficiency,
    get_markov_threshold,
)
from karma_pp.utils.plots import (
    plot_access_fairness_vs_efficiency,
    plot_efficiency_fairness_comparison,
    plot_full_info_policy_and_distribution,
    plot_markov_stationary_violin,
    plot_metrics_table,
    plot_nash_welfare,
    plot_smoothed_reward_over_time,
)
from karma_pp.utils.system_measures import (
    get_access_fairness,
    get_efficiency,
    nash_welfare,
)


# ---------------------------------------------------------------------------
# Shared DB extraction helpers (no simulation execution)
# ---------------------------------------------------------------------------


def _validate_and_resolve_experiments(
    exp_id: Optional[int], group_id: Optional[int]
) -> Tuple[Optional[list[int]], Optional[object], Optional[str]]:
    """
    Validate experiment ID or group ID inputs and return list of experiment IDs.
    Returns (exp_ids, group_obj_or_None, error_message_or_None).
    """
    if not exp_id and not group_id:
        return None, None, (
            "❌ No experiment ID or group ID provided. "
            "Usage: kpp vis <command> <exp_id> OR kpp vis <command> --group-id <group_id>"
        )

    if exp_id and group_id:
        return None, None, "❌ Please provide either an experiment ID OR a group ID, not both."

    if group_id:
        try:
            with Database() as database:
                group = database.exp_group.get(group_id)
                if not group:
                    return None, None, f"❌ Experiment group {group_id} not found."
                exp_ids = database.exp_group.list_members(group_id)
                if not exp_ids:
                    return None, None, f"❌ No experiments found in group {group_id}."
                click.echo(
                    f"📊 Using {len(exp_ids)} experiments from group {group_id} "
                    f"('{group.label}'): {exp_ids}"
                )
                return exp_ids, group, None
        except Exception as e:  # pragma: no cover
            return None, None, f"❌ Error accessing group {group_id}: {e}"

    exp_ids = [exp_id]  # type: ignore[list-item]
    try:
        with Database() as database:
            experiment = database.experiment.get(exp_id)  # type: ignore[arg-type]
            if not experiment:
                return None, None, f"❌ Experiment {exp_id} not found."
            click.echo(f"📊 Using experiment {exp_id}: {experiment.name}")
            return exp_ids, None, None
    except Exception as e:  # pragma: no cover
        return None, None, f"❌ Error accessing experiment {exp_id}: {e}"


def _resolve_efficiency_fairness_points(
    exp_ids_arg: tuple[int, ...],
    group_ids_arg: tuple[int, ...],
) -> Tuple[list[tuple[list[int], str]], Optional[str]]:
    """
    Resolve (exp_ids, label) for each point to plot.
    - Single/multiple exp_ids: one point per experiment, label=str(exp_id)
    - Single group_id: one point per experiment in group, label=str(exp_id)
    - Multiple group_ids: one point per group (averaged), label=group.label
    Returns (points, error) where points = [(exp_ids, label), ...].
    """
    if not exp_ids_arg and not group_ids_arg:
        return [], "❌ Provide exp_ids or --group-id."
    if exp_ids_arg and group_ids_arg:
        return [], "❌ Provide either exp_ids OR --group-id, not both."

    try:
        with Database() as database:
            if exp_ids_arg:
                points = []
                for eid in exp_ids_arg:
                    exp = database.experiment.get(eid)
                    if not exp:
                        return [], f"❌ Experiment {eid} not found."
                    points.append(([eid], exp.name))
                return points, None

            # group_ids
            points = []
            for gid in group_ids_arg:
                group = database.exp_group.get(gid)
                if not group:
                    return [], f"❌ Group {gid} not found."
                members = database.exp_group.list_members(gid)
                if not members:
                    return [], f"❌ No experiments in group {gid}."
                if len(group_ids_arg) == 1:
                    for eid in members:
                        exp = database.experiment.get(eid)
                        label = exp.name if exp else str(eid)
                        points.append(([eid], label))
                else:
                    points.append((members, group.label))
            return points, None
    except Exception as e:  # pragma: no cover
        return [], str(e)


def _load_rewards_access_for_experiment(
    database: object, eid: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load rewards_by_step, access_by_step, and weights for one experiment.
    Returns (rewards_arr, access_arr, weights) or (None, None, None) on failure.
    rewards_arr: (n_steps, n_agents), access_arr: (n_steps, n_agents), weights: (n_agents,)
    """
    metrics = database.metric.filter_by(exp_id=eid)
    if not metrics:
        return None, None, None

    rewards_map = {m.step: m for m in metrics if m.metric_name == "rewards"}
    collectives_map = {m.step: m for m in metrics if m.metric_name == "collectives"}
    reports_map = {m.step: m for m in metrics if m.metric_name == "reports"}

    all_steps = sorted(
        set(rewards_map.keys()) & set(collectives_map.keys()) & set(reports_map.keys())
    )

    rewards_by_step: list[list[float]] = []
    access_by_step: list[list[float]] = []

    for step in all_steps:
        try:
            rewards = json.loads(rewards_map[step].metric_value)
            collectives = json.loads(collectives_map[step].metric_value)
            reports = json.loads(reports_map[step].metric_value)
        except json.JSONDecodeError:
            continue

        if not isinstance(rewards, dict) or any(v is None for v in rewards.values()):
            continue

        agent_ids = sorted(int(aid) for aid in rewards.keys())
        rewards_row = [float(rewards.get(str(aid), 0) or 0) for aid in agent_ids]

        agent_to_collective: dict[int, int] = {}
        if isinstance(collectives, dict):
            for cid_str, agent_list in collectives.items():
                try:
                    cid = int(cid_str)
                except (TypeError, ValueError):
                    continue
                if isinstance(agent_list, list):
                    for aid in agent_list:
                        agent_to_collective[int(aid)] = cid

        access_row: list[float] = []
        for aid in agent_ids:
            cid = agent_to_collective.get(aid)
            if cid is None:
                access_row.append(0.0)
                continue
            report = reports.get(str(cid)) if isinstance(reports, dict) else None
            if not isinstance(report, dict):
                access_row.append(0.0)
                continue
            selected = report.get("selected_outcomes")
            if not isinstance(selected, dict):
                access_row.append(0.0)
                continue
            outcome = selected.get(str(aid))
            if outcome is None:
                access_row.append(0.0)
                continue
            outcome_arr = np.asarray(outcome, dtype=float)
            has_access = bool(np.any(outcome_arr > 0))
            access_row.append(1.0 if has_access else 0.0)

        rewards_by_step.append(rewards_row)
        access_by_step.append(access_row)

    if not rewards_by_step or not access_by_step:
        return None, None, None

    rewards_arr = np.asarray(rewards_by_step, dtype=float)
    access_arr = np.asarray(access_by_step, dtype=float)
    n_agents = rewards_arr.shape[1]
    weights = np.ones(n_agents, dtype=float)
    return rewards_arr, access_arr, weights


def _load_nash_inputs_for_experiment(
    database: object, eid: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load allocation, allocated_urgencies, and weights for nash_welfare.
    Returns (allocation, allocated_urgencies, weights) or (None, None, None).
    """
    metrics = database.metric.filter_by(exp_id=eid)
    if not metrics:
        return None, None, None

    collectives_map = {m.step: m for m in metrics if m.metric_name == "collectives"}
    reports_map = {m.step: m for m in metrics if m.metric_name == "reports"}
    population_map = {m.step: m for m in metrics if m.metric_name == "population_state"}

    all_steps = sorted(
        set(collectives_map.keys()) & set(reports_map.keys()) & set(population_map.keys())
    )
    all_steps = [s for s in all_steps if s > 0]  # skip step 0

    allocation: list[int] = []
    allocation_row_indices: list[int] = []
    urgencies_by_step: list[list[float]] = []

    for step in all_steps:
        try:
            collectives = json.loads(collectives_map[step].metric_value)
            reports = json.loads(reports_map[step].metric_value)
            population_state = json.loads(population_map[step].metric_value)
        except (json.JSONDecodeError, TypeError):
            continue

        agent_states = population_state.get("agent_states") if isinstance(population_state, dict) else {}
        if not isinstance(agent_states, dict):
            continue

        agent_ids = sorted(int(aid) for aid in agent_states.keys())
        urgency_values = []
        for aid in agent_ids:
            state = agent_states.get(str(aid), {})
            if isinstance(state, dict):
                priv = state.get("private")
                urgency_values.append(float(priv) if priv is not None else 0.0)
            else:
                urgency_values.append(0.0)

        agent_to_collective: dict[int, int] = {}
        if isinstance(collectives, dict):
            for cid_str, agent_list in collectives.items():
                try:
                    cid = int(cid_str)
                except (TypeError, ValueError):
                    continue
                if isinstance(agent_list, list):
                    for aid in agent_list:
                        agent_to_collective[int(aid)] = cid

        allocated_agent = None
        for idx, aid in enumerate(agent_ids):
            cid = agent_to_collective.get(aid)
            if cid is None:
                continue
            report = reports.get(str(cid)) if isinstance(reports, dict) else None
            if not isinstance(report, dict):
                continue
            selected = report.get("selected_outcomes")
            if not isinstance(selected, dict):
                continue
            outcome = selected.get(str(aid))
            if outcome is None:
                continue
            outcome_arr = np.asarray(outcome, dtype=float)
            has_access = bool(np.any(outcome_arr > 0))
            if has_access and allocated_agent is None:
                allocated_agent = idx
                break

        row_idx = len(urgencies_by_step)
        urgencies_by_step.append(urgency_values)
        if allocated_agent is not None:
            allocation.append(allocated_agent)
            allocation_row_indices.append(row_idx)

    if not allocation or not urgencies_by_step:
        return None, None, None

    urgencies_arr = np.asarray(urgencies_by_step, dtype=float)
    allocation_indices_arr = np.asarray(allocation_row_indices, dtype=int)
    allocated_urgencies = urgencies_arr[allocation_indices_arr, :]
    n_agents = urgencies_arr.shape[1]
    weights = np.ones(n_agents, dtype=float)
    return np.asarray(allocation, dtype=int), allocated_urgencies, weights


def _load_balances_for_experiments(
    database: object, exp_ids: list[int]
) -> list[dict]:
    """Load balance rows for all experiments. Returns list of {exp_id, step, agent_id, balance}."""
    all_rows: list[dict] = []
    for eid in exp_ids:
        metrics = database.metric.filter_by(exp_id=eid)
        mech_metrics = [m for m in metrics if m.metric_name == "mechanism_state"]
        for metric in mech_metrics:
            try:
                state = json.loads(metric.metric_value)
                balances = state.get("agent_balances")
                if not isinstance(balances, dict):
                    continue
                step = metric.step
                for aid, bal in sorted(balances.items(), key=lambda kv: int(kv[0])):
                    all_rows.append({
                        "exp_id": eid,
                        "step": step,
                        "agent_id": int(aid),
                        "balance": float(bal),
                    })
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    return all_rows


def _extract_markov_urgency_inputs(
    population_cfg: dict,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """Expand agent type configs into per-agent transition matrices, weights, urgency levels."""
    try:
        agent_type_cfgs = population_cfg["parameters"]["agent_type_cfgs"]
    except KeyError as e:
        raise click.ClickException(
            f"Population config missing expected key for Markov metrics: {e}."
        ) from e

    transition_matrices: list[np.ndarray] = []
    weights: list[float] = []
    urgency_levels: list[np.ndarray] = []
    for _type_name, type_cfg in agent_type_cfgs.items():
        try:
            n_agents = int(type_cfg["n_agents"])
            weight = float(type_cfg["weight"])
            transition_matrix = np.asarray(
                type_cfg["agent_model"]["parameters"]["transition_matrix"],
                dtype=float,
            )
        except KeyError as e:
            raise click.ClickException(
                f"Agent type missing transition matrix config key: {e}. "
                "Expected agent_model.parameters.transition_matrix."
            ) from e
        levels = type_cfg["agent_model"]["parameters"].get("urgency_levels")
        if levels is None:
            raise click.ClickException(
                "Agent type missing required key agent_model.parameters.urgency_levels."
            )
        levels_array = np.asarray(levels, dtype=float)
        if levels_array.shape[0] != transition_matrix.shape[0]:
            raise click.ClickException(
                "urgency_levels length must match transition matrix size."
            )
        for _ in range(n_agents):
            transition_matrices.append(transition_matrix)
            weights.append(weight)
            urgency_levels.append(levels_array)

    if not transition_matrices:
        raise click.ClickException("No agent transition matrices found in population config.")
    return transition_matrices, np.asarray(weights, dtype=float), urgency_levels


def _compute_markov_metrics(
    world_cfg: dict,
    population_cfg: dict,
    lambda_star: Optional[float],
    delta_r: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[float], list[float], float]:
    """Compute Markov metrics from configs."""
    try:
        resource_count = float(len(world_cfg["parameters"]["resource_capacities"]))
    except KeyError as e:
        raise click.ClickException(
            f"World config missing resource_capacities: {e}."
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


def _load_experiment_configs(database: object, exp_id: int) -> Tuple[dict, dict]:
    """Load world_cfg and population_cfg from DB for an experiment."""
    exp = database.experiment.get(exp_id)
    if not exp:
        raise click.ClickException(f"Experiment {exp_id} not found.")
    world_row = database.world.get(exp.world_hash)
    if not world_row:
        raise click.ClickException(f"World config for experiment {exp_id} not found.")
    world_cfg = json.loads(world_row.json)
    population_cfg = {
        "parameters": {
            "agent_type_cfgs": database.population.to_scenario_population(exp.population_hash),
        }
    }
    return world_cfg, population_cfg


def _load_rewards_for_experiment(database: object, eid: int) -> Optional[np.ndarray]:
    """Load rewards_by_step (n_steps, n_agents) for one experiment."""
    metrics = database.metric.filter_by(exp_id=eid)
    rewards_metrics = [m for m in metrics if m.metric_name == "rewards"]
    rewards_metrics = [m for m in rewards_metrics if m.step > 0]
    if not rewards_metrics:
        return None

    rows = []
    agent_ids = None
    for m in sorted(rewards_metrics, key=lambda x: x.step):
        try:
            rewards = json.loads(m.metric_value)
        except json.JSONDecodeError:
            continue
        if not isinstance(rewards, dict) or any(v is None for v in rewards.values()):
            continue
        aids = sorted(int(aid) for aid in rewards.keys())
        if agent_ids is None:
            agent_ids = aids
        if aids != agent_ids:
            continue
        rows.append([float(rewards.get(str(aid), 0) or 0) for aid in agent_ids])

    if not rows:
        return None
    return np.asarray(rows, dtype=float)


def _common_vis_options(func):
    """Decorator adding common visualization options."""
    func = click.option(
        "--format", "-f",
        type=click.Choice(["png", "pdf", "svg", "jpg"]),
        default="png",
        show_default=True,
        help="Output image format.",
    )(func)
    func = click.option("--dpi", type=int, default=300, show_default=True, help="Image DPI.")(func)
    func = click.option(
        "--figsize", nargs=2, type=int, default=[12, 8],
        show_default=True, help="Figure size (width height).",
    )(func)
    func = click.option("--style", type=str, default="whitegrid", show_default=True, help="Seaborn style.")(func)
    func = click.option("--palette", type=str, default=None, help="Color palette.")(func)
    func = click.option("--save/--no-save", default=True, show_default=True, help="Save plot to file.")(func)
    func = click.option(
        "--log-level", "-l", default="INFO", show_default=True,
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
        help="Logging level.",
    )(func)
    return func


def _save_or_show(save: bool, output_path: Path, dpi: int) -> None:
    """Save current figure to file and optionally display."""
    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        click.echo(f"💾 Plot saved to: {output_path}")
    click.echo("🖼️  Displaying plot..." if save else "🖼️  Displaying plot (not saving)...")
    plt.show()


# ---------------------------------------------------------------------------
# CLI group and commands
# ---------------------------------------------------------------------------


@click.group()
def vis():
    """Visualization commands for database-backed experiments."""
    pass


@vis.command()
@click.argument("exp_id", type=int, required=False)
@click.option("--group-id", "-g", type=int, help="Experiment group ID.")
@_common_vis_options
def balances(
    exp_id: Optional[int],
    group_id: Optional[int],
    format: str,
    dpi: int,
    figsize: list,
    style: str,
    palette: Optional[str],
    save: bool,
    log_level: str,
) -> None:
    """
    Visualize agent balance (karma) distributions over time.

    Reads mechanism_state metrics and extracts agent_balances for each experiment.
    """
    configure_logging(level=log_level)
    exp_ids, group, error = _validate_and_resolve_experiments(exp_id, group_id)
    if error:
        click.echo(error)
        return

    try:
        with Database() as database:
            for eid in exp_ids or []:
                if not database.experiment.get(eid):
                    click.echo(f"❌ Experiment {eid} not found.")
                    return

            all_rows = _load_balances_for_experiments(database, exp_ids or [])
            if not all_rows:
                click.echo("❌ No valid balance data found for any experiment.")
                return

            df = pd.DataFrame(all_rows)
            click.echo(
                f"✅ Loaded {len(df)} balance records across "
                f"{df['exp_id'].nunique()} experiments, "
                f"{df['agent_id'].nunique()} agents over {df['step'].nunique()} steps"
            )

            avg_df = df.groupby(["step", "agent_id"])["balance"].mean().reset_index()
            avg_df = avg_df.rename(columns={"balance": "avg_balance"})

            sns.set_style(style)
            if palette:
                sns.set_palette(palette)

            fig, ax = plt.subplots(1, 1, figsize=tuple(figsize))
            if group:
                title = f"Average Agent Balance Distributions - {group.label} ({len(exp_ids or [])} Experiments)"
            else:
                title = f"Average Agent Balance Distributions - Experiment {(exp_ids or [0])[0]}"
            fig.suptitle(title, fontsize=16, fontweight="bold")

            sns.violinplot(data=avg_df, x="step", y="avg_balance", ax=ax)
            ax.set_title("Average Balance Distribution Over Time")
            ax.set_xlabel("Step")
            ax.set_ylabel("Average Balance")
            steps_sorted = sorted(avg_df["step"].unique())
            ax.set_xticks(steps_sorted)
            ax.set_xticklabels(steps_sorted)
            plt.tight_layout()

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if group:
                out_dir = Path("data/plots") / f"group_{group.label}"
            else:
                out_dir = Path("data/plots") / f"experiment_{(exp_ids or [0])[0]}"
            output_path = out_dir / f"balances_{timestamp}.{format}"

            _save_or_show(save, output_path, dpi)

            click.echo(f"\n📊 Summary: {df['step'].nunique()} steps, {df['agent_id'].nunique()} agents")
            init = df[df["step"] == 0]
            if len(init) > 0:
                click.echo(f"  Initial Balance Range: {init['balance'].min():.2f} - {init['balance'].max():.2f}")

    except Exception as e:  # pragma: no cover
        click.echo(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


@vis.command(name="efficiency-fairness")
@click.argument("exp_ids", type=int, nargs=-1, required=False)
@click.option("--group-id", "-g", type=int, multiple=True, help="Group ID(s). Single: plot each exp. Multiple: plot avg per group.")
@click.option(
    "--labels",
    type=str,
    multiple=True,
    help="Labels for points (e.g. Random, Optimal). Optional; uses exp_id or group label otherwise.",
)
@_common_vis_options
def efficiency_fairness(
    exp_ids: tuple[int, ...],
    group_id: tuple[int, ...],
    labels: tuple[str, ...],
    format: str,
    dpi: int,
    figsize: list,
    style: str,
    palette: Optional[str],
    save: bool,
    log_level: str,
) -> None:
    """
    Plot (efficiency, access fairness) scatter.

    Single exp_id: plot that experiment. Multiple exp_ids: plot each.
    Single --group-id: plot each experiment in the group.
    Multiple --group-id: plot average per group.
    Use --labels for styled comparison (Random, Optimal, Dictator, Q γ=...).
    """
    configure_logging(level=log_level)
    points_spec, error = _resolve_efficiency_fairness_points(exp_ids, group_id)
    if error:
        click.echo(error)
        return

    try:
        with Database() as database:
            eff_list: list[float] = []
            fair_list: list[float] = []
            labels_list: list[str] = []
            for exp_ids_for_point, default_label in points_spec:
                effs, fairs = [], []
                for eid in exp_ids_for_point:
                    rewards_arr, access_arr, weights = _load_rewards_access_for_experiment(database, eid)
                    if rewards_arr is None or access_arr is None or weights is None:
                        click.echo(f"⚠️  Could not extract rewards/access for experiment {eid}.")
                        continue
                    effs.append(get_efficiency(rewards_arr, weights))
                    fairs.append(get_access_fairness(access_arr, weights))
                if not effs or not fairs:
                    continue
                eff_list.append(float(np.mean(effs)))
                fair_list.append(float(np.mean(fairs)))
                labels_list.append(default_label)

            if not eff_list or not fair_list:
                click.echo("❌ No efficiency/fairness points could be computed.")
                return

            if labels and len(labels) == len(eff_list):
                labels_list = list(labels)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_dir = Path("data/plots") / "experiments"
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(out_dir / f"efficiency_fairness_{timestamp}.{format}")

            # Decide plotting style:
            # - If any label looks like a mechanism comparison or encodes a gamma value,
            #   use the richer comparison plot, which:
            #     * connects gamma points in order,
            #     * colors them from blue (low γ) to yellow (high γ),
            #     * styles known baselines (coin toss, turn-taking, benevolent dictator).
            # - Otherwise, fall back to a simple scatter.
            has_mech_or_gamma_labels = False
            if labels_list:
                for lb in labels_list:
                    lower = lb.strip().lower()
                    if (
                        "gamma" in lower
                        or lower.startswith("γ=")
                        or lower in ("random", "optimal", "benevolent dictator", "coin toss", "turn-taking")
                        or lower.startswith("q ")
                    ):
                        has_mech_or_gamma_labels = True
                        break

            if has_mech_or_gamma_labels:
                plot_efficiency_fairness_comparison(
                    access_fairness=fair_list,
                    efficiency=eff_list,
                    labels=labels_list,
                    save_path=save_path,
                )
            else:
                plot_access_fairness_vs_efficiency(
                    access_fairness=fair_list,
                    efficiency=eff_list,
                    labels=labels_list,
                    save_path=save_path,
                )
            click.echo(f"💾 Plot saved to: {save_path}")
            if format in ("png", "jpg") and save:
                try:
                    img = plt.imread(save_path)
                    fig, ax = plt.subplots(figsize=tuple(figsize))
                    ax.imshow(img)
                    ax.axis("off")
                    plt.tight_layout()
                    plt.show()
                except Exception:
                    pass

    except Exception as e:  # pragma: no cover
        click.echo(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


@vis.command(name="reward-over-time")
@click.argument("exp_id", type=int, required=False)
@click.option("--group-id", "-g", type=int, help="Experiment group ID.")
@click.option("--window", type=int, default=50, show_default=True, help="Rolling window for smoothing.")
@_common_vis_options
def reward_over_time(
    exp_id: Optional[int],
    group_id: Optional[int],
    window: int,
    format: str,
    dpi: int,
    figsize: list,
    style: str,
    palette: Optional[str],
    save: bool,
    log_level: str,
) -> None:
    """
    Plot smoothed per-agent reward over time.

    Uses the first experiment in the group (or single experiment).
    """
    configure_logging(level=log_level)
    exp_ids, group, error = _validate_and_resolve_experiments(exp_id, group_id)
    if error:
        click.echo(error)
        return

    try:
        with Database() as database:
            eid = (exp_ids or [0])[0]
            rewards_by_step = _load_rewards_for_experiment(database, eid)
            if rewards_by_step is None:
                click.echo(f"❌ No rewards data found for experiment {eid}.")
                return

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if group:
                out_dir = Path("data/plots") / f"group_{group.label}"
            else:
                out_dir = Path("data/plots") / f"experiment_{eid}"
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(out_dir / f"reward_over_time_{timestamp}.{format}")

            plot_smoothed_reward_over_time(
                rewards_by_step=rewards_by_step,
                window=window,
                title="Smoothed reward over time",
                save_path=save_path,
            )
            click.echo(f"💾 Plot saved to: {save_path}")
            if format in ("png", "jpg") and save:
                try:
                    img = plt.imread(save_path)
                    fig, ax = plt.subplots(figsize=tuple(figsize))
                    ax.imshow(img)
                    ax.axis("off")
                    plt.tight_layout()
                    plt.show()
                except Exception:
                    pass

    except Exception as e:  # pragma: no cover
        click.echo(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


@vis.command(name="nash-welfare")
@click.argument("exp_id", type=int, required=False)
@click.option("--group-id", "-g", type=int, help="Experiment group ID.")
@_common_vis_options
def nash_welfare_cmd(
    exp_id: Optional[int],
    group_id: Optional[int],
    format: str,
    dpi: int,
    figsize: list,
    style: str,
    palette: Optional[str],
    save: bool,
    log_level: str,
) -> None:
    """
    Plot Nash welfare bar chart for experiments from the DB.

    Requires population_state metrics (for urgencies) and reports (for allocation).
    """
    configure_logging(level=log_level)
    exp_ids, group, error = _validate_and_resolve_experiments(exp_id, group_id)
    if error:
        click.echo(error)
        return

    try:
        with Database() as database:
            nash_values: list[float] = []
            labels_list: list[str] = []
            for eid in exp_ids or []:
                allocation, allocated_urgencies, weights = _load_nash_inputs_for_experiment(database, eid)
                if allocation is None or allocated_urgencies is None or weights is None:
                    click.echo(f"⚠️  Could not compute Nash welfare for experiment {eid}.")
                    continue
                nash_val, _ = nash_welfare(
                    allocation=allocation,
                    urgencies=allocated_urgencies,
                    social_weights=weights,
                )
                nash_values.append(float(nash_val))
                exp = database.experiment.get(eid)
                labels_list.append(exp.name if exp else str(eid))

            if not nash_values:
                click.echo("❌ No Nash welfare values could be computed.")
                return

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if group:
                out_dir = Path("data/plots") / f"group_{group.label}"
            else:
                out_dir = Path("data/plots") / (
                    f"experiment_{(exp_ids or [0])[0]}"
                    if len(exp_ids or []) == 1
                    else "experiments"
                )
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(out_dir / f"nash_welfare_{timestamp}.{format}")

            plot_nash_welfare(
                nash_welfare_values=nash_values,
                labels=labels_list,
                save_path=save_path,
            )
            click.echo(f"💾 Plot saved to: {save_path}")
            if format in ("png", "jpg") and save:
                try:
                    img = plt.imread(save_path)
                    fig, ax = plt.subplots(figsize=tuple(figsize))
                    ax.imshow(img)
                    ax.axis("off")
                    plt.tight_layout()
                    plt.show()
                except Exception:
                    pass

    except Exception as e:  # pragma: no cover
        click.echo(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


@vis.command(name="full-info-policy-distribution")
@click.argument("exp_id", type=int, required=True)
@click.option("--step", type=int, default=None, help="Step to use (default: last).")
@_common_vis_options
def full_info_policy_distribution(
    exp_id: int,
    step: Optional[int],
    format: str,
    dpi: int,
    figsize: list,
    style: str,
    palette: Optional[str],
    save: bool,
    log_level: str,
) -> None:
    """
    Plot learned policy (Bid vs Karma) and karma distribution from FullInfoLearningAgent.

    Requires experiment with FullInfoLearningAgent; policy is read from population_state.
    """
    configure_logging(level=log_level)

    try:
        with Database() as database:
            if not database.experiment.get(exp_id):
                click.echo(f"❌ Experiment {exp_id} not found.")
                return

            metrics = database.metric.filter_by(exp_id=exp_id)
            pop_metrics = [m for m in metrics if m.metric_name == "population_state"]
            if not pop_metrics:
                click.echo("❌ No population_state metrics found.")
                return

            if step is not None:
                pop_by_step = {m.step: m for m in pop_metrics}
                if step not in pop_by_step:
                    click.echo(f"❌ Step {step} not found.")
                    return
                target_metric = pop_by_step[step]
            else:
                target_metric = max(pop_metrics, key=lambda m: m.step)

            pop_state = json.loads(target_metric.metric_value)
            agent_states = pop_state.get("agent_states")
            if not isinstance(agent_states, dict):
                click.echo("❌ Could not parse population_state.")
                return

            pi_arr = None
            d_arr = None
            urgency_levels: list[int] = []
            for _aid, state in agent_states.items():
                if not isinstance(state, dict):
                    continue
                policy = state.get("policy")
                if not isinstance(policy, dict):
                    continue
                pi_raw = policy.get("pi")
                d_raw = policy.get("d")
                if pi_raw is not None and d_raw is not None:
                    pi_arr = np.asarray(pi_raw, dtype=float)
                    d_arr = np.asarray(d_raw, dtype=float)
                    break

            if pi_arr is None or d_arr is None:
                click.echo(
                    "❌ No FullInfoLearningAgent policy (pi, d) found in population_state. "
                    "Run an experiment with FullInfoLearningAgent first."
                )
                return

            nu, nk, _ = pi_arr.shape
            urgency_levels = list(range(nu))
            try:
                pop_cfg = database.population.to_scenario_population(
                    database.experiment.get(exp_id).population_hash
                )
                for _mid, acfg in pop_cfg.items():
                    ul = acfg.get("agent_model", {}).get("parameters", {}).get("urgency_levels")
                    if ul is not None:
                        urgency_levels = [int(u) for u in ul]
                        break
            except Exception:
                pass

            if len(urgency_levels) == 3 and urgency_levels[0] == urgency_levels[1] < urgency_levels[2]:
                urgency_labels = [
                    f"u = {urgency_levels[0]} default",
                    f"u = {urgency_levels[1]} intermediate",
                    f"u = {urgency_levels[2]} urgent",
                ]
            else:
                urgency_labels = [f"u = {u}" for u in urgency_levels]

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_dir = Path("data/plots") / f"experiment_{exp_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(out_dir / f"full_info_policy_distribution_{timestamp}.{format}")

            plot_full_info_policy_and_distribution(
                pi=pi_arr,
                d=d_arr,
                urgency_levels=urgency_levels,
                urgency_labels=urgency_labels,
                save_path=save_path,
            )
            click.echo(f"💾 Plot saved to: {save_path}")
            if format in ("png", "jpg") and save:
                try:
                    img = plt.imread(save_path)
                    fig, ax = plt.subplots(figsize=tuple(figsize))
                    ax.imshow(img)
                    ax.axis("off")
                    plt.tight_layout()
                    plt.show()
                except Exception:
                    pass

    except Exception as e:  # pragma: no cover
        click.echo(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


@vis.command(name="markov-urgency-violin")
@click.argument("exp_id", type=int, required=True)
@click.option("--delta-r", type=float, default=1.0, help="Reward gain factor Delta r.")
@click.option("--lambda-star", type=float, default=None, help="Optional lambda* override.")
@_common_vis_options
def markov_urgency_violin(
    exp_id: int,
    delta_r: float,
    lambda_star: Optional[float],
    format: str,
    dpi: int,
    figsize: list,
    style: str,
    palette: Optional[str],
    save: bool,
    log_level: str,
) -> None:
    """
    Plot Markov stationary urgency distribution violin from experiment config.

    Uses world and population config stored with the experiment.
    """
    configure_logging(level=log_level)

    try:
        with Database() as database:
            world_cfg, population_cfg = _load_experiment_configs(database, exp_id)
            (
                stationary,
                urgency_levels,
                thresholds,
                spike_indices,
                surplus_efficiency,
                _lambda_used,
            ) = _compute_markov_metrics(world_cfg, population_cfg, lambda_star, delta_r)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_dir = Path("data/plots") / f"experiment_{exp_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(out_dir / f"markov_urgency_violin_{timestamp}.{format}")

            plot_markov_stationary_violin(
                urgency_levels=urgency_levels,
                stationary_distributions=stationary,
                spike_indices=spike_indices,
                surplus_efficiency=surplus_efficiency,
                thresholds=thresholds,
                save_path=save_path,
            )
            click.echo(f"💾 Plot saved to: {save_path}")
            if format in ("png", "jpg") and save:
                try:
                    img = plt.imread(save_path)
                    fig, ax = plt.subplots(figsize=tuple(figsize))
                    ax.imshow(img)
                    ax.axis("off")
                    plt.tight_layout()
                    plt.show()
                except Exception:
                    pass

    except Exception as e:  # pragma: no cover
        click.echo(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


@vis.command(name="metrics-table")
@click.argument("exp_ids", type=int, nargs=-1, required=True)
@click.option(
    "--labels",
    type=str,
    multiple=True,
    required=True,
    help="Column labels (one per experiment, e.g. Random, Q-learning, Optimal).",
)
@click.option("--title", type=str, default="Mechanism Metrics Summary", help="Table title.")
@_common_vis_options
def metrics_table(
    exp_ids: tuple[int, ...],
    labels: tuple[str, ...],
    title: str,
    format: str,
    dpi: int,
    figsize: list,
    style: str,
    palette: Optional[str],
    save: bool,
    log_level: str,
) -> None:
    """
    Render a metrics table (Efficiency, Fairness) from DB experiments.

    Example: kpp vis metrics-table 1 2 3 --labels Random --labels "Q-learning" --labels Optimal
    """
    configure_logging(level=log_level)
    if len(labels) != len(exp_ids):
        click.echo("❌ --labels count must match number of experiments.")
        return

    try:
        with Database() as database:
            eff_list: list[float] = []
            fair_list: list[float] = []
            for eid in exp_ids:
                rewards_arr, access_arr, weights = _load_rewards_access_for_experiment(database, eid)
                if rewards_arr is None or access_arr is None or weights is None:
                    click.echo(f"⚠️  Could not extract data for experiment {eid}.")
                    continue
                eff_list.append(get_efficiency(rewards_arr, weights))
                fair_list.append(get_access_fairness(access_arr, weights))

            if len(eff_list) != len(exp_ids) or len(fair_list) != len(exp_ids):
                click.echo("❌ Could not load data for all experiments.")
                return

            row_labels = ["Efficiency", "Fairness"]
            column_labels = list(labels)
            cell_text = [
                [f"{e:.4f}" for e in eff_list],
                [f"{f:.4f}" for f in fair_list],
            ]

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_dir = Path("data/plots")
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(out_dir / f"metrics_table_{timestamp}.{format}")

            plot_metrics_table(
                row_labels=row_labels,
                column_labels=column_labels,
                cell_text=cell_text,
                title=title,
                save_path=save_path,
            )
            click.echo(f"💾 Plot saved to: {save_path}")
            if format in ("png", "jpg") and save:
                try:
                    img = plt.imread(save_path)
                    fig, ax = plt.subplots(figsize=tuple(figsize))
                    ax.imshow(img)
                    ax.axis("off")
                    plt.tight_layout()
                    plt.show()
                except Exception:
                    pass

    except Exception as e:  # pragma: no cover
        click.echo(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    vis()
