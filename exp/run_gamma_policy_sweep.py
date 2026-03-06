"""
Train full-info policies for all 2000x2 gamma scenarios, evaluate them in the
karma-policy scenario, group the evaluation runs, and create a combined
efficiency-vs-fairness plot.

Usage (from repo root):

    python run_gamma_policy_sweep.py

What it does:
  1) Runs all `data/scenarios/2000x2_full_info_gamma_*.yaml` with:
       - 2000 steps
       - seed = 3
  2) Exports the learned policy π from each run to `data/policies/`.
  3) For each exported policy, runs the karma-policy scenario for:
       - 1000 steps
       - 3 runs with seeds 0, 1, 2
     and creates an experiment group named `gamma_[VALUE]` containing those 3 runs.
  4) Plots efficiency vs fairness for groups 38, 39, 40 plus the newly created groups.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from karma_pp.db.base import Database


FULL_INFO_GLOB = "data/scenarios/2000x2_full_info_gamma_*.yaml"
KARMA_POLICY_BASE = Path("data/scenarios/2000x2_karma_policy.yaml")
GENERATED_SCENARIO_DIR = Path("data/scenarios/generated")
POLICY_DIR = Path("data/policies")

FULL_INFO_STEPS = 2000
FULL_INFO_SEED = 3

EVAL_STEPS = 1000
EVAL_SEEDS = (0, 1, 2)

PLOT_BASE_GROUP_IDS = (38, 39, 40)


_GAMMA_RE = re.compile(r"gamma_(\d+(?:\.\d+)?)")


def _run_kpp(scenario_path: Path, *, steps: int, seeds: Iterable[int]) -> None:
    cmd: list[str] = [
        "kpp",
        "run",
        "--scenario",
        str(scenario_path),
        "--steps",
        str(int(steps)),
    ]
    for s in seeds:
        cmd.extend(["--seed", str(int(s))])

    print(f"[run_gamma_policy_sweep] Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _find_latest_finished_experiment_id(
    db: Database, *, name: str, n_steps: int, seed: int
) -> int:
    matches = db.experiment.filter_by(name=name, n_steps=int(n_steps), seed=int(seed))
    matches = [m for m in matches if (m.status or "").lower() == "finished"]
    if not matches:
        raise RuntimeError(
            f"No finished experiment found for name={name!r}, n_steps={n_steps}, seed={seed}."
        )
    return max(matches, key=lambda e: e.exp_id).exp_id


def _export_pi_policy_for_experiment(db: Database, *, exp_id: int, out_path: Path) -> None:
    metrics = db.metric.filter_by(exp_id=exp_id, metric_name="population_state")
    if not metrics:
        raise RuntimeError(f"No population_state metrics found for exp_id={exp_id}.")

    last_metric = max(metrics, key=lambda m: int(m.step) if m.step is not None else -1)
    try:
        pop_state = json.loads(last_metric.metric_value)
    except Exception as e:
        raise RuntimeError(f"Could not JSON-decode population_state for exp_id={exp_id}.") from e

    agent_states = pop_state.get("agent_states", {})
    if not isinstance(agent_states, dict):
        raise RuntimeError(f"population_state.agent_states missing/invalid for exp_id={exp_id}.")

    pi_arr: np.ndarray | None = None
    for _aid, state in agent_states.items():
        if not isinstance(state, dict):
            continue
        policy = state.get("policy")
        if not isinstance(policy, dict):
            continue
        # Skip clone policies that only reference another agent.
        if "reference_agent_id" in policy and "pi" not in policy:
            continue
        pi_raw = policy.get("pi")
        if pi_raw is None:
            continue
        pi_arr = np.asarray(pi_raw, dtype=float)
        break

    if pi_arr is None:
        raise RuntimeError(f"No exportable π policy found in exp_id={exp_id}.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, pi_arr)


def _write_karma_policy_scenario_for_pi(*, policy_path: Path, gamma_value: str) -> Path:
    if not KARMA_POLICY_BASE.is_file():
        raise FileNotFoundError(f"Base karma-policy scenario not found: {KARMA_POLICY_BASE}")

    cfg = yaml.safe_load(KARMA_POLICY_BASE.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Invalid YAML structure in {KARMA_POLICY_BASE}")

    population = cfg.get("population")
    if not isinstance(population, dict) or not population:
        raise RuntimeError(f"Scenario missing population config: {KARMA_POLICY_BASE}")

    # Expect a single population entry (e.g. "pi_policy").
    pop_key = next(iter(population.keys()))
    params = (
        cfg["population"][pop_key]
        .get("agent_model", {})
        .get("parameters", {})
    )
    if not isinstance(params, dict):
        raise RuntimeError(f"Could not locate agent_model.parameters in {KARMA_POLICY_BASE}")

    params["policy_path"] = str(policy_path)

    GENERATED_SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GENERATED_SCENARIO_DIR / f"2000x2_karma_policy_gamma_{gamma_value}.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out_path


def _create_experiment_group(db: Database, *, label: str, exp_ids: list[int]) -> int:
    group = db.exp_group.create(label=label)
    for eid in exp_ids:
        db.exp_group.add_member(group.group_id, int(eid))
    return group.group_id


def _gamma_from_scenario_stem(stem: str) -> str:
    m = _GAMMA_RE.search(stem)
    if not m:
        raise ValueError(f"Could not parse gamma value from scenario name: {stem}")
    return m.group(1)


def main() -> None:
    full_info_scenarios = sorted(Path().glob(FULL_INFO_GLOB))
    if not full_info_scenarios:
        raise FileNotFoundError(f"No scenarios matched {FULL_INFO_GLOB!r}")

    print(
        "[run_gamma_policy_sweep] Found full-info gamma scenarios:",
        [p.name for p in full_info_scenarios],
        flush=True,
    )

    created_group_ids: list[int] = []

    for scen in full_info_scenarios:
        gamma_value = _gamma_from_scenario_stem(scen.stem)

        if gamma_value == "0.00" or gamma_value == "0.10":
            continue

        label = f"gamma_{gamma_value}"

        # 1) Train full-info policy (2000 steps, seed 3)
        _run_kpp(scen, steps=FULL_INFO_STEPS, seeds=[FULL_INFO_SEED])

        with Database() as db:
            exp_id = _find_latest_finished_experiment_id(
                db, name=scen.stem, n_steps=FULL_INFO_STEPS, seed=FULL_INFO_SEED
            )

            # 2) Export π policy
            POLICY_DIR.mkdir(parents=True, exist_ok=True)
            policy_out = POLICY_DIR / f"2000x2_full_info_{label}_seed{FULL_INFO_SEED}_pi.npy"
            _export_pi_policy_for_experiment(db, exp_id=exp_id, out_path=policy_out)
            print(
                f"[run_gamma_policy_sweep] Exported π for {scen.stem} (exp_id={exp_id}) -> {policy_out}",
                flush=True,
            )

        # 3) Evaluate in karma-policy scenario (1000 steps, seeds 0/1/2)
        generated_scenario = _write_karma_policy_scenario_for_pi(
            policy_path=policy_out, gamma_value=gamma_value
        )
        _run_kpp(generated_scenario, steps=EVAL_STEPS, seeds=EVAL_SEEDS)

        with Database() as db:
            eval_exp_ids = [
                _find_latest_finished_experiment_id(
                    db, name=generated_scenario.stem, n_steps=EVAL_STEPS, seed=s
                )
                for s in EVAL_SEEDS
            ]
            group_id = _create_experiment_group(db, label=label, exp_ids=eval_exp_ids)
            created_group_ids.append(group_id)
            print(
                f"[run_gamma_policy_sweep] Created exp_group {group_id} label={label} with exp_ids={eval_exp_ids}",
                flush=True,
            )

    # 4) Plot groups 38/39/40 plus created groups together.
    group_ids_to_plot = list(PLOT_BASE_GROUP_IDS) + created_group_ids
    cmd = ["kpp", "vis", "efficiency-fairness"]
    for gid in group_ids_to_plot:
        cmd.extend(["--group-id", str(int(gid))])

    print(f"[run_gamma_policy_sweep] Plotting groups: {group_ids_to_plot}", flush=True)
    print(f"[run_gamma_policy_sweep] Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

