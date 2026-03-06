from dataclasses import asdict
from pathlib import Path
from typing import Any
import json
import time

import click
import structlog
import yaml
import numpy as np

from karma_pp.logging_config import configure_logging
from karma_pp.db.base import Database
from karma_pp.db.experiment.models import Experiment
from karma_pp.core.simulation import create_components, run_simulation

log = structlog.get_logger(__name__)


def _serialize_for_db(obj: Any) -> str:
    """Convert any object to a JSON string for database storage."""

    def _convert_to_primitive(x: Any) -> Any:
        if x is None:
            return None
        if isinstance(x, (str, int, float, bool)):
            return x
        if hasattr(x, "_asdict"):
            return {k: _convert_to_primitive(v) for k, v in x._asdict().items()}
        if hasattr(x, "__dict__"):
            return {k: _convert_to_primitive(v) for k, v in asdict(x).items()}
        if isinstance(x, (list, tuple)):
            return [_convert_to_primitive(v) for v in x]
        if isinstance(x, dict):
            return {k: _convert_to_primitive(v) for k, v in x.items()}
        if hasattr(x, "tolist"):
            return _convert_to_primitive(x.tolist())
        if hasattr(x, "__array__"):
            return _convert_to_primitive(x.__array__().tolist())
        return str(x)

    primitive = _convert_to_primitive(obj)
    return json.dumps(primitive, default=str)


def _save_metrics(experiment: Experiment, results: list, db: Database) -> None:
    """Persist per-step metrics for a finished experiment."""
    for step, result in enumerate(results):
        db.metric.create(
            exp_id=experiment.exp_id,
            step=step,
            metric_name="collectives",
            metric_value=_serialize_for_db(result.collectives),
        )
        db.metric.create(
            exp_id=experiment.exp_id,
            step=step,
            metric_name="reports",
            metric_value=_serialize_for_db(result.reports),
        )
        db.metric.create(
            exp_id=experiment.exp_id,
            step=step,
            metric_name="rewards",
            metric_value=_serialize_for_db(result.rewards),
        )
        db.metric.create(
            exp_id=experiment.exp_id,
            step=step,
            metric_name="world_state",
            metric_value=_serialize_for_db(result.world_state),
        )
        db.metric.create(
            exp_id=experiment.exp_id,
            step=step,
            metric_name="population_state",
            metric_value=_serialize_for_db(result.population_state),
        )
        db.metric.create(
            exp_id=experiment.exp_id,
            step=step,
            metric_name="mechanism_state",
            metric_value=_serialize_for_db(result.mechanism_state),
        )


def _create_population_hash(db: Database, population_cfg: dict[str, Any]) -> str:
    """Persist agent configs and population members, return population_hash."""
    members: list[dict[str, Any]] = []
    for model_id, cfg in population_cfg.items():
        n_agents = int(cfg["n_agents"])
        weight = float(cfg["weight"])
        agent_model_cfg = cfg["agent_model"]
        agent_hash = db.agent.ensure(agent_model_cfg)
        members.append(
            {
                "model_id": model_id,
                "agent_hash": agent_hash,
                "n_agents": n_agents,
                "weight": weight,
            }
        )
    population_blob = {"members": members}
    return db.population.ensure(population_blob)


@click.command()
@click.option(
    "--scenario",
    "-s",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to unified scenario YAML (world, mechanism, population).",
)
@click.option(
    "--steps",
    type=int,
    default=20,
    show_default=True,
    help="Number of simulation steps.",
)
@click.option(
    "--seed",
    type=int,
    multiple=True,
    default=[42],
    show_default=True,
    help="Random seed (use multiple --seed for several runs).",
)
@click.option(
    "--allow-uncommitted",
    is_flag=True,
    help="Allow running with uncommitted changes (currently not enforced).",
)
@click.option(
    "--log-level",
    "-l",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Set the logging level.",
)
def run(scenario: Path, steps: int, seed: tuple[int, ...], allow_uncommitted: bool, log_level: str):
    """Run a simulation with the given unified scenario and persist it to the database."""
    configure_logging(level=log_level)
    scenario_path = Path(scenario).resolve()
    log.info("loading_scenario", scenario=str(scenario_path), steps=steps, seeds=list(seed))

    with open(scenario_path, "r") as f:
        scenario_cfg = yaml.safe_load(f)

    world_cfg: dict[str, Any] = scenario_cfg["world"]
    mechanism_cfg: dict[str, Any] = scenario_cfg["mechanism"]
    population_cfg: dict[str, Any] = scenario_cfg["population"]

    scenario_name = scenario_path.stem
    seeds = list(seed)
    exp_ids: list[int] = []

    for s in seeds:
        log.info("starting_experiment", scenario=scenario_name, seed=s, steps=steps)

        # Persist configs and create experiment
        with Database() as db:
            world_hash = db.world.ensure(world_cfg)
            mechanism_hash = db.mechanism.ensure(mechanism_cfg)
            population_hash = _create_population_hash(db, population_cfg)

            experiment = db.experiment.create(
                world_hash=world_hash,
                mechanism_hash=mechanism_hash,
                population_hash=population_hash,
                seed=s,
                n_steps=steps,
                git_commit="unknown",
                name=scenario_name,
                comment=f"Auto-generated experiment for seed {s}",
            )
            db.experiment.set_status(experiment.exp_id, status="running")

        # Run simulation (no DB connection held)
        world, mechanism, population = create_components(world_cfg, mechanism_cfg, population_cfg)
        start_time = time.time()
        results = run_simulation(world, population, mechanism, steps, s)
        runtime_s = time.time() - start_time

        # Save metrics and finalize experiment
        with Database() as db:
            _save_metrics(experiment, results, db)
            db.experiment.set_status(
                experiment.exp_id,
                status="finished",
                runtime_s=runtime_s,
            )

        exp_ids.append(experiment.exp_id)
        log.info("experiment_complete", exp_id=experiment.exp_id, seed=s, runtime_s=runtime_s)

    log.info("all_experiments_complete", exp_ids=exp_ids)


if __name__ == "__main__":
    run()
