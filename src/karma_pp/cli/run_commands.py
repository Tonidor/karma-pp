from pathlib import Path
from typing import Any
import click
import structlog
import yaml

from karma_pp.logging_config import configure_logging
from karma_pp.core.simulation import create_components, run_simulation

log = structlog.get_logger(__name__)


def _resolve_path(base_dir: Path, path_str: str) -> Path:
    """Resolve a config file path relative to the scenario file's directory."""
    p = Path(path_str)
    return p if p.is_absolute() else base_dir / p


@click.command()
@click.option('--scenario', type=click.Path(exists=True), required=True)
@click.option('--steps', type=int, default=5)
@click.option('--seed', type=int, default=42)
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), default='INFO')
def run(scenario, steps, seed, log_level):
    """Run a simulation with the given scenario."""
    configure_logging(level=log_level)
    scenario_path = Path(scenario).resolve()
    scenario_dir = scenario_path.parent
    log.info("loading_scenario", scenario=str(scenario_path), steps=steps, seed=seed)

    with open(scenario_path, "r") as f:
        scenario_cfg = yaml.safe_load(f)

    world_path = _resolve_path(scenario_dir, scenario_cfg["world_file"])
    mechanism_path = _resolve_path(scenario_dir, scenario_cfg["mechanism_file"])

    with open(world_path, "r") as f:
        world_cfg = yaml.safe_load(f)
    with open(mechanism_path, "r") as f:
        mechanism_cfg = yaml.safe_load(f)
    population_params: dict[str, Any] = scenario_cfg["population"]

    log.debug(
        "config_loaded",
        world_file=str(world_path),
        mechanism_file=str(mechanism_path),
        population_params=list(population_params.keys()),
    )

    world, mechanism, population = create_components(world_cfg, mechanism_cfg, population_params)
    log.info("components_created", world_type=type(world).__name__, mechanism_type=type(mechanism).__name__)

    results = run_simulation(world, population, mechanism, steps, seed)
    log.info("simulation_complete", num_results=len(results), total_steps=steps)


if __name__ == "__main__":
    run()
