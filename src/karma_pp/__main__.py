from pathlib import Path

import click

from .cli.run_commands import run
from .cli.vis_commands import vis

PROJECT_ROOT = Path(__file__).parent.parent.parent  # ./


@click.group()
def cli():
    """Karma PP CLI."""
    pass


# Add the run commands
cli.add_command(run)
cli.add_command(vis)


if __name__ == "__main__":
    cli()
