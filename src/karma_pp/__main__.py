from pathlib import Path

import click

from .cli.run_commands import run
from .cli.vis_commands import vis
from .cli.db_commands import db

PROJECT_ROOT = Path(__file__).parent.parent.parent  # ./


@click.group()
def cli():
    """Karma PP CLI."""
    pass


cli.add_command(run)
cli.add_command(vis)
cli.add_command(db)


if __name__ == "__main__":
    cli()
