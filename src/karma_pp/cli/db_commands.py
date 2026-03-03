import json
import sys
import time
from pathlib import Path

import click
import yaml

from karma_pp.db.base import Database
from karma_pp.logging_config import configure_logging


@click.group()
def db():
    """Database management commands."""
    pass


@db.command()
@click.argument("repository", type=str)
@click.option(
    "--limit", "-l", type=int, default=10, help="Maximum number of items to show"
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format",
)
@click.option(
    "--log-level",
    "-l",
    default="INFO",
    show_default=True,
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Set the logging level.",
)
def list(repository, limit, format, log_level):
    """List items from a repository."""
    configure_logging(level=log_level)
    try:
        with Database() as database:
            if not hasattr(database, repository):
                click.echo(f"❌ Repository '{repository}' not found.")
                click.echo(f"Available repositories: {list(database.__dict__.keys())}")
                return

            repo = getattr(database, repository)

            if repository == "experiment":
                items = repo.list_all()
                if not items:
                    click.echo("No experiments found in database.")
                    return

                # Sort by most recent first
                items = sorted(items, key=lambda x: x.started_at, reverse=True)[:limit]

                if format == "table":
                    click.echo(
                        f"\nExperiments (showing {len(items)} of "
                        f"{len(repo.list_all())}):"
                    )
                    click.echo("-" * 120)
                    click.echo(
                        f"{'ID':<4} {'Name':<20} {'Seed':<6} {'Steps':<6} "
                        f"{'Status':<10} {'Started':<20} {'Runtime':<8}"
                    )
                    click.echo("-" * 120)

                    for item in items:
                        runtime_str = (
                            f"{item.runtime_s:.1f}s" if item.runtime_s else "N/A"
                        )
                        started_str = (
                            item.started_at.strftime("%Y-%m-%d %H:%M:%S")
                            if item.started_at
                            else "N/A"
                        )
                        click.echo(
                            f"{item.exp_id:<4} {item.name[:19]:<20} {item.seed:<6} "
                            f"{item.n_steps:<6} {item.status:<10} {started_str:<20} "
                            f"{runtime_str:<8}"
                        )

                elif format == "json":
                    import json

                    data = []
                    for item in items:
                        data.append(
                            {
                                "exp_id": item.exp_id,
                                "name": item.name,
                                "seed": item.seed,
                                "n_steps": item.n_steps,
                                "status": item.status,
                                "started_at": (
                                    item.started_at.isoformat()
                                    if item.started_at
                                    else None
                                ),
                                "runtime_s": item.runtime_s,
                                "world_hash": item.world_hash,
                                "mechanism_hash": item.mechanism_hash,
                                "population_hash": item.population_hash,
                            }
                        )
                    click.echo(json.dumps(data, indent=2))

                elif format == "csv":
                    import csv
                    import io

                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow(
                        [
                            "exp_id",
                            "name",
                            "seed",
                            "n_steps",
                            "status",
                            "started_at",
                            "runtime_s",
                            "world_hash",
                            "mechanism_hash",
                            "population_hash",
                        ]
                    )

                    for item in items:
                        writer.writerow(
                            [
                                item.exp_id,
                                item.name,
                                item.seed,
                                item.n_steps,
                                item.status,
                                (
                                    item.started_at.isoformat()
                                    if item.started_at
                                    else None
                                ),
                                item.runtime_s,
                                item.world_hash,
                                item.mechanism_hash,
                                item.population_hash,
                            ]
                        )

                    click.echo(output.getvalue())

            elif repository == "metric":
                items = repo.list_all()
                if not items:
                    click.echo("No metrics found in database.")
                    return

                items = items[:limit]

                if format == "table":
                    click.echo(
                        f"\nMetrics (showing {len(items)} of {len(repo.list_all())}):"
                    )
                    click.echo("-" * 80)
                    click.echo(
                        f"{'Exp ID':<8} {'Step':<6} {'Metric Name':<30} "
                        f"{'Value':<20} {'Recorded':<20}"
                    )
                    click.echo("-" * 80)

                    for item in items:
                        recorded_str = (
                            item.recorded_at.strftime("%Y-%m-%d %H:%M:%S")
                            if item.recorded_at
                            else "N/A"
                        )
                        value_str = (
                            str(item.metric_value)[:19] if item.metric_value else "N/A"
                        )
                        click.echo(
                            f"{item.exp_id:<8} {item.step:<6} "
                            f"{item.metric_name[:29]:<30} {value_str:<20} "
                            f"{recorded_str:<20}"
                        )

                elif format == "json":
                    import json

                    data = []
                    for item in items:
                        data.append(
                            {
                                "exp_id": item.exp_id,
                                "step": item.step,
                                "metric_name": item.metric_name,
                                "metric_value": item.metric_value,
                                "recorded_at": (
                                    item.recorded_at.isoformat()
                                    if item.recorded_at
                                    else None
                                ),
                            }
                        )
                    click.echo(json.dumps(data, indent=2))

                elif format == "csv":
                    import csv
                    import io

                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow(
                        ["exp_id", "step", "metric_name", "metric_value", "recorded_at"]
                    )

                    for item in items:
                        writer.writerow(
                            [
                                item.exp_id,
                                item.step,
                                item.metric_name,
                                item.metric_value,
                                (
                                    item.recorded_at.isoformat()
                                    if item.recorded_at
                                    else None
                                ),
                            ]
                        )

                    click.echo(output.getvalue())

            elif repository == "artifact":
                items = repo.list_all()
                if not items:
                    click.echo("No artifacts found in database.")
                    return

                items = items[:limit]

                if format == "table":
                    click.echo(
                        f"\nArtifacts (showing {len(items)} of {len(repo.list_all())}):"
                    )
                    click.echo("-" * 100)
                    click.echo(
                        f"{'ID':<4} {'Exp ID':<8} {'Name':<25} {'Path':<25} "
                        f"{'SHA256':<12} {'Created':<20}"
                    )
                    click.echo("-" * 100)

                    for item in items:
                        created_str = (
                            item.created_at.strftime("%Y-%m-%d %H:%M:%S")
                            if item.created_at
                            else "N/A"
                        )
                        sha256_short = item.sha256[:11] if item.sha256 else "N/A"
                        click.echo(
                            f"{item.artifact_id:<4} {item.exp_id:<8} "
                            f"{item.name[:24]:<25} {item.path[:24]:<25} "
                            f"{sha256_short:<12} {created_str:<20}"
                        )

                elif format == "json":
                    import json

                    data = []
                    for item in items:
                        data.append(
                            {
                                "artifact_id": item.artifact_id,
                                "exp_id": item.exp_id,
                                "name": item.name,
                                "path": item.path,
                                "sha256": item.sha256,
                                "created_at": (
                                    item.created_at.isoformat()
                                    if item.created_at
                                    else None
                                ),
                            }
                        )
                    click.echo(json.dumps(data, indent=2))

                elif format == "csv":
                    import csv
                    import io

                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow(
                        [
                            "artifact_id",
                            "exp_id",
                            "name",
                            "path",
                            "sha256",
                            "created_at",
                        ]
                    )

                    for item in items:
                        writer.writerow(
                            [
                                item.artifact_id,
                                item.exp_id,
                                item.name,
                                item.path,
                                item.sha256,
                                (
                                    item.created_at.isoformat()
                                    if item.created_at
                                    else None
                                ),
                            ]
                        )

                    click.echo(output.getvalue())

            elif repository == "exp_group":
                items = repo.list_all()
                if not items:
                    click.echo("No experiment groups found in database.")
                    return

                items = items[:limit]

                if format == "table":
                    click.echo(
                        f"\nExperiment Groups (showing {len(items)} of "
                        f"{len(repo.list_all())}):"
                    )
                    click.echo("-" * 80)
                    click.echo(
                        f"{'ID':<4} {'Label':<30} {'Members':<8} {'Created':<20}"
                    )
                    click.echo("-" * 80)

                    for item in items:
                        created_str = (
                            item.created_at.strftime("%Y-%m-%d %H:%M:%S")
                            if item.created_at
                            else "N/A"
                        )
                        # Count members for this group
                        member_count = len(
                            database.exp_group.list_members(item.group_id)
                        )
                        click.echo(
                            f"{item.group_id:<4} {item.label[:29]:<30} "
                            f"{member_count:<8} {created_str:<20}"
                        )

                elif format == "json":
                    import json

                    data = []
                    for item in items:
                        member_count = len(
                            database.exp_group.list_members(item.group_id)
                        )
                        data.append(
                            {
                                "group_id": item.group_id,
                                "label": item.label,
                                "created_at": (
                                    item.created_at.isoformat()
                                    if item.created_at
                                    else None
                                ),
                                "member_count": member_count,
                            }
                        )
                    click.echo(json.dumps(data, indent=2))

                elif format == "csv":
                    import csv
                    import io

                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow(["group_id", "label", "created_at", "member_count"])

                    for item in items:
                        member_count = len(
                            database.exp_group.list_members(item.group_id)
                        )
                        writer.writerow(
                            [
                                item.group_id,
                                item.label,
                                (
                                    item.created_at.isoformat()
                                    if item.created_at
                                    else None
                                ),
                                member_count,
                            ]
                        )

                    click.echo(output.getvalue())

            elif repository in ["world", "population"]:
                items = repo.list_all()
                if not items:
                    click.echo(f"No {repository} configurations found in database.")
                    return

                items = items[:limit]

                if format == "table":
                    click.echo(
                        f"\n{repository.title()} Configurations (showing "
                        f"{len(items)} of {len(repo.list_all())}):"
                    )
                    click.echo("-" * 80)
                    click.echo(f"{'Hash':<65} {'Created':<20}")
                    click.echo("-" * 80)

                    for item in items:
                        created_str = (
                            item.created_at.strftime("%Y-%m-%d %H:%M:%S")
                            if item.created_at
                            else "N/A"
                        )
                        hash_key = f"{repository}_hash"
                        hash_short = (
                            item.__dict__[hash_key][:64]
                            if hasattr(item, hash_key)
                            else "N/A"
                        )
                        click.echo(f"{hash_short:<65} {created_str:<20}")

                elif format == "json":
                    import json

                    data = []
                    for item in items:
                        item_data = item.__dict__.copy()
                        item_data["created_at"] = (
                            item.created_at.isoformat() if item.created_at else None
                        )
                        data.append(item_data)
                    click.echo(json.dumps(data, indent=2))

                elif format == "csv":
                    import csv
                    import io

                    output = io.StringIO()
                    writer = csv.writer(output)

                    # Get the hash column name
                    hash_col = f"{repository}_hash"
                    if repository == "world":
                        # World table still stores JSON
                        writer.writerow([hash_col, "json", "created_at"])
                        for item in items:
                            writer.writerow(
                                [
                                    getattr(item, hash_col, "N/A"),
                                    item.json,
                                    (
                                        item.created_at.isoformat()
                                        if item.created_at
                                        else None
                                    ),
                                ]
                            )
                    else:
                        # Population table no longer stores JSON
                        writer.writerow([hash_col, "created_at"])
                        for item in items:
                            writer.writerow(
                                [
                                    getattr(item, hash_col, "N/A"),
                                    (
                                        item.created_at.isoformat()
                                        if item.created_at
                                        else None
                                    ),
                                ]
                            )

                    click.echo(output.getvalue())

            else:
                click.echo(
                    f"Repository '{repository}' not yet implemented for listing."
                )
                click.echo(f"Available repositories: {list(database.__dict__.keys())}")

    except Exception as e:
        click.echo(f"❌ Error accessing database: {e}")
        import traceback

        traceback.print_exc()


@db.command()
@click.argument("repository", type=str)
@click.argument("id_or_hash", type=str)
@click.option(
    "--log-level",
    "-l",
    default="INFO",
    show_default=True,
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Set the logging level.",
)
def show(repository, id_or_hash, log_level):
    """Show details of a specific item from a repository."""
    configure_logging(level=log_level)
    try:
        with Database() as database:
            if not hasattr(database, repository):
                click.echo(f"❌ Repository '{repository}' not found.")
                click.echo(f"Available repositories: {list(database.__dict__.keys())}")
                return

            repo = getattr(database, repository)

            if repository == "experiment":
                try:
                    exp_id = int(id_or_hash)
                    item = repo.get(exp_id)
                    if not item:
                        click.echo(f"❌ Experiment {exp_id} not found.")
                        return
                except ValueError:
                    click.echo(f"❌ Invalid experiment ID: {id_or_hash}")
                    return

                click.echo(f"\nExperiment {item.exp_id}:")
                click.echo(f"  Name: {item.name}")
                click.echo(f"  Seed: {item.seed}")
                click.echo(f"  Steps: {item.n_steps}")
                click.echo(f"  Status: {item.status}")
                click.echo(f"  Started: {item.started_at}")
                click.echo(f"  Ended: {item.ended_at}")
                click.echo(
                    f"  Runtime: {item.runtime_s}s"
                    if item.runtime_s
                    else "  Runtime: N/A"
                )
                click.echo(f"  World Hash: {item.world_hash}")
                click.echo(f"  Mechanism Hash: {item.mechanism_hash}")
                click.echo(f"  Population Hash: {item.population_hash}")
                click.echo(f"  Git Commit: {item.git_commit}")
                if item.comment:
                    click.echo(f"  Comment: {item.comment}")

            elif repository in ["world", "population"]:
                item = repo.get(id_or_hash)
                if not item:
                    click.echo(
                        f"❌ {repository.title()} configuration with hash "
                        f"{id_or_hash} not found."
                    )
                    return

                hash_col = f"{repository}_hash"
                click.echo(f"\n{repository.title()} Configuration:")
                click.echo(f"  Hash: {getattr(item, hash_col)}")
                click.echo(f"  Created: {item.created_at}")

                if repository == "world":
                    click.echo("  JSON Configuration:")
                    click.echo(f"    {item.json}")

            else:
                click.echo(
                    f"Repository '{repository}' not yet implemented for "
                    f"showing details."
                )

    except Exception as e:
        click.echo(f"❌ Error accessing database: {e}")
        import traceback

        traceback.print_exc()


@db.command()
@click.option(
    "--log-level",
    "-l",
    default="INFO",
    show_default=True,
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Set the logging level.",
)
def info(log_level):
    """Show database information and statistics."""
    configure_logging(level=log_level)
    try:
        with Database() as database:
            click.echo("Database Information:")
            click.echo("=" * 50)

            # Count items in each repository
            for repo_name in [
                "experiment",
                "metric",
                "artifact",
                "world",
                "population",
                "exp_group",
            ]:
                if hasattr(database, repo_name):
                    repo = getattr(database, repo_name)
                    try:
                        count = len(repo.list_all())
                        click.echo(f"{repo_name.title()}: {count} items")
                    except Exception as e:
                        click.echo(f"{repo_name.title()}: Error counting - {e}")

            # Show database path
            from karma_pp.db.base import DB_PATH

            click.echo(f"\nDatabase Path: {DB_PATH}")
            if DB_PATH.exists():
                size_mb = DB_PATH.stat().st_size / (1024 * 1024)
                click.echo(f"Database Size: {size_mb:.2f} MB")
            else:
                click.echo("Database file does not exist")

    except Exception as e:
        click.echo(f"❌ Error accessing database: {e}")
        import traceback

        traceback.print_exc()


@db.command(name="show-metric")
@click.argument("metric_name", type=str)
@click.option("--experiment", "-e", type=int, required=True, help="Experiment ID")
@click.option(
    "--save",
    is_flag=True,
    help="Save metric data to data/metrics/experiment_[exp_id]/[metric_name].yml",
)
@click.option(
    "--log-level",
    "-l",
    default="INFO",
    show_default=True,
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Set the logging level.",
)
def show_metric(metric_name, experiment, save, log_level):
    """Show metrics by name for a specific experiment."""
    configure_logging(level=log_level)
    try:
        with Database() as database:
            # Check if experiment exists
            experiment_obj = database.experiment.get(experiment)
            if not experiment_obj:
                click.echo(f"❌ Experiment {experiment} not found.")
                return

            # Get metrics for this experiment and metric name
            metrics = database.metric.filter_by(
                exp_id=experiment, metric_name=metric_name
            )

            if not metrics:
                click.echo(
                    f"\n❌ No metrics found with name '{metric_name}' for "
                    f"experiment {experiment}."
                )
                return

            # Sort metrics by step
            metrics_sorted = sorted(
                metrics, key=lambda m: m.step if m.step is not None else -1
            )

            # Collect all parsed values into a list
            all_values = []
            for metric in metrics_sorted:
                # Parse the JSON string to get the actual data structure
                parsed_value = json.loads(metric.metric_value)

                # Handle cases where the parsed value is a string representation of a
                # list/array
                if (
                    isinstance(parsed_value, str)
                    and parsed_value.startswith("[")
                    and parsed_value.endswith("]")
                ):
                    try:
                        # Try to parse it as a list (remove brackets and split by
                        # spaces)
                        content = parsed_value[1:-1].strip()
                        if content:
                            # Split by spaces and convert to numbers
                            items = [int(x) for x in content.split() if x.strip()]
                            parsed_value = items
                        else:
                            parsed_value = []
                    except (ValueError, AttributeError):
                        # If parsing fails, keep the original value
                        pass

                all_values.append(parsed_value)

            # Print the complete list of lists
            click.echo(all_values)

            # Save to file if requested
            if save:
                try:
                    # Create the directory structure
                    metrics_dir = Path("data/metrics")
                    exp_dir = metrics_dir / f"experiment_{experiment}"
                    exp_dir.mkdir(parents=True, exist_ok=True)

                    # Create the output file path
                    output_file = exp_dir / f"{metric_name}.yml"

                    # Prepare data for YAML output
                    yaml_data = {
                        "experiment_id": experiment,
                        "metric_name": metric_name,
                        "metric_values": all_values,
                        "total_steps": len(all_values),
                        "experiment_name": (
                            experiment_obj.name if experiment_obj else None
                        ),
                        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }

                    # Write to YAML file
                    with open(output_file, "w") as f:
                        yaml.dump(yaml_data, f, default_flow_style=False, indent=2)

                    click.echo(f"\n💾 Metric data saved to: {output_file}")

                except Exception as save_error:
                    click.echo(f"⚠️  Warning: Failed to save metric data: {save_error}")

    except Exception as e:
        click.echo(f"❌ Error showing metrics: {e}")
        import traceback

        traceback.print_exc()


@db.command(name="export")
@click.argument("experiment", type=int)
@click.option(
    "--log-level",
    "-l",
    default="INFO",
    show_default=True,
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Set the logging level.",
)
def export(experiment, log_level):
    """Export all standard metrics for an experiment to YAML files."""
    configure_logging(level=log_level)
    try:
        with Database() as database:
            # Check if experiment exists
            experiment_obj = database.experiment.get(experiment)
            if not experiment_obj:
                click.echo(f"❌ Experiment {experiment} not found.")
                return

            click.echo(
                f"📊 Exporting metrics for experiment {experiment} "
                f"({experiment_obj.name})..."
            )

            # Define the metrics to export
            metrics_to_export = [
                "agent_publics",
                "agent_privates",
                "agent_balances",
                "agent_commits",
                "agent_transfers",
            ]

            # Create the directory structure
            export_dir = Path("data/metrics") / f"experiment_{experiment}"
            export_dir.mkdir(parents=True, exist_ok=True)

            exported_count = 0
            failed_count = 0

            for metric_name in metrics_to_export:
                try:
                    # Get metrics for this experiment and metric name
                    metrics = database.metric.filter_by(
                        exp_id=experiment, metric_name=metric_name
                    )

                    if not metrics:
                        click.echo(f"  ⚠️  No metrics found for '{metric_name}'")
                        continue

                    # Sort metrics by step
                    metrics_sorted = sorted(
                        metrics, key=lambda m: m.step if m.step is not None else -1
                    )

                    # Collect all parsed values into a list
                    all_values = []
                    for metric in metrics_sorted:
                        # Parse the JSON string to get the actual data structure
                        parsed_value = json.loads(metric.metric_value)

                        # Handle cases where the parsed value is a string
                        # representation of a list/array
                        if (
                            isinstance(parsed_value, str)
                            and parsed_value.startswith("[")
                            and parsed_value.endswith("]")
                        ):
                            try:
                                # Try to parse it as a list (remove brackets and split
                                # by spaces)
                                content = parsed_value[1:-1].strip()
                                if content:
                                    # Split by spaces and convert to numbers
                                    items = [
                                        int(x) for x in content.split() if x.strip()
                                    ]
                                    parsed_value = items
                                else:
                                    parsed_value = []
                            except (ValueError, AttributeError):
                                # If parsing fails, keep the original value
                                pass

                        all_values.append(parsed_value)

                    # Create the output file path
                    output_file = export_dir / f"{metric_name}.yml"

                    # Prepare data for YAML output
                    yaml_data = {
                        "experiment_id": experiment,
                        "metric_name": metric_name,
                        "metric_values": all_values,
                        "total_steps": len(all_values),
                        "experiment_name": experiment_obj.name,
                        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }

                    # Write to YAML file
                    with open(output_file, "w") as f:
                        yaml.dump(yaml_data, f, default_flow_style=False, indent=2)

                    click.echo(
                        f"  ✅ Exported {metric_name} ({len(all_values)} steps) to "
                        f"{output_file}"
                    )
                    exported_count += 1

                except Exception as metric_error:
                    click.echo(f"  ❌ Failed to export {metric_name}: {metric_error}")
                    failed_count += 1

            # Summary
            click.echo("\n📁 Export completed:")
            click.echo(f"  📂 Directory: {export_dir}")
            click.echo(f"  ✅ Successfully exported: {exported_count} metrics")
            if failed_count > 0:
                click.echo(f"  ❌ Failed: {failed_count} metrics")

            if exported_count > 0:
                click.echo(f"\n💾 All exported files are saved in: {export_dir}")

    except Exception as e:
        click.echo(f"❌ Error exporting experiment: {e}")
        import traceback

        traceback.print_exc()


@db.command(name="create-group")
@click.argument("label", type=str)
@click.option(
    "--experiments",
    "-e",
    type=str,
    help='Comma-separated list of experiment IDs (e.g., "8,9,10")',
)
@click.option(
    "--log-level",
    "-l",
    default="INFO",
    show_default=True,
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Set the logging level.",
)
def create_group(label, experiments, log_level):
    """Create a new experiment group with an optional list of experiment IDs."""
    configure_logging(level=log_level)
    try:
        with Database() as database:
            # Create the experiment group
            group = database.exp_group.create(label)
            click.echo(
                f"✅ Created experiment group '{label}' with ID {group.group_id}"
            )

            # Add experiments if provided
            if experiments:
                # Parse comma-separated experiment IDs
                try:
                    exp_ids = [int(exp_id.strip()) for exp_id in experiments.split(",")]
                except ValueError:
                    click.echo(
                        "❌ Error: Invalid experiment ID format. Use comma-separated "
                        "numbers (e.g., '8,9,10')"
                    )
                    return

                added_count = 0
                for exp_id in exp_ids:
                    try:
                        # Check if experiment exists
                        experiment = database.experiment.get(exp_id)
                        if not experiment:
                            click.echo(
                                f"⚠️  Warning: Experiment {exp_id} not found, skipping"
                            )
                            continue

                        # Add to group
                        database.exp_group.add_member(group.group_id, exp_id)
                        click.echo(
                            f"  ✓ Added experiment {exp_id} ({experiment.name}) to "
                            f"group"
                        )
                        added_count += 1

                    except Exception as e:
                        click.echo(
                            f"⚠️  Warning: Failed to add experiment {exp_id}: {e}"
                        )

                if added_count > 0:
                    click.echo(
                        f"\n✅ Successfully added {added_count} experiments to "
                        f"group '{label}'"
                    )
                else:
                    click.echo(f"\n⚠️  No experiments were added to group '{label}'")
            else:
                click.echo(f"ℹ️  Group '{label}' created without any experiments")

            # Show group summary
            click.echo("\nGroup Summary:")
            click.echo(f"  ID: {group.group_id}")
            click.echo(f"  Label: {group.label}")
            click.echo(f"  Created: {group.created_at}")
            click.echo(
                f"  Members: {len(database.exp_group.list_members(group.group_id))}"
            )

    except Exception as e:
        click.echo(f"❌ Error creating experiment group: {e}")
        import traceback

        traceback.print_exc()


@db.command()
@click.argument("repository", type=str)
@click.argument("ids", nargs=-1, type=str)
@click.option(
    "--log-level",
    "-l",
    default="INFO",
    show_default=True,
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Set the logging level.",
)
def delete(repository, ids, log_level):
    """Delete items from a repository by ID or hash."""
    configure_logging(level=log_level)
    if not ids:
        click.echo(
            "❌ No IDs provided. Usage: dcc db delete <repository> "
            "<id1> [id2] [id3] ..."
        )
        return

    try:
        with Database() as database:
            if not hasattr(database, repository):
                click.echo(f"❌ Repository '{repository}' not found.")
                click.echo(f"Available repositories: {list(database.__dict__.keys())}")
                return

            repo = getattr(database, repository)

            if repository == "experiment":
                delete_experiments(database, repo, ids)
            elif repository == "metric":
                delete_metrics(database, repo, ids)
            elif repository == "artifact":
                delete_artifacts(database, repo, ids)
            elif repository == "exp_group":
                delete_exp_groups(database, repo, ids)
            elif repository in ["world", "population"]:
                delete_configs(database, repo, repository, ids)
            else:
                click.echo(
                    f"Repository '{repository}' not yet implemented for deletion."
                )
                click.echo(f"Available repositories: {list(database.__dict__.keys())}")

    except Exception as e:
        click.echo(f"❌ Error deleting from repository '{repository}': {e}")
        import traceback

        traceback.print_exc()


def delete_experiments(database, repo, ids):
    """Delete experiments and their associated data."""
    # Parse experiment IDs
    exp_ids = []
    for id_str in ids:
        try:
            exp_ids.append(int(id_str))
        except ValueError:
            click.echo(f"❌ Invalid experiment ID: {id_str}")
            return

    if not exp_ids:
        click.echo("❌ No valid experiment IDs provided.")
        return

    # Check if experiments exist
    experiments = []
    for exp_id in exp_ids:
        experiment = repo.get(exp_id)
        if not experiment:
            click.echo(f"❌ Experiment {exp_id} not found.")
            return
        experiments.append(experiment)

    # Show experiments to delete
    click.echo("\nExperiments to delete:")
    for experiment in experiments:
        click.echo(
            f"  ID: {experiment.exp_id}, Name: {experiment.name}, "
            f"Seed: {experiment.seed}, Status: {experiment.status}"
        )

    # Count associated data
    total_metrics = 0
    total_artifacts = 0
    for exp_id in exp_ids:
        total_metrics += len(database.metric.filter_by(exp_id=exp_id))
        total_artifacts += len(database.artifact.filter_by(exp_id=exp_id))

    click.echo("\nAssociated data to delete:")
    click.echo(f"  Total Metrics: {total_metrics}")
    click.echo(f"  Total Artifacts: {total_artifacts}")

    # Confirmation prompt
    if not click.confirm(
        f"\n⚠️  Are you sure you want to delete {len(exp_ids)} experiments "
        f"and all their data?"
    ):
        click.echo("Deletion cancelled.")
        return

    click.echo(f"\nDeleting {len(exp_ids)} experiments...")

    # Delete in order: metrics, artifacts, then experiments
    deleted_metrics = 0
    deleted_artifacts = 0

    for exp_id in exp_ids:
        # Delete metrics
        metrics_count = len(database.metric.filter_by(exp_id=exp_id))
        if metrics_count > 0:
            database.conn.execute("DELETE FROM metric WHERE exp_id = ?", (exp_id,))
            deleted_metrics += metrics_count

        # Delete artifacts
        artifacts_count = len(database.artifact.filter_by(exp_id=exp_id))
        if artifacts_count > 0:
            database.conn.execute("DELETE FROM artifact WHERE exp_id = ?", (exp_id,))
            deleted_artifacts += artifacts_count

        # Delete experiment
        database.conn.execute("DELETE FROM experiment WHERE exp_id = ?", (exp_id,))

    click.echo(f"  ✓ Deleted {deleted_metrics} metrics")
    click.echo(f"  ✓ Deleted {deleted_artifacts} artifacts")
    click.echo(f"  ✓ Deleted {len(exp_ids)} experiments")

    click.echo(
        f"\n✅ Successfully deleted {len(exp_ids)} experiments and all "
        f"associated data."
    )

    # Show updated counts
    remaining_experiments = len(database.experiment.list_all())
    remaining_metrics = len(database.metric.list_all())
    remaining_artifacts = len(database.artifact.list_all())

    click.echo("\nRemaining in database:")
    click.echo(f"  Experiments: {remaining_experiments}")
    click.echo(f"  Metrics: {remaining_metrics}")
    click.echo(f"  Artifacts: {remaining_artifacts}")


def delete_metrics(database, repo, ids):
    """Delete metrics by ID."""
    # Parse metric IDs (format: exp_id:step:metric_name)
    metric_keys = []
    for id_str in ids:
        parts = id_str.split(":")
        if len(parts) != 3:
            click.echo(
                f"❌ Invalid metric ID format: {id_str}. Expected format: "
                f"exp_id:step:metric_name"
            )
            return
        try:
            exp_id = int(parts[0])
            step = int(parts[1]) if parts[1] != "null" else None
            metric_name = parts[2]
            metric_keys.append((exp_id, step, metric_name))
        except ValueError:
            click.echo(f"❌ Invalid metric ID: {id_str}")
            return

    if not metric_keys:
        click.echo("❌ No valid metric IDs provided.")
        return

    # Show metrics to delete
    click.echo("\nMetrics to delete:")
    for exp_id, step, metric_name in metric_keys:
        click.echo(f"  Exp: {exp_id}, Step: {step}, Metric: {metric_name}")

    # Confirmation prompt
    if not click.confirm(
        f"\n⚠️  Are you sure you want to delete {len(metric_keys)} metrics?"
    ):
        click.echo("Deletion cancelled.")
        return

    click.echo(f"\nDeleting {len(metric_keys)} metrics...")

    # Delete metrics
    deleted_count = 0
    for exp_id, step, metric_name in metric_keys:
        if step is None:
            database.conn.execute(
                "DELETE FROM metric WHERE exp_id = ? AND step IS NULL AND "
                "metric_name = ?",
                (exp_id, metric_name),
            )
        else:
            database.conn.execute(
                "DELETE FROM metric WHERE exp_id = ? AND step = ? AND metric_name = ?",
                (exp_id, step, metric_name),
            )
        deleted_count += 1

    click.echo(f"  ✓ Deleted {deleted_count} metrics")
    click.echo(f"\n✅ Successfully deleted {deleted_count} metrics.")


def delete_artifacts(database, repo, ids):
    """Delete artifacts by ID."""
    # Parse artifact IDs
    artifact_ids = []
    for id_str in ids:
        try:
            artifact_ids.append(int(id_str))
        except ValueError:
            click.echo(f"❌ Invalid artifact ID: {id_str}")
            return

    if not artifact_ids:
        click.echo("❌ No valid artifact IDs provided.")
        return

    # Check if artifacts exist
    artifacts = []
    for artifact_id in artifact_ids:
        artifact = repo.get(artifact_id)
        if not artifact:
            click.echo(f"❌ Artifact {artifact_id} not found.")
            return
        artifacts.append(artifact)

    # Show artifacts to delete
    click.echo("\nArtifacts to delete:")
    for artifact in artifacts:
        click.echo(
            f"  ID: {artifact.artifact_id}, Name: {artifact.name}, "
            f"Path: {artifact.path}"
        )

    # Confirmation prompt
    if not click.confirm(
        f"\n⚠️  Are you sure you want to delete {len(artifact_ids)} artifacts?"
    ):
        click.echo("Deletion cancelled.")
        return

    click.echo(f"\nDeleting {len(artifact_ids)} artifacts...")

    # Delete artifacts
    for artifact_id in artifact_ids:
        database.conn.execute(
            "DELETE FROM artifact WHERE artifact_id = ?", (artifact_id,)
        )

    click.echo(f"  ✓ Deleted {len(artifact_ids)} artifacts")
    click.echo(f"\n✅ Successfully deleted {len(artifact_ids)} artifacts.")


def delete_configs(database, repo, repo_name, ids):
    """Delete world/population configurations by hash."""
    # Check if configs exist
    configs = []
    for config_hash in ids:
        config = repo.get(config_hash)
        if not config:
            click.echo(
                f"❌ {repo_name.title()} configuration with hash "
                f"{config_hash} not found."
            )
            return
        configs.append(config)

    # Show configs to delete
    click.echo(f"\n{repo_name.title()} configurations to delete:")
    for config in configs:
        hash_col = f"{repo_name}_hash"
        click.echo(f"  Hash: {getattr(config, hash_col)}")

    # Check if any experiments reference these configs
    if repo_name == "world":
        ref_experiments = []
        for config in configs:
            hash_col = f"{repo_name}_hash"
            refs = database.experiment.filter_by(world_hash=getattr(config, hash_col))
            ref_experiments.extend(refs)
    else:  # population
        ref_experiments = []
        for config in configs:
            hash_col = f"{repo_name}_hash"
            refs = database.experiment.filter_by(
                population_hash=getattr(config, hash_col)
            )
            ref_experiments.extend(refs)

    if ref_experiments:
        click.echo(
            f"\n⚠️  Warning: {len(ref_experiments)} experiments reference "
            f"these configurations:"
        )
        for exp in ref_experiments[:5]:  # Show first 5
            click.echo(f"  - Experiment {exp.exp_id}: {exp.name}")
        if len(ref_experiments) > 5:
            click.echo(f"  ... and {len(ref_experiments) - 5} more")
        click.echo(
            "  Deleting these configurations may cause foreign key constraint "
            "violations."
        )

    # Confirmation prompt
    if not click.confirm(
        f"\n⚠️  Are you sure you want to delete {len(configs)} "
        f"{repo_name} configurations?"
    ):
        click.echo("Deletion cancelled.")
        return

    click.echo(f"\nDeleting {len(configs)} {repo_name} configurations...")

    # Delete configs
    for config in configs:
        hash_col = f"{repo_name}_hash"
        database.conn.execute(
            f"DELETE FROM {repo_name} WHERE {hash_col} = ?",
            (getattr(config, hash_col),),
        )

    click.echo(f"  ✓ Deleted {len(configs)} {repo_name} configurations")
    click.echo(f"\n✅ Successfully deleted {len(configs)} {repo_name} configurations.")


def delete_exp_groups(database, repo, ids):
    """Delete experiment groups and their members."""
    # Parse group IDs
    group_ids = []
    for id_str in ids:
        try:
            group_ids.append(int(id_str))
        except ValueError:
            click.echo(f"❌ Invalid group ID: {id_str}")
            return

    if not group_ids:
        click.echo("❌ No valid group IDs provided.")
        return

    # Check if groups exist
    groups = []
    for group_id in group_ids:
        group = repo.get(group_id)
        if not group:
            click.echo(f"❌ Experiment group {group_id} not found.")
            return
        groups.append(group)

    # Check which groups have members
    groups_with_members = []
    groups_without_members = []

    for group in groups:
        member_count = len(database.exp_group.list_members(group.group_id))
        if member_count > 0:
            groups_with_members.append((group, member_count))
        else:
            groups_without_members.append(group)

    # Show groups to delete
    click.echo("\nExperiment groups to delete:")
    for group in groups:
        member_count = len(database.exp_group.list_members(group.group_id))
        click.echo(
            f"  ID: {group.group_id}, Label: '{group.label}', "
            f"Members: {member_count}"
        )

    # Count total members
    total_members = sum(count for _, count in groups_with_members)

    # If groups have members, warn about member records being deleted
    if groups_with_members:
        click.echo("\n⚠️  Warning: The following groups have members:")
        for group, member_count in groups_with_members:
            exp_ids = database.exp_group.list_members(group.group_id)
            click.echo(
                f"  - Group {group.group_id} ('{group.label}'): "
                f"{member_count} members {exp_ids}"
            )
        click.echo(
            "  Group member records will be deleted, but experiments will be "
            "preserved."
        )

    # Confirmation prompt
    if not click.confirm(
        f"\n⚠️  Are you sure you want to delete {len(group_ids)} experiment groups?"
    ):
        click.echo("Deletion cancelled.")
        return

    click.echo(f"\nDeleting {len(group_ids)} experiment groups...")

    # Delete groups and clean up orphaned member records
    for group in groups:
        # Delete orphaned member records first
        database.conn.execute(
            "DELETE FROM exp_group_member WHERE group_id = ?", (group.group_id,)
        )
        # Delete the group
        database.conn.execute(
            "DELETE FROM exp_group WHERE group_id = ?", (group.group_id,)
        )

    total_members = sum(count for _, count in groups_with_members)
    if total_members > 0:
        click.echo(f"  ✓ Preserved {total_members} experiments")
    else:
        click.echo("  ✓ No experiments to preserve")

    click.echo(f"  ✓ Deleted {len(group_ids)} experiment groups")

    if groups_with_members:
        click.echo(f"\n✅ Successfully deleted {len(group_ids)} experiment groups.")
        click.echo(f"✓ {total_members} experiments were preserved.")
    else:
        click.echo(f"\n✅ Successfully deleted {len(group_ids)} experiment groups.")

    # Show updated counts
    remaining_groups = len(database.exp_group.list_all())

    click.echo("\nRemaining in database:")
    click.echo(f"  Experiment Groups: {remaining_groups}")
