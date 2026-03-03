from datetime import datetime

import structlog
import duckdb

from ..base import BaseRepo
from .models import Experiment

log = structlog.get_logger(__name__)


class ExperimentRepo(BaseRepo[Experiment]):
    """
    Repo for the `experiment` table. Inherits generic CRUD from BaseRepo
    and adds a `create()` method that returns a fully-populated Experiment.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        super().__init__(conn, Experiment)

    def create(
        self,
        world_hash: str,
        mechanism_hash: str,
        population_hash: str,
        seed: int,
        n_steps: int,
        git_commit: str,
        name: str,
        comment: str = "",
        runtime_s: float = 0.0,
    ) -> Experiment:
        """
        Insert a new experiment and return the Experiment dataclass
        (including the generated exp_id and timestamp).
        """
        log.info(
            "insert-experiment",
            world_hash=world_hash,
            mechanism_hash=mechanism_hash,
            population_hash=population_hash,
            seed=seed,
            git_commit=git_commit,
            name=name,
        )
        sql = """
            INSERT INTO experiment
                (world_hash, mechanism_hash, population_hash,
                 seed, n_steps, git_commit, name, comment, runtime_s, status)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            RETURNING
                exp_id, world_hash, mechanism_hash, population_hash,
                seed, n_steps, git_commit, name, comment, runtime_s,
                started_at, ended_at, status
        """
        row = self.conn.execute(
            sql,
            (
                world_hash,
                mechanism_hash,
                population_hash,
                seed,
                n_steps,
                git_commit,
                name,
                comment,
                runtime_s,
            ),
        ).fetchone()
        return Experiment(*row)

    def set_status(
        self,
        exp_id: int,
        *,
        status: str,
        runtime_s: float | None = None,
        ended_at: datetime | None = None,
    ) -> None:
        """
        Atomically update status (+ optional runtime / end-timestamp).

        Parameters
        ----------
        exp_id     : primary-key of the row
        status     : 'running' | 'finished' | 'failed'
        runtime_s  : optional seconds runtime
        ended_at   : optional explicit timestamp (defaults to NOW)
        """
        cols, vals = ["status"], [status]

        if runtime_s is not None:
            cols.append("runtime_s")
            vals.append(runtime_s)

        cols.append("ended_at")
        vals.append(ended_at or "CURRENT_TIMESTAMP")

        set_clause = ", ".join(
            f"{c} = ?" if v != "CURRENT_TIMESTAMP" else f"{c} = {v}"
            for c, v in zip(cols, vals)
        )
        sql = f"UPDATE {self.table} SET {set_clause} WHERE exp_id = ?"
        self.conn.execute(sql, [v for v in vals if v != "CURRENT_TIMESTAMP"] + [exp_id])
