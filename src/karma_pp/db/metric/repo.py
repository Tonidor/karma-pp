from typing import Optional

import duckdb

from ..base import BaseRepo
from .models import Metric


class MetricRepo(BaseRepo[Metric]):
    """
    CRUD helper for the `metric` table.  `create()` inserts a new metric row
    and returns a fully populated `Metric` dataclass (including the
    auto-generated timestamp).
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        super().__init__(conn, Metric)

    def create(
        self,
        exp_id: int,
        metric_name: str,
        metric_value: float,
        step: Optional[int] = None,
    ) -> Metric:
        """
        Insert a metric and return the freshly created `Metric` object.

        Parameters
        ----------
        exp_id        : primary-key of the experiment this metric belongs to
        metric_name   : e.g. "accuracy", "loss", "runtime_s"
        metric_value  : numeric value of the metric
        step          : optional epoch/episode/iteration; leave `None` for
                        final or single-point metrics
        """
        sql = """
            INSERT INTO metric
                   (exp_id, step, metric_name, metric_value)
            VALUES (?, ?, ?, ?)
            RETURNING exp_id, step, metric_name, metric_value, recorded_at
        """
        row = self.conn.execute(
            sql, (exp_id, step, metric_name, metric_value)
        ).fetchone()

        return Metric(*row)
