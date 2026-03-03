from typing import List

import duckdb

from ..base import BaseRepo
from .models import Artifact


class ArtifactRepo(BaseRepo[Artifact]):
    """
    CRUD helper for the `artifact` table.
    Adds `create()` (insert & return) and `list_for_experiment()`
    convenience methods.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        super().__init__(conn, Artifact)

    def create(
        self,
        exp_id: int,
        name: str,
        path: str,
        sha256: str,
    ) -> Artifact:
        """
        Insert a new artifact row and return the populated dataclass.
        """
        sql = """
            INSERT INTO artifact
                   (exp_id, name, path, sha256)
            VALUES (?, ?, ?, ?)
            RETURNING artifact_id, exp_id, name, path, sha256, created_at
        """
        row = self.conn.execute(sql, (exp_id, name, path, sha256)).fetchone()
        return Artifact(*row)

    def list_for_experiment(self, exp_id: int) -> List[Artifact]:
        """
        Fetch every artifact associated with a given experiment.
        """
        return self.filter_by(exp_id=exp_id)
