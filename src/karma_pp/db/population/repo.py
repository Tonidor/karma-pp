import hashlib
import json
from typing import Any

import duckdb

from ..base import BaseRepo
from .models import PopulationConfig, PopulationMember


class PopulationRepo(BaseRepo[PopulationConfig]):
    """
    Repo for canonical population definitions.

    A population is content-addressable by a canonical JSON representation of
    its member list. Membership is stored normalized in `population_member`
    for easy querying and joins, so we don't duplicate the structure in the
    `population` table.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        super().__init__(conn, PopulationConfig)

    @staticmethod
    def _canonicalize_members(blob: dict[str, Any]) -> str:
        """
        Return canonical JSON for the population's member list only.

        Expected blob format:
          {"members": [{"model_id": str, "agent_hash": str, "n_agents": int, "weight": float}, ...]}
        """
        members = blob.get("members")
        if isinstance(members, list):
            # Ensure deterministic ordering for hashing. JSON dump below will preserve list order.
            members_sorted = sorted(
                members,
                key=lambda m: (
                    str(m.get("model_id", "")),
                    str(m.get("agent_hash", "")),
                    int(m.get("n_agents", 0)),
                    float(m.get("weight", 0.0)),
                ),
            )
            # Only hash the normalized members list, not any outer metadata.
            return json.dumps(members_sorted, sort_keys=True, separators=(",", ":"))
        # Fallback: hash the entire blob if no members list is present
        return json.dumps(blob, sort_keys=True, separators=(",", ":"))

    def ensure(self, blob: dict[str, Any]) -> str:
        """
        Ensure population exists; also upsert normalized membership rows.

        Expected blob format:
          {"members": [{"model_id": str, "agent_hash": str, "n_agents": int, "weight": float}, ...]}
        """
        canonical_json = self._canonicalize_members(blob)
        h = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

        # Insert population header row if new
        self.conn.execute(
            """
            INSERT OR IGNORE INTO population
              (population_hash)
            VALUES (?)
            """,
            (h,),
        )

        members = blob.get("members", [])
        if not isinstance(members, list):
            raise ValueError("population blob must contain list field 'members'.")

        sql = """
            INSERT OR IGNORE INTO population_member
              (population_hash, model_id, agent_hash, n_agents, weight)
            VALUES (?, ?, ?, ?, ?)
        """
        for m in members:
            self.conn.execute(
                sql,
                (
                    h,
                    str(m["model_id"]),
                    str(m["agent_hash"]),
                    int(m["n_agents"]),
                    float(m["weight"]),
                ),
            )

        return h

    def members(self, population_hash: str) -> list[PopulationMember]:
        rows = self.conn.execute(
            """
            SELECT id, population_hash, model_id, agent_hash, n_agents, weight
            FROM population_member
            WHERE population_hash = ?
            ORDER BY model_id
            """,
            (population_hash,),
        ).fetchall()
        return [PopulationMember(*row) for row in rows]

    def to_scenario_population(self, population_hash: str) -> dict[str, Any]:
        """
        Reconstruct a scenario-style population config:

        {
          "<model_id>": {
            "n_agents": int,
            "weight": float,
            "agent_model": { ... }  # loaded from agent.json
          },
          ...
        }
        """
        rows = self.conn.execute(
            """
            SELECT
              pm.model_id,
              pm.agent_hash,
              pm.n_agents,
              pm.weight,
              a.json
            FROM population_member pm
            JOIN agent a
              ON a.agent_hash = pm.agent_hash
            WHERE pm.population_hash = ?
            ORDER BY pm.model_id
            """,
            (population_hash,),
        ).fetchall()

        population_cfg: dict[str, Any] = {}
        for model_id, agent_hash, n_agents, weight, agent_json in rows:
            agent_cfg = json.loads(agent_json)
            population_cfg[str(model_id)] = {
                "n_agents": int(n_agents),
                "weight": float(weight),
                "agent_model": agent_cfg,
                "agent_hash": agent_hash,
            }
        return population_cfg

