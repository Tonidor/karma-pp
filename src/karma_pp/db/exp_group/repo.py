from typing import List

import duckdb

from ...db.base import BaseRepo
from .models import ExpGroup, ExpGroupMember


class ExpGroupRepo(BaseRepo[ExpGroup]):
    """
    CRUD helper for the `exp_group` table.
    Adds convenience helpers to attach experiments
    and to list all members of a group.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        super().__init__(conn, ExpGroup)

    def create(self, label: str) -> ExpGroup:
        """
        Insert a new exp-group and return the populated dataclass.
        """
        sql = """
            INSERT INTO exp_group (label)
            VALUES (?)
            RETURNING group_id, label, created_at
        """
        row = self.conn.execute(sql, (label,)).fetchone()
        return ExpGroup(*row)

    def add_member(self, group_id: int, exp_id: int) -> ExpGroupMember:
        """
        Attach an experiment to an exp-group.
        Uses INSERT OR IGNORE so calling twice is harmless.
        """
        sql = """
            INSERT OR IGNORE INTO exp_group_member (group_id, exp_id)
            VALUES (?, ?)
            RETURNING id, group_id, exp_id
        """
        row = self.conn.execute(sql, (group_id, exp_id)).fetchone()
        return ExpGroupMember(*row)

    def list_members(self, group_id: int) -> List[int]:
        """
        Return all experiment IDs that belong to the given group.
        """
        sql = "SELECT exp_id FROM exp_group_member WHERE group_id = ?"
        rows = self.conn.execute(sql, (group_id,)).fetchall()
        return [r[0] for r in rows]


class ExpGroupMemberRepo(BaseRepo[ExpGroupMember]):
    """
    Direct CRUD access to the `exp_group_member` link table.
    Most projects won't need this explicitly, but it's here for
    completeness and symmetry with the other repos.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        super().__init__(conn, ExpGroupMember)
