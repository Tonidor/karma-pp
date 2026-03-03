from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ExpGroup:
    """
    One row in the `exp_group` table.
    """

    __table__ = "exp_group"
    __pk__ = "group_id"

    group_id: int
    label: str
    created_at: datetime


@dataclass(frozen=True)
class ExpGroupMember:
    """
    Linking row between an exp-group and an experiment.
    """

    __table__ = "exp_group_member"
    __pk__ = "id"

    id: int
    group_id: int
    exp_id: int
