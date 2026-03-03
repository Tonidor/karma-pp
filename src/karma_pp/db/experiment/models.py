from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class Experiment:
    """
    A row in the `experiment` table.
    """

    __table__ = "experiment"
    __pk__ = "exp_id"

    exp_id: int
    world_hash: str
    mechanism_hash: str
    population_hash: str
    seed: int
    n_steps: int
    git_commit: str
    name: str
    comment: Optional[str]
    runtime_s: Optional[float]
    started_at: datetime
    ended_at: Optional[datetime]
    status: Optional[str]
