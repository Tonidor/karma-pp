from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Artifact:
    """
    A row in the `artifact` table.
    """

    __table__ = "artifact"
    __pk__ = "artifact_id"

    artifact_id: int
    exp_id: int
    name: str
    path: str
    sha256: str
    created_at: datetime
