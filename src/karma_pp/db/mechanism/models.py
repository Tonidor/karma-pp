from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class MechanismConfig:
    """Single row in the `mechanism` table - stores mechanism configuration as JSON."""

    __table__ = "mechanism"
    __pk__ = "mechanism_hash"

    mechanism_hash: str
    json: str
    created_at: datetime

