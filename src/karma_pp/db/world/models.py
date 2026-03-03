from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class WorldConfig:
    """Single row in the `world` table - stores world configuration as JSON."""

    __table__ = "world"
    __pk__ = "world_hash"

    world_hash: str
    json: str
    created_at: datetime
