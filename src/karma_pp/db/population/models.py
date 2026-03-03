from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class PopulationConfig:
    """Single row in the `population` table."""

    __table__ = "population"
    __pk__ = "population_hash"

    population_hash: str
    created_at: datetime


@dataclass(frozen=True)
class PopulationMember:
    """Single row in the `population_member` table."""

    __table__ = "population_member"
    __pk__ = "id"

    id: int
    population_hash: str
    model_id: str
    agent_hash: str
    n_agents: int
    weight: float

