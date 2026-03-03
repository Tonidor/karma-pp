from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class AgentConfig:
    """Single row in the `agent` table - stores agent model configuration as JSON."""

    __table__ = "agent"
    __pk__ = "agent_hash"

    agent_hash: str
    json: str
    created_at: datetime

