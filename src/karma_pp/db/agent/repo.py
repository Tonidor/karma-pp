from ..base import HashedJSONRepo
from .models import AgentConfig


class AgentRepo(HashedJSONRepo[AgentConfig]):
    """Repo for canonical agent model definitions."""

    hash_col = "agent_hash"

    def __init__(self, conn):
        super().__init__(conn, AgentConfig)

