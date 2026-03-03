from ..base import HashedJSONRepo
from .models import WorldConfig


class WorldRepo(HashedJSONRepo[WorldConfig]):
    """Repo for canonical world definitions."""

    hash_col = "world_hash"

    def __init__(self, conn):
        super().__init__(conn, WorldConfig)
