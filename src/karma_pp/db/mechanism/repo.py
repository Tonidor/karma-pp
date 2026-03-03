from ..base import HashedJSONRepo
from .models import MechanismConfig


class MechanismRepo(HashedJSONRepo[MechanismConfig]):
    """Repo for canonical mechanism definitions."""

    hash_col = "mechanism_hash"

    def __init__(self, conn):
        super().__init__(conn, MechanismConfig)

