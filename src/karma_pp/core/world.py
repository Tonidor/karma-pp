from abc import ABC, abstractmethod

import numpy as np

class World[
    WORLD_STATE,
    WORLD_DYNAMICS,
    WORLD_OBSERVATION,
    OUTCOME,
    SIGNAL,
    COLLECTIVE_ACTION,
    REPORT,
](ABC):
    """Interface for a world."""

    @abstractmethod
    def initialize(
        self,
        n_agents: int,
        rng: np.random.Generator,
    ) -> tuple[WORLD_STATE, WORLD_DYNAMICS]:
        """Create initial world state."""
        ...

    @abstractmethod
    def get_collectives(
        self,
        world_state: WORLD_STATE,
        agent_ids: list[int],
    ) -> dict[int, list[int]]:
        """Get all possible collectives for the given agent ids."""
        ...

    @abstractmethod
    def get_observations(
        self,
        agent_ids: list[int],
        world_state: WORLD_STATE,
    ) -> dict[int, WORLD_OBSERVATION]:
        """Project world state into the agent-visible world observation."""
        ...

    @abstractmethod
    def filter_actions(
        self,
        world_state: WORLD_STATE,
        agent_actions: dict[int, list[tuple[OUTCOME, SIGNAL]]],
        collective: list[int],
    ) -> COLLECTIVE_ACTION:
        """Filter infeasible joint actions and map to decision space."""
        ...

    @abstractmethod
    def update_state(
        self,
        previous: WORLD_STATE | None,
        reports: dict[int, REPORT | None],  # collective_id -> report
        rng: np.random.Generator,
    ) -> WORLD_STATE:
        """Return world state; initialize when timestep is 0."""
        ...
