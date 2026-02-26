from abc import ABC, abstractmethod

import numpy as np

class Mechanism[
    OUTCOME,
    SIGNAL,
    MECHANISM_STATE,
    MECHANISM_DYNAMICS,
    MECHANISM_OBSERVATION,
    REPORT,
    RESOLUTION,
    COLLECTIVE_ACTION,
](ABC):
    """Interface for a mechanism."""

    @abstractmethod
    def initialize(
        self,
        agent_weights: dict[int, int],
        rng: np.random.Generator,
    ) -> tuple[MECHANISM_STATE, MECHANISM_DYNAMICS]:
        """Create initial mechanism state."""
        ...

    @abstractmethod
    def run(
        self,
        mechanism_state: MECHANISM_STATE,
        collective_action: COLLECTIVE_ACTION,
        rng: np.random.Generator,
    ) -> REPORT:
        """Run the mechanism."""
        ...

    @abstractmethod
    def get_resolutions(
        self,
        mechanism_state: MECHANISM_STATE,
        collective_action: COLLECTIVE_ACTION,
        report: REPORT,
    ) -> dict[int, RESOLUTION]:
        """Build agent_id -> resolution mapping from action + report."""
        ...

    @abstractmethod
    def update_state(
        self,
        previous: MECHANISM_STATE | None,
        report: REPORT | None,
        rng: np.random.Generator,
    ) -> MECHANISM_STATE:
        """Update mechanism state; initialize when timestep is 0."""
        ...
