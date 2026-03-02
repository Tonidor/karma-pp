"""Tests for TurnTakingMechanism."""

import numpy as np
import pytest

from karma_pp.impl.mechanisms.turn_taking import TurnTakingMechanism, TurnTakingReport
from karma_pp.core.types import CollectiveAction


def _make_rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _simple_collective_action(
    signals: list[list[float]],
    agent_weights: list[int] | None = None,
) -> CollectiveAction:
    """Helper: 2 agents, 2 decisions, single resource."""
    n_agents = len(signals)
    n_decisions = len(signals[0])
    agent_ids = list(range(n_agents))
    if agent_weights is None:
        agent_weights = [1] * n_agents
    agent_outcomes = [[(False,), (True,)] for _ in range(n_agents)]
    decisions_to_outcomes = [[0] * n_agents, [1] * n_agents][:n_decisions]
    return CollectiveAction(
        agent_ids=agent_ids,
        agent_weights=agent_weights,
        decisions=list(range(n_decisions)),
        signals=signals,
        decisions_to_outcomes=decisions_to_outcomes,
        agent_outcomes=agent_outcomes,
    )


class TestTurnTakingMechanism:
    def setup_method(self):
        self.mech = TurnTakingMechanism()
        self.rng = _make_rng()

    def test_weights_1_and_2_give_approx_1_3_and_2_3_of_turns(self):
        """With weights 1 and 2, agent 0 should get ~1/3 of turns, agent 1 ~2/3."""
        state, _ = self.mech.initialize(agent_weights={0: 1, 1: 2}, rng=_make_rng(42))
        rng = _make_rng(123)
        turn_counts = {0: 0, 1: 0}
        n_steps = 300
        for _ in range(n_steps):
            # Both agents prefer decision 1 (highest signal)
            ca = _simple_collective_action([[1.0, 2.0], [1.0, 2.0]], agent_weights=[1, 2])
            report = self.mech.run(state, ca, rng)
            turn_counts[report.turn_holder_agent_id] += 1
            state = self.mech.update_state(state, {0: report}, rng)
        # Allow ±5% tolerance
        assert abs(turn_counts[0] / n_steps - 1 / 3) < 0.05
        assert abs(turn_counts[1] / n_steps - 2 / 3) < 0.05

    def test_equal_weights_give_approx_equal_turns(self):
        """With equal weights, both agents get ~50% of turns."""
        state, _ = self.mech.initialize(agent_weights={0: 1, 1: 1}, rng=_make_rng(42))
        rng = _make_rng(456)
        turn_counts = {0: 0, 1: 0}
        for _ in range(200):
            ca = _simple_collective_action([[1.0, 2.0], [1.0, 2.0]], agent_weights=[1, 1])
            report = self.mech.run(state, ca, rng)
            turn_counts[report.turn_holder_agent_id] += 1
            state = self.mech.update_state(state, {0: report}, rng)
        assert abs(turn_counts[0] / 200 - 0.5) < 0.08
        assert abs(turn_counts[1] / 200 - 0.5) < 0.08

    def test_raises_on_empty_weights(self):
        with pytest.raises(ValueError, match="strictly positive"):
            agent_weights = np.array([1, 0])
            ca = _simple_collective_action([[1.0, 2.0], [1.0, 2.0]], agent_weights=[1, 0])
            state, _ = self.mech.initialize(agent_weights={0: 1, 1: 1}, rng=self.rng)
            self.mech.run(state, ca, self.rng)
