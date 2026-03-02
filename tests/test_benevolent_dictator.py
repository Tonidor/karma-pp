"""Tests for BenevolentDictatorMechanism."""

import numpy as np
import pytest

from karma_pp.impl.mechanisms.benevolent_dictator import (
    BenevolentDictatorMechanism,
    DictatorReport,
)
from karma_pp.core.types import CollectiveAction


def _make_rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _simple_collective_action(
    signals: list[list[float]],
) -> CollectiveAction:
    """Helper: 2 agents, 2 decisions, single resource."""
    n_agents = len(signals)
    n_decisions = len(signals[0])
    agent_ids = list(range(n_agents))
    agent_weights = [1] * n_agents
    # outcomes: False=no resource, True=got it
    agent_outcomes = [[(False,), (True,)] for _ in range(n_agents)]
    # decisions_to_outcomes[d][i]: which outcome index agent i gets under decision d
    decisions_to_outcomes = [[0] * n_agents, [1] * n_agents][:n_decisions]
    return CollectiveAction(
        agent_ids=agent_ids,
        agent_weights=agent_weights,
        decisions=list(range(n_decisions)),
        signals=signals,
        decisions_to_outcomes=decisions_to_outcomes,
        agent_outcomes=agent_outcomes,
    )


class TestBenevolentDictatorMechanism:
    def setup_method(self):
        self.mech = BenevolentDictatorMechanism()
        self.rng = _make_rng()

    def test_initialize_returns_none_state_and_dynamics(self):
        state, dynamics = self.mech.initialize(agent_weights={0: 1, 1: 1}, rng=self.rng)
        assert state is None
        assert dynamics is None

    def test_update_state_always_returns_none(self):
        state, _ = self.mech.initialize(agent_weights={0: 1}, rng=self.rng)
        assert self.mech.update_state(previous=state, reports={}, rng=self.rng) is None
        assert self.mech.update_state(previous=None, reports={}, rng=self.rng) is None

    def test_run_selects_highest_total_reward(self):
        """With signals [[1, 10], [1, 10]] the second decision has higher total."""
        ca = _simple_collective_action([[1.0, 10.0], [1.0, 10.0]])
        state, _ = self.mech.initialize(agent_weights={0: 1, 1: 1}, rng=self.rng)
        # Use a seed that does not flip a tie — signals are [2, 20] so no tie.
        report = self.mech.run(mechanism_state=state, collective_action=ca, rng=_make_rng(0))
        assert report.selected_decision == 1  # second decision index

    def test_run_weights_rewards_proportionally(self):
        """Higher-weight agents' rewards count more. Agent 0 (weight 2): [10, 0], Agent 1 (weight 1): [0, 10].
        Weighted decision 0 = 20, decision 1 = 10 → decision 0 selected."""
        ca = _simple_collective_action([[10.0, 0.0], [0.0, 10.0]])
        ca = CollectiveAction(
            agent_ids=ca.agent_ids,
            agent_weights=[2, 1],  # agent 0 has weight 2, agent 1 has weight 1
            decisions=ca.decisions,
            signals=ca.signals,
            decisions_to_outcomes=ca.decisions_to_outcomes,
            agent_outcomes=ca.agent_outcomes,
        )
        state, _ = self.mech.initialize(agent_weights={0: 2, 1: 1}, rng=self.rng)
        report = self.mech.run(mechanism_state=state, collective_action=ca, rng=_make_rng(0))
        assert report.selected_decision == 0

    def test_run_breaks_ties_uniformly(self):
        """Equal total rewards → both decisions should sometimes be chosen."""
        ca = _simple_collective_action([[5.0, 5.0], [5.0, 5.0]])
        state, _ = self.mech.initialize(agent_weights={0: 1, 1: 1}, rng=self.rng)
        choices = {
            self.mech.run(state, ca, _make_rng(seed)).selected_decision
            for seed in range(40)
        }
        assert 0 in choices and 1 in choices, "Tie-breaking should select both decisions"

    def test_run_raises_on_no_decisions(self):
        empty_ca = CollectiveAction(
            agent_ids=[0],
            agent_weights=[1],
            decisions=[],
            signals=[],
            decisions_to_outcomes=[],
            agent_outcomes=[[]],
        )
        state, _ = self.mech.initialize(agent_weights={0: 1}, rng=self.rng)
        with pytest.raises(ValueError, match="No feasible decisions"):
            self.mech.run(state, empty_ca, self.rng)

    def test_get_resolutions_keys_match_agent_ids(self):
        ca = _simple_collective_action([[1.0, 2.0], [1.0, 2.0]])
        state, _ = self.mech.initialize(agent_weights={0: 1, 1: 1}, rng=self.rng)
        report = self.mech.run(state, ca, self.rng)
        resolutions = self.mech.get_resolutions(state, ca, report)
        assert set(resolutions.keys()) == {0, 1}

    def test_get_resolutions_selected_outcome_matches_report(self):
        ca = _simple_collective_action([[1.0, 5.0], [1.0, 5.0]])
        state, _ = self.mech.initialize(agent_weights={0: 1, 1: 1}, rng=self.rng)
        report = self.mech.run(state, ca, _make_rng(0))
        resolutions = self.mech.get_resolutions(state, ca, report)
        for agent_id, res in resolutions.items():
            assert res.selected_outcome == report.selected_outcomes[agent_id]
