"""Tests for KarmaMechanism and its components."""

import numpy as np
import pytest

from karma_pp.impl.mechanisms.karma.karma_mechanism import (
    KarmaMechanism,
    KarmaState,
    KarmaDynamics,
)
from karma_pp.impl.mechanisms.karma.max_sum_selection_rule import MaxSumSelectionRule
from karma_pp.impl.mechanisms.karma.proportional_redistribution_rule import ProportionalRedistributionRule
from karma_pp.core.types import CollectiveAction
from karma_pp.utils.loading_utils import Config


def _make_rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _karma_mechanism_cfg() -> tuple[Config, Config]:
    sel: Config = {
        "code": "karma_pp.impl.mechanisms.karma.max_sum_selection_rule.MaxSumSelectionRule",
        "parameters": {},
    }
    red: Config = {
        "code": "karma_pp.impl.mechanisms.karma.proportional_redistribution_rule.ProportionalRedistributionRule",
        "parameters": {},
    }
    return sel, red


def _simple_ca(n_agents: int, commits: list[list[int]]) -> CollectiveAction:
    """Single-resource collective action with two decisions (no-one / agent-0 gets it)."""
    agent_ids = list(range(n_agents))
    agent_weights = [1] * n_agents
    # decisions[d][i] = list of agents that get resource under decision d
    decisions = [[], [0]]
    # agent_outcomes[i] = [(False,), (True,)] for each agent
    agent_outcomes = [[(False,), (True,)] for _ in range(n_agents)]
    # decisions_to_outcomes[d][i] = outcome index agent i gets under decision d
    decisions_to_outcomes = [
        [0] * n_agents,   # decision 0: everyone gets null outcome
        [1, *([0] * (n_agents - 1))],  # decision 1: agent 0 gets resource, rest null
    ]
    return CollectiveAction(
        agent_ids=agent_ids,
        agent_weights=agent_weights,
        decisions=decisions,
        signals=commits,
        decisions_to_outcomes=decisions_to_outcomes,
        agent_outcomes=agent_outcomes,
    )


class TestKarmaMechanismInitialize:
    def setup_method(self):
        sel, red = _karma_mechanism_cfg()
        self.mech = KarmaMechanism(
            selection_rule=sel,
            redistribution_rule=red,
            weight_karma_ratio=5.0,
            max_balance=25,
        )

    def test_balances_proportional_to_weights(self):
        weights = {0: 2, 1: 3}
        state, dynamics = self.mech.initialize(agent_weights=weights, rng=_make_rng())
        assert state.agent_balances[0] == int(2 * 5.0)
        assert state.agent_balances[1] == int(3 * 5.0)

    def test_dynamics_contains_agent_weights(self):
        weights = {0: 1, 1: 2}
        _, dynamics = self.mech.initialize(agent_weights=weights, rng=_make_rng())
        assert dynamics.agent_weights == weights

    def test_max_balance_equals_sum_of_balances(self):
        weights = {0: 1, 1: 1}
        _, dynamics = self.mech.initialize(agent_weights=weights, rng=_make_rng())
        assert dynamics.max_balance == 25  # configured in setup


class TestKarmaMechanismRun:
    def setup_method(self):
        sel, red = _karma_mechanism_cfg()
        self.mech = KarmaMechanism(sel, red, weight_karma_ratio=2.0, max_balance=10)
        self.weights = {0: 1, 1: 1}
        self.state, _ = self.mech.initialize(self.weights, _make_rng())

    def test_run_returns_report_with_transfers(self):
        # commits: agent 0 bids 1 for decision 1, agent 1 bids 0
        ca = _simple_ca(2, [[0, 1], [0, 0]])
        report = self.mech.run(self.state, ca, _make_rng(0))
        assert isinstance(report.transfers, dict)
        assert set(report.transfers.keys()) == {0, 1}

    def test_karma_is_conserved_after_update(self):
        """Total karma must be identical before and after a step."""
        ca = _simple_ca(2, [[0, 1], [0, 0]])
        report = self.mech.run(self.state, ca, _make_rng(0))
        new_state = self.mech.update_state(self.state, {0: report}, _make_rng(0))
        total_before = sum(self.state.agent_balances.values())
        total_after = sum(new_state.agent_balances.values())
        assert total_before == total_after

    def test_update_state_raises_on_none_previous(self):
        ca = _simple_ca(2, [[0, 0], [0, 0]])
        report = self.mech.run(self.state, ca, _make_rng(0))
        with pytest.raises(ValueError, match="non-None previous state"):
            self.mech.update_state(None, {0: report}, _make_rng())

    def test_balances_never_go_negative(self):
        """Run 50 steps and verify no balance ever goes negative."""
        state = self.state
        rng = _make_rng(7)
        for _ in range(50):
            max_bal = max(state.agent_balances.values())
            b0, b1 = state.agent_balances[0], state.agent_balances[1]
            commits = [[0, b0], [0, b1]]
            ca = _simple_ca(2, commits)
            report = self.mech.run(state, ca, rng)
            state = self.mech.update_state(state, {0: report}, rng)
        for b in state.agent_balances.values():
            assert b >= 0, f"Balance went negative: {b}"


class TestMaxSumSelectionRule:
    def setup_method(self):
        self.rule = MaxSumSelectionRule()
        self.rng = np.random.default_rng(0)

    def test_selects_max_total_commit(self):
        commits = [[0, 5], [0, 5]]  # decision 1 has total 10, decision 0 has 0
        idx = self.rule(commits, self.rng)
        assert idx == 1

    def test_tie_returns_valid_index(self):
        commits = [[3, 3], [3, 3]]  # equal totals — either index is valid
        idx = self.rule(commits, self.rng)
        assert idx in (0, 1)

    def test_tie_is_uniform_over_many_draws(self):
        commits = [[3, 3], [3, 3]]
        rng = np.random.default_rng(42)
        counts = [0, 0]
        for _ in range(1000):
            counts[self.rule(commits, rng)] += 1
        # Each decision should be chosen ~50% of the time (allow ±5%)
        assert abs(counts[0] / 1000 - 0.5) < 0.05

    def test_returns_valid_index(self):
        for s in range(10):
            commits = [
                [int(np.random.default_rng(s).integers(0, 5)) for _ in range(3)]
                for _ in range(4)
            ]
            idx = self.rule(commits, np.random.default_rng(s))
            assert 0 <= idx < 3


class TestProportionalRedistributionRule:
    def setup_method(self):
        self.rule = ProportionalRedistributionRule()
        self.rng = np.random.default_rng(0)

    def test_transfers_are_zero_sum(self):
        transfer = self.rule([2, 3, 1], [1, 1, 1], self.rng)
        assert sum(transfer) == 0, f"Transfer not zero-sum: {transfer}"

    def test_no_agent_pays_more_than_committed(self):
        commits = [2, 3, 1]
        transfer = self.rule(commits, [1, 1, 1], self.rng)
        for i, t in enumerate(transfer):
            assert t >= -commits[i], f"Agent {i} pays more than committed"

    def test_single_agent(self):
        transfer = self.rule([5], [1], self.rng)
        assert transfer[0] == 0  # single agent: net transfer is 0

    def test_all_zero_commits(self):
        transfer = self.rule([0, 0], [1, 1], self.rng)
        assert all(t == 0 for t in transfer)

    def test_raises_on_empty_weights(self):
        with pytest.raises(ValueError, match="non-empty"):
            self.rule([], [], self.rng)
