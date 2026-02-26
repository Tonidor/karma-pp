"""Tests for ResourceWorld."""

import numpy as np
import pytest

from karma_pp.impl.worlds.resource_world.resource_world import ResourceWorld


def _make_rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


Outcome = tuple[bool]
Signal = list[int]


def _null_action(n_resources: int) -> list[tuple[Outcome, Signal]]:
    """Agent requests nothing (null outcome with signal 0)."""
    return [(tuple(False for _ in range(n_resources)), [0] * n_resources)]


def _resource_action(r: int, n_resources: int, bid: int = 1) -> list[tuple[Outcome, Signal]]:
    """Agent requests resource r."""
    actions = [(tuple(False for _ in range(n_resources)), [0] * n_resources)]
    outcome = tuple(i == r for i in range(n_resources))
    actions.append((outcome, [bid if i == r else 0 for i in range(n_resources)]))
    return actions


class TestResourceWorldInitialize:
    def test_world_state_has_correct_capacities(self):
        world = ResourceWorld(resource_capacities=[2, 3])
        state, dynamics = world.initialize(n_agents=4, rng=_make_rng())
        assert state.resource_capacities == [2, 3]
        assert dynamics.resource_capacities == [2, 3]


class TestResourceWorldFilterActions:
    def setup_method(self):
        self.world = ResourceWorld(resource_capacities=[1])  # one resource, cap 1

    def _state(self):
        state, _ = self.world.initialize(n_agents=2, rng=_make_rng())
        return state

    def test_feasible_actions_returned(self):
        state = self._state()
        # Both agents can request the resource; at most 1 can get it (cap=1)
        actions = [
            _resource_action(0, 1, bid=2),
            _resource_action(0, 1, bid=1),
        ]
        ca = self.world.filter_actions(state, actions, [0, 1])
        assert len(ca.decisions) > 0

    def test_no_more_than_capacity_agents_assigned(self):
        state = self._state()
        actions = [
            _resource_action(0, 1, bid=1),
            _resource_action(0, 1, bid=1),
        ]
        ca = self.world.filter_actions(state, actions, [0, 1])
        for decision in ca.decisions:
            # decision is list of lists: decision[r] = agents getting resource r
            assert len(decision[0]) <= 1, "capacity violated"

    def test_agents_can_request_null_outcome(self):
        state = self._state()
        actions = [_null_action(1), _null_action(1)]
        ca = self.world.filter_actions(state, actions, [0, 1])
        assert len(ca.decisions) >= 1

    def test_agent_ids_preserved_in_output(self):
        state = self._state()
        actions = [_null_action(1), _null_action(1)]
        ca = self.world.filter_actions(state, actions, [0, 1])
        assert ca.agent_ids == [0, 1]

    def test_signals_shape_matches_decisions(self):
        state = self._state()
        actions = [_resource_action(0, 1, bid=3), _null_action(1)]
        ca = self.world.filter_actions(state, actions, [0, 1])
        n_agents = 2
        n_decisions = len(ca.decisions)
        assert len(ca.signals) == n_agents
        for row in ca.signals:
            assert len(row) == n_decisions

    def test_mismatched_ids_raises(self):
        state = self._state()
        actions = [_null_action(1)]
        with pytest.raises(ValueError):
            self.world.filter_actions(state, actions, [0, 1])  # 2 ids, 1 action


class TestResourceWorldMultiResource:
    def setup_method(self):
        self.world = ResourceWorld(resource_capacities=[1, 1])

    def test_feasibility_respected_per_resource(self):
        state, _ = self.world.initialize(n_agents=3, rng=_make_rng())
        n = 2
        actions = [
            _resource_action(0, n, bid=2),
            _resource_action(1, n, bid=2),
            _resource_action(0, n, bid=1),
        ]
        ca = self.world.filter_actions(state, actions, [0, 1, 2])
        for decision in ca.decisions:
            assert len(decision[0]) <= 1
            assert len(decision[1]) <= 1

    def test_update_state_is_identity(self):
        state, _ = self.world.initialize(n_agents=2, rng=_make_rng())
        new_state = self.world.update_state(state, report=None, rng=_make_rng())
        assert new_state.resource_capacities == state.resource_capacities
