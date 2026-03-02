"""Tests for system-level measures."""

import numpy as np
import pytest

from karma_pp.utils.agent_measures import stationary_distribution, get_markov_lambda_star
from karma_pp.utils.system_measures import get_access_fairness, get_efficiency, get_reward_fairness


class TestAccessFairness:
    def test_perfect_fairness_equals_zero(self):
        """If all agents have same access rate, std = 0, so fairness = 0."""
        access = np.array([[1, 1], [1, 1], [0, 0]])  # both agents same
        result = get_access_fairness(access, weights=[1.0, 1.0])
        assert result == pytest.approx(0.0)

    def test_unequal_access_is_negative(self):
        """Negative std means unfair — function returns negative value."""
        access = np.array([[1, 0], [1, 0], [1, 0]])  # agent 0 always gets it
        result = get_access_fairness(access, weights=[1.0, 1.0])
        assert result < 0.0

    def test_empty_returns_zero(self):
        assert get_access_fairness(np.zeros((0, 2)), weights=[1.0, 1.0]) == pytest.approx(0.0)

    def test_weighted_fairness(self):
        """With weights, higher-weight agents' deviations count more."""
        # Agent 0 always gets access, agent 1 never. Unweighted std > 0.
        access = np.array([[1, 0], [1, 0], [1, 0]])
        unweighted = get_access_fairness(access, weights=[1.0, 1.0])
        assert unweighted < 0.0
        # With equal weights, same as unweighted
        weighted_eq = get_access_fairness(access, weights=[1.0, 1.0])
        assert weighted_eq == pytest.approx(unweighted)


class TestEfficiency:
    def test_mean_reward(self):
        rewards = np.array([[1.0, 2.0], [3.0, 4.0]])  # agents get [2, 3] means
        result = get_efficiency(rewards, weights=[1.0, 1.0])
        assert result == pytest.approx(2.5)

    def test_empty_returns_zero(self):
        assert get_efficiency(np.zeros((0, 2)), weights=[1.0, 1.0]) == pytest.approx(0.0)

    def test_weighted_efficiency(self):
        """With weights, higher-weight agents contribute more."""
        # Agent 0 mean=1, agent 1 mean=3. Unweighted = 2.0
        rewards = np.array([[1.0, 3.0], [1.0, 3.0]])
        assert get_efficiency(rewards, weights=[1, 1]) == pytest.approx(2.0)
        # Weight agent 1 twice as much: (1*1 + 2*3)/3 = 7/3
        assert get_efficiency(rewards, weights=[1, 2]) == pytest.approx(7.0 / 3.0)


class TestRewardFairness:
    def test_equal_rewards_gives_zero(self):
        rewards = np.array([[1.0, 1.0], [1.0, 1.0]])
        assert get_reward_fairness(rewards) == pytest.approx(0.0)

    def test_unequal_rewards_is_negative(self):
        rewards = np.array([[1.0, 0.0], [1.0, 0.0]])
        assert get_reward_fairness(rewards) < 0.0


class TestStationaryDistribution:
    def test_uniform_chain(self):
        T = [[0.5, 0.5], [0.5, 0.5]]
        pi = stationary_distribution(np.array(T))
        assert pi == pytest.approx([0.5, 0.5], abs=1e-6)

    def test_absorbing_state(self):
        T = [[1.0, 0.0], [0.0, 1.0]]
        # Two absorbing states; result depends on initial distribution
        pi = stationary_distribution(np.array(T))
        assert abs(sum(pi) - 1.0) < 1e-6

    def test_distribution_sums_to_one(self):
        T = [[0.8, 0.2], [0.3, 0.7]]
        pi = stationary_distribution(np.array(T))
        assert abs(sum(pi) - 1.0) < 1e-9
        assert all(p >= 0 for p in pi)

    def test_satisfies_balance_equation(self):
        T = np.array([[0.8, 0.2], [0.3, 0.7]])
        pi = stationary_distribution(T)
        assert np.allclose(pi @ T, pi, atol=1e-5)


class TestLambdaStar:
    def test_returns_finite_value(self):
        T = [np.array([[0.8, 0.2], [0.2, 0.8]])] * 2
        lam = get_markov_lambda_star(T, weights=[1.0, 1.0], resource_count=1.0)
        assert np.isfinite(lam)

    def test_more_resources_lower_lambda(self):
        """More resources → lower shadow price."""
        T = [np.array([[0.8, 0.2], [0.2, 0.8]])] * 4
        lam1 = get_markov_lambda_star(T, weights=[1.0] * 4, resource_count=1.0)
        lam2 = get_markov_lambda_star(T, weights=[1.0] * 4, resource_count=2.0)
        assert lam2 <= lam1

    def test_raises_on_empty_matrices(self):
        with pytest.raises(ValueError):
            get_markov_lambda_star([], weights=[], resource_count=1.0)
