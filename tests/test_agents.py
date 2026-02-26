"""Tests for resource agent implementations."""

import numpy as np
import pytest

from karma_pp.impl.agents.resource_agent import ResourceAgentObservation
from karma_pp.impl.agents.random_resource_agent import RandomResourceAgent
from karma_pp.impl.agents.truthful_resource_agent import TruthfulResourceAgent
from karma_pp.impl.agents.q_learning_resource_agent import QLearningResourceAgent, QLearningPolicyState
from karma_pp.impl.agents.optimal_bidding_resource_agent import OptimalBiddingResourceAgent, OptimalBiddingPolicyState
from karma_pp.impl.mechanisms.karma.karma_mechanism import KarmaDynamics, KarmaState
from karma_pp.impl.mechanisms.karma.max_sum_selection_rule import MaxSumSelectionRule
from karma_pp.impl.mechanisms.karma.proportional_redistribution_rule import ProportionalRedistributionRule
from karma_pp.impl.worlds.resource_world.resource_world import ResourceWorldDynamics
from karma_pp.core.types import AgentState


def _make_rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


_TRANSITION = [[0.8, 0.2], [0.2, 0.8]]
_URGENCY_LEVELS = [0, 1]
_REWARD = [0.0]
_PENALTY = [-1.0]


def _world_dynamics() -> ResourceWorldDynamics:
    return ResourceWorldDynamics(resource_capacities=[1])


def _karma_dynamics(max_balance: int = 10) -> KarmaDynamics:
    return KarmaDynamics(
        selection_rule=MaxSumSelectionRule(),
        redistribution_rule=ProportionalRedistributionRule(),
        weight_karma_ratio=5.0,
        max_balance=max_balance,
        agent_weights={0: 1, 1: 1},
    )


def _karma_state(balance: int = 5) -> KarmaState:
    return KarmaState(agent_balances={0: balance, 1: balance})


class TestResourceAgentOutcomeReward:
    """Verify the shared _outcome_reward helper."""

    def setup_method(self):
        self.agent = TruthfulResourceAgent(
            transition_matrix=_TRANSITION,
            initial_urgency=0,
            urgency_levels=_URGENCY_LEVELS,
            reward_per_resource=_REWARD,
            no_resource_penalty=_PENALTY,
        )

    def test_null_outcome_with_urgency_1_gives_penalty(self):
        reward = self.agent._outcome_reward(urgency=1, outcome=(False,))
        # urgency=1, no resource: 1 * (0*0 + (-1)*(1)) = -1
        assert reward == pytest.approx(-1.0)

    def test_resource_outcome_with_urgency_1_gives_zero(self):
        reward = self.agent._outcome_reward(urgency=1, outcome=(True,))
        # urgency=1, got resource: 1 * (0*1 + (-1)*0) = 0
        assert reward == pytest.approx(0.0)

    def test_zero_urgency_always_zero(self):
        for outcome in [(False,), (True,)]:
            assert self.agent._outcome_reward(0, outcome) == pytest.approx(0.0)


class TestRandomResourceAgent:
    def setup_method(self):
        self.agent = RandomResourceAgent(
            transition_matrix=_TRANSITION,
            initial_urgency=0,
            urgency_levels=_URGENCY_LEVELS,
            reward_per_resource=_REWARD,
            no_resource_penalty=_PENALTY,
        )
        self.agent_state = self.agent.initialize(_world_dynamics(), _karma_dynamics(), _make_rng())

    def test_initialize_returns_correct_policy(self):
        assert self.agent_state.policy is None

    def test_get_action_returns_correct_number_of_signals(self):
        from karma_pp.impl.agents.random_resource_agent import RandomObservation
        obs = RandomObservation(resource_capacities=[1], agent_balance=5)
        actions = self.agent.get_action(self.agent_state, obs, _make_rng())
        # one null outcome + one resource outcome = 2 actions
        assert len(actions) == 2

    def test_bids_within_balance(self):
        from karma_pp.impl.agents.random_resource_agent import RandomObservation
        balance = 3
        obs = RandomObservation(resource_capacities=[1], agent_balance=balance)
        for seed in range(50):
            actions = self.agent.get_action(self.agent_state, obs, _make_rng(seed))
            for _, signal in actions:
                assert 0 <= signal <= balance

    def test_adapt_returns_unchanged_state(self):
        from karma_pp.impl.agents.random_resource_agent import RandomObservation
        from karma_pp.impl.mechanisms.karma.karma_mechanism import KarmaResolution
        obs = RandomObservation(resource_capacities=[1], agent_balance=5)
        res = KarmaResolution(agent_id=0, selected_outcome=(False,), transfer=0, outcome_scores=[-1.0, 0.0], n_agents=2)
        new_state = self.agent.adapt(self.agent_state, obs, res, reward=-1.0, timestep=1, rng=_make_rng())
        assert new_state is self.agent_state


class TestTruthfulResourceAgent:
    def setup_method(self):
        self.agent = TruthfulResourceAgent(
            transition_matrix=_TRANSITION,
            initial_urgency=1,
            urgency_levels=_URGENCY_LEVELS,
            reward_per_resource=_REWARD,
            no_resource_penalty=_PENALTY,
        )
        self.state = self.agent.initialize(_world_dynamics(), None, _make_rng())

    def test_reports_truthful_rewards(self):
        obs = ResourceAgentObservation(resource_capacities=[1])
        actions = self.agent.get_action(self.state, obs, _make_rng())
        # outcomes: (False,) -> reward -1, (True,) -> reward 0
        outcome_signal = {outcome: signal for outcome, signal in actions}
        assert outcome_signal[(False,)] == pytest.approx(-1.0)
        assert outcome_signal[(True,)] == pytest.approx(0.0)


class TestQLearningResourceAgent:
    def setup_method(self):
        self.agent = QLearningResourceAgent(
            transition_matrix=_TRANSITION,
            initial_urgency=0,
            urgency_levels=_URGENCY_LEVELS,
            reward_per_resource=_REWARD,
            no_resource_penalty=_PENALTY,
            alpha=0.1,
            epsilon=1.0,
            gamma=0.99,
            epsilon_min=0.01,
            epsilon_decay=0.99,
        )
        self.karma_dyn = _karma_dynamics(max_balance=10)
        self.agent_state = self.agent.initialize(_world_dynamics(), self.karma_dyn, _make_rng())

    def test_epsilon_stored_in_policy_not_on_model(self):
        assert self.agent_state.policy.epsilon == pytest.approx(1.0)
        assert not hasattr(self.agent, 'epsilon')

    def test_max_balance_stored_in_policy(self):
        assert self.agent_state.policy.max_balance == 10

    def test_adapt_decays_epsilon_per_agent(self):
        from karma_pp.impl.agents.q_learning_resource_agent import QLearningObservation
        from karma_pp.impl.mechanisms.karma.karma_mechanism import KarmaResolution
        obs = QLearningObservation(resource_capacities=[1], agent_balance=5)
        _ = self.agent.get_action(self.agent_state, obs, _make_rng())
        res = KarmaResolution(agent_id=0, selected_outcome=(False,), transfer=0, outcome_scores=[], n_agents=2)
        new_state = self.agent.adapt(self.agent_state, obs, res, reward=-1.0, timestep=1, rng=_make_rng())
        assert new_state.policy.epsilon < self.agent_state.policy.epsilon
        assert new_state.policy.epsilon >= self.agent.epsilon_min

    def test_adapt_updates_q_table(self):
        from karma_pp.impl.agents.q_learning_resource_agent import QLearningObservation
        from karma_pp.impl.mechanisms.karma.karma_mechanism import KarmaResolution
        obs = QLearningObservation(resource_capacities=[1], agent_balance=5)
        _ = self.agent.get_action(self.agent_state, obs, _make_rng())
        res = KarmaResolution(agent_id=0, selected_outcome=(False,), transfer=0, outcome_scores=[], n_agents=2)
        new_state = self.agent.adapt(self.agent_state, obs, res, reward=-1.0, timestep=1, rng=_make_rng())
        assert len(new_state.policy.Q) > 0

    def test_independent_epsilons_for_two_agents(self):
        """Each agent should have its own epsilon, not shared via the model."""
        from karma_pp.impl.agents.q_learning_resource_agent import QLearningObservation
        from karma_pp.impl.mechanisms.karma.karma_mechanism import KarmaResolution

        state1 = self.agent.initialize(_world_dynamics(), self.karma_dyn, _make_rng(0))
        state2 = self.agent.initialize(_world_dynamics(), self.karma_dyn, _make_rng(1))
        obs = QLearningObservation(resource_capacities=[1], agent_balance=5)
        res = KarmaResolution(agent_id=0, selected_outcome=(False,), transfer=0, outcome_scores=[], n_agents=2)

        # Only adapt state1 multiple times
        _ = self.agent.get_action(state1, obs, _make_rng())
        for _ in range(10):
            state1 = self.agent.adapt(state1, obs, res, reward=-1.0, timestep=1, rng=_make_rng())

        # state2 should be unaffected
        assert state2.policy.epsilon == pytest.approx(1.0)
        assert state1.policy.epsilon < 1.0


class TestOptimalBiddingResourceAgent:
    def setup_method(self):
        self.agent = OptimalBiddingResourceAgent(
            transition_matrix=_TRANSITION,
            initial_urgency=1,
            urgency_levels=_URGENCY_LEVELS,
            reward_per_resource=_REWARD,
            no_resource_penalty=_PENALTY,
        )
        self.karma_dyn = _karma_dynamics(max_balance=10)
        self.agent_state = self.agent.initialize(_world_dynamics(), self.karma_dyn, _make_rng())

    def test_max_balance_and_n_outcomes_in_policy(self):
        policy = self.agent_state.policy
        assert policy.max_balance == 10
        assert policy.n_outcomes == 2  # null + 1 resource

    def test_bids_within_balance(self):
        from karma_pp.impl.agents.optimal_bidding_resource_agent import OptimalBiddingObservation
        balance = 4
        obs = OptimalBiddingObservation(resource_capacities=[1], agent_balance=balance)
        actions = self.agent.get_action(self.agent_state, obs, _make_rng())
        for _, signal in actions:
            assert 0 <= signal <= balance

    def test_adapt_updates_last_balance(self):
        from karma_pp.impl.agents.optimal_bidding_resource_agent import OptimalBiddingObservation
        from karma_pp.impl.mechanisms.karma.karma_mechanism import KarmaResolution
        obs = OptimalBiddingObservation(resource_capacities=[1], agent_balance=3)
        res = KarmaResolution(agent_id=0, selected_outcome=(False,), transfer=-1, outcome_scores=[], n_agents=2)
        new_state = self.agent.adapt(self.agent_state, obs, res, reward=-1.0, timestep=1, rng=_make_rng())
        assert new_state.policy.last_balance == 3


class TestResourceAgentUrgencyTransition:
    """Verify Markov urgency transitions work correctly."""

    def setup_method(self):
        self.agent = TruthfulResourceAgent(
            transition_matrix=_TRANSITION,
            initial_urgency=0,
            urgency_levels=_URGENCY_LEVELS,
            reward_per_resource=_REWARD,
            no_resource_penalty=_PENALTY,
        )

    def test_initial_urgency_matches_config(self):
        state = self.agent.initialize(_world_dynamics(), None, _make_rng())
        assert state.private == 0

    def test_urgency_stays_in_valid_levels(self):
        from karma_pp.impl.mechanisms.benevolent_dictator import DictatorResolution
        state = self.agent.initialize(_world_dynamics(), None, _make_rng())
        res = DictatorResolution(agent_id=0, selected_outcome=(False,), outcome_scores=[])
        obs = ResourceAgentObservation(resource_capacities=[1])
        rng = _make_rng(99)
        for _ in range(200):
            state = self.agent.update_state(state, obs, res, rng)
            assert state.private in _URGENCY_LEVELS

    def test_invalid_urgency_raises(self):
        with pytest.raises(ValueError, match="not found"):
            self.agent._state_idx_from_urgency(99)
