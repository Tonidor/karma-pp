from dataclasses import dataclass, field
from itertools import product

import numpy as np

from karma_pp.impl.agents.resource_agent import ResourceAgentObservation, ResourceAgent
from karma_pp.impl.mechanisms.karma.karma_mechanism import KarmaDynamics, KarmaResolution, KarmaState
from karma_pp.impl.worlds.resource_world.resource_world import ResourceWorldDynamics, ResourceWorldState
from karma_pp.core.types import AgentState, PopulationState

StateKey = tuple[int, int]
ActionKey = tuple[int, ...]
Outcome = tuple[bool]
Commit = int


def _valid_actions(balance: int, n_outcomes: int) -> list[ActionKey]:
    return [a for a in product(range(balance + 1), repeat=n_outcomes)]


@dataclass(frozen=True)
class QLearningObservation(ResourceAgentObservation):
    """Observation for Q-learning agent; extends base with agent_balance."""

    agent_balance: int


@dataclass
class QLearningPolicyState:
    """Per-agent Q-learning state.

    Note: last_state and last_action are written in-place by _get_action so
    that adapt() (called in the same timestep) can read the (s, a) pair for
    the Q-update.  All other state transitions return new objects.
    """

    Q: dict[tuple[StateKey, ActionKey], float] = field(default_factory=dict)
    last_state: StateKey | None = None
    last_action: ActionKey | None = None
    epsilon: float = 0.5      # per-agent exploration rate, decayed in adapt()
    max_balance: int = 0      # set once from mechanism_dynamics at initialize time


class QLearningResourceAgent(
    ResourceAgent[
        ResourceWorldState,
        KarmaState,
        QLearningPolicyState,
        KarmaResolution,
    ]
):
    """Q-learning agent for karma mechanism."""

    def __init__(
        self,
        transition_matrix: list[list[float]],
        initial_urgency: int,
        urgency_levels: list[int],
        reward_per_resource: list[float],
        no_resource_penalty: list[float],
        alpha: float = 0.1,
        epsilon: float = 0.5,
        gamma: float = 0.99,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
    ):
        super().__init__(
            transition_matrix=transition_matrix,
            initial_urgency=initial_urgency,
            urgency_levels=urgency_levels,
            reward_per_resource=reward_per_resource,
            no_resource_penalty=no_resource_penalty,
        )
        self.alpha = float(alpha)
        self.init_epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

    def _initialize_policy(
        self,
        world_dynamics: ResourceWorldDynamics,
        mechanism_dynamics: KarmaDynamics,
        rng: np.random.Generator,
    ) -> QLearningPolicyState:
        return QLearningPolicyState(
            Q={},
            last_state=None,
            last_action=None,
            epsilon=self.init_epsilon,
            max_balance=mechanism_dynamics.max_balance,
        )

    def get_observation(
        self,
        agent_id: int,
        agent_state: AgentState[int, QLearningPolicyState],
        world_state: ResourceWorldState,
        mechanism_state: KarmaState,
        population_state: PopulationState[int, QLearningPolicyState],
        membership: tuple[int, int],
        rng: np.random.Generator,
    ) -> QLearningObservation:
        balance = mechanism_state.agent_balances[agent_id]
        return QLearningObservation(
            resource_capacities=world_state.resource_capacities,
            agent_balance=balance,
        )

    def _get_action(
        self,
        agent_state: AgentState[int, QLearningPolicyState],
        outcomes: list[Outcome],
        observation: QLearningObservation,
        rng: np.random.Generator,
    ) -> list[Commit]:
        policy = agent_state.policy
        n_outcomes = len(outcomes)
        balance = int(np.clip(observation.agent_balance, 0, policy.max_balance))
        state: StateKey = (int(agent_state.private), balance)
        actions = _valid_actions(balance, n_outcomes)
        if not actions:
            chosen = tuple([0] * n_outcomes)
            policy.last_state = state
            policy.last_action = chosen
            return list(chosen)
        q_vals = [policy.Q.get((state, a), 0.0) for a in actions]
        idx = int(rng.integers(0, len(actions))) if rng.random() < policy.epsilon else int(np.argmax(q_vals))
        chosen = actions[idx]
        # Store (state, action) so adapt() can perform the Q-update.
        # This is intentional in-place mutation of the mutable policy dataclass;
        # it avoids threading an extra return value through the interface.
        policy.last_state = state
        policy.last_action = chosen
        return list(chosen)

    def adapt(
        self,
        previous: AgentState[int, QLearningPolicyState],
        observation: QLearningObservation,
        resolution: KarmaResolution,
        reward: float,
        timestep: int,
        rng: np.random.Generator,
    ) -> AgentState[int, QLearningPolicyState]:
        del resolution, rng
        if timestep == -1:
            return AgentState(private=previous.private, policy=previous.policy)
        policy = previous.policy
        new_policy = QLearningPolicyState(
            Q=dict(policy.Q),
            last_state=policy.last_state,
            last_action=policy.last_action,
            epsilon=policy.epsilon,
            max_balance=policy.max_balance,
        )
        s = policy.last_state
        a = policy.last_action
        n_outcomes = len(a) if a is not None else 0
        if s is not None and a is not None and n_outcomes > 0:
            balance_next = int(np.clip(observation.agent_balance, 0, policy.max_balance))
            s_next: StateKey = (int(previous.private), balance_next)
            actions_next = _valid_actions(balance_next, n_outcomes)
            max_q_next = max((policy.Q.get((s_next, an), 0.0) for an in actions_next), default=0.0)
            key = (s, a)
            old_q = policy.Q.get(key, 0.0)
            td_target = float(reward) + self.gamma * max_q_next
            new_policy.Q[key] = old_q + self.alpha * (td_target - old_q)

        # Decay exploration rate per agent, clipping at epsilon_min.
        new_policy.epsilon = max(self.epsilon_min, policy.epsilon * self.epsilon_decay)
        return AgentState(private=previous.private, policy=new_policy)
