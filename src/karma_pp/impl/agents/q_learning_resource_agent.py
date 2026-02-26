from dataclasses import dataclass, field
from itertools import product

import numpy as np

from karma_pp.impl.agents.resource_agent import ResourceAgentObservation, ResourceAgent
from karma_pp.impl.mechanisms.karma.karma_mechanism import KarmaDynamics, KarmaResolution, KarmaState
from karma_pp.impl.worlds.resource_world.resource_world import ResourceWorldDynamics, ResourceWorldState
from karma_pp.src.types import AgentState, PopulationState

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
    Q: dict[tuple[StateKey, ActionKey], float] = field(default_factory=dict)
    last_state: StateKey | None = None
    last_action: ActionKey | None = None


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
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self._max_balance = 0

    def _initialize_policy(
        self,
        world_dynamics: ResourceWorldDynamics,
        mechanism_dynamics: KarmaDynamics,
        rng: np.random.Generator,
    ) -> QLearningPolicyState:
        self._max_balance = mechanism_dynamics.max_balance
        return QLearningPolicyState(Q={}, last_state=None, last_action=None)

    def get_observation(
        self,
        agent_id: int,
        agent_state: AgentState[int, QLearningPolicyState],
        world_state: ResourceWorldState,
        mechanism_state: KarmaState,
        population_state: PopulationState[int, QLearningPolicyState],
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
        memory = agent_state.policy
        n_outcomes = len(outcomes)
        balance = int(np.clip(observation.agent_balance, 0, self._max_balance))
        state: StateKey = (int(agent_state.private), balance)
        actions = _valid_actions(balance, n_outcomes)
        if not actions:
            return [0] * n_outcomes
        q_vals = [memory.Q.get((state, a), 0.0) for a in actions]
        idx = int(rng.integers(0, len(actions))) if rng.random() < self.epsilon else int(np.argmax(q_vals))
        chosen = actions[idx]
        memory.last_state = state
        memory.last_action = chosen
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
        del resolution, timestep, rng
        memory = previous.policy
        new_memory = QLearningPolicyState(
            Q=dict(memory.Q),
            last_state=memory.last_state,
            last_action=memory.last_action,
        )
        s = memory.last_state
        a = memory.last_action
        n_outcomes = len(a) if a is not None else 0
        if s is not None and a is not None and n_outcomes > 0:
            balance_next = int(np.clip(observation.agent_balance, 0, self._max_balance))
            s_next: StateKey = (int(previous.private), balance_next)
            actions_next = _valid_actions(balance_next, n_outcomes)
            max_q_next = max((memory.Q.get((s_next, an), 0.0) for an in actions_next), default=0.0)
            key = (s, a)
            old_q = memory.Q.get(key, 0.0)
            td_target = float(reward) + self.gamma * max_q_next
            new_memory.Q[key] = old_q + self.alpha * (td_target - old_q)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return AgentState(private=previous.private, policy=new_memory)
