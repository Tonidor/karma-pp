from dataclasses import dataclass
from itertools import product

import numpy as np
import structlog

from karma_pp.core.world import World
from karma_pp.core.types import CollectiveAction

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ResourceWorldDynamics:
    resource_capacities: list[int]
    collective_size_distribution: dict[int, float]  # size -> probability


@dataclass(frozen=True)
class ResourceWorldState:
    """World state holding resource capacities (immutable per run)."""
    public_states: dict[int, None]  # agent_id -> public state
    resource_capacities: list[int]

    @property
    def n_agents(self) -> int:
        """Number of agents in the population."""
        return len(self.public_states)

    @property
    def agent_ids(self) -> list[int]:
        """List of agent ids in the population."""
        return list(self.public_states.keys())


Outcome = tuple[bool]  # (N_resources,)
Signal = list[int]  # (N_resources,)
Decision = list[list[int]]  # (N_resources,) each element: list of agent indices that get that resource

class ResourceWorld[PRIVATE_STATE, REPORT](
    World[
        ResourceWorldState,
        ResourceWorldDynamics,
        PRIVATE_STATE,
        Outcome,
        Signal,
        CollectiveAction[Outcome, Decision],
        REPORT,
    ]
):
    """A resource world."""

    def __init__(
        self,
        collective_size_distribution: dict[int, float],
        resource_capacities: list[int],
    ) -> None:
        """Initialize a resource world with a distribution over collective sizes.

        collective_size_distribution maps group_size -> probability, and is expected
        to already be normalized (sum to 1.0, up to numerical tolerance).
        """
        size_values = np.array(sorted(collective_size_distribution.keys()), dtype=int)
        if np.any(size_values <= 0):
            raise ValueError("collective_size_distribution keys (sizes) must be positive integers.")
        self.size_values = size_values

        probs = np.array([float(collective_size_distribution[s]) for s in size_values], dtype=float)
        if np.any(probs < 0):
            raise ValueError("collective_size_distribution entries must be non-negative.")
        self.size_probs = probs / probs.sum()

        self.resource_capacities = resource_capacities
        self.resource_count = len(resource_capacities)

    def initialize(
        self,
        agent_ids: list[int],
        rng: np.random.Generator,
    ) -> tuple[ResourceWorldState, ResourceWorldDynamics]:
        del rng
        # Initialize trivial public state per agent; can later hold richer public info.
        public_states: dict[int, None] = {agent_id: None for agent_id in agent_ids}
        world_state = ResourceWorldState(
            public_states=public_states,
            resource_capacities=self.resource_capacities,
        )
        collective_size_distribution = dict(
            zip(self.size_values.tolist(), self.size_probs.tolist())
        )
        world_dynamics = ResourceWorldDynamics(
            resource_capacities=self.resource_capacities,
            collective_size_distribution=collective_size_distribution,
        )
        return world_state, world_dynamics

    def get_collectives(
        self,
        world_state: ResourceWorldState,
        rng: np.random.Generator,
    ) -> dict[int, list[int]]:
        """Randomly partition agents into collectives with random sizes.

        Sizes are drawn sequentially from the configured size distribution until
        all agents are assigned. Every agent appears in exactly one collective.
        """
        n_agents = world_state.n_agents
        if n_agents == 0:
            return {}
        
        # Sample a composition of n_agents via truncated draws.
        sizes: list[int] = []
        remaining = n_agents
        while remaining > 0:
            mask = self.size_values <= remaining
            p_trunc = self.size_probs[mask]
            p_trunc = p_trunc / p_trunc.sum()
            k = int(rng.choice(self.size_values[mask], p=p_trunc))
            sizes.append(k)
            remaining -= k

        # Randomly assign agents to these blocks.
        agent_ids = world_state.agent_ids
        rng.shuffle(agent_ids)
        collectives: dict[int, list[int]] = {}
        start = 0
        for cid, size in enumerate(sizes):
            end = start + size
            collectives[cid] = agent_ids[start:end]
            start = end
        return collectives

    def get_observations(
        self,
        agent_ids: list[int],
        world_state: ResourceWorldState,
    ) -> dict[int, int]:
        del world_state
        return {agent_id: self.resource_count for agent_id in agent_ids}

    def filter_actions(
        self,
        world_state: ResourceWorldState,
        agent_weights: dict[int, int],
        agent_actions: dict[int, list[tuple[Outcome, Signal]]],
    ) -> CollectiveAction[Outcome, Decision]:
        if set(agent_weights.keys()) != set(agent_actions.keys()):
            raise ValueError(
                "agent_weights and agent_actions must have matching agent ids: "
                f"weights={sorted(agent_weights.keys())} vs actions={sorted(agent_actions.keys())}"
            )
        agent_ids = sorted(agent_actions.keys())
        per_agent_actions = [agent_actions[agent_id] for agent_id in agent_ids]
        agent_outcomes = [[a[0] for a in actions] for actions in per_agent_actions]
        agent_signals = [[a[1] for a in actions] for actions in per_agent_actions]

        for i, outcomes in enumerate(agent_outcomes):
            if not isinstance(outcomes, list) or len(outcomes) < 1:
                raise ValueError("Outcomes must be a non-empty list of outcome vectors.")
            for outcome in outcomes:
                if len(outcome) != self.resource_count:
                    raise ValueError("Each outcome must be a tuple of length equal to the resource count.")
            if len(agent_signals[i]) != len(outcomes):
                raise ValueError("Each signal vector must match the number of outcomes.")

        possible_decisions: list[Decision] = []
        decisions_to_outcomes: list[list[int]] = []
        for choice in product(*[range(len(outcomes)) for outcomes in agent_outcomes]):
            chosen = [agent_outcomes[i][choice[i]] for i in range(len(agent_outcomes))]
            decision = [
                [i for i, outcome in enumerate(chosen) if outcome[r]]
                for r in range(self.resource_count)
            ]
            if self._is_feasible(decision):
                possible_decisions.append(decision)
                decisions_to_outcomes.append(list(choice))

        if not possible_decisions:
            raise ValueError("No feasible collective actions found.")

        collective_commits = [
            [agent_signals[i][decisions_to_outcomes[d][i]] for d in range(len(possible_decisions))]
            for i in range(len(agent_outcomes))
        ]
        decision_total_commits = np.asarray(collective_commits).sum(axis=0).tolist()
        log.info(
            "filter_actions",
            possible_decisions=possible_decisions,
            decision_total_commits=decision_total_commits,
        )

        return CollectiveAction[Outcome, Decision](
            agent_ids=agent_ids,
            agent_weights=[agent_weights[agent_id] for agent_id in agent_ids],
            decisions=possible_decisions,
            signals=collective_commits,
            decisions_to_outcomes=decisions_to_outcomes,
            agent_outcomes=agent_outcomes,
        )

    def _is_feasible(self, decision: Decision) -> bool:
        """Check if allocation is feasible given resource capacities."""
        return all(
            len(decision[r]) <= self.resource_capacities[r]
            for r in range(self.resource_count)
        )

    def update_state(
        self,
        previous: ResourceWorldState,
        reports: dict[int, REPORT],
        rng: np.random.Generator,
    ) -> ResourceWorldState:
        del reports, rng
        return previous
