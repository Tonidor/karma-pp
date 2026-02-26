from dataclasses import dataclass
from itertools import product

import numpy as np
import structlog

from karma_pp.src.world import World
from karma_pp.src.types import CollectiveAction

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ResourceWorldDynamics:
    resource_capacities: list[int]


@dataclass(frozen=True)
class ResourceWorldState:
    """World state holding resource capacities (immutable per run)."""

    resource_capacities: list[int]
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

    def __init__(self, resource_capacities: list[int], no_resource_penalty: float = 1.0):
        self.resource_capacities = resource_capacities
        self.resource_count = len(resource_capacities)
        self.no_resource_penalty = no_resource_penalty

    def initialize(
        self,
        n_agents: int,
        rng: np.random.Generator,
    ) -> tuple[ResourceWorldState, ResourceWorldDynamics]:
        del n_agents, rng
        world_state = ResourceWorldState(resource_capacities=self.resource_capacities)
        world_dynamics = ResourceWorldDynamics(
            resource_capacities=self.resource_capacities,
        )
        return world_state, world_dynamics

    def get_observations(
        self,
        agent_ids: list[int],
        world_state: ResourceWorldState,
    ) -> dict[int, int]:
        return {agent_id: self.resource_count for agent_id in agent_ids}

    def filter_actions(
        self,
        world_state: ResourceWorldState,
        agent_actions: list[list[tuple[Outcome, Signal]]],
        agent_ids: list[int],
    ) -> CollectiveAction[Outcome, Decision]:
        if len(agent_ids) != len(agent_actions):
            raise ValueError("agent_ids length must match agent_actions.")
        agent_outcomes = [[a[0] for a in actions] for actions in agent_actions]
        agent_signals = [[a[1] for a in actions] for actions in agent_actions]

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
        report: REPORT,
        rng: np.random.Generator,
    ) -> ResourceWorldState:
        return previous
