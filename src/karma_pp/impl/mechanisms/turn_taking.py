"""TurnTaking mechanism: selects the agent most deserving of a turn (by turn_count/weight), then picks the decision with highest signal from that agent."""

from dataclasses import dataclass

import numpy as np
import structlog

from karma_pp.core.mechanism import Mechanism
from karma_pp.core.types import CollectiveAction, Resolution

log = structlog.get_logger(__name__)

TurnTakingState = dict[int, int]  # agent_id -> turn count
TurnTakingDynamics = type(None)
Observation = type(None)
Signal = list[float]


@dataclass
class TurnTakingReport[OUTCOME, DECISION]:
    """Report emitted by the TurnTaking mechanism for one timestep."""

    selected_decision: DECISION
    selected_outcomes: dict[int, OUTCOME]
    turn_holder_agent_id: int


class TurnTakingMechanism[OUTCOME, DECISION](
    Mechanism[
        OUTCOME,
        Signal,
        TurnTakingState,
        TurnTakingDynamics,
        Observation,
        TurnTakingReport[OUTCOME, DECISION],
        Resolution[OUTCOME],
        CollectiveAction[OUTCOME, DECISION],
    ]
):
    """
    Maintains turn counts per agent. Each step: select the agent most deserving
    of a turn (lowest turn_count / weight, so higher-weight agents get proportionally
    more turns). Ties are broken uniformly at random.
    """

    def initialize(
        self,
        agent_weights: dict[int, int],
        rng: np.random.Generator,
    ) -> tuple[TurnTakingState, TurnTakingDynamics]:
        del rng
        return {aid: 0 for aid in agent_weights}, None

    def run(
        self,
        mechanism_state: TurnTakingState,
        collective_action: CollectiveAction[OUTCOME, DECISION],
        rng: np.random.Generator,
    ) -> TurnTakingReport[OUTCOME, DECISION]:
        possible_decisions = collective_action.decisions
        if not possible_decisions:
            raise ValueError("No feasible decisions in collective action.")

        signals = np.asarray(collective_action.signals, dtype=np.float64)
        if signals.ndim != 2:
            raise ValueError("Signals must be a 2D array.")
        agent_ids = collective_action.agent_ids
        n_decisions = len(possible_decisions)
        if signals.shape[1] != n_decisions:
            raise ValueError("Signals columns must match number of decisions.")

        turn_counts = mechanism_state
        agent_weights = np.asarray(collective_action.agent_weights, dtype=np.float64)
        if len(agent_weights) != len(agent_ids):
            raise ValueError(
                f"agent_weights length {len(agent_weights)} must match number of agents {len(agent_ids)}"
            )
        if np.any(agent_weights <= 0):
            raise ValueError("agent_weights must be strictly positive.")
        # Priority = turn_count / weight; lower = more deserving (higher-weight agents get more turns)
        counts_for_participants = np.array(
            [turn_counts.get(aid, 0) for aid in agent_ids], dtype=np.float64
        )
        priorities = counts_for_participants / agent_weights
        min_priority = np.min(priorities)
        tied_indices = np.where(np.isclose(priorities, min_priority))[0].tolist()
        turn_holder_idx = int(rng.choice(tied_indices))
        turn_holder_id = agent_ids[turn_holder_idx]

        agent_signals = signals[turn_holder_idx]
        best_value = float(np.max(agent_signals))
        best_indices = np.flatnonzero(np.isclose(agent_signals, best_value))
        selected_idx = int(rng.choice(best_indices))
        selected_decision: DECISION = possible_decisions[selected_idx]

        decisions_to_outcomes = collective_action.decisions_to_outcomes
        agent_outcomes = collective_action.agent_outcomes
        selected_outcomes = {
            agent_id: agent_outcomes[row_idx][decisions_to_outcomes[selected_idx][row_idx]]
            for row_idx, agent_id in enumerate(agent_ids)
        }

        log.debug(
            "turn_taking_selected",
            turn_holder=turn_holder_id,
            selected_idx=selected_idx,
            turn_counts={aid: turn_counts.get(aid, 0) for aid in agent_ids},
        )

        return TurnTakingReport(
            selected_decision=selected_decision,
            selected_outcomes=selected_outcomes,
            turn_holder_agent_id=turn_holder_id,
        )

    def get_resolutions(
        self,
        mechanism_state: TurnTakingState,
        collective_action: CollectiveAction[OUTCOME, DECISION],
        report: TurnTakingReport[OUTCOME, DECISION],
    ) -> dict[int, Resolution[OUTCOME]]:
        del mechanism_state, collective_action

        agent_ids = list(report.selected_outcomes.keys())
        resolutions: dict[int, Resolution[OUTCOME]] = {}
        for agent_id in agent_ids:
            selected_outcome = report.selected_outcomes[agent_id]
            resolutions[agent_id] = Resolution(
                agent_id=agent_id,
                selected_outcome=selected_outcome,
            )
        return resolutions

    def update_state(
        self,
        previous: TurnTakingState,
        reports: dict[int, TurnTakingReport[OUTCOME, DECISION]],
        rng: np.random.Generator,
    ) -> TurnTakingState:
        del rng
        new_state = dict(previous)
        for report in reports.values():
            holder = report.turn_holder_agent_id
            new_state[holder] = previous.get(holder, 0) + 1
        return new_state
