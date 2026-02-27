"""CoinToss mechanism: selects a decision uniformly at random, ignoring collective signals."""

from dataclasses import dataclass

import numpy as np
import structlog

from karma_pp.core.mechanism import Mechanism
from karma_pp.core.types import CollectiveAction, Resolution

log = structlog.get_logger(__name__)

CoinTossState = type(None)
CoinTossDynamics = type(None)
Observation = type(None)
Signal = list[int] | list[float]


@dataclass
class CoinTossReport[OUTCOME, DECISION]:
    """Report emitted by the CoinToss mechanism for one timestep."""

    selected_decision: DECISION
    selected_outcomes: dict[int, OUTCOME]


class CoinTossMechanism[OUTCOME, DECISION](
    Mechanism[
        OUTCOME,
        Signal,
        CoinTossState,
        CoinTossDynamics,
        Observation,
        CoinTossReport[OUTCOME, DECISION],
        Resolution[OUTCOME],
        CollectiveAction[OUTCOME, DECISION],
    ]
):
    """
    Selects a feasible collective decision uniformly at random.
    Ignores all collective signals.
    """

    def initialize(
        self,
        agent_weights: dict[int, int],
        rng: np.random.Generator,
    ) -> tuple[CoinTossState, CoinTossDynamics]:
        return None, None

    def run(
        self,
        mechanism_state: CoinTossState,
        collective_action: CollectiveAction[OUTCOME, DECISION],
        rng: np.random.Generator,
    ) -> CoinTossReport[OUTCOME, DECISION]:
        del mechanism_state

        possible_decisions = collective_action.decisions
        if not possible_decisions:
            raise ValueError("No feasible decisions in collective action.")

        n_decisions = len(possible_decisions)
        selected_idx = int(rng.integers(0, n_decisions))
        selected_decision: DECISION = possible_decisions[selected_idx]

        decisions_to_outcomes = collective_action.decisions_to_outcomes
        agent_outcomes = collective_action.agent_outcomes
        agent_ids = collective_action.agent_ids
        selected_outcomes = {
            agent_id: agent_outcomes[row_idx][decisions_to_outcomes[selected_idx][row_idx]]
            for row_idx, agent_id in enumerate(agent_ids)
        }
        log.info(
            "coin_toss_selected",
            selected_idx=selected_idx,
            n_decisions=n_decisions,
        )
        return CoinTossReport(
            selected_decision=selected_decision,
            selected_outcomes=selected_outcomes,
        )

    def get_resolutions(
        self,
        mechanism_state: CoinTossState,
        collective_action: CollectiveAction[OUTCOME, DECISION],
        report: CoinTossReport[OUTCOME, DECISION],
    ) -> dict[int, Resolution[OUTCOME]]:
        del mechanism_state

        agent_ids = collective_action.agent_ids

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
        previous: CoinTossState | None,
        report: CoinTossReport[OUTCOME, DECISION] | None,
        rng: np.random.Generator,
    ) -> CoinTossState:
        return None
