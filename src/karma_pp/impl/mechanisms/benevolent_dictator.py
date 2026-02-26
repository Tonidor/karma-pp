from dataclasses import dataclass
from types import NoneType

import numpy as np
import structlog

from karma_pp.src.mechanism import Mechanism
from karma_pp.src.types import CollectiveAction, Resolution

log = structlog.get_logger(__name__)

BenevolentDictatorState = NoneType
BenevolentDictatorDynamics = NoneType
Observation = NoneType
Signal = list[float]


@dataclass
class DictatorReport[OUTCOME, DECISION]:
    """Report emitted by the benevolent dictator mechanism for one timestep."""

    selected_decision: DECISION
    selected_outcomes: dict[int, OUTCOME]  # agent_id -> outcome (populated by get_resolutions)


@dataclass(frozen=True)
class DictatorResolution[OUTCOME, DECISION](Resolution[OUTCOME]):
    """Per-agent view of dictator decision and candidate outcome scores."""

    outcome_scores: list[float]


class BenevolentDictatorMechanism[OUTCOME, DECISION](
    Mechanism[
        OUTCOME,                             # OUTCOME
        Signal,                              # SIGNAL
        BenevolentDictatorState,             # MECHANISM_STATE
        BenevolentDictatorDynamics,          # MECHANISM_DYNAMICS
        Observation,                         # MECHANISM_OBSERVATION
        DictatorReport[OUTCOME, DECISION],   # REPORT
        DictatorResolution[OUTCOME, DECISION],  # RESOLUTION
        CollectiveAction[OUTCOME, DECISION], # COLLECTIVE_ACTION
    ]
):
    """
    Selects the feasible collective decision with the highest total reported reward.

    The mechanism assumes `collective_action.signals` already contains truthful per-agent
    rewards for each feasible decision.
    """

    def initialize(
        self,
        agent_weights: dict[int, int],
        rng: np.random.Generator,
    ) -> tuple[BenevolentDictatorState, BenevolentDictatorDynamics]:
        return BenevolentDictatorState(), BenevolentDictatorDynamics()

    def get_dynamics(self) -> BenevolentDictatorDynamics:
        return BenevolentDictatorDynamics()

    def run(
        self,
        mechanism_state: BenevolentDictatorState,
        collective_action: CollectiveAction[OUTCOME, DECISION],
        rng: np.random.Generator,
    ) -> DictatorReport[OUTCOME, DECISION]:
        del mechanism_state

        possible_decisions = collective_action.decisions
        if not possible_decisions:
            raise ValueError("No feasible decisions in collective action.")

        collective_rewards = np.asarray(collective_action.signals, dtype=np.float64)  # (N, D)
        if collective_rewards.ndim != 2:
            raise ValueError("Collective signals must be a 2D array-like structure.")

        n_agents, n_decisions = collective_rewards.shape
        if n_decisions != len(possible_decisions):
            raise ValueError("Signals must provide one value per feasible decision.")

        if not np.all(np.isfinite(collective_rewards)):
            raise ValueError("All reported rewards must be finite numbers.")

        total_reward = np.sum(collective_rewards, axis=0)  # (D,)
        best_value = float(np.max(total_reward))
        best_indices = np.flatnonzero(np.isclose(total_reward, best_value))
        selected_idx = int(rng.choice(best_indices))
        selected_decision: DECISION = possible_decisions[selected_idx]

        # Selected outcome per agent based on selected decision
        decisions_to_outcomes = collective_action.decisions_to_outcomes  # (N_decisions, N_agents)
        agent_outcomes = collective_action.agent_outcomes
        agent_ids = collective_action.agent_ids
        selected_outcomes = {
            agent_id: agent_outcomes[row_idx][decisions_to_outcomes[selected_idx][row_idx]]
            for row_idx, agent_id in enumerate(agent_ids)
        }
        log.info(
            "benevolent_dictator_selected",
            selected_idx=selected_idx,
            selected_value=best_value,
            total_reward=total_reward.tolist(),
        )
        return DictatorReport(
            selected_decision=selected_decision,
            selected_outcomes=selected_outcomes,
        )

    def get_resolutions(
        self,
        mechanism_state: BenevolentDictatorState,
        collective_action: CollectiveAction[OUTCOME, DECISION],
        report: DictatorReport[OUTCOME, DECISION],
    ) -> dict[int, DictatorResolution[OUTCOME, DECISION]]:
        del mechanism_state

        collective_rewards = np.asarray(collective_action.signals, dtype=np.float64)  # (N, D)
        decisions_to_outcomes = collective_action.decisions_to_outcomes
        agent_ids = collective_action.agent_ids
        n_agents = len(agent_ids)
        total_reward = np.sum(collective_rewards, axis=0) if collective_rewards.size else np.array([])

        agent_outcomes = collective_action.agent_outcomes
        resolutions: dict[int, DictatorResolution[OUTCOME, DECISION]] = {}
        for row_idx, agent_id in enumerate(agent_ids):
            if decisions_to_outcomes and total_reward.size > 0:
                n_outcomes = len(agent_outcomes[row_idx])
                scores = np.full(n_outcomes, -np.inf, dtype=float)
                for d in range(len(decisions_to_outcomes)):
                    outcome_idx = decisions_to_outcomes[d][row_idx]
                    scores[outcome_idx] = max(scores[outcome_idx], float(total_reward[d]))
                outcome_scores = scores.tolist()
            else:
                outcome_scores = []

            selected_outcome = report.selected_outcomes[agent_id]
            resolutions[agent_id] = DictatorResolution(
                agent_id=agent_id,
                selected_outcome=selected_outcome,
                outcome_scores=outcome_scores,
            )
        return resolutions

    def update_state(
        self,
        previous: BenevolentDictatorState | None,
        report: DictatorReport[OUTCOME, DECISION] | None,
        rng: np.random.Generator,
    ) -> BenevolentDictatorState:
        # Stateless mechanism; state is intentionally unchanged over time.
        return previous
