from dataclasses import dataclass

import numpy as np
import structlog

from karma_pp.core.mechanism import Mechanism
from karma_pp.core.types import CollectiveAction, Resolution

log = structlog.get_logger(__name__)

# The benevolent dictator is stateless; both state and dynamics are represented as None.
BenevolentDictatorState = type(None)
BenevolentDictatorDynamics = type(None)
Observation = type(None)
Signal = list[float]


@dataclass
class DictatorReport[OUTCOME, DECISION]:
    """Report emitted by the benevolent dictator mechanism for one timestep."""

    selected_decision: DECISION
    selected_outcomes: dict[int, OUTCOME]  # agent_id -> outcome (populated by get_resolutions)


class BenevolentDictatorMechanism[OUTCOME, DECISION](
    Mechanism[
        OUTCOME,                             # OUTCOME
        Signal,                              # SIGNAL
        BenevolentDictatorState,             # MECHANISM_STATE
        BenevolentDictatorDynamics,          # MECHANISM_DYNAMICS
        Observation,                         # MECHANISM_OBSERVATION
        DictatorReport[OUTCOME, DECISION],   # REPORT
        Resolution[OUTCOME],  # RESOLUTION
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
        # Stateless mechanism: both state and dynamics are None.
        return None, None

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

        # Weight each agent's reward by their agent_weight for proportional consideration
        agent_weights = np.asarray(collective_action.agent_weights, dtype=np.float64)
        if len(agent_weights) != n_agents:
            raise ValueError(
                f"agent_weights length {len(agent_weights)} must match number of agents {n_agents}"
            )
        total_reward = (collective_rewards * agent_weights[:, np.newaxis]).sum(axis=0)  # (D,)
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
        log.debug(
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
        previous: BenevolentDictatorState,
        reports: dict[int, DictatorReport[OUTCOME, DECISION]],
        rng: np.random.Generator,
    ) -> BenevolentDictatorState:
        return previous
