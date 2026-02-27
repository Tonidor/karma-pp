from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import structlog

from karma_pp.core.mechanism import Mechanism
from karma_pp.core.types import CollectiveAction, Resolution
from karma_pp.utils.loading_utils import Config, instantiate

log = structlog.get_logger(__name__)

Commit = list[int]  # SIGNAL


@dataclass
class KarmaReport[OUTCOME, DECISION]:
    """Report after a timestep."""

    selected_decision: DECISION
    transfers: dict[int, int]  # agent_id -> transfer
    selected_outcomes: dict[int, OUTCOME]  # agent_id -> outcome


@dataclass
class KarmaState:
    """Maps agent_id to karma balance."""

    agent_balances: dict[int, int]


@dataclass
class KarmaObservation:
    agent_balance: int


@dataclass(frozen=True)
class KarmaResolution[OUTCOME, DECISION](Resolution[OUTCOME]):
    """Karma mechanism resolution with transfer and outcome scores."""

    transfer: int
    outcome_scores: list[float]
    n_agents: int


@runtime_checkable
class SelectionRule(Protocol):
    """Map collective signals to probability distribution over decisions."""

    def __call__(
        self,
        collective_commits: list[list[int]],
    ) -> list[float]:
        """Return probability distribution over decisions.

        Args:
            collective_commits: Collective signal matrix, shape (N_agents, N_decisions).
                collective_commits[agent_row][d] = agent's signal for decision d.

        Returns:
            Probability distribution over decisions, length N_decisions.
        """
        ...

    def compute_gamma(
        self,
        b: int,
        v_b: dict[int, float],
        n_agents: int,
    ) -> float:
        """Probability that an ego agent bidding b wins the resource.

        Uses a mean-field approximation: the N-1 opponents each draw their bid
        i.i.d. from v_b.  Used by FullInfoLearningAgent to compute the outcome
        probability γ[o|b](d, π) required by Algorithm 1.

        Args:
            b: Ego agent's bid.
            v_b: Population-level bid distribution {bid: probability}.
            n_agents: Total number of agents (including ego).

        Returns:
            P(ego gets resource | ego bids b).
        """
        ...


@runtime_checkable
class RedistributionRule(Protocol):
    """Map collective commits and agent weights to transfers and probabilities."""

    def __call__(
        self,
        selected_commits: list[int],
        agent_weights: list[int],
    ) -> tuple[list[list[int]], list[float]]:
        """Return transfer vector and probabilities.

        Args:
            selected_commits: The commits for the selected decision (N_agents,).
            agent_weights: The weights of the agents (N_agents,).

        Returns:
            The transfer vectors for the agents (N_transfer_vectors, N_agents).
            The probabilities of the transfer vectors (N_transfer_vectors,).
        """
        ...

    def compute_kappa(
        self,
        k: int,
        b: int,
        v_b: dict[int, float],
        n_agents: int,
        gamma_b: float = 0.5,
    ) -> tuple[dict[int, float], dict[int, float]]:
        """Distributions over next karma balance k' for won and lost outcomes.

        Assumes agents bid 0 for all outcomes except their own winning outcome,
        so only the winner's bid enters the redistribution pool.  Used by
        FullInfoLearningAgent to compute κ[k'|k,b,o](d,π) (Algorithm 1).

        Args:
            k: Ego's current karma balance.
            b: Ego's bid for the resource outcome.
            v_b: Population-level bid distribution {bid: probability}.
                 Used to marginalise over winning bids in the lost case.
            n_agents: Total number of agents (including ego).
            gamma_b: P(ego wins | ego bids b).  Required for the lost case.

        Returns:
            (kappa_won, kappa_lost): each a {k_next: probability} dict.
        """
        ...


@dataclass(frozen=True)
class KarmaDynamics:
    selection_rule: SelectionRule
    redistribution_rule: RedistributionRule
    weight_karma_ratio: float
    max_balance: int
    agent_weights: dict[int, int]  # agent_id -> weight; used by redistribution rule


class KarmaMechanism[OUTCOME, DECISION](
    Mechanism[
        OUTCOME,                             # OUTCOME
        Commit,                              # SIGNAL
        KarmaState,                          # MECHANISM_STATE
        KarmaDynamics,                       # MECHANISM_DYNAMICS
        KarmaObservation,                    # MECHANISM_OBSERVATION
        KarmaReport[OUTCOME, DECISION],      # REPORT
        KarmaResolution[OUTCOME, DECISION],  # RESOLUTION
        CollectiveAction[OUTCOME, DECISION], # COLLECTIVE_ACTION
    ]
):
    """A karma mechanism."""

    def __init__(
        self,
        selection_rule: Config,
        redistribution_rule: Config,
        weight_karma_ratio: float,
    ) -> None:
        """Initialize the karma mechanism."""
        self.selection_rule: SelectionRule = instantiate(selection_rule["code"], selection_rule["parameters"])
        self.redistribution_rule: RedistributionRule = instantiate(redistribution_rule["code"], redistribution_rule["parameters"])
        self.weight_karma_ratio = weight_karma_ratio
        self.max_balance: int | None = None
        # Stored at initialize() time so run() can pass correct weights to redistribution_rule.
        self._agent_weights: dict[int, int] = {}

    def initialize(
        self,
        agent_weights: dict[int, int],
        rng: np.random.Generator,
    ) -> tuple[KarmaState, KarmaDynamics]:
        agent_balances = {
            agent_id: int(weight * self.weight_karma_ratio)
            for agent_id, weight in agent_weights.items()
        }
        if any(b < 0 for b in agent_balances.values()):
            raise ValueError("Agent balances must be non-negative.")
        self.max_balance = sum(agent_balances.values())
        self._agent_weights = dict(agent_weights)
        dynamics = KarmaDynamics(
            selection_rule=self.selection_rule,
            redistribution_rule=self.redistribution_rule,
            weight_karma_ratio=self.weight_karma_ratio,
            max_balance=self.max_balance,
            agent_weights=agent_weights,
        )
        log.debug("mechanism_state_initialized", agent_balances=agent_balances)
        return KarmaState(agent_balances=agent_balances), dynamics

    def get_observations(
        self,
        agent_ids: list[int],
        mechanism_state: KarmaState,
    ) -> dict[int, KarmaObservation]:
        observations = {}
        for agent_id in agent_ids:
            balance = mechanism_state.agent_balances[agent_id]
            observations[agent_id] = KarmaObservation(agent_balance=balance)
        return observations

    def run(
        self,
        mechanism_state: KarmaState,
        collective_action: CollectiveAction[OUTCOME, DECISION],
        rng: np.random.Generator,
    ) -> KarmaReport[OUTCOME, DECISION]:
        """Select a collective decision and compute agent transfers.

        Agent order is taken only from collective_action.agent_ids; do not use
        mechanism_state.agent_balances key order.
        """
        decisions = collective_action.decisions
        commits = collective_action.signals  # (N_agents, N_decisions)
        agent_ids = collective_action.agent_ids
        n_agents = len(agent_ids)
        n_decisions = len(decisions)

        log.debug("mechanism_run_start", n_decisions=n_decisions)
        if len(mechanism_state.agent_balances) != n_agents:
            raise ValueError("Mechanism state does not match number of agents.")
        if n_decisions != len(commits[0]) if commits else 0:
            raise ValueError("Signals columns must match number of decisions.")
        for aid in agent_ids:
            if aid not in mechanism_state.agent_balances:
                raise ValueError("Every collective_action.agent_id must be in mechanism state.")

        # Validate each agent's signals against their balance
        for row_idx, agent_id in enumerate(agent_ids):
            balance = mechanism_state.agent_balances[agent_id]
            row = commits[row_idx]
            if len(row) != n_decisions:
                raise ValueError("Each agent must have one signal per decision.")
            if any(c < 0 or c > balance for c in row):
                raise ValueError("Commits must be between 0 and the agent's balance.")

        # Select decision
        probs = self.selection_rule(commits)
        selected_idx = int(rng.choice(len(probs), p=probs))
        collective_decision: DECISION = decisions[selected_idx]

        # Transfers for selected decision: one signal per agent (column selected_idx).
        # Agent weights are ordered to match collective_action.agent_ids.
        commits_selected = [commits[row_idx][selected_idx] for row_idx in range(n_agents)]
        agent_weights_list = [
            int(self._agent_weights.get(agent_id, 1))
            for agent_id in agent_ids
        ]
        possible_transfers, transfer_probs = self.redistribution_rule(
            commits_selected,
            agent_weights_list,
        )
        transfer_list = possible_transfers[
            int(rng.choice(len(possible_transfers), p=transfer_probs))
        ]
        transfers = {agent_id: transfer_list[row_idx] for row_idx, agent_id in enumerate(agent_ids)}

        # Selected outcome per agent based on selected decision
        decisions_to_outcomes = collective_action.decisions_to_outcomes  # (N_decisions, N_agents)
        agent_outcomes = collective_action.agent_outcomes
        selected_outcomes = {
            agent_id: agent_outcomes[row_idx][decisions_to_outcomes[selected_idx][row_idx]]
            for row_idx, agent_id in enumerate(agent_ids)
        }
        log.info("mechanism_run_complete", selected_decision=collective_decision, transfers=transfers)

        return KarmaReport[OUTCOME, DECISION](
            selected_decision=collective_decision,
            transfers=transfers,
            selected_outcomes=selected_outcomes,
        )

    def get_resolutions(
        self,
        mechanism_state: KarmaState,
        collective_action: CollectiveAction[OUTCOME, DECISION],
        report: KarmaReport[OUTCOME, DECISION],
    ) -> dict[int, KarmaResolution[OUTCOME, DECISION]]:
        """Build per-agent resolution. Agent order from collective_action.agent_ids only.

        Uses decisions_to_outcomes to convert selected decision to outcomes.
        """
        commits = collective_action.signals  # (N_agents, N_decisions)
        decisions_to_outcomes = collective_action.decisions_to_outcomes  # (N_decisions, N_agents)
        agent_ids = collective_action.agent_ids
        n_agents = len(agent_ids)
        n_decisions = len(decisions_to_outcomes)

        agent_outcomes = collective_action.agent_outcomes

        resolutions: dict[int, KarmaResolution[OUTCOME, DECISION]] = {}
        for row_idx, agent_id in enumerate(agent_ids):
            n_outcomes = len(agent_outcomes[row_idx])
            outcome_scores = [float("-inf")] * n_outcomes
            for d in range(n_decisions):
                outcome_idx = decisions_to_outcomes[d][row_idx]
                total_commit_d = sum(commits[i][d] for i in range(n_agents))
                commit_without_agent = total_commit_d - commits[row_idx][d]
                outcome_scores[outcome_idx] = max(outcome_scores[outcome_idx], float(commit_without_agent))

            selected_outcome = report.selected_outcomes[agent_id]
            transfer = report.transfers.get(agent_id, 0)
            resolutions[agent_id] = KarmaResolution(
                agent_id=agent_id,
                selected_outcome=selected_outcome,
                transfer=transfer,
                outcome_scores=outcome_scores,
                n_agents=n_agents,
            )
        return resolutions

    def update_state(
        self,
        previous: KarmaState | None,
        report: KarmaReport[OUTCOME, DECISION] | None,
        rng: np.random.Generator,
    ) -> KarmaState:
        """Apply transfers from report to produce the next karma state.

        Note: call initialize() before the simulation loop to create the initial
        state. This method requires both a valid previous state and a report.
        """
        if previous is None:
            raise ValueError(
                "KarmaMechanism.update_state requires a non-None previous state. "
                "Call initialize() to create the initial mechanism state."
            )
        if report is None:
            raise ValueError("KarmaMechanism.update_state requires a non-None report.")

        new_balances = {
            agent_id: previous.agent_balances[agent_id] + report.transfers.get(agent_id, 0)
            for agent_id in previous.agent_balances
        }
        log.info("mechanism_state_updated", balances=new_balances)
        return KarmaState(agent_balances=new_balances)
