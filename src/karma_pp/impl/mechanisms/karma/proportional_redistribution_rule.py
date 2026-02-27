from itertools import combinations_with_replacement

import numpy as np
import structlog

from karma_pp.impl.mechanisms.karma.karma_mechanism import RedistributionRule

log = structlog.get_logger(__name__)


class ProportionalRedistributionRule(RedistributionRule):
    """Proportional redistribution rule."""

    def __call__(
        self,
        collective_commits: list[int],
        agent_weights: list[int],
    ) -> tuple[list[list[int]], list[float]]:
        """Return all possible proportional redistributions of commits + remainder.
        
        Args:
            collective_commits: Commitment from each agent
            agent_weights: Weight of each agent
            
        Returns:
            tuple: (transfer_vectors, probabilities) where probabilities are weighted
                   by agent weights for remainder distribution
        """
        n_agents = len(agent_weights)
        if n_agents == 0:
            raise ValueError("agent_weights must be non-empty.")
        if any(weight <= 0 for weight in agent_weights):
            raise ValueError("agent_weights must be strictly positive.")

        agent_weights_arr = np.asarray(agent_weights, dtype=np.int64)
        total_weight = int(agent_weights_arr.sum())

        collective_commits_arr = np.asarray(collective_commits, dtype=np.int64)
        if collective_commits_arr.ndim != 1 or collective_commits_arr.shape[0] != n_agents:
            raise ValueError(
                f"Collective commits length {len(collective_commits)} != n_agents {n_agents}"
            )

        # Calculate base distribution
        total_commits = int(collective_commits_arr.sum())
        equal_share = int(total_commits // total_weight)
        remainder = int(total_commits % total_weight)
        base_transfers_arr = equal_share * agent_weights_arr

        # Generate all valid remainder distributions
        transfers, probs = self._generate_remainder_distributions(
            base_transfers_arr,
            collective_commits_arr,
            agent_weights_arr,
            remainder,
        )

        # Validate outputs
        self._validate_outputs(collective_commits_arr, transfers, probs, n_agents)

        log.info(
            "proportional_redistribution",
            selected_decision_agent_commits=collective_commits_arr.tolist(),
            transfer_options=transfers,
            transfer_probs=probs,
        )
        return transfers, probs

    def _generate_remainder_distributions(
        self,
        base_transfers: np.ndarray,
        collective_commits: np.ndarray,
        agent_weights: np.ndarray,
        remainder: int,
    ) -> tuple[list[list[int]], list[float]]:
        """Generate all valid ways to distribute remainder tokens.
        
        Args:
            base_transfers: Base transfer amounts (equal share × weight)
            collective_commits: Original commitments
            agent_weights: Agent weights (max extra tokens per agent)
            remainder: Number of remainder tokens to distribute
            
        Returns:
            tuple: (transfers array, probabilities array)
        """
        if remainder == 0:
            # No remainder: single deterministic solution
            net_transfer = (base_transfers - collective_commits).astype(np.int64, copy=False)
            return [net_transfer.tolist()], [1.0]

        # Enumerate all ways to distribute remainder tokens
        # Constraint: agent i can receive at most agent_weights[i] extra tokens
        
        transfers_list: list[list[int]] = []
        probs_list: list[float] = []
        n_agents = int(agent_weights.shape[0])

        for combo in combinations_with_replacement(range(n_agents), remainder):
            # Count extra tokens for each agent
            extra_tokens = np.zeros(n_agents, dtype=np.int64)
            for agent_idx in combo:
                extra_tokens[agent_idx] += 1
            
            # Check weight constraint
            if np.any(extra_tokens > agent_weights):
                continue  # Invalid: exceeds agent capacity
            
            # Compute net transfer
            net_transfer = (base_transfers + extra_tokens - collective_commits).astype(np.int64, copy=False)
            transfers_list.append(net_transfer.tolist())
            
            # Probability proportional to product of weights
            combo_weights = agent_weights[list(combo)]
            probs_list.append(float(np.prod(combo_weights, dtype=np.float64)))

        # Normalize probabilities
        if not transfers_list:
            raise ValueError("No valid remainder redistribution found for given weights.")
        probs_arr = np.asarray(probs_list, dtype=np.float64)
        probs_sum = float(probs_arr.sum())
        if probs_sum <= 0.0:
            raise ValueError("Redistribution probabilities must have positive mass.")
        probs_arr /= probs_sum
        probs = probs_arr.tolist()

        return transfers_list, probs

    def _validate_outputs(
        self,
        collective_commits: np.ndarray,
        transfers: list[list[int]],
        probs: list[float],
        n_agents: int,
    ) -> None:
        """Validate transfer vectors and probabilities."""
        transfers_arr = np.asarray(transfers, dtype=np.int64)
        probs_arr = np.asarray(probs, dtype=np.float64)

        # Check dimensions
        if transfers_arr.ndim != 2 or transfers_arr.shape[1] != n_agents:
            raise ValueError(
                "Transfer vector length mismatch with n_agents"
            )
        
        # Check zero-sum property
        if not np.all(np.sum(transfers_arr, axis=1) == 0):
            raise ValueError("Transfers do not sum to 0 for at least one outcome.")
        
        # Check probability distribution
        if not np.isclose(probs_arr.sum(), 1.0):
            raise ValueError(
                f"Probs do not sum to 1.0: {probs_arr.sum()} != 1.0"
            )

        # Check that no agent pays more than their commitment
        min_allowed = -collective_commits.reshape(1, n_agents)
        if np.any(transfers_arr < min_allowed):
            raise ValueError("At least one agent pays more than their commitment.")

    def compute_kappa(
        self,
        k: int,
        b: int,
        v_b: dict[int, float],
        n_agents: int,
        gamma_b: float = 0.5,
    ) -> tuple[dict[int, float], dict[int, float]]:
        """Distributions over next karma balance for won and lost outcomes.

        The rule is symmetric: every agent pays their bid for the selected
        outcome and receives a proportional share of the pool.  Under the
        zero-bid-for-no-resource convention only the winner's bid enters the
        pool, so:

        Won case  (ego bid b, ego receives resource):
            Pool = b.  Each of N agents receives floor(b/N); the remainder
            b % N tokens are each assigned to a uniformly random agent.
            Ego's net: floor(b/N) [± 1] − b.

        Lost case (ego bid 0, winner bid b_win):
            Pool = b_win drawn from the conditional distribution of winning
            bids given ego lost.  For N=2 this is exact; for N>2 uses the
            distribution of the maximum of N-1 i.i.d. draws from v_b with
            a 0.5 tie-break approximation.
            Ego's net: floor(b_win/N) [± 1].
        """
        # ---- Won case -------------------------------------------------------
        won_result: dict[int, float] = {}
        base = b // n_agents
        rem = b % n_agents
        k_base = k + base - b          # floor(b/N) − b
        won_result[k_base] = (n_agents - rem) / n_agents
        if rem > 0:
            won_result[k_base + 1] = rem / n_agents

        # ---- Lost case -------------------------------------------------------
        lost_result: dict[int, float] = {}
        p_lose = 1.0 - gamma_b
        if p_lose < 1e-15:
            # Ego virtually always wins; lost case is never used.
            lost_result[k] = 1.0
            return won_result, lost_result

        # Build CDF of a single opponent draw from v_b.
        max_bid = max(v_b.keys(), default=0)
        cdf = [0.0] * (max_bid + 2)
        for bid_val, prob in v_b.items():
            if 0 <= bid_val <= max_bid:
                cdf[bid_val + 1] += prob
        for i in range(1, len(cdf)):
            cdf[i] += cdf[i - 1]

        def cdf_at(x: int) -> float:
            if x < 0:
                return 0.0
            if x >= len(cdf) - 1:
                return 1.0
            return cdf[x + 1]

        n_opp = n_agents - 1

        # Distribution of b_win = max of n_opp i.i.d. draws from v_b.
        for b_win in range(max_bid + 1):
            p_max = cdf_at(b_win) ** n_opp - cdf_at(b_win - 1) ** n_opp
            if p_max < 1e-15:
                continue

            # Conditional weight: P(b_win AND ego lost) / P(ego lost).
            if b_win > b:
                w = p_max / p_lose
            elif b_win == b:
                # Approximate: ego loses a tie with prob 0.5 (exact for N=2).
                w = 0.5 * p_max / p_lose
            else:
                continue  # b_win < b → ego would have won, skip

            if w < 1e-15:
                continue

            # Ego paid 0; pool = b_win; same formula as won case.
            base_l = b_win // n_agents
            rem_l = b_win % n_agents
            k_next_0 = k + base_l           # floor(b_win/N)
            k_next_1 = k + base_l + 1
            lost_result[k_next_0] = lost_result.get(k_next_0, 0.0) + w * (n_agents - rem_l) / n_agents
            if rem_l > 0:
                lost_result[k_next_1] = lost_result.get(k_next_1, 0.0) + w * rem_l / n_agents

        # Renormalise to guard against floating-point drift.
        total = sum(lost_result.values())
        if total > 1e-15:
            lost_result = {kn: p / total for kn, p in lost_result.items()}

        return won_result, lost_result
