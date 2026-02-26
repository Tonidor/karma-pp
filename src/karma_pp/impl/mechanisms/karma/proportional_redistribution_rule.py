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
