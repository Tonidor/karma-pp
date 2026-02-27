from math import comb

import numpy as np
import structlog

from karma_pp.impl.mechanisms.karma.karma_mechanism import SelectionRule

log = structlog.get_logger(__name__)


class MaxSumSelectionRule(SelectionRule):
    """Selection rule that chooses outcomes with maximum total
    commitment."""

    def __call__(
        self, collective_commits: list[list[int]]
    ) -> list[float]:
        """Return probability distribution favoring outcomes with higher total
        commits."""
        collective_commits = np.asarray(collective_commits, dtype=np.int64)

        # Calculate total commits for each outcome
        total_commits = np.sum(
            collective_commits, axis=0
        )  # Sum across agents for each outcome

        # Find the maximum total commit
        max_total = np.max(total_commits)

        # Create probability distribution: 1 for max outcomes, 0 for others
        probs = np.where(total_commits == max_total, 1.0, 0.0)

        # Normalize to sum to 1
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
        else:
            # If all are zero, use uniform distribution
            n_outcomes = collective_commits.shape[1]
            probs = np.ones(n_outcomes) / n_outcomes

        log.info("max_sum_selection", total_commits=total_commits.tolist(), max_total=int(max_total), probs=probs.tolist())
        return probs.tolist()

    def compute_gamma(
        self,
        b: int,
        v_b: dict[int, float],
        n_agents: int,
    ) -> float:
        """P(ego wins resource | ego bids b, N-1 opponents draw i.i.d. from v_b).

        Under the convention that agents bid 0 for all outcomes except their own
        winning outcome, MaxSumSelectionRule reduces to picking the highest bidder.
        Ties among t co-maximum bidders are broken uniformly (each wins with 1/t).

        For general N, uses the exact binomial expansion over the number j of
        opponents who also bid exactly b:

            γ(b) = Σ_{j=0}^{N-1} C(N-1,j) · v_b[b]^j · F(b-1)^{N-1-j} · 1/(j+1)

        where F(b-1) = Σ_{b'<b} v_b[b'].
        """
        n_opp = n_agents - 1
        f_below = sum(p for bid, p in v_b.items() if bid < b)   # P(opponent bids < b)
        p_eq = v_b.get(b, 0.0)                                   # P(opponent bids = b)

        gamma = 0.0
        for j in range(n_opp + 1):
            gamma += comb(n_opp, j) * (p_eq ** j) * (f_below ** (n_opp - j)) / (j + 1)

        return float(np.clip(gamma, 0.0, 1.0))
