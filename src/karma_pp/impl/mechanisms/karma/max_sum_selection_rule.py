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
