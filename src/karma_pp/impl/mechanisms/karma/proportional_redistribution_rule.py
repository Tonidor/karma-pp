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
        rng: np.random.Generator,
    ) -> list[int]:
        """Sample and return a proportional net transfer vector.

        Each agent receives floor(pool / total_weight) * w_i karma. The
        remainder pool % total_weight tokens are distributed via slot-expansion
        sampling: each agent i contributes w_i slots; remainder slots are drawn
        uniformly without replacement.

        Args:
            collective_commits: Commitment from each agent for the selected decision.
            agent_weights: Weight of each agent.
            rng: Random number generator for remainder allocation.

        Returns:
            Net transfer for each agent, summing to 0.
        """
        n_agents = len(agent_weights)
        if n_agents == 0:
            raise ValueError("agent_weights must be non-empty.")
        if any(weight <= 0 for weight in agent_weights):
            raise ValueError("agent_weights must be strictly positive.")

        weights = np.asarray(agent_weights, dtype=np.int64)
        commits = np.asarray(collective_commits, dtype=np.int64)
        if commits.shape[0] != n_agents:
            raise ValueError(
                f"Collective commits length {len(collective_commits)} != n_agents {n_agents}"
            )

        total_weight = int(weights.sum())
        total_commits = int(commits.sum())
        base_per_weight = total_commits // total_weight
        remainder = total_commits % total_weight

        # Base: each agent gets base_per_weight * w_i karma back.
        base = base_per_weight * weights

        # Remainder: sample without replacement from the slot pool.
        # Slot pool: agent i appears w_i times → probability proportional to w_i.
        extra = np.zeros(n_agents, dtype=np.int64)
        if remainder > 0:
            slots = np.repeat(np.arange(n_agents), weights)
            chosen = rng.choice(len(slots), size=remainder, replace=False)
            chosen_agents, counts = np.unique(slots[chosen], return_counts=True)
            extra[chosen_agents] = counts

        net = base + extra - commits
        log.debug(
            "proportional_redistribution",
            commits=commits.tolist(),
            net_transfer=net.tolist(),
        )
        assert int(net.sum()) == 0, f"Transfer not zero-sum: {net.tolist()}"
        return net.tolist()

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
