import numpy as np
import structlog

from karma_pp.impl.mechanisms.karma.karma_mechanism import RedistributionRule

log = structlog.get_logger(__name__)


class WinnersPayRedistributionRule(RedistributionRule):
    """Redistribution rule where winners pay their full bid and receive nothing."""

    def __call__(
        self,
        collective_commits: list[int],
        agent_weights: list[int],
    ) -> tuple[list[list[int]], list[float]]:
        """Return transfer vectors where only losers receive karma.

        Args:
            collective_commits: Commitment from each agent for the selected
                decision.
            agent_weights: Weight of each agent used for redistribution.

        Returns:
            (transfer_vectors, probabilities): transfer_vectors[i] is a list of
            integer net transfers for each agent summing to 0.
        """
        n_agents = len(collective_commits)
        if n_agents == 0:
            raise ValueError("collective_commits must be non-empty.")
        if any(w <= 0 for w in agent_weights):
            raise ValueError("agent_weights must be strictly positive.")

        commits = np.asarray(collective_commits, dtype=np.int64)
        weights = np.asarray(agent_weights, dtype=np.int64)

        max_commit = int(commits.max())
        if max_commit == 0:
            # No one bid anything — no karma moves.
            transfer = np.zeros(n_agents, dtype=np.int64).tolist()
            log.info("winners_pay_redistribution", pool=0, transfer=transfer)
            return [transfer], [1.0]

        # Identify winners (non-zero bidders) and losers.
        is_winner = commits > 0
        is_loser = ~is_winner
        n_losers = int(is_loser.sum())

        pool = int(commits[is_winner].sum())

        # Base share per loser weighted by their weights (all agents may be losers).
        loser_weights = weights[is_loser]
        total_loser_weight = int(loser_weights.sum())
        base_share = pool // total_loser_weight
        base_transfers = np.zeros(n_agents, dtype=np.int64)
        base_transfers[is_loser] = base_share * loser_weights
        remainder = pool % total_loser_weight

        if remainder == 0:
            transfer = (base_transfers - commits).tolist()
            log.info("winners_pay_redistribution", pool=pool, transfer=transfer)
            return [transfer], [1.0]

        # Distribute remainder tokens to losers weighted by their weights.
        loser_indices = np.where(is_loser)[0]

        transfers_list: list[list[int]] = []
        probs_list: list[float] = []

        # Enumerate all ways to assign `remainder` extra tokens among losers.
        from itertools import combinations_with_replacement

        for combo in combinations_with_replacement(range(len(loser_indices)), remainder):
            extra_for_losers = np.zeros(len(loser_indices), dtype=np.int64)
            for idx in combo:
                extra_for_losers[idx] += 1

            # Each loser can receive at most 1 extra token (weight=1 agents).
            # Generalised: at most agent_weight[i] extra tokens.
            if np.any(extra_for_losers > loser_weights):
                continue

            loser_extra = np.zeros(n_agents, dtype=np.int64)
            loser_extra[loser_indices] = extra_for_losers

            net = -commits + base_transfers + loser_extra
            transfers_list.append(net.tolist())

            combo_weights = loser_weights[list(combo)]
            probs_list.append(float(np.prod(combo_weights, dtype=np.float64)))

        if not transfers_list:
            raise ValueError("No valid remainder redistribution found.")

        probs_arr = np.asarray(probs_list, dtype=np.float64)
        probs_arr /= probs_arr.sum()

        log.info(
            "winners_pay_redistribution",
            pool=pool,
            n_losers=n_losers,
            n_options=len(transfers_list),
        )
        return transfers_list, probs_arr.tolist()

    def compute_kappa(
        self,
        k: int,
        b: int,
        v_b: dict[int, float],
        n_agents: int,
        gamma_b: float = 0.5,
    ) -> tuple[dict[int, float], dict[int, float]]:
        """Distributions over next karma balance for won and lost outcomes.

        Won case (ego bid b, ego wins resource):
            Ego pays b in full and receives nothing.
            k_next = k − b  (deterministic).

        Lost case (ego bid 0, winner bid b_win):
            Ego receives floor(b_win / (N−1)) [± 1 for remainder].
            b_win is marginalised over the conditional distribution of the
            winning bid given ego lost.  For N=2 this is exact; for N>2 uses
            the distribution of the maximum of N-1 i.i.d. draws from v_b with
            a 0.5 tie-break approximation (exact for N=2).
        """
        # ---- Won case -------------------------------------------------------
        # Ego pays b, receives 0 back.
        won_result: dict[int, float] = {k - b: 1.0}

        # ---- Lost case -------------------------------------------------------
        lost_result: dict[int, float] = {}
        p_lose = 1.0 - gamma_b
        if p_lose < 1e-15:
            lost_result[k] = 1.0
            return won_result, lost_result

        n_losers = n_agents - 1  # under zero-bid convention, only 1 winner

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

        for b_win in range(max_bid + 1):
            p_max = cdf_at(b_win) ** n_opp - cdf_at(b_win - 1) ** n_opp
            if p_max < 1e-15:
                continue

            if b_win > b:
                w = p_max / p_lose
            elif b_win == b:
                w = 0.5 * p_max / p_lose
            else:
                continue  # b_win < b → ego would have won

            if w < 1e-15:
                continue

            # Ego receives floor(b_win / n_losers) [± 1 for remainder].
            base = b_win // n_losers
            rem = b_win % n_losers
            k_next_0 = k + base
            k_next_1 = k + base + 1
            lost_result[k_next_0] = (
                lost_result.get(k_next_0, 0.0) + w * (n_losers - rem) / n_losers
            )
            if rem > 0:
                lost_result[k_next_1] = (
                    lost_result.get(k_next_1, 0.0) + w * rem / n_losers
                )

        # Renormalise.
        total = sum(lost_result.values())
        if total > 1e-15:
            lost_result = {kn: p / total for kn, p in lost_result.items()}

        return won_result, lost_result
