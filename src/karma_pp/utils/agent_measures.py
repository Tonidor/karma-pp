import numpy as np


def _validate_transition_matrix(transition_matrix: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    transition_matrix = np.asarray(transition_matrix, dtype=float)
    if transition_matrix.ndim != 2 or transition_matrix.shape[0] != transition_matrix.shape[1]:
        raise ValueError("Transition matrix must be square with shape (K, K).")
    if np.any(transition_matrix < -atol):
        raise ValueError("Transition matrix entries must be non-negative.")
    row_sums = np.sum(transition_matrix, axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        raise ValueError("Each transition matrix row must sum to 1.")
    transition_matrix = np.clip(transition_matrix, 0.0, 1.0)
    transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)
    return transition_matrix


def stationary_distribution(
    transition_matrix: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 10000,
) -> np.ndarray:
    """
    Compute stationary distribution pi for a finite-state Markov chain.

    Returns:
        pi with shape (K,), satisfying pi @ T ~= pi and sum(pi)=1.
    """
    transition_matrix = _validate_transition_matrix(transition_matrix)
    n_states = transition_matrix.shape[0]
    pi = np.full(n_states, 1.0 / n_states, dtype=float)
    for _ in range(max_iter):
        pi_next = pi @ transition_matrix
        if np.max(np.abs(pi_next - pi)) < tol:
            pi = pi_next
            break
        pi = pi_next
    pi = np.clip(pi, 0.0, 1.0)
    pi = pi / np.sum(pi)
    return pi


def _resolve_urgency_levels(
    transition_matrices: list[np.ndarray],
    urgency_levels: np.ndarray | list[np.ndarray] | None,
) -> list[np.ndarray]:
    resolved: list[np.ndarray] = []
    if urgency_levels is None:
        for transition_matrix in transition_matrices:
            k = transition_matrix.shape[0]
            resolved.append(np.arange(k, dtype=float))
        return resolved

    if isinstance(urgency_levels, np.ndarray):
        shared_levels = np.asarray(urgency_levels, dtype=float)
        for transition_matrix in transition_matrices:
            k = transition_matrix.shape[0]
            if shared_levels.shape[0] != k:
                raise ValueError(
                    "Shared urgency_levels length must match each agent's matrix size."
                )
            resolved.append(shared_levels)
        return resolved

    if len(urgency_levels) != len(transition_matrices):
        raise ValueError("Per-agent urgency_levels must match number of agents.")
    for levels_i, transition_matrix in zip(urgency_levels, transition_matrices):
        levels_i = np.asarray(levels_i, dtype=float)
        k = transition_matrix.shape[0]
        if levels_i.shape[0] != k:
            raise ValueError("Urgency levels length must match transition matrix size.")
        resolved.append(levels_i)
    return resolved


def get_markov_stationary_distributions(
    transition_matrices: list[np.ndarray],
) -> list[np.ndarray]:
    """Return one stationary distribution per agent transition matrix."""
    return [stationary_distribution(np.asarray(t, dtype=float)) for t in transition_matrices]


def get_markov_effective_capacity(resource_count: float, n_agents: int) -> float:
    """Effective capacity is capped by number of agents."""
    if resource_count < 0:
        raise ValueError("resource_count must be non-negative.")
    return float(min(resource_count, n_agents))


def _solve_lambda_star(
    stationary_distributions: list[np.ndarray],
    urgency_levels: list[np.ndarray],
    weights: np.ndarray,
    effective_capacity: float,
    epsilon: float = 1e-9,
) -> float:
    """Solve lambda* from sum_i P_i(u >= w_i/lambda*) = effective_capacity."""
    if effective_capacity <= epsilon:
        return 0.0

    def served_mass(lam: float) -> float:
        if lam <= 0.0:
            return 0.0
        total = 0.0
        for pi_i, u_i, w_i in zip(stationary_distributions, urgency_levels, weights):
            threshold = w_i / lam
            total += float(np.sum(pi_i[u_i >= threshold]))
        return total

    max_served = served_mass(1e12)
    if effective_capacity > max_served + epsilon:
        return float("inf")

    low, high = 0.0, 1.0
    while served_mass(high) < effective_capacity:
        high *= 2.0
        if high > 1e12:
            raise ValueError("Could not bracket lambda* for given resources/agents.")

    for _ in range(200):
        mid = 0.5 * (low + high)
        if served_mass(mid) < effective_capacity:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def get_markov_lambda_star(
    transition_matrices: list[np.ndarray],
    weights: np.ndarray | list[float],
    resource_count: float,
    urgency_levels: np.ndarray | list[np.ndarray] | None = None,
    epsilon: float = 1e-9,
) -> float:
    """Compute lambda* from resources and per-agent urgency Markov chains."""
    if not transition_matrices:
        raise ValueError("transition_matrices must be a non-empty list.")
    transition_matrices_np = [_validate_transition_matrix(np.asarray(t, dtype=float)) for t in transition_matrices]
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.shape[0] != len(transition_matrices_np):
        raise ValueError("weights must have one entry per agent.")
    if np.any(weights <= 0):
        raise ValueError("weights must be strictly positive.")

    urgency_levels_i = _resolve_urgency_levels(transition_matrices_np, urgency_levels)
    stationary = get_markov_stationary_distributions(transition_matrices_np)
    effective_capacity = get_markov_effective_capacity(resource_count, len(transition_matrices_np))
    return _solve_lambda_star(
        stationary_distributions=stationary,
        urgency_levels=urgency_levels_i,
        weights=weights,
        effective_capacity=effective_capacity,
        epsilon=epsilon,
    )


def get_markov_threshold(weight: float, lambda_star: float, epsilon: float = 1e-9) -> float:
    """Threshold ubar_i = w_i / lambda* with edge-case handling."""
    if weight <= 0:
        raise ValueError("weight must be strictly positive.")
    if lambda_star < 0:
        raise ValueError("lambda_star must be non-negative.")
    if lambda_star == 0.0:
        return float("inf")
    if np.isinf(lambda_star):
        return 0.0
    return float(weight / max(lambda_star, epsilon))


def get_markov_spike_index(
    stationary_distribution: np.ndarray,
    urgency_levels: np.ndarray,
    threshold: float,
    epsilon: float = 1e-9,
) -> float:
    """Return spike index phi_i for one agent."""
    pi_i = np.asarray(stationary_distribution, dtype=float)
    u_i = np.asarray(urgency_levels, dtype=float)
    if pi_i.shape[0] != u_i.shape[0]:
        raise ValueError("stationary_distribution and urgency_levels length mismatch.")
    if np.isinf(threshold):
        return 0.0
    if threshold <= epsilon:
        positive_mass = float(np.sum(pi_i[u_i >= threshold]))
        return float("inf") if positive_mass > epsilon else 0.0
    mask_spike = u_i >= threshold
    return float(np.sum(pi_i[mask_spike] * ((u_i[mask_spike] / threshold) - 1.0)))


def get_markov_surplus_efficiency(
    stationary_distribution: np.ndarray,
    urgency_levels: np.ndarray,
    threshold: float,
    delta_r: float = 1.0,
    epsilon: float = 1e-9,
) -> float:
    """Return surplus efficiency e_i for one agent."""
    pi_i = np.asarray(stationary_distribution, dtype=float)
    u_i = np.asarray(urgency_levels, dtype=float)
    if pi_i.shape[0] != u_i.shape[0]:
        raise ValueError("stationary_distribution and urgency_levels length mismatch.")
    mask = u_i >= threshold
    p_served = float(np.sum(pi_i[mask]))
    if p_served <= epsilon:
        return 0.0
    conditional_urgency = float(np.sum(pi_i[mask] * u_i[mask]) / p_served)
    return float(conditional_urgency * delta_r)
