import numpy as np


def get_access_fairness(access_by_step):
    """
    Access fairness is the standard deviation of the fraction of resource access by each agent.

    Args:
        access_by_step: shape (num_steps, num_agents)
    """
    access_by_step = np.asarray(access_by_step)
    if access_by_step.shape[0] == 0:
        return 0.0
    frac_success_by_agent = np.sum(access_by_step, axis=0) / access_by_step.shape[0]
    return -float(np.std(frac_success_by_agent))


def get_efficiency(rewards_by_step):
    """
    Efficiency is the mean of mean reward per agent.

    Args:
        rewards_by_step: shape (num_steps, num_agents)
    """
    rewards_by_step = np.asarray(rewards_by_step)
    if rewards_by_step.size == 0:
        return 0.0
    return float(np.mean(np.mean(rewards_by_step, axis=0)))


def get_reward_fairness(rewards_by_step):
    """
    Ex-post reward fairness is the negative standard deviation of mean rewards per agent.

    Formula: rf_T = -std_{i in N}(bar_zeta_T^i)
    where bar_zeta_T^i = (1/T) * sum_{s=0}^{T-1} zeta^i[t_s]

    Args:
        rewards_by_step: shape (num_steps, num_agents) where rewards_by_step[t, i] is the reward for agent i at step t

    Returns:
        reward_fairness: Negative standard deviation of mean rewards across agents
    """
    rewards_by_step = np.asarray(rewards_by_step)
    if rewards_by_step.size == 0:
        return 0.0
    mean_rewards_per_agent = np.mean(rewards_by_step, axis=0)
    return -float(np.std(mean_rewards_per_agent))


def nash_welfare(allocation, urgencies, social_weights, epsilon=1e-6):
    """
    Calculates Nash Welfare with epsilon safety to prevent log(0).
    """
    allocation = np.asarray(allocation)
    _n_steps, n_agents = urgencies.shape
    utilities = np.zeros(n_agents)

    if allocation.size == 0:
        utilities_safe = utilities + epsilon
        return float(np.sum(social_weights * np.log(utilities_safe))), utilities

    for t, agent_idx in enumerate(allocation):
        utilities[agent_idx] += urgencies[t, agent_idx]

    utilities_safe = utilities + epsilon
    log_utility = np.sum(social_weights * np.log(utilities_safe))
    return log_utility, utilities
