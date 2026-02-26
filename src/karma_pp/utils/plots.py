import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_access_fairness_vs_efficiency(
    access_fairness: list[float],
    efficiency: list[float],
    labels: list[str],
    save_path: str = "data/plots/access_fairness_vs_efficiency.png",
) -> None:
    """Plot efficiency vs access fairness scatter for multiple scenarios."""
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=1.5)
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (eff, fair, label) in enumerate(zip(efficiency, access_fairness, labels)):
        ax.scatter(
            eff,
            fair,
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            s=150,
            label=label,
            edgecolors="black",
            linewidths=1.5,
            alpha=0.8,
        )
    ax.set_xlabel("Efficiency", fontsize=14)
    ax.set_ylabel("Access Fairness", fontsize=14)
    ax.set_title("Efficiency vs. Access Fairness", fontsize=16, fontweight="bold")
    ax.legend(loc="best", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_efficiency_fairness_comparison(
    access_fairness: list[float],
    efficiency: list[float],
    labels: list[str],
    save_path: str = "data/plots/efficiency_fairness_comparison.png",
) -> None:
    """Efficiency vs access fairness for Random, Optimal, Dictator, and Q variants."""
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=1.5)
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

    fig, ax = plt.subplots(figsize=(10, 6))

    random_indices = [i for i, label in enumerate(labels) if label.strip().lower() == "random"]
    optimal_indices = [i for i, label in enumerate(labels) if label.strip().lower() == "optimal"]
    dictator_indices = [
        i for i, label in enumerate(labels) if label.strip().lower() == "benevolent dictator"
    ]
    q_indices = [i for i, label in enumerate(labels) if label.strip().lower().startswith("q ")]

    # Q-learners: same marker family, diverging colors and connecting line
    if q_indices:
        cmap = plt.get_cmap("RdYlBu_r")
        q_colors = [cmap(i / max(1, len(q_indices) - 1)) for i in range(len(q_indices))]
        eff_q = [efficiency[i] for i in q_indices]
        fair_q = [access_fairness[i] for i in q_indices]
        ax.plot(
            eff_q,
            fair_q,
            color="gray",
            linestyle="-",
            linewidth=1.5,
            alpha=0.7,
            zorder=0,
        )
        for rank, idx in enumerate(q_indices):
            ax.scatter(
                efficiency[idx],
                access_fairness[idx],
                c=[q_colors[rank]],
                marker="o",
                s=150,
                label=labels[idx],
                edgecolors="black",
                linewidths=1.5,
                alpha=0.9,
                zorder=1,
            )

    # Random: dark circle
    for idx in random_indices:
        ax.scatter(
            efficiency[idx],
            access_fairness[idx],
            c="#2f2f2f",
            marker="s",
            s=150,
            label=labels[idx],
            edgecolors="black",
            linewidths=1.5,
            alpha=0.9,
            zorder=2,
        )

    # Optimal: distinct yellow star
    for idx in optimal_indices:
        ax.scatter(
            efficiency[idx],
            access_fairness[idx],
            c="#ffd60a",
            marker="*",
            s=260,
            label=labels[idx],
            edgecolors="black",
            linewidths=1.5,
            alpha=0.95,
            zorder=3,
        )

    # Benevolent Dictator: red star
    for idx in dictator_indices:
        ax.scatter(
            efficiency[idx],
            access_fairness[idx],
            c="#d00000",
            marker="*",
            s=280,
            label=labels[idx],
            edgecolors="black",
            linewidths=1.5,
            alpha=0.95,
            zorder=4,
        )

    # Fallback style for any additional labels.
    handled = set(random_indices + optimal_indices + dictator_indices + q_indices)
    for idx, label in enumerate(labels):
        if idx in handled:
            continue
        ax.scatter(
            efficiency[idx],
            access_fairness[idx],
            c="#1f77b4",
            marker="^",
            s=150,
            label=label,
            edgecolors="black",
            linewidths=1.5,
            alpha=0.9,
            zorder=1,
        )

    ax.set_xlabel("Efficiency", fontsize=14)
    ax.set_ylabel("Access Fairness", fontsize=14)
    ax.set_title("Efficiency vs. Access Fairness", fontsize=16, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_nash_welfare(
    nash_welfare_values: list[float],
    labels: list[str],
    save_path: str = "data/plots/nash_welfare.png",
) -> None:
    """Plot Nash welfare bar chart for multiple scenarios."""
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=1.5)
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (value, label) in enumerate(zip(nash_welfare_values, labels)):
        ax.bar(label, value, color=colors[i % len(colors)], alpha=0.8)
    ax.set_xlabel("Mechanism", fontsize=14)
    ax.set_ylabel("Nash Welfare", fontsize=14)
    ax.set_title("Nash Welfare vs. Mechanism", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_smoothed_reward_over_time(
    rewards_by_step: np.ndarray,
    window: int = 50,
    title: str = "Smoothed reward over time",
    save_path: str = "data/plots/reward_over_time.png",
) -> None:
    """
    Plot per-agent and mean reward over time with a rolling mean to show learning.

    Args:
        rewards_by_step: shape (n_steps, n_agents), reward per agent per timestep.
        window: rolling window size for smoothing.
        title: plot title.
        save_path: path to save the figure.
    """
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=1.5)
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

    n_steps, n_agents = rewards_by_step.shape
    window_half = max(1, window // 2)
    smoothed = np.zeros_like(rewards_by_step, dtype=float)
    for i in range(n_steps):
        start = max(0, i - window_half)
        end = min(n_steps, i + window_half + 1)
        smoothed[i] = np.mean(rewards_by_step[start:end], axis=0)

    timesteps = np.arange(1, n_steps + 1, dtype=float)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(10, 6))

    for j in range(n_agents):
        ax.plot(
            timesteps,
            smoothed[:, j],
            color=colors[j % len(colors)],
            label=f"Agent {j}",
            alpha=0.9,
        )
    mean_reward = np.mean(smoothed, axis=1)
    ax.plot(
        timesteps,
        mean_reward,
        color="black",
        linestyle="--",
        label="Mean",
        linewidth=2,
    )

    ax.set_xlabel("Timestep", fontsize=14)
    ax.set_ylabel("Smoothed reward", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(loc="best", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_markov_stationary_violin(
    urgency_levels: list[np.ndarray],
    stationary_distributions: list[np.ndarray],
    spike_indices: list[float],
    surplus_efficiency: list[float],
    thresholds: list[float] | None = None,
    save_path: str = "data/plots/markov_urgency_violin.png",
    sample_size: int = 2000,
) -> None:
    """Plot stationary urgency distributions per agent with spike/surplus annotations."""
    if not (len(urgency_levels) == len(stationary_distributions) == len(spike_indices) == len(surplus_efficiency)):
        raise ValueError("All input lists must have the same length.")
    if thresholds is not None and len(thresholds) != len(urgency_levels):
        raise ValueError("thresholds length must match number of agents.")

    sns.set_theme(style="whitegrid")
    sns.set(font_scale=1.2)
    sns.set_context("paper", font_scale=1.3, rc={"lines.linewidth": 2.0})

    rng = np.random.default_rng(0)
    datasets: list[np.ndarray] = []
    for u_i, pi_i in zip(urgency_levels, stationary_distributions):
        u_i = np.asarray(u_i, dtype=float)
        pi_i = np.asarray(pi_i, dtype=float)
        datasets.append(rng.choice(u_i, size=sample_size, p=pi_i))

    n_agents = len(datasets)
    positions = np.arange(1, n_agents + 1, dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, n_agents * 1.6), 6))
    violin = ax.violinplot(datasets, positions=positions, showmeans=True, showextrema=True, widths=0.8)
    for body in violin["bodies"]:
        body.set_facecolor("#8ecae6")
        body.set_edgecolor("black")
        body.set_alpha(0.75)

    for idx, x in enumerate(positions):
        ymax = float(np.max(urgency_levels[idx]))
        annotation = (
            f"$\\phi_{{{idx}}}$={spike_indices[idx]:.3f}\n"
            f"$e_{{{idx}}}$={surplus_efficiency[idx]:.3f}"
        )
        ax.text(
            x + 0.28,
            ymax,
            annotation,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#666"},
        )

        if thresholds is not None and np.isfinite(thresholds[idx]):
            threshold = float(thresholds[idx])
            ax.hlines(
                y=threshold,
                xmin=x - 0.32,
                xmax=x + 0.32,
                colors="#d62728",
                linestyles="--",
                linewidth=2.0,
                zorder=3,
            )
            ax.text(
                x,
                threshold,
                f"ū={threshold:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#d62728",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([f"Agent {i}" for i in range(n_agents)], rotation=0)
    ax.set_xlabel("Agent", fontsize=13)
    ax.set_ylabel("Urgency", fontsize=13)
    ax.set_title("Stationary urgency distribution by agent", fontsize=15, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics_table(
    row_labels: list[str],
    column_labels: list[str],
    cell_text: list[list[str]],
    title: str = "Mechanism Metrics Summary",
    save_path: str = "data/plots/metrics_table.png",
) -> None:
    """Render a metrics table as a PNG image."""
    if not row_labels or not column_labels or not cell_text:
        raise ValueError("row_labels, column_labels, and cell_text must be non-empty.")
    if len(cell_text) != len(row_labels):
        raise ValueError("cell_text row count must match row_labels.")
    if any(len(row) != len(column_labels) for row in cell_text):
        raise ValueError("Each cell_text row length must match column_labels.")

    sns.set_theme(style="whitegrid")
    fig_w = max(8.0, 1.8 * len(column_labels) + 2.0)
    fig_h = max(4.0, 0.8 * len(row_labels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=column_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.35)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#dce6f1")
            cell.set_text_props(weight="bold")
        if col == -1:
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(weight="bold")
        cell.set_edgecolor("#666666")
        cell.set_linewidth(0.8)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()