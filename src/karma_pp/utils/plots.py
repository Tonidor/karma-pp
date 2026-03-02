import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_access_fairness_vs_efficiency(
    access_fairness: list[float],
    efficiency: list[float],
    labels: list[str],
    save_path: str = "data/plots/access_fairness_vs_efficiency.png",
    sweep_values: list[float] | None = None,
) -> None:
    """Plot efficiency vs access fairness scatter for multiple scenarios.

    When sweep_values is provided, points are connected by a line in order,
    all use circle markers, and colors vary from blue (low) to yellow (high).
    """
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=1.5)
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

    fig, ax = plt.subplots(figsize=(10, 6))

    if sweep_values is not None and len(sweep_values) == len(efficiency):
        # Parameter sweep: line + circles, blue-to-yellow colormap
        sweep_arr = np.asarray(sweep_values)
        eff_arr = np.asarray(efficiency)
        fair_arr = np.asarray(access_fairness)
        order = np.argsort(sweep_arr)
        eff_ordered = eff_arr[order]
        fair_ordered = fair_arr[order]
        sweep_ordered = sweep_arr[order]
        labels_ordered = [labels[i] for i in order]

        cmap = plt.get_cmap("YlGnBu_r")  # blue (low) -> yellow (high)
        norm = plt.Normalize(vmin=sweep_ordered.min(), vmax=sweep_ordered.max())
        colors = [cmap(norm(v)) for v in sweep_ordered]

        ax.plot(
            eff_ordered,
            fair_ordered,
            color="gray",
            linestyle="-",
            linewidth=1.5,
            alpha=0.7,
            zorder=0,
        )
        for i, (eff, fair, label) in enumerate(zip(eff_ordered, fair_ordered, labels_ordered)):
            ax.scatter(
                eff,
                fair,
                c=[colors[i]],
                marker="o",
                s=150,
                label=label,
                edgecolors="black",
                linewidths=1.5,
                alpha=0.9,
                zorder=1,
            )
    else:
        # Non-sweep: distinct colors and markers per scenario
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        markers = ["o", "s", "^"]
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


def plot_full_info_policy_and_distribution(
    pi: np.ndarray,
    d: np.ndarray,
    urgency_levels: list[int],
    urgency_labels: list[str] | None = None,
    save_path: str = "data/plots/full_info_policy_distribution.png",
) -> None:
    """Plot learned policy (Bid vs Karma) and Karma distribution per urgency level.

    Args:
        pi: Policy array shape (nu, nk, nk), π[u, k, b] = P(bid b | urgency u, karma k).
        d: Distribution array shape (nu, nk), d[u, k] = P(karma k | urgency u).
        urgency_levels: List of urgency values, length nu.
        urgency_labels: Optional column titles (default: f"u = {u}" for each u).
        save_path: Path to save the figure.
    """
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=1.2)
    sns.set_context("paper", font_scale=1.3, rc={"lines.linewidth": 2.0})

    nu, nk, _ = pi.shape
    if d.shape != (nu, nk):
        raise ValueError(f"d shape {d.shape} must match (nu={nu}, nk={nk}).")
    if len(urgency_levels) != nu:
        raise ValueError(f"urgency_levels length {len(urgency_levels)} must equal nu={nu}.")

    labels = urgency_labels or [f"u = {u}" for u in urgency_levels]
    if len(labels) != nu:
        raise ValueError(f"urgency_labels length must equal nu={nu}.")

    fig, axes = plt.subplots(2, nu, figsize=(4 * nu, 8), sharex="col")
    if nu == 1:
        axes = np.array([axes])
    axes_policy = axes[0]
    axes_dist = axes[1]

    karma_edges = np.arange(nk + 1) - 0.5
    karma_centers = np.arange(nk)
    max_karma = nk - 1

    for u_idx in range(nu):
        ax_p = axes_policy[u_idx] if nu > 1 else axes_policy
        ax_d = axes_dist[u_idx] if nu > 1 else axes_dist

        # --- Policy heatmap (Bid vs Karma) ---
        # Build matrix: rows = bid b, cols = karma k. Valid only for b <= k.
        policy_mat = np.full((nk, nk), np.nan, dtype=float)
        for k in range(nk):
            for b in range(k + 1):
                policy_mat[b, k] = pi[u_idx, k, b]

        cmap = plt.get_cmap("Reds")
        cmap.set_bad(color="#c0c0c0", alpha=0.8)
        im = ax_p.pcolormesh(
            karma_edges,
            karma_edges,
            policy_mat,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            shading="flat",
        )
        ax_p.set_xlim(-0.5, max_karma + 0.5)
        ax_p.set_ylim(-0.5, max_karma + 0.5)
        ax_p.set_aspect("equal")
        ax_p.set_xlabel("Karma", fontsize=12)
        ax_p.set_ylabel("Bid", fontsize=12)
        ax_p.set_title(labels[u_idx], fontsize=13, fontweight="bold")
        ax_p.grid(True, alpha=0.3)
        ax_p.set_xticks(karma_centers[:: max(1, nk // 10)])
        ax_p.set_yticks(karma_centers[:: max(1, nk // 10)])

        # --- Distribution bar chart ---
        dist_vals = d[u_idx, :]
        ax_d.bar(
            karma_centers,
            dist_vals,
            width=0.8,
            color="#1f77b4",
            alpha=0.8,
            edgecolor="#2f2f2f",
            linewidth=1.0,
        )
        ax_d.set_xlim(-0.5, max_karma + 0.5)
        ax_d.set_ylabel("Distribution", fontsize=12)
        ax_d.set_xlabel("Karma", fontsize=12)
        ax_d.grid(True, axis="y", alpha=0.3)
        ax_d.set_xticks(karma_centers[:: max(1, nk // 10)])
        tick_positions = karma_centers[:: max(1, nk // 10)]
        tick_labels = [str(int(x)) for x in tick_positions]
        if len(tick_positions) > 0 and tick_positions[-1] == max_karma and max_karma >= 10:
            tick_labels[-1] = f"≥{max_karma}"
        ax_d.set_xticklabels(tick_labels)

    # Shared y-axis labels for rows
    fig.text(0.02, 0.75, "Bid", fontsize=13, va="center", rotation="vertical")
    fig.text(0.02, 0.25, "Distribution", fontsize=13, va="center", rotation="vertical")
    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_efficiency_fairness_mechanisms(
    full_info_efficiency: list[float],
    full_info_fairness: list[float],
    full_info_gammas: list[float],
    benevolent_dictator_eff: float,
    benevolent_dictator_fair: float,
    turn_taking_eff: float,
    turn_taking_fair: float,
    coin_toss_eff: float,
    coin_toss_fair: float,
    save_path: str = "data/plots/efficiency_fairness_mechanisms.png",
) -> None:
    """Plot efficiency vs fairness: full info (gamma sweep) + benevolent dictator + turn-taking + coin toss."""
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=1.5)
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

    fig, ax = plt.subplots(figsize=(10, 6))

    # Full info: line + circles with gamma colormap
    sweep_arr = np.asarray(full_info_gammas)
    eff_arr = np.asarray(full_info_efficiency)
    fair_arr = np.asarray(full_info_fairness)
    order = np.argsort(sweep_arr)
    eff_ordered = eff_arr[order]
    fair_ordered = fair_arr[order]
    sweep_ordered = sweep_arr[order]

    cmap = plt.get_cmap("YlGnBu_r")
    norm = plt.Normalize(vmin=sweep_ordered.min(), vmax=sweep_ordered.max())
    colors = [cmap(norm(v)) for v in sweep_ordered]

    ax.plot(
        eff_ordered,
        fair_ordered,
        color="gray",
        linestyle="-",
        linewidth=1.5,
        alpha=0.7,
        zorder=0,
    )
    for i, (eff, fair) in enumerate(zip(eff_ordered, fair_ordered)):
        ax.scatter(
            eff,
            fair,
            c=[colors[i]],
            marker="o",
            s=120,
            label="Full info (α sweep)" if i == 0 else None,
            edgecolors="black",
            linewidths=1.5,
            alpha=0.9,
            zorder=1,
        )

    # Benevolent dictator
    ax.scatter(
        benevolent_dictator_eff,
        benevolent_dictator_fair,
        c="#d00000",
        marker="*",
        s=280,
        label="Benevolent Dictator",
        edgecolors="black",
        linewidths=1.5,
        alpha=0.95,
        zorder=4,
    )

    # Turn-taking
    ax.scatter(
        turn_taking_eff,
        turn_taking_fair,
        c="#2ca02c",
        marker="^",
        s=180,
        label="Turn-taking",
        edgecolors="black",
        linewidths=1.5,
        alpha=0.9,
        zorder=3,
    )

    # Coin toss
    ax.scatter(
        coin_toss_eff,
        coin_toss_fair,
        c="#1f77b4",
        marker="s",
        s=150,
        label="Coin toss",
        edgecolors="black",
        linewidths=1.5,
        alpha=0.9,
        zorder=2,
    )

    ax.set_xlabel("Efficiency", fontsize=14)
    ax.set_ylabel("Access Fairness", fontsize=14)
    ax.set_title("Efficiency vs. Access Fairness (1000 agents, 3 runs)", fontsize=16, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
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