# karma-pp

## Installation

```bash
pip install -e .
```

## Commands

The CLI is available via `kpp`.

### `run` — Run a simulation

Runs a single simulation with the given scenario.

```bash
kpp run [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--scenario` | (required) | Path to scenario YAML |
| `--steps` | 5 | Number of simulation steps |
| `--seed` | 42 | Random seed |
| `--log-level` | INFO | Log level (DEBUG, INFO, WARNING, ERROR) |

### `vis` — Create visualizations

Generates plots and metrics from simulation outputs.

```bash
kpp vis <PLOT_NAME> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--scenario` | (required) | Scenario path(s), can be repeated |
| `--label` | (from filename) | Custom labels per scenario |
| `--steps` | 1000 | Simulation steps |
| `--seed` | 42 | Random seed |
| `--window` | 50 | Rolling window for `reward_over_time` smoothing |
| `--out` | `data/plots/<plot>.png` | Output path |
| `--log-level` | INFO | Log level |

**Plot types:**

| Plot | Description |
|------|-------------|
| `access_fairness_vs_efficiency` | Scatter of efficiency vs access fairness for multiple scenarios |
| `nash_welfare` | Bar chart of Nash welfare per scenario |
| `reward_over_time` | Smoothed per-agent and mean reward over timesteps |
| `efficiency_fairness_comparison` | Compares Random, Optimal, Benevolent Dictator, and Q-learner variants. Requires 4 `--scenario` values in order: random, optimal, dictator, q-learner. |
| `markov_urgency_metrics` | YAML output of Markov metrics (lambda*, spike index, surplus efficiency) per scenario |
| `markov_urgency_violin` | Violin plot of stationary urgency distributions with spike/surplus annotations |
| `metrics_table` | Table of efficiency, fairness, spike index, surplus efficiency, and lambda* for Random vs Q-learning vs Optimal. Requires 3 `--scenario` values in order: random, q-learner, optimal. |

**Additional options for specific plots:**

| Option | Default | Used by |
|--------|---------|---------|
| `--n-runs` | 3 | efficiency_fairness_comparison, metrics_table |
| `--seed-base` | 42 | efficiency_fairness_comparison, metrics_table |
| `--delta-r` | 1.0 | markov_urgency_metrics |
| `--lambda-star` | (computed) | markov_urgency_metrics, markov_urgency_violin |

## Examples

```bash
# Run a short simulation
kpp run --scenario data/scenarios/2x2_benevolent_dictator.yaml --steps 100

# Plot efficiency vs fairness for one scenario
kpp vis access_fairness_vs_efficiency --scenario data/scenarios/2x2_benevolent_dictator.yaml --steps 500

# Compare multiple scenarios with custom labels
kpp vis nash_welfare --scenario data/scenarios/2x2_randoms.yaml --scenario data/scenarios/2x2_benevolent_dictator.yaml --label "Random" --label "Dictator"

# Export Markov metrics to YAML
kpp vis markov_urgency_metrics --scenario data/scenarios/2x2_benevolent_dictator.yaml --out data/plots/metrics.yaml

# Generate comparison plot (Random vs Optimal vs Dictator vs Q-learners)
# Order: random, optimal, dictator, q-learner. Saves to data/plots/ by default.
kpp vis efficiency_fairness_comparison \
  --scenario data/scenarios/2x2_randoms.yaml \
  --scenario data/scenarios/2x2_optimal_bidders.yaml \
  --scenario data/scenarios/2x2_benevolent_dictator.yaml \
  --scenario data/scenarios/2x2_q_learners.yaml \
  --steps 200 --n-runs 2

# Generate metrics table (Random vs Q-learning vs Optimal)
# Order: random, q-learner, optimal. Saves to data/plots/ by default.
kpp vis metrics_table \
  --scenario data/scenarios/2x2_randoms.yaml \
  --scenario data/scenarios/2x2_q_learners.yaml \
  --scenario data/scenarios/2x2_optimal_bidders.yaml \
  --steps 200 --n-runs 2
```
