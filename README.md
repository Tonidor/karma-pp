# karma-pp

A multi-agent coordination simulation framework implementing a karma mechanism for resource allocation, evaluated against a benevolent dictator baseline and random/optimal bidding strategies.

## Installation

```bash
pip install -e .
```

For development (includes pytest):

```bash
pip install -e ".[dev]"
```

Requires **Python 3.12+**.

## Running tests

```bash
pytest
```

## Commands

The CLI is available via `kpp`. Scenario files can be passed as any path — file references within them are resolved relative to the scenario file itself, so the commands work from any directory.

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
kpp run --scenario data/scenarios/200x2_benevolent_dictator.yaml --steps 100

# Run with debug logging and a different seed
kpp run --scenario data/scenarios/200x2_coin_toss.yaml --steps 500 --seed 7 --log-level DEBUG

# Plot efficiency vs fairness for one scenario
kpp vis access_fairness_vs_efficiency --scenario data/scenarios/200x2_benevolent_dictator.yaml --steps 500

# Compare multiple scenarios with custom labels
kpp vis nash_welfare \
  --scenario data/scenarios/200x2_benevolent_dictator.yaml \
  --scenario data/scenarios/200x2_turn_taking.yaml \
  --label "Dictator" --label "Turn-taking"

# Export Markov metrics to YAML
kpp vis markov_urgency_metrics --scenario data/scenarios/200x2_full_info_gamma_0.50.yaml --out data/plots/metrics.yaml

# Generate comparison plot across scenario variants
# Scenarios must be passed in a consistent order
kpp vis efficiency_fairness_comparison \
  --scenario data/scenarios/200x2_full_info_gamma_0.00.yaml \
  --scenario data/scenarios/200x2_full_info_gamma_0.30.yaml \
  --scenario data/scenarios/200x2_full_info_gamma_0.60.yaml \
  --scenario data/scenarios/200x2_full_info_gamma_0.90.yaml \
  --steps 200 --n-runs 2

# Generate metrics table across scenario variants
# Scenarios must be passed in a consistent order
kpp vis metrics_table \
  --scenario data/scenarios/200x2_full_info_gamma_0.10.yaml \
  --scenario data/scenarios/200x2_full_info_gamma_0.50.yaml \
  --scenario data/scenarios/200x2_full_info_gamma_0.90.yaml \
  --steps 200 --n-runs 2
```

## Project structure

```
data/
  scenarios/    # Unified experiment entry points (YAML with world, mechanism, population)
src/karma_pp/
  core/         # Abstract interfaces (Agent, Mechanism, World, Population, Simulation)
  impl/
    agents/     # TruthfulResourceAgent, RandomResourceAgent, QLearningResourceAgent, OptimalBiddingResourceAgent
    mechanisms/ # KarmaMechanism, BenevolentDictatorMechanism
    worlds/     # ResourceWorld
  utils/        # Metrics (efficiency, fairness, Markov measures)
  cli/          # Click commands (run, vis)
tests/          # pytest test suite
```

## Adding a new scenario

1. Create a unified scenario YAML in `data/scenarios/` with top-level `world`, `mechanism`, and `population` keys.
2. Run with `kpp run --scenario data/scenarios/my_scenario.yaml`.
