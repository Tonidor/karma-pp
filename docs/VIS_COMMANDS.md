# Visualization Commands

All visualization commands use data from the database only. No simulation execution is performed.

## Working Commands

### 1. Balances (Karma Distribution)

Visualize agent balance (karma) distributions over time as violin plots.

```bash
kpp vis balances 1
kpp vis balances --group-id 2
```

### 2. Efficiency vs Fairness

Plot (efficiency, access fairness) scatter. Supports single/multiple experiments and groups.

```bash
# Single experiment
kpp vis efficiency-fairness 1

# Multiple experiments
kpp vis efficiency-fairness 1 2 3

# Single group (one point per experiment)
kpp vis efficiency-fairness -g 1

# Multiple groups (average per group)
kpp vis efficiency-fairness -g 1 -g 2

# With labels for styled comparison (Random, Optimal, Dictator, Q γ=...)
kpp vis efficiency-fairness -g 1 --labels Random --labels Optimal --labels "Benevolent Dictator" --labels "Q γ=0.9"
```

### 3. Reward Over Time

Plot smoothed per-agent reward over time.

```bash
kpp vis reward-over-time 1
kpp vis reward-over-time 1 --window 50
```

### 4. Nash Welfare

Plot Nash welfare bar chart for experiments.

```bash
kpp vis nash-welfare 1
kpp vis nash-welfare --group-id 2
```

### 5. Full-Info Policy Distribution

Plot learned policy (Bid vs Karma) and karma distribution from FullInfoLearningAgent.

```bash
kpp vis full-info-policy-distribution 1
kpp vis full-info-policy-distribution 1 --step 100  # use specific step
```

### 6. Markov Urgency Violin

Plot Markov stationary urgency distribution from experiment config.

```bash
kpp vis markov-urgency-violin 1
kpp vis markov-urgency-violin 1 --delta-r 1.0 --lambda-star 0.5
```

### 7. Metrics Table

Render a table of Efficiency and Fairness for multiple experiments.

```bash
kpp vis metrics-table 1 2 3 --labels Random --labels "Q-learning" --labels Optimal
```

## Common Options

Most commands support:

- `--format`, `-f`: png, pdf, svg, jpg (default: png)
- `--dpi`: Image DPI (default: 300)
- `--figsize`: Width and height (default: 12 8)
- `--style`: Seaborn style (default: whitegrid)
- `--palette`: Color palette name
- `--save` / `--no-save`: Save to file (default: save)
- `--log-level`, `-l`: DEBUG, INFO, WARNING, ERROR, CRITICAL

## Output Locations

- Single experiment: `data/plots/experiment_<id>/`
- Group: `data/plots/group_<label>/`
- Multi-experiment: `data/plots/experiments/`
- Metrics table: `data/plots/`
