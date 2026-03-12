# Run Guide

## Setup

```bash
uv sync
```

## Core model runs

```bash
uv run python main.py --stochastic --pessimistic-profit-floor 0
```

## Sensitivity analysis

```bash
uv run python scripts/sensitivity.py
```

## Visualizations

```bash
uv run python scripts/visualize.py
```

## Outputs

- `reports/summary.md`
- `reports/sensitivity_rehab.csv`
- `reports/figures/`
