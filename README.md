# Prescriptive Portfolio Optimization

Python + Jupyter project for the BANA Properties Inc. residential portfolio optimization final.

## Quick start

1. Create / update the environment:

```bash
uv sync
```

2. Launch JupyterLab:

```bash
uv run jupyter lab
```

3. Open `notebooks/ARV_2.ipynb` or start with `notebooks/00_project_setup.ipynb`.

## Project layout

- `data/raw/` raw inputs (copied from source)
- `data/processed/` derived datasets
- `notebooks/` analysis notebooks
- `reports/` generated summaries and tables
- `src/portfolio_opt/` reusable Python modules

## Running the model from CLI

```bash
uv run python main.py \
  --rehab-cost-column "Rehab Costs" \
  --budget 5000000 \
  --min-roi 0.15 \
  --sfh-min 0.10 \
  --sfh-max 0.30 \
  --zip-max 0.40
```

Outputs:
- `reports/summary.md`
- `data/processed/selected_arv.csv`
- `data/processed/selected_dinkelbach.csv`

## Stochastic model

```bash
uv run python main.py \
  --rehab-cost-column "Rehab Costs" \
  --stochastic \
  --pessimistic-profit-floor 0
```

## Sensitivity analysis

```bash
uv run python scripts/sensitivity.py \
  --rehab-cost-column "Rehab Costs" \
  --multipliers "0.9,1.0,1.1,1.2"
```

## Visualizations

```bash
uv run python scripts/visualize.py
```

