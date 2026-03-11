from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from portfolio_opt.data import (  # noqa: E402
    DataConfig,
    add_stochastic_columns,
    load_raw_csv,
    prepare_portfolio_data,
)
from portfolio_opt.optimization import (  # noqa: E402
    PortfolioConstraints,
    solve_arv_profit_maximization,
    solve_dinkelbach_roi_maximization,
    solve_stochastic_expected_profit_maximization,
)
from portfolio_opt.scenarios import SCENARIOS  # noqa: E402


sns.set_theme(style="whitegrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate model comparison visualizations.")
    parser.add_argument(
        "--data",
        default="data/raw/quartile_update_costs.csv",
        help="Path to raw CSV data.",
    )
    parser.add_argument(
        "--rehab-cost-column",
        default="Rehab Costs",
        help="Column to use for rehab costs.",
    )
    parser.add_argument("--budget", type=float, default=5_000_000)
    parser.add_argument("--min-roi", type=float, default=0.15)
    parser.add_argument("--sfh-min", type=float, default=0.10)
    parser.add_argument("--sfh-max", type=float, default=0.30)
    parser.add_argument("--zip-max", type=float, default=None)
    parser.add_argument("--zip-col", default="Zip")
    parser.add_argument(
        "--pessimistic-profit-floor",
        type=float,
        default=None,
        help="Optional pessimistic profit floor for stochastic model.",
    )
    parser.add_argument(
        "--sensitivity-file",
        default="reports/sensitivity_rehab.csv",
        help="Optional sensitivity results CSV to plot.",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/figures",
        help="Output directory for plots.",
    )
    return parser.parse_args()


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, dpi=200, bbox_inches="tight")


def main() -> None:
    args = parse_args()

    df_raw = load_raw_csv(Path(args.data))
    config = DataConfig(rehab_cost_col=args.rehab_cost_column)
    df = prepare_portfolio_data(df_raw, config)
    df = add_stochastic_columns(df, SCENARIOS, config)

    constraints = PortfolioConstraints(
        budget=args.budget,
        min_roi_per_property=args.min_roi,
        sfh_share_min=args.sfh_min,
        sfh_share_max=args.sfh_max,
        zip_max_share=args.zip_max,
        zip_col=args.zip_col,
    )

    arv = solve_arv_profit_maximization(df, constraints)
    dinkelbach = solve_dinkelbach_roi_maximization(df, constraints)
    stochastic = solve_stochastic_expected_profit_maximization(
        df,
        constraints,
        pessimistic_profit_floor=args.pessimistic_profit_floor,
    )

    summary = pd.DataFrame(
        [
            {
                "model": "ARV Profit Max",
                "total_profit": arv.total_profit,
                "roi": arv.roi,
                "selected_properties": len(arv.selected),
            },
            {
                "model": "Dinkelbach ROI Max",
                "total_profit": dinkelbach.total_profit,
                "roi": dinkelbach.roi,
                "selected_properties": len(dinkelbach.selected),
            },
            {
                "model": "Stochastic Expected Profit",
                "total_profit": stochastic.total_profit,
                "roi": stochastic.roi,
                "selected_properties": len(stochastic.selected),
            },
        ]
    )

    out_dir = Path(args.out_dir)

    # Plot 1: Total profit by model
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=summary, x="model", y="total_profit", ax=ax, color="#4C72B0")
    ax.set_title("Total Profit by Model")
    ax.set_ylabel("Total Profit ($)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, out_dir, "profit_by_model.png")
    plt.close(fig)

    # Plot 2: ROI by model
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=summary, x="model", y="roi", ax=ax, color="#55A868")
    ax.set_title("Portfolio ROI by Model")
    ax.set_ylabel("ROI")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, out_dir, "roi_by_model.png")
    plt.close(fig)

    # Plot 3: Selected properties count
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=summary,
        x="model",
        y="selected_properties",
        ax=ax,
        color="#C44E52",
    )
    ax.set_title("Number of Selected Properties")
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, out_dir, "count_by_model.png")
    plt.close(fig)

    # Plot 4: Stochastic scenario profit totals
    scenario_totals = []
    for name in SCENARIOS:
        col = f"profit_{name}"
        scenario_totals.append(
            {
                "scenario": name,
                "total_profit": stochastic.selected[col].sum(),
            }
        )
    scenario_df = pd.DataFrame(scenario_totals)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=scenario_df, x="scenario", y="total_profit", ax=ax)
    ax.set_title("Stochastic Portfolio Profit by Scenario")
    ax.set_ylabel("Total Profit ($)")
    ax.set_xlabel("")
    _save(fig, out_dir, "stochastic_profit_by_scenario.png")
    plt.close(fig)

    # Plot 5: Sensitivity curve (if file exists)
    sens_path = Path(args.sensitivity_file)
    if sens_path.exists():
        sens = pd.read_csv(sens_path)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=sens, x="rehab_multiplier", y="roi", marker="o", ax=ax)
        ax.set_title("ROI Sensitivity to Rehab Costs")
        ax.set_ylabel("ROI")
        ax.set_xlabel("Rehab Cost Multiplier")
        _save(fig, out_dir, "sensitivity_roi_rehab.png")
        plt.close(fig)

    summary.to_csv(out_dir / "model_summary.csv", index=False)
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
