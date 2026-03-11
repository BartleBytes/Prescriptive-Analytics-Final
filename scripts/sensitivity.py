from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from portfolio_opt.data import DataConfig, load_raw_csv, prepare_portfolio_data  # noqa: E402
from portfolio_opt.optimization import (  # noqa: E402
    PortfolioConstraints,
    solve_dinkelbach_roi_maximization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sensitivity analysis on rehab costs.")
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
        "--multipliers",
        default="0.9,1.0,1.1,1.2",
        help="Comma-separated rehab cost multipliers.",
    )
    parser.add_argument(
        "--output",
        default="reports/sensitivity_rehab.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df_raw = load_raw_csv(Path(args.data))
    config = DataConfig(rehab_cost_col=args.rehab_cost_column)
    base_df = prepare_portfolio_data(df_raw, config)

    constraints = PortfolioConstraints(
        budget=args.budget,
        min_roi_per_property=args.min_roi,
        sfh_share_min=args.sfh_min,
        sfh_share_max=args.sfh_max,
        zip_max_share=args.zip_max,
        zip_col=args.zip_col,
    )

    multipliers = [float(x.strip()) for x in args.multipliers.split(",") if x.strip()]
    results: list[dict[str, float]] = []

    for mult in multipliers:
        df = base_df.copy()
        df[config.rehab_cost_col] = df[config.rehab_cost_col] * mult
        df = prepare_portfolio_data(df, config)

        result = solve_dinkelbach_roi_maximization(df, constraints)
        results.append(
            {
                "rehab_multiplier": mult,
                "total_profit": result.total_profit,
                "roi": result.roi,
                "selected_properties": len(result.selected),
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
