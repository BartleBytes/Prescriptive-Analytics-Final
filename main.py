from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from portfolio_opt.data import (  
    DataConfig,
    add_stochastic_columns,
    load_raw_csv,
    prepare_portfolio_data,
)
from portfolio_opt.scenarios import SCENARIOS  
from portfolio_opt.optimization import (  
    PortfolioConstraints,
    solve_arv_profit_maximization,
    solve_dinkelbach_roi_maximization,
    solve_stochastic_expected_profit_maximization,
)


def write_summary(path: Path, arv_result, dinkelbach_result, stochastic_result=None) -> None:
    lines = [
        "# Portfolio Optimization Summary",
        "",
        "## ARV Profit Maximization",
        f"Status: {arv_result.status}",
        f"Total Investment: ${arv_result.total_investment:,.0f}",
        f"Total Profit: ${arv_result.total_profit:,.0f}",
        f"Portfolio ROI: {arv_result.roi:.2%}",
        f"Selected Properties: {len(arv_result.selected)}",
        "",
        "## Dinkelbach ROI Maximization",
        f"Status: {dinkelbach_result.status}",
        f"Total Investment: ${dinkelbach_result.total_investment:,.0f}",
        f"Total Profit: ${dinkelbach_result.total_profit:,.0f}",
        f"Portfolio ROI: {dinkelbach_result.roi:.2%}",
        f"Selected Properties: {len(dinkelbach_result.selected)}",
    ]
    if stochastic_result is not None:
        lines.extend(
            [
                "",
                "## Stochastic Expected Profit Maximization",
                f"Status: {stochastic_result.status}",
                f"Total Investment: ${stochastic_result.total_investment:,.0f}",
                f"Expected Total Profit: ${stochastic_result.total_profit:,.0f}",
                f"Expected Portfolio ROI: {stochastic_result.roi:.2%}",
                f"Selected Properties: {len(stochastic_result.selected)}",
            ]
        )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run portfolio optimization models.")
    parser.add_argument(
        "--data",
        default="data/raw/quartile_update_costs.csv",
        help="Path to raw CSV data.",
    )
    parser.add_argument(
        "--rehab-cost-column",
        default="Rehab Costs",
        help="Column to use for rehab costs (e.g., 'Rehab Costs' or 'Ave Update Total Cost').",
    )
    parser.add_argument("--budget", type=float, default=5_000_000)
    parser.add_argument("--min-roi", type=float, default=0.15)
    parser.add_argument("--sfh-min", type=float, default=0.10)
    parser.add_argument("--sfh-max", type=float, default=0.30)
    parser.add_argument(
        "--zip-max",
        type=float,
        default=None,
        help="Optional max share of total investment allowed in any one zip code.",
    )
    parser.add_argument(
        "--zip-col",
        default="Zip",
        help="Column name for zip codes in the dataset.",
    )
    parser.add_argument(
        "--min-investment",
        type=float,
        default=None,
        help="Optional minimum total investment to avoid trivial solutions.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Run stochastic expected-profit optimization.",
    )
    parser.add_argument(
        "--pessimistic-profit-floor",
        type=float,
        default=None,
        help="Optional floor on total pessimistic profit (e.g., 0).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write reports and processed outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_csv(data_path)
    config = DataConfig(rehab_cost_col=args.rehab_cost_column)
    df = prepare_portfolio_data(df_raw, config)
    df = add_stochastic_columns(df, SCENARIOS, config)

    constraints = PortfolioConstraints(
        budget=args.budget,
        min_roi_per_property=args.min_roi,
        sfh_share_min=args.sfh_min,
        sfh_share_max=args.sfh_max,
        min_total_investment=args.min_investment,
        zip_max_share=args.zip_max,
        zip_col=args.zip_col,
    )

    arv_result = solve_arv_profit_maximization(df, constraints)
    dinkelbach_result = solve_dinkelbach_roi_maximization(df, constraints)

    processed_dir = output_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    arv_result.selected.to_csv(processed_dir / "selected_arv.csv", index=False)
    dinkelbach_result.selected.to_csv(
        processed_dir / "selected_dinkelbach.csv", index=False
    )

    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stochastic_result = None
    if args.stochastic:
        stochastic_result = solve_stochastic_expected_profit_maximization(
            df,
            constraints,
            pessimistic_profit_floor=args.pessimistic_profit_floor,
        )
        stochastic_result.selected.to_csv(
            processed_dir / "selected_stochastic.csv", index=False
        )

    write_summary(reports_dir / "summary.md", arv_result, dinkelbach_result, stochastic_result)

    print("Wrote:")
    print(f"- {processed_dir / 'selected_arv.csv'}")
    print(f"- {processed_dir / 'selected_dinkelbach.csv'}")
    if stochastic_result is not None:
        print(f"- {processed_dir / 'selected_stochastic.csv'}")
    print(f"- {reports_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
