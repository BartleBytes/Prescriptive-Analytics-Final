from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class DataConfig:
    purchase_price_col: str = "Purchase Price"
    resale_price_col: str = "Resale Price"
    rehab_cost_col: str = "Rehab Costs"
    closing_cost_col: str = "Closing Costs"
    property_type_col: str = "Property Type"
    sfh_flag_col: str = "Condo/SFH"




NUMERIC_COLUMNS_DEFAULT = (
    "Property price (USD)",
    "Price cut amount (USD)",
    "Living area",
    "Price per living area unit (USD)",
    "Est Sales price (sq ft)",
    "Purchase Price",
    "Resale Price",
    "Update Costs (sq ft) by Q",
    "Rehab Costs",
    "Closing Costs",
    "Ave Update Total Cost",
    "Total Investment",
    "Ave Profit",
    "Ave Return",
)


def _to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _to_numeric(df, NUMERIC_COLUMNS_DEFAULT)
    return df


def prepare_portfolio_data(
    df: pd.DataFrame,
    config: DataConfig | None = None,
) -> pd.DataFrame:
    """Normalize numeric fields and compute investment, profit, and ROI.

    Returns a copy of the dataframe with added columns:
    - total_investment
    - profit
    - roi
    - is_sfh
    """
    config = config or DataConfig()
    df = df.copy()

    df = _to_numeric(
        df,
        [
            config.purchase_price_col,
            config.resale_price_col,
            config.rehab_cost_col,
            config.closing_cost_col,
            config.sfh_flag_col,
        ],
    )

    df["total_investment"] = (
        df[config.purchase_price_col]
        + df[config.rehab_cost_col]
        + df[config.closing_cost_col]
    )
    df["profit"] = df[config.resale_price_col] - df["total_investment"]
    df["roi"] = df["profit"] / df["total_investment"]
    df["is_sfh"] = df[config.sfh_flag_col].fillna(0).astype(int)

    return df


def add_stochastic_columns(
    df: pd.DataFrame,
    scenarios: dict[str, dict[str, float]] | None = None,
    config: DataConfig | None = None,
) -> pd.DataFrame:
    config = config or DataConfig()
    if scenarios is None:
        raise ValueError("scenarios must be provided")
    df = df.copy()

    df["expected_profit"] = 0.0

    for name, spec in scenarios.items():
        resale_col = f"resale_price_{name}"
        rehab_col = f"rehab_cost_{name}"
        total_investment_col = f"total_investment_{name}"
        profit_col = f"profit_{name}"
        roi_col = f"roi_{name}"

        df[resale_col] = (
            df[config.resale_price_col].astype(float) * spec["resale_multiplier"]
        )
        df[rehab_col] = (
            df[config.rehab_cost_col].astype(float) * spec["rehab_multiplier"]
        )

        df[total_investment_col] = (
            df[config.purchase_price_col]
            + df[rehab_col]
            + df[config.closing_cost_col]
        )
        df[profit_col] = df[resale_col] - df[total_investment_col]
        df[roi_col] = df[profit_col] / df[total_investment_col]

        df["expected_profit"] += spec["prob"] * df[profit_col]

    df["expected_roi"] = df["expected_profit"] / df["total_investment"]

    return df
