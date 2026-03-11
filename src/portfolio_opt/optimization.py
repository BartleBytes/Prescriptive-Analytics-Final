from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import pulp


@dataclass(frozen=True)
class PortfolioConstraints:
    budget: float = 5_000_000
    min_roi_per_property: float = 0.15
    sfh_share_min: float = 0.10
    sfh_share_max: float = 0.30
    min_total_investment: float | None = None
    zip_max_share: float | None = None
    zip_col: str = "Zip"


@dataclass(frozen=True)
class OptimizationResult:
    status: str
    total_investment: float
    total_profit: float
    roi: float
    selected: pd.DataFrame


def _filter_by_min_roi(
    df: pd.DataFrame,
    min_roi: float,
    roi_col: str = "roi",
) -> pd.DataFrame:
    if min_roi is None:
        return df
    return df[df[roi_col] >= min_roi].copy()


def _build_problem(
    df: pd.DataFrame,
    constraints: PortfolioConstraints,
    objective_profit_col: str = "profit",
    investment_col: str = "total_investment",
    objective_coeff_profit: float = 1.0,
    objective_coeff_investment: float = 0.0,
    pessimistic_profit_floor: float | None = None,
) -> tuple[pulp.LpProblem, dict[int, pulp.LpVariable]]:
    problem = pulp.LpProblem("portfolio_optimization", pulp.LpMaximize)
    buy = {i: pulp.LpVariable(f"buy_{i}", cat="Binary") for i in df.index}

    total_investment = pulp.lpSum(
        df.loc[i, investment_col] * buy[i] for i in df.index
    )
    total_profit = pulp.lpSum(
        df.loc[i, objective_profit_col] * buy[i] for i in df.index
    )

    problem += objective_coeff_profit * total_profit - objective_coeff_investment * total_investment

    problem += total_investment <= constraints.budget

    if constraints.min_total_investment is not None:
        problem += total_investment >= constraints.min_total_investment

    if constraints.sfh_share_min is not None:
        sfh_investment = pulp.lpSum(
            df.loc[i, investment_col] * df.loc[i, "is_sfh"] * buy[i]
            for i in df.index
        )
        problem += sfh_investment >= constraints.sfh_share_min * total_investment

    if constraints.sfh_share_max is not None:
        sfh_investment = pulp.lpSum(
            df.loc[i, investment_col] * df.loc[i, "is_sfh"] * buy[i]
            for i in df.index
        )
        problem += sfh_investment <= constraints.sfh_share_max * total_investment

    if constraints.zip_max_share is not None and constraints.zip_col in df.columns:
        for zip_code, group in df.groupby(constraints.zip_col):
            zip_investment = pulp.lpSum(
                group.loc[i, investment_col] * buy[i] for i in group.index
            )
            problem += zip_investment <= constraints.zip_max_share * total_investment

    if pessimistic_profit_floor is not None and "profit_pessimistic" in df.columns:
        total_pess_profit = pulp.lpSum(
            df.loc[i, "profit_pessimistic"] * buy[i] for i in df.index
        )
        problem += total_pess_profit >= pessimistic_profit_floor

    return problem, buy


def _solve_problem(problem: pulp.LpProblem) -> str:
    solver = pulp.PULP_CBC_CMD(msg=False)
    problem.solve(solver)
    return pulp.LpStatus[problem.status]


def _extract_solution(df: pd.DataFrame, buy: dict[int, pulp.LpVariable], status: str) -> OptimizationResult:
    selected_idx = [i for i, var in buy.items() if var.value() == 1]
    selected = df.loc[selected_idx].copy()

    total_investment = selected["total_investment"].sum()
    total_profit = selected["profit"].sum()
    roi = (total_profit / total_investment) if total_investment else 0.0

    return OptimizationResult(
        status=status,
        total_investment=float(total_investment),
        total_profit=float(total_profit),
        roi=float(roi),
        selected=selected,
    )


def solve_arv_profit_maximization(
    df: pd.DataFrame,
    constraints: PortfolioConstraints,
) -> OptimizationResult:
    filtered = _filter_by_min_roi(df, constraints.min_roi_per_property, roi_col="roi")
    problem, buy = _build_problem(
        filtered,
        constraints,
        objective_coeff_profit=1.0,
        objective_coeff_investment=0.0,
    )
    status = _solve_problem(problem)
    return _extract_solution(filtered, buy, status)


def solve_dinkelbach_roi_maximization(
    df: pd.DataFrame,
    constraints: PortfolioConstraints,
    tol: float = 1e-5,
    max_iter: int = 50,
) -> OptimizationResult:
    filtered = _filter_by_min_roi(df, constraints.min_roi_per_property, roi_col="roi")
    lam = 0.0
    best_result: OptimizationResult | None = None

    for _ in range(max_iter):
        problem, buy = _build_problem(
            filtered,
            constraints,
            objective_coeff_profit=1.0,
            objective_coeff_investment=lam,
        )
        status = _solve_problem(problem)
        result = _extract_solution(filtered, buy, status)

        if result.total_investment <= 0:
            best_result = result
            break

        new_lam = result.total_profit / result.total_investment
        improvement = abs(new_lam - lam)
        lam = new_lam
        best_result = result

        if improvement < tol:
            break

    return best_result if best_result is not None else _extract_solution(filtered, {}, "NoSolution")


def _extract_solution_expected_profit(
    df: pd.DataFrame,
    buy: dict[int, pulp.LpVariable],
    status: str,
) -> OptimizationResult:
    selected_idx = [i for i, var in buy.items() if var.value() == 1]
    selected = df.loc[selected_idx].copy()

    total_investment = selected["total_investment"].sum()
    total_profit = selected["expected_profit"].sum()
    roi = (total_profit / total_investment) if total_investment else 0.0

    selected["expected_roi"] = selected["expected_profit"] / selected["total_investment"]

    return OptimizationResult(
        status=status,
        total_investment=float(total_investment),
        total_profit=float(total_profit),
        roi=float(roi),
        selected=selected,
    )


def solve_stochastic_expected_profit_maximization(
    df: pd.DataFrame,
    constraints: PortfolioConstraints,
    pessimistic_profit_floor: float | None = None,
) -> OptimizationResult:
    filtered = _filter_by_min_roi(
        df,
        constraints.min_roi_per_property,
        roi_col="expected_roi",
    )
    problem, buy = _build_problem(
        filtered,
        constraints,
        objective_profit_col="expected_profit",
        investment_col="total_investment",
        objective_coeff_profit=1.0,
        objective_coeff_investment=0.0,
        pessimistic_profit_floor=pessimistic_profit_floor,
    )
    status = _solve_problem(problem)
    return _extract_solution_expected_profit(filtered, buy, status)
