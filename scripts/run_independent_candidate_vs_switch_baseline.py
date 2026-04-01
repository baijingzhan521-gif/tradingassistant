from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.service import BacktestAssumptions, BacktestService
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.services.strategy_service import StrategyService
from scripts.run_simple_candidate_v2_regime_switch_fixed_calendar import (
    DEFAULT_SWITCH_DATE,
    run_regime_switch_case,
)
from scripts.run_switch_axis_band_management_one_rule_gate import attach_axis_band_risk_group
from scripts.run_switch_portfolio_blend_one_rule_gate import build_switch_managed_trades
from scripts.run_trend_pullback_standalone_one_rule_gate import resolve_end_from_history


SYMBOL = "BTC/USDT:USDT"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "backtests" / "independent_candidate_vs_switch_baseline"
SWITCH_SCENARIO_KIND = "switch_simple_candidate_v2_then_challenger_managed"

DEFAULT_SIMPLE_PROFILE = "swing_trend_simple_candidate_v2"
DEFAULT_CHALLENGER_STRATEGY_PROFILE = "swing_trend_long_regime_short_no_reversal_no_aux_v1"
DEFAULT_CHALLENGER_OVERLAY_PROFILE = "be_if_no_extension_within_3bars_after_tp1_and_veto_hold_at_trend_96_98"

WINDOWS = ("full_2020", "two_year")
COST_SCENARIOS = {
    "base": {},
    "stress_x2": {"taker_fee_bps": 10.0, "slippage_bps": 4.0},
    "stress_x3": {"taker_fee_bps": 15.0, "slippage_bps": 6.0},
}

PF_TOLERANCE = 0.05
DD_TOLERANCE_R = 1.5
LONG_GUARD_TOLERANCE_R = 2.0
TOP3_SHARE_MAX = 65.0
BEST_YEAR_SHARE_MAX = 80.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Head-to-head: independent candidate vs fixed switch baseline.")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--simple-profile", default=DEFAULT_SIMPLE_PROFILE)
    parser.add_argument("--challenger-strategy-profile", default=DEFAULT_CHALLENGER_STRATEGY_PROFILE)
    parser.add_argument("--challenger-overlay-profile", default=DEFAULT_CHALLENGER_OVERLAY_PROFILE)
    parser.add_argument("--switch-date", default=DEFAULT_SWITCH_DATE)
    parser.add_argument("--candidate-profile", required=True)
    parser.add_argument("--primary-start", default="2020-01-01")
    parser.add_argument("--secondary-start", default=DEFAULT_SWITCH_DATE)
    parser.add_argument("--end", default=None, help="Defaults to latest available data at runtime.")
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--step-days", type=int, default=90)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_candidate_service(*, assumption_overrides: dict[str, Any] | None = None) -> BacktestService:
    assumptions = {
        "exit_profile": "long_scaled1_3_short_fixed1_5",
        "take_profit_mode": "scaled",
        "long_exit": {"take_profit_mode": "scaled", "scaled_tp1_r": 1.0, "scaled_tp2_r": 3.0},
        "short_exit": {"take_profit_mode": "fixed_r", "fixed_take_profit_r": 1.5},
        "swing_detection_mode": "confirmed",
        "cache_dir": "artifacts/backtests/cache",
    }
    if assumption_overrides:
        assumptions.update(assumption_overrides)
    return BacktestService(
        ohlcv_service=OhlcvService(get_exchange_client_factory()),
        strategy_service=StrategyService(),
        assumptions=BacktestAssumptions(**assumptions),
    )


def normalize_trade_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    for column in ("signal_time", "entry_time", "exit_time"):
        result[column] = pd.to_datetime(result[column], utc=True)
    result["year"] = result["signal_time"].dt.year.astype(int)
    result["month"] = result["signal_time"].dt.strftime("%Y-%m")
    result["quarter"] = result["signal_time"].dt.tz_localize(None).dt.to_period("Q").astype(str)
    return result.sort_values(["cost_scenario", "window", "profile_kind", "entry_time"]).reset_index(drop=True)


def _window_days(start: datetime, end: datetime) -> float:
    return max((end - start).total_seconds() / 86400.0, 1e-9)


def _cagr_pct_from_weighted(frame: pd.DataFrame, *, window_start: datetime, window_end: datetime) -> float:
    if frame.empty:
        return 0.0
    compounded = float((1.0 + frame["weighted_pnl_pct"] / 100.0).prod())
    years = _window_days(window_start, window_end) / 365.25
    if years <= 0.0:
        return 0.0
    if compounded <= 0.0:
        return -100.0
    return round(((compounded ** (1.0 / years)) - 1.0) * 100.0, 4)


def _geometric_return_pct(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    return round(float((1.0 + frame["weighted_pnl_pct"] / 100.0).prod() - 1.0) * 100.0, 4)


def _profit_factor(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    wins = float(frame.loc[frame["weighted_pnl_r"] > 0.0, "weighted_pnl_r"].sum())
    losses = float(-frame.loc[frame["weighted_pnl_r"] < 0.0, "weighted_pnl_r"].sum())
    if losses <= 0.0:
        return 999.0 if wins > 0.0 else 0.0
    return round(wins / losses, 4)


def _max_dd_r(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    ordered = frame.sort_values("entry_time")
    cumulative = ordered["weighted_pnl_r"].cumsum()
    drawdown = cumulative - cumulative.cummax()
    return round(abs(float(drawdown.min())), 4)


def summarize_weighted_frame(
    *,
    frame: pd.DataFrame,
    cost_scenario: str,
    window: str,
    window_start: datetime,
    window_end: datetime,
    profile_kind: str,
    strategy_profile: str,
) -> dict[str, Any]:
    trades = int(len(frame))
    return {
        "cost_scenario": cost_scenario,
        "window": window,
        "window_start": window_start.date().isoformat(),
        "window_end": window_end.date().isoformat(),
        "profile_kind": profile_kind,
        "strategy_profile": strategy_profile,
        "trades": trades,
        "avg_size": round(float(frame["size_multiplier"].mean()), 4) if trades else 0.0,
        "win_rate_pct": round(float((frame["weighted_pnl_r"] > 0.0).mean() * 100.0), 2) if trades else 0.0,
        "profit_factor": _profit_factor(frame),
        "expectancy_r": round(float(frame["weighted_pnl_r"].mean()), 4) if trades else 0.0,
        "cum_r": round(float(frame["weighted_pnl_r"].sum()), 4) if trades else 0.0,
        "max_dd_r": _max_dd_r(frame),
        "geometric_return_pct": _geometric_return_pct(frame),
        "additive_return_pct": round(float(frame["weighted_pnl_pct"].sum()), 4) if trades else 0.0,
        "cagr_pct": _cagr_pct_from_weighted(frame, window_start=window_start, window_end=window_end),
    }


def build_side_summary(weighted_trades: pd.DataFrame) -> list[dict[str, Any]]:
    if weighted_trades.empty:
        return []
    rows: list[dict[str, Any]] = []
    grouped = weighted_trades.groupby(["cost_scenario", "window", "profile_kind", "strategy_profile", "side"], sort=True, observed=True)
    for (cost_scenario, window, profile_kind, strategy_profile, side), group in grouped:
        rows.append(
            {
                "cost_scenario": cost_scenario,
                "window": window,
                "profile_kind": profile_kind,
                "strategy_profile": strategy_profile,
                "side": side,
                "trades": int(len(group)),
                "avg_size": round(float(group["size_multiplier"].mean()), 4),
                "profit_factor": _profit_factor(group),
                "cum_r": round(float(group["weighted_pnl_r"].sum()), 4),
                "geometric_return_pct": _geometric_return_pct(group),
            }
        )
    return rows


def build_yearly_geometric_returns(weighted_trades: pd.DataFrame) -> list[dict[str, Any]]:
    if weighted_trades.empty:
        return []
    rows: list[dict[str, Any]] = []
    grouped = weighted_trades.groupby(
        ["cost_scenario", "window", "profile_kind", "strategy_profile", "year"], sort=True, observed=True
    )
    for (cost_scenario, window, profile_kind, strategy_profile, year), group in grouped:
        rows.append(
            {
                "cost_scenario": cost_scenario,
                "window": window,
                "profile_kind": profile_kind,
                "strategy_profile": strategy_profile,
                "year": int(year),
                "trades": int(len(group)),
                "geometric_return_pct": _geometric_return_pct(group),
                "additive_return_pct": round(float(group["weighted_pnl_pct"].sum()), 4),
            }
        )
    return rows


def build_cost_sensitivity(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(summary_rows)
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(["window", "profile_kind", "strategy_profile"], sort=True, observed=True)
    for (window, profile_kind, strategy_profile), group in grouped:
        base = group[group["cost_scenario"] == "base"]
        if base.empty:
            continue
        base_row = base.iloc[0]
        for stress_name in ("stress_x2", "stress_x3"):
            stress = group[group["cost_scenario"] == stress_name]
            if stress.empty:
                continue
            stress_row = stress.iloc[0]
            rows.append(
                {
                    "window": window,
                    "profile_kind": profile_kind,
                    "strategy_profile": strategy_profile,
                    "stress_scenario": stress_name,
                    "base_geometric_return_pct": round(float(base_row["geometric_return_pct"]), 4),
                    "stress_geometric_return_pct": round(float(stress_row["geometric_return_pct"]), 4),
                    "delta_geometric_return_pct": round(
                        float(stress_row["geometric_return_pct"] - base_row["geometric_return_pct"]), 4
                    ),
                    "base_cagr_pct": round(float(base_row["cagr_pct"]), 4),
                    "stress_cagr_pct": round(float(stress_row["cagr_pct"]), 4),
                    "delta_cagr_pct": round(float(stress_row["cagr_pct"] - base_row["cagr_pct"]), 4),
                    "base_profit_factor": round(float(base_row["profit_factor"]), 4),
                    "stress_profit_factor": round(float(stress_row["profit_factor"]), 4),
                    "delta_profit_factor": round(float(stress_row["profit_factor"] - base_row["profit_factor"]), 4),
                    "base_max_dd_r": round(float(base_row["max_dd_r"]), 4),
                    "stress_max_dd_r": round(float(stress_row["max_dd_r"]), 4),
                    "delta_max_dd_r": round(float(stress_row["max_dd_r"] - base_row["max_dd_r"]), 4),
                }
            )
    return rows


def build_trade_concentration(weighted_trades: pd.DataFrame, yearly_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if weighted_trades.empty:
        return []
    yearly = pd.DataFrame(yearly_rows)
    rows: list[dict[str, Any]] = []
    grouped = weighted_trades.groupby(["cost_scenario", "window", "profile_kind", "strategy_profile"], sort=True, observed=True)
    for (cost_scenario, window, profile_kind, strategy_profile), group in grouped:
        positive = group.loc[group["weighted_pnl_r"] > 0.0, "weighted_pnl_r"].sort_values(ascending=False)
        total_positive = float(positive.sum())
        top3_positive = float(positive.head(3).sum())
        top3_share = (top3_positive / total_positive * 100.0) if total_positive > 0.0 else 100.0

        yearly_sub = yearly[
            (yearly["cost_scenario"] == cost_scenario)
            & (yearly["window"] == window)
            & (yearly["profile_kind"] == profile_kind)
            & (yearly["strategy_profile"] == strategy_profile)
        ]
        pos_year_returns = yearly_sub.loc[yearly_sub["geometric_return_pct"] > 0.0, "geometric_return_pct"]
        if not pos_year_returns.empty:
            best_year_geo = float(pos_year_returns.max())
            best_year_share = float((best_year_geo / pos_year_returns.sum()) * 100.0)
        else:
            best_year_geo = 0.0
            best_year_share = 100.0
        rows.append(
            {
                "cost_scenario": cost_scenario,
                "window": window,
                "profile_kind": profile_kind,
                "strategy_profile": strategy_profile,
                "trades": int(len(group)),
                "top3_trades_pnl_share_pct": round(top3_share, 4),
                "best_year_geometric_pct": round(best_year_geo, 4),
                "best_year_geometric_pct_share": round(best_year_share, 4),
            }
        )
    return rows


def generate_folds(
    *,
    start: datetime,
    end: datetime,
    train_days: int,
    test_days: int,
    step_days: int,
) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    train_start = start
    train_end = train_start + timedelta(days=train_days)
    fold = 1
    while train_end + timedelta(days=test_days) <= end:
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)
        folds.append(
            {
                "fold": fold,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        fold += 1
        train_start = train_start + timedelta(days=step_days)
        train_end = train_start + timedelta(days=train_days)
    return folds


def build_fold_panel(*, weighted_trades: pd.DataFrame, folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if weighted_trades.empty:
        return rows
    base_only = weighted_trades[
        (weighted_trades["cost_scenario"] == "base") & (weighted_trades["window"] == "full_2020")
    ].copy()
    for fold in folds:
        oos = base_only[
            (base_only["entry_time"] >= fold["test_start"]) & (base_only["entry_time"] < fold["test_end"])
        ].copy()
        baseline_oos = oos[oos["profile_kind"] == "switch_baseline"].copy()
        baseline_summary = summarize_weighted_frame(
            frame=baseline_oos,
            cost_scenario="base",
            window="oos_fold",
            window_start=fold["test_start"],
            window_end=fold["test_end"],
            profile_kind="switch_baseline",
            strategy_profile="switch_baseline",
        )
        candidate_oos = oos[oos["profile_kind"] == "candidate"].copy()
        candidate_summary = summarize_weighted_frame(
            frame=candidate_oos,
            cost_scenario="base",
            window="oos_fold",
            window_start=fold["test_start"],
            window_end=fold["test_end"],
            profile_kind="candidate",
            strategy_profile=str(candidate_oos["strategy_profile"].iloc[0]) if not candidate_oos.empty else "candidate",
        )
        rows.append(
            {
                "fold": int(fold["fold"]),
                "train_start": fold["train_start"].date().isoformat(),
                "train_end": fold["train_end"].date().isoformat(),
                "test_start": fold["test_start"].date().isoformat(),
                "test_end": fold["test_end"].date().isoformat(),
                "candidate_trades": int(candidate_summary["trades"]),
                "baseline_trades": int(baseline_summary["trades"]),
                "candidate_geometric_return_pct": round(float(candidate_summary["geometric_return_pct"]), 4),
                "baseline_geometric_return_pct": round(float(baseline_summary["geometric_return_pct"]), 4),
                "delta_vs_baseline_geometric_return_pct": round(
                    float(candidate_summary["geometric_return_pct"] - baseline_summary["geometric_return_pct"]), 4
                ),
                "candidate_cagr_pct": round(float(candidate_summary["cagr_pct"]), 4),
                "baseline_cagr_pct": round(float(baseline_summary["cagr_pct"]), 4),
                "candidate_profit_factor": round(float(candidate_summary["profit_factor"]), 4),
                "baseline_profit_factor": round(float(baseline_summary["profit_factor"]), 4),
                "candidate_cum_r": round(float(candidate_summary["cum_r"]), 4),
                "baseline_cum_r": round(float(baseline_summary["cum_r"]), 4),
            }
        )
    return rows


def build_oos_summary(fold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(fold_rows)
    if frame.empty:
        return {
            "candidate_geometric_return_pct": 0.0,
            "baseline_geometric_return_pct": 0.0,
            "delta_vs_baseline_geometric_return_pct": 0.0,
            "oos_available": False,
        }
    candidate_factor = float((1.0 + frame["candidate_geometric_return_pct"] / 100.0).prod())
    baseline_factor = float((1.0 + frame["baseline_geometric_return_pct"] / 100.0).prod())
    candidate_geo = round((candidate_factor - 1.0) * 100.0, 4)
    baseline_geo = round((baseline_factor - 1.0) * 100.0, 4)
    return {
        "candidate_geometric_return_pct": candidate_geo,
        "baseline_geometric_return_pct": baseline_geo,
        "delta_vs_baseline_geometric_return_pct": round(candidate_geo - baseline_geo, 4),
        "oos_available": True,
    }


def _lookup(
    rows: list[dict[str, Any]],
    *,
    cost_scenario: str | None = None,
    window: str | None = None,
    profile_kind: str | None = None,
    side: str | None = None,
) -> dict[str, Any]:
    for row in rows:
        if cost_scenario is not None and row.get("cost_scenario") != cost_scenario:
            continue
        if window is not None and row.get("window") != window:
            continue
        if profile_kind is not None and row.get("profile_kind") != profile_kind:
            continue
        if side is not None and row.get("side") != side:
            continue
        return row
    raise KeyError(
        f"missing row cost_scenario={cost_scenario}, window={window}, profile_kind={profile_kind}, side={side}"
    )


def build_comparison_decision(
    *,
    summary_rows: list[dict[str, Any]],
    side_rows: list[dict[str, Any]],
    concentration_rows: list[dict[str, Any]],
    oos_summary: dict[str, Any],
    candidate_profile: str,
) -> dict[str, Any]:
    baseline_base = _lookup(summary_rows, cost_scenario="base", window="full_2020", profile_kind="switch_baseline")
    candidate_base = _lookup(summary_rows, cost_scenario="base", window="full_2020", profile_kind="candidate")
    baseline_x2 = _lookup(summary_rows, cost_scenario="stress_x2", window="full_2020", profile_kind="switch_baseline")
    candidate_x2 = _lookup(summary_rows, cost_scenario="stress_x2", window="full_2020", profile_kind="candidate")
    baseline_x3 = _lookup(summary_rows, cost_scenario="stress_x3", window="full_2020", profile_kind="switch_baseline")
    candidate_x3 = _lookup(summary_rows, cost_scenario="stress_x3", window="full_2020", profile_kind="candidate")
    baseline_long_two_year = _lookup(
        side_rows, cost_scenario="base", window="two_year", profile_kind="switch_baseline", side="LONG"
    )
    candidate_long_two_year = _lookup(side_rows, cost_scenario="base", window="two_year", profile_kind="candidate", side="LONG")
    candidate_concentration = _lookup(
        concentration_rows, cost_scenario="base", window="full_2020", profile_kind="candidate"
    )

    pass_base_geo = float(candidate_base["geometric_return_pct"]) > float(baseline_base["geometric_return_pct"])
    pass_base_cagr = float(candidate_base["cagr_pct"]) > float(baseline_base["cagr_pct"])
    pass_base_pf = float(candidate_base["profit_factor"]) >= float(baseline_base["profit_factor"]) - PF_TOLERANCE
    pass_base_max_dd = float(candidate_base["max_dd_r"]) <= float(baseline_base["max_dd_r"]) + DD_TOLERANCE_R
    pass_stress_x2_geo = float(candidate_x2["geometric_return_pct"]) > 0.0
    pass_stress_x2_pf = float(candidate_x2["profit_factor"]) >= 1.0
    pass_stress_x3_geo = float(candidate_x3["geometric_return_pct"]) > 0.0
    pass_stress_x3_pf = float(candidate_x3["profit_factor"]) >= 1.0
    pass_long_guard = float(candidate_long_two_year["cum_r"]) >= float(baseline_long_two_year["cum_r"]) - LONG_GUARD_TOLERANCE_R
    pass_oos_geo = bool(oos_summary.get("oos_available", False)) and float(oos_summary["delta_vs_baseline_geometric_return_pct"]) >= 0.0
    pass_top3 = float(candidate_concentration["top3_trades_pnl_share_pct"]) <= TOP3_SHARE_MAX
    pass_best_year = float(candidate_concentration["best_year_geometric_pct_share"]) <= BEST_YEAR_SHARE_MAX

    all_pass = (
        pass_base_geo
        and pass_base_cagr
        and pass_base_pf
        and pass_base_max_dd
        and pass_stress_x2_geo
        and pass_stress_x2_pf
        and pass_stress_x3_geo
        and pass_stress_x3_pf
        and pass_long_guard
        and pass_oos_geo
        and pass_top3
        and pass_best_year
    )
    status = "candidate_beats_switch" if all_pass else "switch_baseline_retained"
    return {
        "candidate_profile": candidate_profile,
        "base_full_2020_candidate_geometric_return_pct": round(float(candidate_base["geometric_return_pct"]), 4),
        "base_full_2020_baseline_geometric_return_pct": round(float(baseline_base["geometric_return_pct"]), 4),
        "base_full_2020_candidate_cagr_pct": round(float(candidate_base["cagr_pct"]), 4),
        "base_full_2020_baseline_cagr_pct": round(float(baseline_base["cagr_pct"]), 4),
        "base_full_2020_candidate_profit_factor": round(float(candidate_base["profit_factor"]), 4),
        "base_full_2020_baseline_profit_factor": round(float(baseline_base["profit_factor"]), 4),
        "base_full_2020_candidate_max_dd_r": round(float(candidate_base["max_dd_r"]), 4),
        "base_full_2020_baseline_max_dd_r": round(float(baseline_base["max_dd_r"]), 4),
        "stress_x2_full_2020_candidate_geometric_return_pct": round(float(candidate_x2["geometric_return_pct"]), 4),
        "stress_x3_full_2020_candidate_geometric_return_pct": round(float(candidate_x3["geometric_return_pct"]), 4),
        "two_year_long_delta_r": round(float(candidate_long_two_year["cum_r"] - baseline_long_two_year["cum_r"]), 4),
        "candidate_top3_trades_pnl_share_pct": round(float(candidate_concentration["top3_trades_pnl_share_pct"]), 4),
        "candidate_best_year_geometric_pct_share": round(float(candidate_concentration["best_year_geometric_pct_share"]), 4),
        "oos_candidate_geometric_return_pct": round(float(oos_summary["candidate_geometric_return_pct"]), 4),
        "oos_baseline_geometric_return_pct": round(float(oos_summary["baseline_geometric_return_pct"]), 4),
        "oos_delta_vs_baseline_geometric_return_pct": round(float(oos_summary["delta_vs_baseline_geometric_return_pct"]), 4),
        "pass_base_geo": pass_base_geo,
        "pass_base_cagr": pass_base_cagr,
        "pass_base_pf": pass_base_pf,
        "pass_base_max_dd": pass_base_max_dd,
        "pass_stress_x2_geo": pass_stress_x2_geo,
        "pass_stress_x2_pf": pass_stress_x2_pf,
        "pass_stress_x3_geo": pass_stress_x3_geo,
        "pass_stress_x3_pf": pass_stress_x3_pf,
        "pass_long_guard": pass_long_guard,
        "pass_top3_concentration": pass_top3,
        "pass_best_year_concentration": pass_best_year,
        "pass_oos_geo_non_negative": pass_oos_geo,
        "status": status,
        "next_route": (
            "satisfied"
            if status == "candidate_beats_switch"
            else "unsatisfied_move_to_next_round_or_stop"
        ),
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_None_"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        values: list[str] = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(
    *,
    summary_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    decision_row: dict[str, Any],
    candidate_profile: str,
) -> str:
    base_full = [
        row for row in summary_rows if row["cost_scenario"] == "base" and row["window"] == "full_2020"
    ]
    base_full = sorted(base_full, key=lambda row: row["profile_kind"])
    conclusion = (
        "candidate beats fixed switch baseline under all required constraints."
        if decision_row["status"] == "candidate_beats_switch"
        else "candidate does not beat fixed switch baseline; keep switch as primary baseline."
    )
    return "\n".join(
        [
            "# Independent Candidate vs Switch Baseline",
            "",
            "- baseline: fixed switch (`simple_candidate_v2 -> challenger_managed`) + fixed management layer",
            f"- candidate: `{candidate_profile}`",
            "- gate: fixed M1-style compare, no parameter search",
            "",
            "## Base Full_2020 Summary",
            "",
            markdown_table(
                base_full,
                [
                    ("profile_kind", "Profile Kind"),
                    ("strategy_profile", "Profile"),
                    ("trades", "Trades"),
                    ("profit_factor", "PF"),
                    ("cum_r", "Cum R"),
                    ("max_dd_r", "MaxDD R"),
                    ("geometric_return_pct", "Geom %"),
                    ("cagr_pct", "CAGR %"),
                ],
            ),
            "",
            "## Fold Panel",
            "",
            markdown_table(
                fold_rows,
                [
                    ("fold", "Fold"),
                    ("candidate_geometric_return_pct", "Cand Geom %"),
                    ("baseline_geometric_return_pct", "Switch Geom %"),
                    ("delta_vs_baseline_geometric_return_pct", "Delta %"),
                ],
            ),
            "",
            "## Decision",
            "",
            markdown_table(
                [decision_row],
                [
                    ("candidate_profile", "Candidate"),
                    ("base_full_2020_candidate_geometric_return_pct", "Cand Base Geom %"),
                    ("base_full_2020_baseline_geometric_return_pct", "Switch Base Geom %"),
                    ("base_full_2020_candidate_cagr_pct", "Cand Base CAGR %"),
                    ("base_full_2020_baseline_cagr_pct", "Switch Base CAGR %"),
                    ("oos_delta_vs_baseline_geometric_return_pct", "OOS Delta %"),
                    ("status", "Status"),
                ],
            ),
            "",
            f"- {conclusion}",
        ]
    )


def main() -> None:
    args = parse_args()
    configure_logging()

    switch_date = parse_date(args.switch_date)
    primary_start = parse_date(args.primary_start)
    secondary_start = parse_date(args.secondary_start)
    if secondary_start != switch_date:
        raise ValueError("secondary-start must equal switch-date to keep cut fixed.")
    if secondary_start <= primary_start:
        raise ValueError("secondary-start must be later than primary-start")
    requested_end = parse_date(args.end) if args.end else datetime.now(timezone.utc)

    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    # Candidate history end first, so switch run can align to same terminal date.
    candidate_history_service = make_candidate_service()
    candidate_enriched = candidate_history_service.prepare_enriched_history(
        exchange=args.exchange,
        market_type=args.market_type,
        symbol=args.symbol,
        strategy_profile=args.candidate_profile,
        start=primary_start,
        end=requested_end,
    )
    candidate_resolved_end = resolve_end_from_history({args.candidate_profile: candidate_enriched})

    switch_case = run_regime_switch_case(
        symbol=args.symbol,
        exchange=args.exchange,
        market_type=args.market_type,
        simple_profile=args.simple_profile,
        challenger_strategy_profile=args.challenger_strategy_profile,
        challenger_overlay_profile=args.challenger_overlay_profile,
        switch_date=switch_date,
        primary_start=primary_start,
        requested_end=min(requested_end, candidate_resolved_end),
        cost_scenarios=COST_SCENARIOS,
    )
    resolved_end: datetime = switch_case["resolved_end"]
    windows = {
        "full_2020": (primary_start, resolved_end),
        "two_year": (secondary_start, resolved_end),
    }

    switch_raw = switch_case["trade_frame"]
    switch_raw = switch_raw[switch_raw["scenario_kind"] == SWITCH_SCENARIO_KIND].copy()
    switch_labeled, _coverage = attach_axis_band_risk_group(switch_raw)
    switch_managed = build_switch_managed_trades(switch_labeled)
    switch_managed["profile_kind"] = "switch_baseline"
    switch_managed["strategy_profile"] = "switch_simple_candidate_v2_then_challenger_managed"
    switch_managed["size_multiplier"] = switch_managed["switch_size_multiplier"].astype(float)
    switch_managed["weighted_pnl_r"] = switch_managed["switch_weighted_pnl_r"].astype(float)
    switch_managed["weighted_pnl_pct"] = switch_managed["switch_weighted_pnl_pct"].astype(float)

    candidate_rows: list[dict[str, Any]] = []
    for cost_scenario, overrides in COST_SCENARIOS.items():
        service = make_candidate_service(assumption_overrides=overrides)
        for window, (window_start, window_end) in windows.items():
            summary, trades = service.run_symbol_strategy_with_enriched_frames(
                symbol=args.symbol,
                strategy_profile=args.candidate_profile,
                start=window_start,
                end=window_end,
                enriched_frames=candidate_enriched,
            )
            _ = summary
            for trade in trades:
                row = {
                    **asdict(trade),
                    "cost_scenario": cost_scenario,
                    "window": window,
                    "profile_kind": "candidate",
                    "strategy_profile": args.candidate_profile,
                    "size_multiplier": 1.0,
                    "weighted_pnl_r": float(trade.pnl_r),
                    "weighted_pnl_pct": float(trade.pnl_pct),
                }
                candidate_rows.append(row)
    candidate_frame = normalize_trade_frame(pd.DataFrame(candidate_rows) if candidate_rows else pd.DataFrame())

    baseline_frame = switch_managed[
        (switch_managed["entry_time"] >= pd.Timestamp(primary_start))
        & (switch_managed["entry_time"] <= pd.Timestamp(resolved_end))
    ].copy()
    baseline_frame = normalize_trade_frame(baseline_frame)

    all_frames = [frame for frame in (baseline_frame, candidate_frame) if not frame.empty]
    weighted_trades = normalize_trade_frame(pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame())

    summary_rows: list[dict[str, Any]] = []
    for cost_scenario in COST_SCENARIOS:
        for window, (window_start, window_end) in windows.items():
            for profile_kind, strategy_profile in (
                ("switch_baseline", "switch_simple_candidate_v2_then_challenger_managed"),
                ("candidate", args.candidate_profile),
            ):
                subset = weighted_trades[
                    (weighted_trades["cost_scenario"] == cost_scenario)
                    & (weighted_trades["window"] == window)
                    & (weighted_trades["profile_kind"] == profile_kind)
                ].copy()
                summary_rows.append(
                    summarize_weighted_frame(
                        frame=subset,
                        cost_scenario=cost_scenario,
                        window=window,
                        window_start=window_start,
                        window_end=window_end,
                        profile_kind=profile_kind,
                        strategy_profile=strategy_profile,
                    )
                )

    side_rows = build_side_summary(weighted_trades)
    yearly_rows = build_yearly_geometric_returns(weighted_trades)
    cost_rows = build_cost_sensitivity(summary_rows)
    concentration_rows = build_trade_concentration(weighted_trades, yearly_rows)
    folds = generate_folds(
        start=primary_start,
        end=resolved_end,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
    )
    fold_rows = build_fold_panel(weighted_trades=weighted_trades, folds=folds)
    oos_summary = build_oos_summary(fold_rows)
    decision_row = build_comparison_decision(
        summary_rows=summary_rows,
        side_rows=side_rows,
        concentration_rows=concentration_rows,
        oos_summary=oos_summary,
        candidate_profile=args.candidate_profile,
    )
    report = build_report(
        summary_rows=summary_rows,
        fold_rows=fold_rows,
        decision_row=decision_row,
        candidate_profile=args.candidate_profile,
    )

    write_csv(output_dir / "summary_all.csv", summary_rows)
    write_csv(output_dir / "side_summary_all.csv", side_rows)
    write_csv(output_dir / "yearly_geometric_returns.csv", yearly_rows)
    write_csv(output_dir / "cost_sensitivity.csv", cost_rows)
    write_csv(output_dir / "fold_panel.csv", fold_rows)
    write_csv(output_dir / "comparison_decision.csv", [decision_row])
    (output_dir / "report.md").write_text(report + "\n", encoding="utf-8")
    print(f"saved independent candidate vs switch baseline artifacts to {output_dir}")


if __name__ == "__main__":
    main()
