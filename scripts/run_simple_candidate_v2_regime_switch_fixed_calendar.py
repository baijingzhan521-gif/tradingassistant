from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.service import BacktestAssumptions, BacktestService, BacktestTrade
from app.core.logging import configure_logging
from app.core.exceptions import ExternalServiceError
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.services.strategy_service import StrategyService
from scripts.post_tp1_managed_replay import build_service
from scripts.run_range_failure_vs_challenger_managed import (
    COST_SCENARIOS,
    DEFAULT_BASELINE_OVERLAY_PROFILE,
    DEFAULT_BASELINE_STRATEGY_PROFILE,
    baseline_label,
    ensure_output_dir,
    parse_date,
    resolve_end_from_history,
    run_managed_baseline_with_helper,
    write_csv,
)


SYMBOL = "BTC/USDT:USDT"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "backtests" / "simple_candidate_v2_regime_switch_fixed_calendar"
DEFAULT_SWITCH_DATE = "2024-03-19"
SIMPLE_PROFILE = "swing_trend_simple_candidate_v2"
WINDOW_STARTS = {
    "full_2020": "2020-01-01",
    "two_year": DEFAULT_SWITCH_DATE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-calendar regime switch: simple_candidate_v2 -> challenger_managed.")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--simple-profile", default=SIMPLE_PROFILE)
    parser.add_argument("--challenger-strategy-profile", default=DEFAULT_BASELINE_STRATEGY_PROFILE)
    parser.add_argument("--challenger-overlay-profile", default=DEFAULT_BASELINE_OVERLAY_PROFILE)
    parser.add_argument("--switch-date", default=DEFAULT_SWITCH_DATE)
    parser.add_argument("--primary-start", default=WINDOW_STARTS["full_2020"])
    parser.add_argument("--secondary-start", default=WINDOW_STARTS["two_year"])
    parser.add_argument("--end", default=None, help="Defaults to latest available data at runtime.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def make_simple_service(*, assumption_overrides: dict[str, Any] | None = None) -> BacktestService:
    assumptions = {
        "exit_profile": "simple_candidate_v2_fixed_calendar",
        "take_profit_mode": "scaled",
        "scaled_tp1_r": 1.0,
        "scaled_tp2_r": 3.0,
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


def utc_from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def pf_from_frame(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    wins = float(frame.loc[frame["pnl_r"] > 0.0, "pnl_r"].sum())
    losses = float(-frame.loc[frame["pnl_r"] < 0.0, "pnl_r"].sum())
    if losses <= 0.0:
        return None
    return round(wins / losses, 4)


def geometric_return_pct(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    compounded = float((1.0 + frame["pnl_pct"] / 100.0).prod() - 1.0)
    return round(compounded * 100.0, 4)


def additive_return_pct(frame: pd.DataFrame) -> float:
    return round(float(frame["pnl_pct"].sum()), 4) if not frame.empty else 0.0


def cagr_pct(*, frame: pd.DataFrame, window_start: datetime, window_end: datetime) -> float:
    if frame.empty:
        return 0.0
    elapsed_days = max((window_end - window_start).total_seconds() / 86400.0, 1e-9)
    years = elapsed_days / 365.25
    if years <= 0.0:
        return 0.0
    factor = float((1.0 + frame["pnl_pct"] / 100.0).prod())
    if factor <= 0.0:
        return -100.0
    return round(((factor ** (1.0 / years)) - 1.0) * 100.0, 4)


def annotate_trades(
    trades: list[BacktestTrade],
    *,
    cost_scenario: str,
    window: str,
    scenario_kind: str,
    scenario_label: str,
    switch_date: datetime,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trade in trades:
        row = asdict(trade)
        signal_time = pd.to_datetime(row["signal_time"], utc=True).to_pydatetime()
        row.update(
            {
                "cost_scenario": cost_scenario,
                "window": window,
                "scenario_kind": scenario_kind,
                "scenario_label": scenario_label,
                "segment": "pre_cutover" if signal_time < switch_date else "post_cutover",
            }
        )
        rows.append(row)
    return rows


def build_switch_rows(
    *,
    simple_trades: list[BacktestTrade],
    challenger_trades: list[BacktestTrade],
    cost_scenario: str,
    window: str,
    switch_date: datetime,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trade in simple_trades:
        signal_time = pd.to_datetime(trade.signal_time, utc=True).to_pydatetime()
        if signal_time < switch_date:
            row = asdict(trade)
            row.update(
                {
                    "cost_scenario": cost_scenario,
                    "window": window,
                    "scenario_kind": "switch_simple_candidate_v2_then_challenger_managed",
                    "scenario_label": "swing_trend_simple_candidate_v2 -> challenger_managed",
                    "segment": "pre_cutover",
                }
            )
            rows.append(row)
    for trade in challenger_trades:
        signal_time = pd.to_datetime(trade.signal_time, utc=True).to_pydatetime()
        if signal_time >= switch_date:
            row = asdict(trade)
            row.update(
                {
                    "cost_scenario": cost_scenario,
                    "window": window,
                    "scenario_kind": "switch_simple_candidate_v2_then_challenger_managed",
                    "scenario_label": "swing_trend_simple_candidate_v2 -> challenger_managed",
                    "segment": "post_cutover",
                }
            )
            rows.append(row)
    return rows


def build_trade_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "cost_scenario",
                "window",
                "scenario_kind",
                "scenario_label",
                "segment",
                "signal_time",
                "entry_time",
                "exit_time",
                "pnl_pct",
                "pnl_r",
            ]
        )
    frame = pd.DataFrame(rows)
    for column in ("signal_time", "entry_time", "exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    frame["year"] = frame["signal_time"].dt.year.astype(int)
    frame["month"] = frame["signal_time"].dt.strftime("%Y-%m")
    frame["quarter"] = frame["signal_time"].dt.tz_localize(None).dt.to_period("Q").astype(str)
    return frame.sort_values(["cost_scenario", "window", "scenario_kind", "entry_time"]).reset_index(drop=True)


def summarize_by_group(
    frame: pd.DataFrame,
    *,
    group_cols: list[str],
    include_cagr: bool = False,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, sort=True, observed=True):
        key_tuple = (keys,) if not isinstance(keys, tuple) else keys
        row = {column: value for column, value in zip(group_cols, key_tuple)}
        row.update(
            {
                "trades": int(len(group)),
                "profit_factor": pf_from_frame(group),
                "expectancy_r": round(float(group["pnl_r"].mean()), 4),
                "cum_r": round(float(group["pnl_r"].sum()), 4),
                "max_dd_r": round(
                    abs(float((group.sort_values("entry_time")["pnl_r"].cumsum() - group.sort_values("entry_time")["pnl_r"].cumsum().cummax()).min())),
                    4,
                ),
                "geometric_return_pct": geometric_return_pct(group),
                "additive_return_pct": additive_return_pct(group),
            }
        )
        if include_cagr and window_start is not None and window_end is not None:
            row["cagr_pct"] = cagr_pct(frame=group, window_start=window_start, window_end=window_end)
        rows.append(row)
    return rows


def build_summary_rows(frame: pd.DataFrame, window_bounds: dict[str, tuple[datetime, datetime]]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(["cost_scenario", "window", "scenario_kind", "scenario_label"], sort=True, observed=True)
    for (cost_scenario, window, scenario_kind, scenario_label), group in grouped:
        window_start, window_end = window_bounds[window]
        rows.append(
            {
                "cost_scenario": cost_scenario,
                "window": window,
                "scenario_kind": scenario_kind,
                "scenario_label": scenario_label,
                "window_start": window_start.date().isoformat(),
                "window_end": window_end.date().isoformat(),
                "trades": int(len(group)),
                "win_rate_pct": round(float((group["pnl_r"] > 0.0).mean() * 100.0), 2),
                "profit_factor": pf_from_frame(group),
                "expectancy_r": round(float(group["pnl_r"].mean()), 4),
                "cum_r": round(float(group["pnl_r"].sum()), 4),
                "max_dd_r": round(
                    abs(float((group.sort_values("entry_time")["pnl_r"].cumsum() - group.sort_values("entry_time")["pnl_r"].cumsum().cummax()).min())),
                    4,
                ),
                "avg_holding_bars": round(float(group["bars_held"].mean()), 2),
                "geometric_return_pct": geometric_return_pct(group),
                "additive_return_pct": additive_return_pct(group),
                "cagr_pct": cagr_pct(frame=group, window_start=window_start, window_end=window_end),
            }
        )
    return rows


def build_monthly_delta_summary(
    frame: pd.DataFrame,
    *,
    baseline_kind: str,
    switch_kind: str,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    grouped = frame[frame["scenario_kind"].isin([baseline_kind, switch_kind])].groupby(
        ["cost_scenario", "window", "segment", "month"],
        sort=True,
        observed=True,
    )
    for (cost_scenario, window, segment, month), group in grouped:
        baseline_r = float(group.loc[group["scenario_kind"] == baseline_kind, "pnl_r"].sum())
        switch_r = float(group.loc[group["scenario_kind"] == switch_kind, "pnl_r"].sum())
        rows.append(
            {
                "cost_scenario": cost_scenario,
                "window": window,
                "segment": segment,
                "month": month,
                "baseline_r": round(baseline_r, 4),
                "switch_r": round(switch_r, 4),
                "delta_r": round(switch_r - baseline_r, 4),
            }
        )
    return rows


def build_quarterly_delta_summary(
    frame: pd.DataFrame,
    *,
    baseline_kind: str,
    switch_kind: str,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    grouped = frame[frame["scenario_kind"].isin([baseline_kind, switch_kind])].groupby(
        ["cost_scenario", "window", "segment", "quarter"],
        sort=True,
        observed=True,
    )
    for (cost_scenario, window, segment, quarter), group in grouped:
        baseline_r = float(group.loc[group["scenario_kind"] == baseline_kind, "pnl_r"].sum())
        switch_r = float(group.loc[group["scenario_kind"] == switch_kind, "pnl_r"].sum())
        rows.append(
            {
                "cost_scenario": cost_scenario,
                "window": window,
                "segment": segment,
                "quarter": quarter,
                "baseline_r": round(baseline_r, 4),
                "switch_r": round(switch_r, 4),
                "delta_r": round(switch_r - baseline_r, 4),
            }
        )
    return rows


def build_switch_decision(
    *,
    summary_rows: list[dict[str, Any]],
    side_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    baseline_kind: str,
    switch_kind: str,
) -> dict[str, Any]:
    def lookup(rows: list[dict[str, Any]], *, cost: str, window: str, kind: str, extra: tuple[str, Any] | None = None) -> dict[str, Any]:
        for row in rows:
            if row["cost_scenario"] == cost and row["window"] == window and row["scenario_kind"] == kind:
                if extra is None or row[extra[0]] == extra[1]:
                    return row
        raise KeyError(f"Missing row for cost={cost}, window={window}, kind={kind}, extra={extra}")

    baseline_base = lookup(summary_rows, cost="base", window="full_2020", kind=baseline_kind)
    switch_base = lookup(summary_rows, cost="base", window="full_2020", kind=switch_kind)
    baseline_stress = lookup(summary_rows, cost="stress_x2", window="full_2020", kind=baseline_kind)
    switch_stress = lookup(summary_rows, cost="stress_x2", window="full_2020", kind=switch_kind)
    baseline_long = lookup(side_rows, cost="base", window="two_year", kind=baseline_kind, extra=("side", "LONG"))
    switch_long = lookup(side_rows, cost="base", window="two_year", kind=switch_kind, extra=("side", "LONG"))
    switch_pre = lookup(segment_rows, cost="base", window="full_2020", kind=switch_kind, extra=("segment", "pre_cutover"))
    switch_post = lookup(segment_rows, cost="base", window="full_2020", kind=switch_kind, extra=("segment", "post_cutover"))
    baseline_pre = lookup(segment_rows, cost="base", window="full_2020", kind=baseline_kind, extra=("segment", "pre_cutover"))
    baseline_post = lookup(segment_rows, cost="base", window="full_2020", kind=baseline_kind, extra=("segment", "post_cutover"))

    pass_geo = float(switch_base["geometric_return_pct"]) > float(baseline_base["geometric_return_pct"])
    pass_pf = float(switch_base["profit_factor"] or 0.0) > float(baseline_base["profit_factor"] or 0.0)
    pass_dd = float(switch_base["max_dd_r"]) <= float(baseline_base["max_dd_r"]) + 2.0
    pass_stress_geo = float(switch_stress["geometric_return_pct"]) > 0.0
    pass_stress_pf = float(switch_stress["profit_factor"] or 0.0) >= 1.0
    pass_long_guard = float(switch_long["cum_r"]) >= float(baseline_long["cum_r"]) - 2.0
    promoted = pass_geo and pass_pf and pass_dd and pass_stress_geo and pass_stress_pf and pass_long_guard

    return {
        "baseline_scenario": baseline_kind,
        "switch_scenario": switch_kind,
        "base_full_2020_baseline_geometric_return_pct": round(float(baseline_base["geometric_return_pct"]), 4),
        "base_full_2020_switch_geometric_return_pct": round(float(switch_base["geometric_return_pct"]), 4),
        "base_full_2020_delta_geometric_return_pct": round(
            float(switch_base["geometric_return_pct"] - baseline_base["geometric_return_pct"]), 4
        ),
        "base_full_2020_baseline_cagr_pct": round(float(baseline_base["cagr_pct"]), 4),
        "base_full_2020_switch_cagr_pct": round(float(switch_base["cagr_pct"]), 4),
        "base_full_2020_delta_cagr_pct": round(float(switch_base["cagr_pct"] - baseline_base["cagr_pct"]), 4),
        "base_full_2020_baseline_profit_factor": round(float(baseline_base["profit_factor"]), 4),
        "base_full_2020_switch_profit_factor": round(float(switch_base["profit_factor"]), 4),
        "base_full_2020_baseline_max_dd_r": round(float(baseline_base["max_dd_r"]), 4),
        "base_full_2020_switch_max_dd_r": round(float(switch_base["max_dd_r"]), 4),
        "stress_full_2020_baseline_geometric_return_pct": round(float(baseline_stress["geometric_return_pct"]), 4),
        "stress_full_2020_switch_geometric_return_pct": round(float(switch_stress["geometric_return_pct"]), 4),
        "stress_full_2020_baseline_profit_factor": round(float(baseline_stress["profit_factor"]), 4),
        "stress_full_2020_switch_profit_factor": round(float(switch_stress["profit_factor"]), 4),
        "pre_cutover_simple_candidate_cum_r": round(float(switch_pre["cum_r"]), 4),
        "pre_cutover_baseline_cum_r": round(float(baseline_pre["cum_r"]), 4),
        "post_cutover_switch_cum_r": round(float(switch_post["cum_r"]), 4),
        "post_cutover_baseline_cum_r": round(float(baseline_post["cum_r"]), 4),
        "pass_base_geo": pass_geo,
        "pass_base_pf": pass_pf,
        "pass_base_max_dd": pass_dd,
        "pass_stress_geo": pass_stress_geo,
        "pass_stress_pf": pass_stress_pf,
        "pass_secondary_long_guard": pass_long_guard,
        "status": "promoted_calendar_switch_candidate" if promoted else "rejected_switch_fragile_or_unprofitable",
        "next_route": "treat_switch_as_active_management_candidate"
        if promoted
        else "keep_challenger_managed_and_do_not_add_more_switch_rules",
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_None_"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def build_report(
    *,
    summary_rows: list[dict[str, Any]],
    side_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    monthly_rows: list[dict[str, Any]],
    quarterly_rows: list[dict[str, Any]],
    decision_row: dict[str, Any],
    simple_profile: str,
    challenger_strategy_profile: str,
    challenger_overlay_profile: str,
    switch_date: datetime,
) -> str:
    base_summary = [row for row in summary_rows if row["cost_scenario"] == "base" and row["window"] == "full_2020"]
    base_summary = sorted(base_summary, key=lambda row: float(row["geometric_return_pct"]), reverse=True)
    base_side = [row for row in side_rows if row["cost_scenario"] == "base" and row["window"] == "full_2020"]
    base_side = sorted(base_side, key=lambda row: (row["scenario_kind"], row["side"]))
    base_segment = [row for row in segment_rows if row["cost_scenario"] == "base" and row["window"] == "full_2020"]
    base_segment = sorted(base_segment, key=lambda row: (row["scenario_kind"], row["segment"]))
    base_monthly = [row for row in monthly_rows if row["cost_scenario"] == "base" and row["window"] == "full_2020"]
    base_monthly = sorted(base_monthly, key=lambda row: (row["scenario_kind"], row["month"]))
    base_quarterly = [row for row in quarterly_rows if row["cost_scenario"] == "base" and row["window"] == "full_2020"]
    base_quarterly = sorted(base_quarterly, key=lambda row: (row["scenario_kind"], row["quarter"]))
    return "\n".join(
        [
            "# Simple Candidate V2 Regime Switch",
            "",
            f"- switch date: `{switch_date.date().isoformat()}`",
            f"- simple profile: `{simple_profile}`",
            f"- challenger managed: `{challenger_strategy_profile}` + `{challenger_overlay_profile}`",
            "- 说明：这是固定日历切换，不搜索切点，不新增 profile。",
            "",
            "## Full Window Summary",
            "",
            markdown_table(
                base_summary,
                [
                    ("scenario_kind", "Scenario"),
                    ("trades", "Trades"),
                    ("profit_factor", "PF"),
                    ("cum_r", "Cum R"),
                    ("max_dd_r", "MaxDD R"),
                    ("geometric_return_pct", "Geom Return %"),
                    ("cagr_pct", "CAGR %"),
                ],
            ),
            "",
            "## Segment Summary",
            "",
            markdown_table(
                base_segment,
                [
                    ("scenario_kind", "Scenario"),
                    ("segment", "Segment"),
                    ("trades", "Trades"),
                    ("profit_factor", "PF"),
                    ("cum_r", "Cum R"),
                    ("geometric_return_pct", "Geom Return %"),
                ],
            ),
            "",
            "## Side Summary",
            "",
            markdown_table(
                base_side,
                [
                    ("scenario_kind", "Scenario"),
                    ("side", "Side"),
                    ("trades", "Trades"),
                    ("profit_factor", "PF"),
                    ("cum_r", "Cum R"),
                    ("geometric_return_pct", "Geom Return %"),
                ],
            ),
            "",
            "## Monthly Summary",
            "",
            markdown_table(
                base_monthly,
                [
                    ("scenario_kind", "Scenario"),
                    ("month", "Month"),
                    ("trades", "Trades"),
                    ("cum_r", "Cum R"),
                    ("geometric_return_pct", "Geom Return %"),
                ],
            ),
            "",
            "## Quarterly Summary",
            "",
            markdown_table(
                base_quarterly,
                [
                    ("scenario_kind", "Scenario"),
                    ("quarter", "Quarter"),
                    ("trades", "Trades"),
                    ("cum_r", "Cum R"),
                    ("geometric_return_pct", "Geom Return %"),
                ],
            ),
            "",
            "## Decision",
            "",
            markdown_table(
                [decision_row],
                [
                    ("base_full_2020_baseline_geometric_return_pct", "Baseline Geom %"),
                    ("base_full_2020_switch_geometric_return_pct", "Switch Geom %"),
                    ("base_full_2020_delta_geometric_return_pct", "Delta Geom %"),
                    ("base_full_2020_baseline_cagr_pct", "Baseline CAGR %"),
                    ("base_full_2020_switch_cagr_pct", "Switch CAGR %"),
                    ("base_full_2020_baseline_profit_factor", "Baseline PF"),
                    ("base_full_2020_switch_profit_factor", "Switch PF"),
                    ("base_full_2020_baseline_max_dd_r", "Baseline MaxDD"),
                    ("base_full_2020_switch_max_dd_r", "Switch MaxDD"),
                    ("stress_full_2020_switch_geometric_return_pct", "Stress Switch Geom %"),
                    ("stress_full_2020_switch_profit_factor", "Stress Switch PF"),
                    ("status", "Status"),
                ],
            ),
            "",
            "## Conclusion",
            "",
            (
                "固定日历切换通过。"
                if decision_row["status"] == "promoted_calendar_switch_candidate"
                else "固定日历切换没有稳稳压过 challenger_managed。"
            ),
            f"- 前段增量：switch 前段 `cum_r` `{decision_row['pre_cutover_simple_candidate_cum_r']:.4f}R` 对比 baseline `{decision_row['pre_cutover_baseline_cum_r']:.4f}R`。",
            f"- 后段稳定性：switch 后段 `cum_r` `{decision_row['post_cutover_switch_cum_r']:.4f}R`，baseline 后段 `{decision_row['post_cutover_baseline_cum_r']:.4f}R`。",
            f"- 这条线只允许固定切换，不再引入额外切点。",
        ]
    )


def run_simple_candidate_window(
    *,
    service: BacktestService,
    symbol: str,
    strategy_profile: str,
    exchange: str,
    market_type: str,
    start: datetime,
    end: datetime,
) -> list[BacktestTrade]:
    report = service.run(
        exchange=exchange,
        market_type=market_type,
        symbols=[symbol],
        strategy_profiles=[strategy_profile],
        start=start,
        end=end,
    )
    return report.trades


def run_regime_switch_case(
    *,
    symbol: str,
    exchange: str,
    market_type: str,
    simple_profile: str,
    challenger_strategy_profile: str,
    challenger_overlay_profile: str,
    switch_date: datetime,
    primary_start: datetime,
    requested_end: datetime | None,
    cost_scenarios: dict[str, dict[str, Any]] = COST_SCENARIOS,
    enriched_history: dict[str, dict[str, pd.DataFrame]] | None = None,
    resolved_end: datetime | None = None,
) -> dict[str, Any]:
    if requested_end is None:
        requested_end = datetime.now(timezone.utc)
    if enriched_history is None or resolved_end is None:
        history_service = build_service()
        enriched_history = {
            challenger_strategy_profile: history_service.prepare_enriched_history(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                strategy_profile=challenger_strategy_profile,
                start=primary_start,
                end=requested_end,
            ),
            simple_profile: history_service.prepare_enriched_history(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                strategy_profile=simple_profile,
                start=primary_start,
                end=requested_end,
            ),
        }
        resolved_end = resolve_end_from_history(enriched_history)
    windows = {
        "full_2020": (primary_start, resolved_end),
        "two_year": (switch_date, resolved_end),
    }

    all_rows: list[dict[str, Any]] = []

    for cost_scenario, overrides in cost_scenarios.items():
        for window, (window_start, window_end) in windows.items():
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    challenger_service = build_service(assumption_overrides=overrides)
                    simple_service = make_simple_service(assumption_overrides=overrides)
                    _, challenger_trades = run_managed_baseline_with_helper(
                        service=challenger_service,
                        symbol=symbol,
                        baseline_strategy_profile=challenger_strategy_profile,
                        baseline_overlay_profile=challenger_overlay_profile,
                        start=window_start,
                        end=window_end,
                        enriched_frames=enriched_history[challenger_strategy_profile],
                    )
                    simple_trades = run_simple_candidate_window(
                        service=simple_service,
                        symbol=symbol,
                        strategy_profile=simple_profile,
                        exchange=exchange,
                        market_type=market_type,
                        start=window_start,
                        end=window_end,
                    )
                    switch_rows = build_switch_rows(
                        simple_trades=simple_trades,
                        challenger_trades=challenger_trades,
                        cost_scenario=cost_scenario,
                        window=window,
                        switch_date=switch_date,
                    )
                    break
                except ExternalServiceError as exc:
                    last_error = exc
                    if attempt == 2:
                        raise
                    time.sleep(2.0 * (attempt + 1))

            all_rows.extend(
                annotate_trades(
                    challenger_trades,
                    cost_scenario=cost_scenario,
                    window=window,
                    scenario_kind="always_challenger_managed",
                    scenario_label=f"{challenger_strategy_profile} + {challenger_overlay_profile}",
                    switch_date=switch_date,
                )
            )
            all_rows.extend(
                annotate_trades(
                    simple_trades,
                    cost_scenario=cost_scenario,
                    window=window,
                    scenario_kind="always_simple_candidate_v2",
                    scenario_label=simple_profile,
                    switch_date=switch_date,
                )
            )
            all_rows.extend(switch_rows)

    trade_frame = build_trade_frame(all_rows)
    summary_rows = build_summary_rows(trade_frame, windows)
    side_rows = summarize_by_group(
        trade_frame,
        group_cols=["cost_scenario", "window", "scenario_kind", "scenario_label", "side"],
    )
    segment_rows = summarize_by_group(
        trade_frame,
        group_cols=["cost_scenario", "window", "scenario_kind", "scenario_label", "segment"],
    )
    monthly_rows = summarize_by_group(
        trade_frame,
        group_cols=["cost_scenario", "window", "scenario_kind", "scenario_label", "month"],
    )
    quarterly_rows = summarize_by_group(
        trade_frame,
        group_cols=["cost_scenario", "window", "scenario_kind", "scenario_label", "quarter"],
    )
    yearly_rows = summarize_by_group(
        trade_frame,
        group_cols=["cost_scenario", "window", "scenario_kind", "scenario_label", "year"],
    )

    monthly_delta_rows = build_monthly_delta_summary(
        trade_frame,
        baseline_kind="always_challenger_managed",
        switch_kind="switch_simple_candidate_v2_then_challenger_managed",
    )
    quarterly_delta_rows = build_quarterly_delta_summary(
        trade_frame,
        baseline_kind="always_challenger_managed",
        switch_kind="switch_simple_candidate_v2_then_challenger_managed",
    )
    decision_row = build_switch_decision(
        summary_rows=summary_rows,
        side_rows=side_rows,
        segment_rows=segment_rows,
        baseline_kind="always_challenger_managed",
        switch_kind="switch_simple_candidate_v2_then_challenger_managed",
    )
    report = build_report(
        summary_rows=summary_rows,
        side_rows=side_rows,
        segment_rows=segment_rows,
        monthly_rows=monthly_rows,
        quarterly_rows=quarterly_rows,
        decision_row=decision_row,
        simple_profile=simple_profile,
        challenger_strategy_profile=challenger_strategy_profile,
        challenger_overlay_profile=challenger_overlay_profile,
        switch_date=switch_date,
    )

    return {
        "switch_date": switch_date,
        "resolved_end": resolved_end,
        "windows": windows,
        "trade_frame": trade_frame,
        "summary_rows": summary_rows,
        "side_rows": side_rows,
        "segment_rows": segment_rows,
        "monthly_rows": monthly_rows,
        "quarterly_rows": quarterly_rows,
        "yearly_rows": yearly_rows,
        "monthly_delta_rows": monthly_delta_rows,
        "quarterly_delta_rows": quarterly_delta_rows,
        "decision_row": decision_row,
        "report": report,
    }


def main() -> None:
    args = parse_args()
    configure_logging()

    switch_date = utc_from_iso(args.switch_date)
    primary_start = parse_date(args.primary_start)
    secondary_start = parse_date(args.secondary_start)
    if secondary_start != switch_date:
        raise ValueError("secondary-start must equal switch-date to keep the cut fixed.")
    if secondary_start <= primary_start:
        raise ValueError("secondary-start must be later than primary-start")
    requested_end = parse_date(args.end) if args.end else datetime.now(timezone.utc)

    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)
    case = run_regime_switch_case(
        symbol=args.symbol,
        exchange=args.exchange,
        market_type=args.market_type,
        simple_profile=args.simple_profile,
        challenger_strategy_profile=args.challenger_strategy_profile,
        challenger_overlay_profile=args.challenger_overlay_profile,
        switch_date=switch_date,
        primary_start=primary_start,
        requested_end=requested_end,
    )

    write_csv(output_dir / "summary_all.csv", case["summary_rows"])
    write_csv(output_dir / "side_summary_all.csv", case["side_rows"])
    write_csv(output_dir / "segment_summary.csv", case["segment_rows"])
    write_csv(output_dir / "monthly_pnl_summary.csv", case["monthly_rows"])
    write_csv(output_dir / "quarterly_pnl_summary.csv", case["quarterly_rows"])
    write_csv(output_dir / "yearly_geometric_returns.csv", case["yearly_rows"])
    write_csv(output_dir / "monthly_switch_delta_summary.csv", case["monthly_delta_rows"])
    write_csv(output_dir / "quarterly_switch_delta_summary.csv", case["quarterly_delta_rows"])
    write_csv(output_dir / "comparison_decision.csv", [case["decision_row"]])
    case["trade_frame"].to_csv(output_dir / "trades_all.csv", index=False)
    (output_dir / "report.md").write_text(case["report"] + "\n", encoding="utf-8")
    print(f"saved fixed-calendar regime switch artifacts to {output_dir}")


if __name__ == "__main__":
    main()
