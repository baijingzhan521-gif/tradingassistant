from __future__ import annotations

import argparse
import json
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

from app.backtesting.service import BacktestAssumptions, BacktestReport, BacktestService
from app.core.exceptions import ExternalServiceError
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.services.strategy_service import StrategyService
from scripts.post_tp1_managed_replay import build_service
from scripts.run_range_failure_vs_challenger_managed import (
    ensure_output_dir,
    parse_date,
    resolve_end_from_history,
    trades_frame,
    write_csv,
)
from scripts.run_simple_candidate_v2_regime_switch_fixed_calendar import (
    DEFAULT_SWITCH_DATE,
    run_regime_switch_case,
)


SYMBOL = "BTC/USDT:USDT"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "backtests" / "switch_candidate_vs_ct_block80"
WINDOWS = ("two_year", "full_2020")
WINDOW_STARTS = {
    "two_year": "2024-03-19",
    "full_2020": "2020-01-01",
}
COST_SCENARIOS = {
    "base": {},
    "stress_x2": {"taker_fee_bps": 10.0, "slippage_bps": 4.0},
    "stress_x3": {"taker_fee_bps": 15.0, "slippage_bps": 6.0},
}

SWITCH_KIND = "switch_candidate"
CT_KIND = "ct_block80_candidate"
SWITCH_SCENARIO_KIND = "switch_simple_candidate_v2_then_challenger_managed"

MIN_TRADES = 10
MIN_ACCEPTABLE_PF = 1.0
MAX_PF_UNDERPERF_DELTA = 1.0
MAX_DD_ALLOWED_WORSE_R = 12.0
MAX_DD_ABS_CAP_R = 14.0
LONG_GUARD_MAX_LOSS_R = 2.0
MAX_TOP3_SHARE_PCT = 65.0
MAX_BEST_YEAR_SHARE_PCT = 80.0
MAX_BEST_MONTH_SHARE_PCT = 40.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Head-to-head: fixed switch candidate vs ct_block80 under aligned windows and cost stress."
    )
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--simple-profile", default="swing_trend_simple_candidate_v2")
    parser.add_argument(
        "--challenger-strategy-profile",
        default="swing_trend_long_regime_short_no_reversal_no_aux_v1",
    )
    parser.add_argument(
        "--challenger-overlay-profile",
        default="be_if_no_extension_within_3bars_after_tp1_and_veto_hold_at_trend_96_98",
    )
    parser.add_argument("--switch-date", default=DEFAULT_SWITCH_DATE)
    parser.add_argument("--ct-profile", default="swing_exhaustion_divergence_ct_block80_v1_btc")
    parser.add_argument("--primary-start", default=WINDOW_STARTS["full_2020"])
    parser.add_argument("--secondary-start", default=WINDOW_STARTS["two_year"])
    parser.add_argument("--end", default=None, help="Defaults to latest available data at runtime.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def utc_from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def make_ct_service(*, assumption_overrides: dict[str, Any] | None = None) -> BacktestService:
    assumptions = {
        "exit_profile": "exhaustion_divergence_strategy_defined",
        "take_profit_mode": "scaled",
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


def run_ct_window_with_retry(
    *,
    service: BacktestService,
    symbol: str,
    strategy_profile: str,
    exchange: str,
    market_type: str,
    start: datetime,
    end: datetime,
) -> BacktestReport:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return service.run(
                exchange=exchange,
                market_type=market_type,
                symbols=[symbol],
                strategy_profiles=[strategy_profile],
                start=start,
                end=end,
            )
        except ExternalServiceError as exc:
            last_error = exc
            if attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError("unreachable")


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


def cagr_pct(frame: pd.DataFrame, *, window_start: datetime, window_end: datetime) -> float:
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


def max_drawdown_r_from_frame(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    ordered = frame.sort_values("entry_time")
    cumulative = ordered["pnl_r"].cumsum()
    drawdown = cumulative - cumulative.cummax()
    return round(abs(float(drawdown.min())), 4)


def build_summary_row_from_frame(
    *,
    cost_scenario: str,
    window: str,
    window_start: datetime,
    window_end: datetime,
    profile_kind: str,
    strategy_profile: str,
    profile_label: str,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    frame_local = frame.copy()
    trade_count = int(len(frame_local))
    return {
        "cost_scenario": cost_scenario,
        "window": window,
        "window_start": window_start.date().isoformat(),
        "window_end": window_end.date().isoformat(),
        "profile_kind": profile_kind,
        "strategy_profile": strategy_profile,
        "profile_label": profile_label,
        "trades": trade_count,
        "win_rate_pct": round(float((frame_local["pnl_r"] > 0.0).mean() * 100.0), 2) if trade_count else 0.0,
        "profit_factor": pf_from_frame(frame_local),
        "expectancy_r": round(float(frame_local["pnl_r"].mean()), 4) if trade_count else 0.0,
        "cum_r": round(float(frame_local["pnl_r"].sum()), 4) if trade_count else 0.0,
        "max_dd_r": max_drawdown_r_from_frame(frame_local),
        "avg_holding_bars": round(float(frame_local["bars_held"].mean()), 2)
        if trade_count and "bars_held" in frame_local.columns
        else 0.0,
        "geometric_return_pct": geometric_return_pct(frame_local),
        "additive_return_pct": additive_return_pct(frame_local),
        "cagr_pct": cagr_pct(frame_local, window_start=window_start, window_end=window_end),
    }


def build_side_summary(trades_all: pd.DataFrame) -> list[dict[str, Any]]:
    if trades_all.empty:
        return []
    rows: list[dict[str, Any]] = []
    grouped = trades_all.groupby(
        ["cost_scenario", "window", "profile_kind", "strategy_profile", "profile_label", "side"],
        observed=True,
        sort=True,
    )
    for keys, group in grouped:
        cost_scenario, window, profile_kind, strategy_profile, profile_label, side = keys
        rows.append(
            {
                "cost_scenario": cost_scenario,
                "window": window,
                "profile_kind": profile_kind,
                "strategy_profile": strategy_profile,
                "profile_label": profile_label,
                "side": side,
                "trades": int(len(group)),
                "profit_factor": pf_from_frame(group),
                "expectancy_r": round(float(group["pnl_r"].mean()), 4),
                "cum_r": round(float(group["pnl_r"].sum()), 4),
                "max_dd_r": max_drawdown_r_from_frame(group),
                "geometric_return_pct": geometric_return_pct(group),
            }
        )
    return rows


def build_yearly_geometric_returns(trades_all: pd.DataFrame) -> list[dict[str, Any]]:
    if trades_all.empty:
        return []
    rows: list[dict[str, Any]] = []
    grouped = trades_all.groupby(
        ["cost_scenario", "window", "profile_kind", "strategy_profile", "profile_label", "year"],
        observed=True,
        sort=True,
    )
    for keys, group in grouped:
        cost_scenario, window, profile_kind, strategy_profile, profile_label, year = keys
        rows.append(
            {
                "cost_scenario": cost_scenario,
                "window": window,
                "profile_kind": profile_kind,
                "strategy_profile": strategy_profile,
                "profile_label": profile_label,
                "year": int(year),
                "trades": int(len(group)),
                "geometric_return_pct": geometric_return_pct(group),
                "additive_return_pct": additive_return_pct(group),
            }
        )
    return rows


def build_cost_sensitivity(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(summary_rows)
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(["window", "profile_kind", "strategy_profile", "profile_label"], observed=True, sort=True)
    for keys, group in grouped:
        window, profile_kind, strategy_profile, profile_label = keys
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
                    "profile_label": profile_label,
                    "stress_scenario": stress_name,
                    "base_trades": int(base_row["trades"]),
                    "stress_trades": int(stress_row["trades"]),
                    "base_geometric_return_pct": round(float(base_row["geometric_return_pct"]), 4),
                    "stress_geometric_return_pct": round(float(stress_row["geometric_return_pct"]), 4),
                    "delta_geometric_return_pct": round(
                        float(stress_row["geometric_return_pct"] - base_row["geometric_return_pct"]), 4
                    ),
                    "base_cagr_pct": round(float(base_row["cagr_pct"]), 4),
                    "stress_cagr_pct": round(float(stress_row["cagr_pct"]), 4),
                    "delta_cagr_pct": round(float(stress_row["cagr_pct"] - base_row["cagr_pct"]), 4),
                    "base_profit_factor": round(float(base_row["profit_factor"] or 0.0), 4),
                    "stress_profit_factor": round(float(stress_row["profit_factor"] or 0.0), 4),
                    "delta_profit_factor": round(
                        float((stress_row["profit_factor"] or 0.0) - (base_row["profit_factor"] or 0.0)), 4
                    ),
                    "base_max_dd_r": round(float(base_row["max_dd_r"]), 4),
                    "stress_max_dd_r": round(float(stress_row["max_dd_r"]), 4),
                    "delta_max_dd_r": round(float(stress_row["max_dd_r"] - base_row["max_dd_r"]), 4),
                }
            )
    return rows


def build_concentration_summary(trades_all: pd.DataFrame) -> list[dict[str, Any]]:
    if trades_all.empty:
        return []
    rows: list[dict[str, Any]] = []
    base_full = trades_all[(trades_all["cost_scenario"] == "base") & (trades_all["window"] == "full_2020")]
    grouped = base_full.groupby(["profile_kind", "strategy_profile", "profile_label"], observed=True, sort=True)
    for keys, group in grouped:
        profile_kind, strategy_profile, profile_label = keys
        total_r = float(group["pnl_r"].sum())
        denom = abs(total_r)
        ordered = group.sort_values("pnl_r", ascending=False)
        top3 = float(ordered.head(3)["pnl_r"].sum()) if not ordered.empty else 0.0
        year_r = group.groupby(group["signal_time"].dt.year)["pnl_r"].sum()
        month_r = group.groupby(group["signal_time"].dt.strftime("%Y-%m"))["pnl_r"].sum()
        quarter_r = group.groupby(group["signal_time"].dt.tz_localize(None).dt.to_period("Q").astype(str))["pnl_r"].sum()
        best_year = float(year_r.max()) if not year_r.empty else 0.0
        best_month = float(month_r.max()) if not month_r.empty else 0.0
        best_quarter = float(quarter_r.max()) if not quarter_r.empty else 0.0
        rows.append(
            {
                "profile_kind": profile_kind,
                "strategy_profile": strategy_profile,
                "profile_label": profile_label,
                "trades": int(len(group)),
                "cum_r": round(total_r, 4),
                "top3_trade_pnl_share_pct": round(top3 / denom * 100.0, 2) if denom > 0.0 else None,
                "best_year_pnl_share_pct": round(best_year / denom * 100.0, 2) if denom > 0.0 else None,
                "best_month_pnl_share_pct": round(best_month / denom * 100.0, 2) if denom > 0.0 else None,
                "best_quarter_pnl_share_pct": round(best_quarter / denom * 100.0, 2) if denom > 0.0 else None,
                "positive_years": int((year_r > 0).sum()),
                "negative_years": int((year_r < 0).sum()),
            }
        )
    return rows


def _lookup_summary(rows: list[dict[str, Any]], *, cost: str, window: str, kind: str) -> dict[str, Any]:
    for row in rows:
        if row["cost_scenario"] == cost and row["window"] == window and row["profile_kind"] == kind:
            return row
    raise KeyError(f"Missing summary row for cost={cost}, window={window}, kind={kind}")


def _lookup_side(
    rows: list[dict[str, Any]], *, cost: str, window: str, kind: str, side: str
) -> dict[str, Any] | None:
    for row in rows:
        if (
            row["cost_scenario"] == cost
            and row["window"] == window
            and row["profile_kind"] == kind
            and row["side"] == side
        ):
            return row
    return None


def _lookup_concentration(rows: list[dict[str, Any]], *, kind: str) -> dict[str, Any]:
    for row in rows:
        if row["profile_kind"] == kind:
            return row
    raise KeyError(f"Missing concentration row for kind={kind}")


def build_comparison_decision(
    *,
    summary_rows: list[dict[str, Any]],
    side_rows: list[dict[str, Any]],
    concentration_rows: list[dict[str, Any]],
    switch_label: str,
    ct_profile: str,
) -> dict[str, Any]:
    switch_base = _lookup_summary(summary_rows, cost="base", window="full_2020", kind=SWITCH_KIND)
    ct_base = _lookup_summary(summary_rows, cost="base", window="full_2020", kind=CT_KIND)
    switch_stress_x2 = _lookup_summary(summary_rows, cost="stress_x2", window="full_2020", kind=SWITCH_KIND)
    switch_stress_x3 = _lookup_summary(summary_rows, cost="stress_x3", window="full_2020", kind=SWITCH_KIND)
    ct_stress_x2 = _lookup_summary(summary_rows, cost="stress_x2", window="full_2020", kind=CT_KIND)
    ct_stress_x3 = _lookup_summary(summary_rows, cost="stress_x3", window="full_2020", kind=CT_KIND)

    switch_long_two_year = _lookup_side(
        side_rows, cost="base", window="two_year", kind=SWITCH_KIND, side="LONG"
    )
    ct_long_two_year = _lookup_side(side_rows, cost="base", window="two_year", kind=CT_KIND, side="LONG")
    switch_short_full = _lookup_side(side_rows, cost="base", window="full_2020", kind=SWITCH_KIND, side="SHORT")
    ct_short_full = _lookup_side(side_rows, cost="base", window="full_2020", kind=CT_KIND, side="SHORT")
    switch_concentration = _lookup_concentration(concentration_rows, kind=SWITCH_KIND)

    switch_pf = float(switch_base["profit_factor"] or 0.0)
    ct_pf = float(ct_base["profit_factor"] or 0.0)
    switch_dd = float(switch_base["max_dd_r"])
    ct_dd = float(ct_base["max_dd_r"])
    switch_long = float((switch_long_two_year or {}).get("cum_r", 0.0))
    ct_long = float((ct_long_two_year or {}).get("cum_r", 0.0))
    switch_short = float((switch_short_full or {}).get("cum_r", 0.0))
    ct_short = float((ct_short_full or {}).get("cum_r", 0.0))

    top3_share = float(switch_concentration["top3_trade_pnl_share_pct"] or 0.0)
    best_year_share = float(switch_concentration["best_year_pnl_share_pct"] or 0.0)
    best_month_share = float(switch_concentration["best_month_pnl_share_pct"] or 0.0)

    pass_head_to_head_geo = float(switch_base["geometric_return_pct"]) > float(ct_base["geometric_return_pct"])
    pass_base_pf_guard = switch_pf >= max(MIN_ACCEPTABLE_PF, ct_pf - MAX_PF_UNDERPERF_DELTA)
    pass_base_max_dd_guard = switch_dd <= min(MAX_DD_ABS_CAP_R, ct_dd + MAX_DD_ALLOWED_WORSE_R)
    pass_stress_x2_positive = float(switch_stress_x2["geometric_return_pct"]) > 0.0
    pass_stress_x3_positive = float(switch_stress_x3["geometric_return_pct"]) > 0.0
    pass_two_year_long_guard = switch_long >= (ct_long - LONG_GUARD_MAX_LOSS_R)
    pass_trade_floor = int(switch_base["trades"]) >= MIN_TRADES
    pass_top3_concentration = top3_share <= MAX_TOP3_SHARE_PCT
    pass_best_year_concentration = best_year_share <= MAX_BEST_YEAR_SHARE_PCT
    pass_best_month_concentration = best_month_share <= MAX_BEST_MONTH_SHARE_PCT

    promoted = (
        pass_head_to_head_geo
        and pass_base_pf_guard
        and pass_base_max_dd_guard
        and pass_stress_x2_positive
        and pass_stress_x3_positive
        and pass_two_year_long_guard
        and pass_trade_floor
        and pass_top3_concentration
        and pass_best_year_concentration
        and pass_best_month_concentration
    )
    status = (
        "promoted_new_baseline_candidate"
        if promoted
        else "keep_ct_block80_as_independent_pool_member"
    )
    return {
        "switch_candidate_label": switch_label,
        "ct_profile": ct_profile,
        "base_full_2020_switch_geometric_return_pct": round(float(switch_base["geometric_return_pct"]), 4),
        "base_full_2020_ct_geometric_return_pct": round(float(ct_base["geometric_return_pct"]), 4),
        "base_full_2020_delta_geometric_return_pct": round(
            float(switch_base["geometric_return_pct"] - ct_base["geometric_return_pct"]), 4
        ),
        "base_full_2020_switch_cagr_pct": round(float(switch_base["cagr_pct"]), 4),
        "base_full_2020_ct_cagr_pct": round(float(ct_base["cagr_pct"]), 4),
        "base_full_2020_delta_cagr_pct": round(float(switch_base["cagr_pct"] - ct_base["cagr_pct"]), 4),
        "base_full_2020_switch_profit_factor": round(switch_pf, 4),
        "base_full_2020_ct_profit_factor": round(ct_pf, 4),
        "base_full_2020_delta_profit_factor": round(switch_pf - ct_pf, 4),
        "base_full_2020_switch_max_dd_r": round(switch_dd, 4),
        "base_full_2020_ct_max_dd_r": round(ct_dd, 4),
        "base_full_2020_delta_max_dd_r": round(switch_dd - ct_dd, 4),
        "stress_x2_full_2020_switch_geometric_return_pct": round(float(switch_stress_x2["geometric_return_pct"]), 4),
        "stress_x2_full_2020_ct_geometric_return_pct": round(float(ct_stress_x2["geometric_return_pct"]), 4),
        "stress_x3_full_2020_switch_geometric_return_pct": round(float(switch_stress_x3["geometric_return_pct"]), 4),
        "stress_x3_full_2020_ct_geometric_return_pct": round(float(ct_stress_x3["geometric_return_pct"]), 4),
        "base_two_year_long_delta_r": round(switch_long - ct_long, 4),
        "base_full_2020_short_delta_r": round(switch_short - ct_short, 4),
        "switch_top3_trade_pnl_share_pct": round(top3_share, 2),
        "switch_best_year_pnl_share_pct": round(best_year_share, 2),
        "switch_best_month_pnl_share_pct": round(best_month_share, 2),
        "pass_head_to_head_geo": pass_head_to_head_geo,
        "pass_base_pf_guard": pass_base_pf_guard,
        "pass_base_max_dd_guard": pass_base_max_dd_guard,
        "pass_stress_x2_positive": pass_stress_x2_positive,
        "pass_stress_x3_positive": pass_stress_x3_positive,
        "pass_two_year_long_guard": pass_two_year_long_guard,
        "pass_trade_floor": pass_trade_floor,
        "pass_top3_concentration": pass_top3_concentration,
        "pass_best_year_concentration": pass_best_year_concentration,
        "pass_best_month_concentration": pass_best_month_concentration,
        "status": status,
        "next_route": "promote_switch_candidate_as_new_research_baseline"
        if promoted
        else "freeze_regime_switch_and_keep_ct_block80_in_independent_pool",
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
    side_rows: list[dict[str, Any]],
    yearly_rows: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
    concentration_rows: list[dict[str, Any]],
    decision_row: dict[str, Any],
    switch_label: str,
    ct_profile: str,
) -> str:
    base_primary = [
        row for row in summary_rows if row["cost_scenario"] == "base" and row["window"] == "full_2020"
    ]
    base_primary = sorted(base_primary, key=lambda row: row["profile_kind"])
    base_side = [
        row for row in side_rows if row["cost_scenario"] == "base" and row["window"] == "full_2020"
    ]
    base_side = sorted(base_side, key=lambda row: (row["profile_kind"], row["side"]))
    base_yearly = [
        row for row in yearly_rows if row["cost_scenario"] == "base" and row["window"] == "full_2020"
    ]
    base_yearly = sorted(base_yearly, key=lambda row: (row["profile_kind"], int(row["year"])))
    concentration_sorted = sorted(concentration_rows, key=lambda row: row["profile_kind"])
    cost_sorted = sorted(cost_rows, key=lambda row: (row["profile_kind"], row["window"], row["stress_scenario"]))

    conclusion = (
        "固定切换候选通过一次性并列升格门，可作为后续研究 baseline。"
        if decision_row["status"] == "promoted_new_baseline_candidate"
        else "固定切换候选未同时通过收益/风险/稳健门槛，保留 ct_block80 为独立候选池成员。"
    )
    return "\n".join(
        [
            "# Switch Candidate vs CT Block80",
            "",
            f"- switch candidate: `{switch_label}`",
            f"- ct candidate: `{ct_profile}`",
            "- 只做一次 head-to-head；不新增 profile，不搜索切点，不做收益直接相加。",
            "",
            "## Base Full_2020 Summary",
            "",
            markdown_table(
                base_primary,
                [
                    ("profile_kind", "Kind"),
                    ("trades", "Trades"),
                    ("profit_factor", "PF"),
                    ("cum_r", "Cum R"),
                    ("max_dd_r", "MaxDD R"),
                    ("geometric_return_pct", "Geom Return %"),
                    ("cagr_pct", "CAGR %"),
                ],
            ),
            "",
            "## Base Full_2020 Side Summary",
            "",
            markdown_table(
                base_side,
                [
                    ("profile_kind", "Kind"),
                    ("side", "Side"),
                    ("trades", "Trades"),
                    ("profit_factor", "PF"),
                    ("cum_r", "Cum R"),
                    ("geometric_return_pct", "Geom Return %"),
                ],
            ),
            "",
            "## Base Full_2020 Yearly Geometric Returns",
            "",
            markdown_table(
                base_yearly,
                [
                    ("profile_kind", "Kind"),
                    ("year", "Year"),
                    ("trades", "Trades"),
                    ("geometric_return_pct", "Geom Return %"),
                ],
            ),
            "",
            "## Cost Sensitivity",
            "",
            markdown_table(
                cost_sorted,
                [
                    ("profile_kind", "Kind"),
                    ("window", "Window"),
                    ("stress_scenario", "Stress"),
                    ("base_geometric_return_pct", "Base Geom %"),
                    ("stress_geometric_return_pct", "Stress Geom %"),
                    ("delta_geometric_return_pct", "Delta Geom %"),
                    ("base_profit_factor", "Base PF"),
                    ("stress_profit_factor", "Stress PF"),
                ],
            ),
            "",
            "## Concentration",
            "",
            markdown_table(
                concentration_sorted,
                [
                    ("profile_kind", "Kind"),
                    ("trades", "Trades"),
                    ("cum_r", "Cum R"),
                    ("top3_trade_pnl_share_pct", "Top3 Share %"),
                    ("best_year_pnl_share_pct", "Best Year Share %"),
                    ("best_month_pnl_share_pct", "Best Month Share %"),
                ],
            ),
            "",
            "## Decision",
            "",
            markdown_table(
                [decision_row],
                [
                    ("base_full_2020_switch_geometric_return_pct", "Switch Geom %"),
                    ("base_full_2020_ct_geometric_return_pct", "CT Geom %"),
                    ("base_full_2020_switch_cagr_pct", "Switch CAGR %"),
                    ("base_full_2020_ct_cagr_pct", "CT CAGR %"),
                    ("base_full_2020_switch_profit_factor", "Switch PF"),
                    ("base_full_2020_ct_profit_factor", "CT PF"),
                    ("base_full_2020_switch_max_dd_r", "Switch MaxDD"),
                    ("base_full_2020_ct_max_dd_r", "CT MaxDD"),
                    ("stress_x2_full_2020_switch_geometric_return_pct", "Switch StressX2 Geom %"),
                    ("stress_x3_full_2020_switch_geometric_return_pct", "Switch StressX3 Geom %"),
                    ("base_two_year_long_delta_r", "Two-Year LONG Delta R"),
                    ("status", "Status"),
                ],
            ),
            "",
            "## Conclusion",
            "",
            f"- {conclusion}",
        ]
    )


def main() -> None:
    args = parse_args()
    configure_logging()

    switch_date = utc_from_iso(args.switch_date)
    primary_start = parse_date(args.primary_start)
    secondary_start = parse_date(args.secondary_start)
    if secondary_start != switch_date:
        raise ValueError("secondary-start must equal switch-date to keep this comparison fixed-calendar.")
    if secondary_start <= primary_start:
        raise ValueError("secondary-start must be later than primary-start")
    requested_end = parse_date(args.end) if args.end else datetime.now(timezone.utc)

    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    history_service = build_service()
    shared_history = {
        args.simple_profile: history_service.prepare_enriched_history(
            exchange=args.exchange,
            market_type=args.market_type,
            symbol=args.symbol,
            strategy_profile=args.simple_profile,
            start=primary_start,
            end=requested_end,
        ),
        args.challenger_strategy_profile: history_service.prepare_enriched_history(
            exchange=args.exchange,
            market_type=args.market_type,
            symbol=args.symbol,
            strategy_profile=args.challenger_strategy_profile,
            start=primary_start,
            end=requested_end,
        ),
        args.ct_profile: history_service.prepare_enriched_history(
            exchange=args.exchange,
            market_type=args.market_type,
            symbol=args.symbol,
            strategy_profile=args.ct_profile,
            start=primary_start,
            end=requested_end,
        ),
    }
    resolved_end = resolve_end_from_history(shared_history)

    switch_case = run_regime_switch_case(
        symbol=args.symbol,
        exchange=args.exchange,
        market_type=args.market_type,
        simple_profile=args.simple_profile,
        challenger_strategy_profile=args.challenger_strategy_profile,
        challenger_overlay_profile=args.challenger_overlay_profile,
        switch_date=switch_date,
        primary_start=primary_start,
        requested_end=requested_end,
        cost_scenarios=COST_SCENARIOS,
        enriched_history={
            args.simple_profile: shared_history[args.simple_profile],
            args.challenger_strategy_profile: shared_history[args.challenger_strategy_profile],
        },
        resolved_end=resolved_end,
    )

    switch_label = (
        f"{args.simple_profile} -> {args.challenger_strategy_profile} + {args.challenger_overlay_profile}"
    )
    switch_trade_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    switch_trade_source: pd.DataFrame = switch_case["trade_frame"]
    for cost_scenario in COST_SCENARIOS:
        for window in WINDOWS:
            window_start, window_end = switch_case["windows"][window]
            mask = (
                (switch_trade_source["cost_scenario"] == cost_scenario)
                & (switch_trade_source["window"] == window)
                & (switch_trade_source["scenario_kind"] == SWITCH_SCENARIO_KIND)
            )
            frame = switch_trade_source.loc[mask].copy()
            frame["profile_kind"] = SWITCH_KIND
            frame["strategy_profile"] = SWITCH_SCENARIO_KIND
            frame["profile_label"] = switch_label
            frame = normalize_trade_frame(frame)
            switch_trade_frames.append(frame)
            summary_rows.append(
                build_summary_row_from_frame(
                    cost_scenario=cost_scenario,
                    window=window,
                    window_start=window_start,
                    window_end=window_end,
                    profile_kind=SWITCH_KIND,
                    strategy_profile=SWITCH_SCENARIO_KIND,
                    profile_label=switch_label,
                    frame=frame,
                )
            )

    ct_trade_frames: list[pd.DataFrame] = []
    for cost_scenario, overrides in COST_SCENARIOS.items():
        ct_service = make_ct_service(assumption_overrides=overrides)
        for window in WINDOWS:
            start = secondary_start if window == "two_year" else primary_start
            report = run_ct_window_with_retry(
                service=ct_service,
                symbol=args.symbol,
                strategy_profile=args.ct_profile,
                exchange=args.exchange,
                market_type=args.market_type,
                start=start,
                end=resolved_end,
            )
            frame = trades_frame(
                cost_scenario=cost_scenario,
                window=window,
                window_start=start,
                window_end=resolved_end,
                profile_kind=CT_KIND,
                strategy_profile=args.ct_profile,
                profile_label=args.ct_profile,
                trades=[asdict(item) for item in report.trades],
            )
            frame = normalize_trade_frame(frame)
            ct_trade_frames.append(frame)
            summary_rows.append(
                build_summary_row_from_frame(
                    cost_scenario=cost_scenario,
                    window=window,
                    window_start=start,
                    window_end=resolved_end,
                    profile_kind=CT_KIND,
                    strategy_profile=args.ct_profile,
                    profile_label=args.ct_profile,
                    frame=frame,
                )
            )

    switch_trades_all = pd.concat(switch_trade_frames, ignore_index=True) if switch_trade_frames else pd.DataFrame()
    ct_trades_all = pd.concat(ct_trade_frames, ignore_index=True) if ct_trade_frames else pd.DataFrame()
    trades_all = pd.concat([switch_trades_all, ct_trades_all], ignore_index=True)

    side_rows = build_side_summary(trades_all)
    yearly_rows = build_yearly_geometric_returns(trades_all)
    cost_rows = build_cost_sensitivity(summary_rows)
    concentration_rows = build_concentration_summary(trades_all)
    decision_row = build_comparison_decision(
        summary_rows=summary_rows,
        side_rows=side_rows,
        concentration_rows=concentration_rows,
        switch_label=switch_label,
        ct_profile=args.ct_profile,
    )

    report = build_report(
        summary_rows=summary_rows,
        side_rows=side_rows,
        yearly_rows=yearly_rows,
        cost_rows=cost_rows,
        concentration_rows=concentration_rows,
        decision_row=decision_row,
        switch_label=switch_label,
        ct_profile=args.ct_profile,
    )

    write_csv(output_dir / "summary_all.csv", summary_rows)
    write_csv(output_dir / "side_summary_all.csv", side_rows)
    write_csv(output_dir / "yearly_geometric_returns.csv", yearly_rows)
    write_csv(output_dir / "cost_sensitivity.csv", cost_rows)
    write_csv(output_dir / "concentration_summary.csv", concentration_rows)
    write_csv(output_dir / "comparison_decision.csv", [decision_row])
    switch_trades_all.to_csv(output_dir / "switch_trades.csv", index=False)
    ct_trades_all.to_csv(output_dir / "ct_block80_trades.csv", index=False)
    (output_dir / "report.md").write_text(report + "\n", encoding="utf-8")
    (output_dir / "meta.json").write_text(
        json.dumps(
            {
                "switch_candidate_label": switch_label,
                "ct_profile": args.ct_profile,
                "switch_date": switch_date.date().isoformat(),
                "resolved_end": resolved_end.isoformat(),
                "cost_scenarios": COST_SCENARIOS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved switch candidate vs ct_block80 artifacts to {output_dir}")


if __name__ == "__main__":
    main()
