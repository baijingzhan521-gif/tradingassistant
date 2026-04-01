from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
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
from scripts.run_simple_candidate_v2_regime_switch_fixed_calendar import (
    DEFAULT_SWITCH_DATE,
    run_regime_switch_case,
)
from scripts.run_switch_axis_band_management_one_rule_gate import attach_axis_band_risk_group


SYMBOL = "BTC/USDT:USDT"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "backtests" / "switch_portfolio_blend_one_rule_gate"
SWITCH_SCENARIO_KIND = "switch_simple_candidate_v2_then_challenger_managed"

DEFAULT_SIMPLE_PROFILE = "swing_trend_simple_candidate_v2"
DEFAULT_CHALLENGER_STRATEGY_PROFILE = "swing_trend_long_regime_short_no_reversal_no_aux_v1"
DEFAULT_CHALLENGER_OVERLAY_PROFILE = "be_if_no_extension_within_3bars_after_tp1_and_veto_hold_at_trend_96_98"
DEFAULT_CT_PROFILE = "swing_exhaustion_divergence_ct_block80_v1_btc"

WINDOWS = ("full_2020", "two_year")
COST_SCENARIOS = {
    "base": {},
    "stress_x2": {"taker_fee_bps": 10.0, "slippage_bps": 4.0},
    "stress_x3": {"taker_fee_bps": 15.0, "slippage_bps": 6.0},
}

BASE_SIDE_MULTIPLIER = {"LONG": 1.15, "SHORT": 0.79}
BASE_AXIS_ACTIVE_LONG_K = 1.15

PRESETS: dict[str, dict[str, float]] = {
    "flat_switch100_ct0": {"switch_weight": 1.00, "ct_weight": 0.00},
    "switch95_ct05": {"switch_weight": 0.95, "ct_weight": 0.05},
    "switch90_ct10": {"switch_weight": 0.90, "ct_weight": 0.10},
    "switch85_ct15": {"switch_weight": 0.85, "ct_weight": 0.15},
}
BASELINE_PRESET = "flat_switch100_ct0"

M1_PF_TOLERANCE = 0.05
M1_DD_TOLERANCE_R = 1.5
LONG_GUARD_TOLERANCE_R = 2.0
TOP3_SHARE_MAX = 65.0
BEST_YEAR_SHARE_MAX = 80.0
SATISFACTION_CAGR_TARGET = 22.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-rule portfolio blend gate: switch baseline + ct_block80.")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--simple-profile", default=DEFAULT_SIMPLE_PROFILE)
    parser.add_argument("--challenger-strategy-profile", default=DEFAULT_CHALLENGER_STRATEGY_PROFILE)
    parser.add_argument("--challenger-overlay-profile", default=DEFAULT_CHALLENGER_OVERLAY_PROFILE)
    parser.add_argument("--switch-date", default=DEFAULT_SWITCH_DATE)
    parser.add_argument("--ct-profile", default=DEFAULT_CT_PROFILE)
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


def validate_preset_weights(presets: dict[str, dict[str, float]]) -> None:
    if BASELINE_PRESET not in presets:
        raise ValueError(f"Missing baseline preset: {BASELINE_PRESET}")
    for name, cfg in presets.items():
        switch_weight = float(cfg["switch_weight"])
        ct_weight = float(cfg["ct_weight"])
        if switch_weight < 0.0 or ct_weight < 0.0:
            raise ValueError(f"Negative sleeve weight is not allowed: {name}")
        if abs((switch_weight + ct_weight) - 1.0) > 1e-9:
            raise ValueError(
                f"Invalid sleeve weights for {name}: switch_weight + ct_weight must equal 1.0"
            )


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
    sort_keys = [key for key in ("cost_scenario", "window", "preset", "entry_time") if key in result.columns]
    if sort_keys:
        result = result.sort_values(sort_keys)
    return result.reset_index(drop=True)


def build_switch_managed_trades(trade_frame: pd.DataFrame) -> pd.DataFrame:
    if trade_frame.empty:
        result = trade_frame.copy()
        result["switch_size_multiplier"] = 0.0
        result["switch_weighted_pnl_r"] = 0.0
        result["switch_weighted_pnl_pct"] = 0.0
        return result

    result = trade_frame.copy()
    raw_size = result["side"].map(BASE_SIDE_MULTIPLIER).astype(float)
    long_active = (result["side"] == "LONG") & (result["risk_group"] == "active")
    raw_size = raw_size * long_active.map({True: float(BASE_AXIS_ACTIVE_LONG_K), False: 1.0}).astype(float)

    norm_scope = result[(result["cost_scenario"] == "base") & (result["window"] == "full_2020")].index
    raw_avg = float(raw_size.loc[norm_scope].mean()) if len(norm_scope) else 1.0
    normalize_scale = (1.0 / raw_avg) if raw_avg > 0.0 else 1.0
    size_multiplier = raw_size * normalize_scale

    result["switch_size_multiplier"] = size_multiplier
    result["switch_weighted_pnl_r"] = result["pnl_r"].astype(float) * size_multiplier
    result["switch_weighted_pnl_pct"] = result["pnl_pct"].astype(float) * size_multiplier
    return result


def trades_to_frame(
    *,
    trades: list[Any],
    cost_scenario: str,
    window: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade in trades:
        row = asdict(trade)
        row["cost_scenario"] = cost_scenario
        row["window"] = window
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in ("signal_time", "entry_time", "exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def build_blended_trades(
    *,
    switch_managed: pd.DataFrame,
    ct_trades: pd.DataFrame,
    preset: str,
) -> pd.DataFrame:
    cfg = PRESETS[preset]
    switch_weight = float(cfg["switch_weight"])
    ct_weight = float(cfg["ct_weight"])

    frames: list[pd.DataFrame] = []

    if not switch_managed.empty:
        switch_frame = switch_managed.copy()
        switch_frame["preset"] = preset
        switch_frame["source"] = "switch"
        switch_frame["source_weight"] = switch_weight
        switch_frame["size_multiplier"] = switch_frame["switch_size_multiplier"].astype(float) * switch_weight
        switch_frame["weighted_pnl_r"] = switch_frame["switch_weighted_pnl_r"].astype(float) * switch_weight
        switch_frame["weighted_pnl_pct"] = switch_frame["switch_weighted_pnl_pct"].astype(float) * switch_weight
        frames.append(switch_frame)

    if not ct_trades.empty:
        ct_frame = ct_trades.copy()
        ct_frame["preset"] = preset
        ct_frame["source"] = "ct_block80"
        ct_frame["source_weight"] = ct_weight
        ct_frame["size_multiplier"] = ct_weight
        ct_frame["weighted_pnl_r"] = ct_frame["pnl_r"].astype(float) * ct_weight
        ct_frame["weighted_pnl_pct"] = ct_frame["pnl_pct"].astype(float) * ct_weight
        frames.append(ct_frame)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return normalize_trade_frame(combined)


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
    preset: str,
) -> dict[str, Any]:
    trades = int(len(frame))
    return {
        "cost_scenario": cost_scenario,
        "window": window,
        "window_start": window_start.date().isoformat(),
        "window_end": window_end.date().isoformat(),
        "preset": preset,
        "trades": trades,
        "avg_size": round(float(frame["size_multiplier"].mean()), 4) if trades else 0.0,
        "switch_weight": round(float(PRESETS[preset]["switch_weight"]), 4),
        "ct_weight": round(float(PRESETS[preset]["ct_weight"]), 4),
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
    grouped = weighted_trades.groupby(["cost_scenario", "window", "preset", "side"], sort=True, observed=True)
    for (cost_scenario, window, preset, side), group in grouped:
        rows.append(
            {
                "cost_scenario": cost_scenario,
                "window": window,
                "preset": preset,
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
    grouped = weighted_trades.groupby(["cost_scenario", "window", "preset", "year"], sort=True, observed=True)
    for (cost_scenario, window, preset, year), group in grouped:
        rows.append(
            {
                "cost_scenario": cost_scenario,
                "window": window,
                "preset": preset,
                "year": int(year),
                "trades": int(len(group)),
                "geometric_return_pct": _geometric_return_pct(group),
                "additive_return_pct": round(float(group["weighted_pnl_pct"].sum()), 4),
            }
        )
    return rows


def build_cost_sensitivity(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(summary_rows)
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(["window", "preset"], sort=True, observed=True)
    for (window, preset), group in grouped:
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
                    "preset": preset,
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
    grouped = weighted_trades.groupby(["cost_scenario", "window", "preset"], sort=True, observed=True)
    for (cost_scenario, window, preset), group in grouped:
        positive = group.loc[group["weighted_pnl_r"] > 0.0, "weighted_pnl_r"].sort_values(ascending=False)
        total_positive = float(positive.sum())
        top3_positive = float(positive.head(3).sum())
        top3_share = (top3_positive / total_positive * 100.0) if total_positive > 0.0 else 100.0

        yearly_sub = yearly[
            (yearly["cost_scenario"] == cost_scenario)
            & (yearly["window"] == window)
            & (yearly["preset"] == preset)
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
                "preset": preset,
                "trades": int(len(group)),
                "avg_size": round(float(group["size_multiplier"].mean()), 4),
                "top3_trades_pnl_share_pct": round(top3_share, 4),
                "best_year_geometric_pct": round(best_year_geo, 4),
                "best_year_geometric_pct_share": round(best_year_share, 4),
            }
        )
    return rows


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
        flat_oos = oos[oos["preset"] == BASELINE_PRESET].copy()
        flat_summary = summarize_weighted_frame(
            frame=flat_oos,
            cost_scenario="base",
            window="oos_fold",
            window_start=fold["test_start"],
            window_end=fold["test_end"],
            preset=BASELINE_PRESET,
        )
        for preset in PRESETS:
            preset_oos = oos[oos["preset"] == preset].copy()
            preset_summary = summarize_weighted_frame(
                frame=preset_oos,
                cost_scenario="base",
                window="oos_fold",
                window_start=fold["test_start"],
                window_end=fold["test_end"],
                preset=preset,
            )
            rows.append(
                {
                    "fold": int(fold["fold"]),
                    "train_start": fold["train_start"].date().isoformat(),
                    "train_end": fold["train_end"].date().isoformat(),
                    "test_start": fold["test_start"].date().isoformat(),
                    "test_end": fold["test_end"].date().isoformat(),
                    "preset": preset,
                    "trades": int(preset_summary["trades"]),
                    "avg_size": round(float(preset_summary["avg_size"]), 4),
                    "profit_factor": round(float(preset_summary["profit_factor"]), 4),
                    "cum_r": round(float(preset_summary["cum_r"]), 4),
                    "max_dd_r": round(float(preset_summary["max_dd_r"]), 4),
                    "geometric_return_pct": round(float(preset_summary["geometric_return_pct"]), 4),
                    "cagr_pct": round(float(preset_summary["cagr_pct"]), 4),
                    "flat_geometric_return_pct": round(float(flat_summary["geometric_return_pct"]), 4),
                    "delta_vs_flat_geometric_return_pct": round(
                        float(preset_summary["geometric_return_pct"] - flat_summary["geometric_return_pct"]), 4
                    ),
                }
            )
    return rows


def build_oos_summary(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(fold_rows)
    if frame.empty:
        return [
            {
                "preset": preset,
                "folds": 0,
                "trades": 0,
                "geometric_return_pct": 0.0,
                "cagr_proxy_pct": 0.0,
                "cum_r_sum": 0.0,
                "profit_factor_proxy": 0.0,
                "max_dd_r_avg": 0.0,
                "positive_fold_ratio": 0.0,
                "oos_available": False,
            }
            for preset in PRESETS
        ]
    rows: list[dict[str, Any]] = []
    for preset, group in frame.groupby("preset", sort=True, observed=True):
        geo_factor = float((1.0 + group["geometric_return_pct"] / 100.0).prod())
        cagr_factor = float((1.0 + group["cagr_pct"] / 100.0).prod())
        total_trades = int(group["trades"].sum())
        pf_proxy = (
            float((group["profit_factor"] * group["trades"]).sum()) / max(total_trades, 1)
            if total_trades
            else 0.0
        )
        rows.append(
            {
                "preset": preset,
                "folds": int(len(group)),
                "trades": total_trades,
                "geometric_return_pct": round((geo_factor - 1.0) * 100.0, 4),
                "cagr_proxy_pct": round((cagr_factor - 1.0) * 100.0, 4),
                "cum_r_sum": round(float(group["cum_r"].sum()), 4),
                "profit_factor_proxy": round(pf_proxy, 4),
                "max_dd_r_avg": round(float(group["max_dd_r"].mean()), 4),
                "positive_fold_ratio": round(float((group["cum_r"] > 0.0).mean()), 4),
                "oos_available": True,
            }
        )
    return rows


def fetch_btc_yearly_returns(*, symbol: str, start: datetime, end: datetime) -> dict[int, float]:
    service = OhlcvService(get_exchange_client_factory())
    frame = service.fetch_ohlcv_range(
        exchange="binance",
        market_type="perpetual",
        symbol=symbol,
        timeframe="1d",
        start=start,
        end=end + timedelta(days=1),
    )
    frame["year"] = frame["timestamp"].dt.year.astype(int)
    results: dict[int, float] = {}
    for year, group in frame.groupby("year", sort=True):
        first_close = float(group.iloc[0]["close"])
        last_close = float(group.iloc[-1]["close"])
        ret_pct = (last_close / first_close - 1.0) * 100.0
        results[int(year)] = round(ret_pct, 4)
    return results


def build_btc_yearly_comparison(
    *,
    yearly_rows: list[dict[str, Any]],
    btc_yearly_returns: dict[int, float],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    frame = pd.DataFrame(yearly_rows)
    if frame.empty:
        return [], {preset: 0 for preset in PRESETS}
    subset = frame[(frame["cost_scenario"] == "base") & (frame["window"] == "full_2020")].copy()
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for preset in PRESETS:
        preset_rows = subset[subset["preset"] == preset].copy().sort_values("year")
        win_count = 0
        total_count = 0
        for _, row in preset_rows.iterrows():
            year = int(row["year"])
            if year not in btc_yearly_returns:
                continue
            strategy_geo = float(row["geometric_return_pct"])
            btc_ret = float(btc_yearly_returns[year])
            outperform = strategy_geo > btc_ret
            if outperform:
                win_count += 1
            total_count += 1
            rows.append(
                {
                    "row_type": "year",
                    "preset": preset,
                    "year": year,
                    "strategy_geometric_return_pct": round(strategy_geo, 4),
                    "btc_buy_hold_return_pct": round(btc_ret, 4),
                    "delta_strategy_minus_btc_pct": round(strategy_geo - btc_ret, 4),
                    "outperform_btc": outperform,
                    "outperform_years": "",
                    "total_years": "",
                    "outperform_rate_pct": "",
                }
            )
        counts[preset] = win_count
        rows.append(
            {
                "row_type": "aggregate",
                "preset": preset,
                "year": "aggregate",
                "strategy_geometric_return_pct": "",
                "btc_buy_hold_return_pct": "",
                "delta_strategy_minus_btc_pct": "",
                "outperform_btc": "",
                "outperform_years": int(win_count),
                "total_years": int(total_count),
                "outperform_rate_pct": round((win_count / total_count) * 100.0, 2) if total_count else 0.0,
            }
        )
    return rows, counts


def _lookup(
    rows: list[dict[str, Any]],
    *,
    cost_scenario: str | None = None,
    window: str | None = None,
    preset: str | None = None,
    side: str | None = None,
) -> dict[str, Any]:
    for row in rows:
        if cost_scenario is not None and row.get("cost_scenario") != cost_scenario:
            continue
        if window is not None and row.get("window") != window:
            continue
        if preset is not None and row.get("preset") != preset:
            continue
        if side is not None and row.get("side") != side:
            continue
        return row
    raise KeyError(
        f"missing row cost_scenario={cost_scenario}, window={window}, preset={preset}, side={side}"
    )


def build_promotion_decision(
    *,
    summary_rows: list[dict[str, Any]],
    side_rows: list[dict[str, Any]],
    concentration_rows: list[dict[str, Any]],
    oos_summary_rows: list[dict[str, Any]],
    btc_outperform_counts: dict[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    flat_base_primary = _lookup(summary_rows, cost_scenario="base", window="full_2020", preset=BASELINE_PRESET)
    flat_two_year_long = _lookup(side_rows, cost_scenario="base", window="two_year", preset=BASELINE_PRESET, side="LONG")
    flat_oos = _lookup(oos_summary_rows, preset=BASELINE_PRESET)
    flat_btc_years = int(btc_outperform_counts.get(BASELINE_PRESET, 0))

    candidate_rows: list[dict[str, Any]] = []
    for preset in PRESETS:
        if preset == BASELINE_PRESET:
            continue
        base_primary = _lookup(summary_rows, cost_scenario="base", window="full_2020", preset=preset)
        stress_x2 = _lookup(summary_rows, cost_scenario="stress_x2", window="full_2020", preset=preset)
        stress_x3 = _lookup(summary_rows, cost_scenario="stress_x3", window="full_2020", preset=preset)
        two_year_long = _lookup(side_rows, cost_scenario="base", window="two_year", preset=preset, side="LONG")
        concentration = _lookup(concentration_rows, cost_scenario="base", window="full_2020", preset=preset)
        oos = _lookup(oos_summary_rows, preset=preset)
        candidate_btc_years = int(btc_outperform_counts.get(preset, 0))

        pass_base_geo = float(base_primary["geometric_return_pct"]) > float(flat_base_primary["geometric_return_pct"])
        pass_base_cagr = float(base_primary["cagr_pct"]) > float(flat_base_primary["cagr_pct"])
        pass_base_pf = float(base_primary["profit_factor"]) >= float(flat_base_primary["profit_factor"]) - M1_PF_TOLERANCE
        pass_base_max_dd = float(base_primary["max_dd_r"]) <= float(flat_base_primary["max_dd_r"]) + M1_DD_TOLERANCE_R
        pass_stress_x2_geo = float(stress_x2["geometric_return_pct"]) > 0.0
        pass_stress_x2_pf = float(stress_x2["profit_factor"]) >= 1.0
        pass_stress_x3_geo = float(stress_x3["geometric_return_pct"]) > 0.0
        pass_stress_x3_pf = float(stress_x3["profit_factor"]) >= 1.0
        pass_long_guard = float(two_year_long["cum_r"]) >= float(flat_two_year_long["cum_r"]) - LONG_GUARD_TOLERANCE_R
        pass_top3 = float(concentration["top3_trades_pnl_share_pct"]) <= TOP3_SHARE_MAX
        pass_best_year = float(concentration["best_year_geometric_pct_share"]) <= BEST_YEAR_SHARE_MAX
        pass_oos_geo = bool(oos.get("oos_available", False)) and (
            float(oos["geometric_return_pct"]) - float(flat_oos["geometric_return_pct"])
        ) >= 0.0
        pass_target_cagr_22 = float(base_primary["cagr_pct"]) >= SATISFACTION_CAGR_TARGET
        pass_btc_non_regression = candidate_btc_years >= flat_btc_years

        all_pass_m1 = (
            pass_base_geo
            and pass_base_cagr
            and pass_base_pf
            and pass_base_max_dd
            and pass_stress_x2_geo
            and pass_stress_x2_pf
            and pass_stress_x3_geo
            and pass_stress_x3_pf
            and pass_long_guard
            and pass_top3
            and pass_best_year
            and pass_oos_geo
        )
        promoted = all_pass_m1 and pass_target_cagr_22 and pass_btc_non_regression

        candidate_rows.append(
            {
                "preset": preset,
                "base_geometric_return_pct": round(float(base_primary["geometric_return_pct"]), 4),
                "base_cagr_pct": round(float(base_primary["cagr_pct"]), 4),
                "base_profit_factor": round(float(base_primary["profit_factor"]), 4),
                "base_max_dd_r": round(float(base_primary["max_dd_r"]), 4),
                "stress_x2_geometric_return_pct": round(float(stress_x2["geometric_return_pct"]), 4),
                "stress_x2_profit_factor": round(float(stress_x2["profit_factor"]), 4),
                "stress_x3_geometric_return_pct": round(float(stress_x3["geometric_return_pct"]), 4),
                "stress_x3_profit_factor": round(float(stress_x3["profit_factor"]), 4),
                "two_year_long_cum_r": round(float(two_year_long["cum_r"]), 4),
                "top3_trades_pnl_share_pct": round(float(concentration["top3_trades_pnl_share_pct"]), 4),
                "best_year_geometric_pct_share": round(float(concentration["best_year_geometric_pct_share"]), 4),
                "oos_geometric_return_pct": round(float(oos["geometric_return_pct"]), 4),
                "oos_delta_vs_flat_geometric_return_pct": round(
                    float(oos["geometric_return_pct"] - flat_oos["geometric_return_pct"]), 4
                ),
                "candidate_btc_outperform_years": candidate_btc_years,
                "flat_btc_outperform_years": flat_btc_years,
                "pass_base_geo": pass_base_geo,
                "pass_base_cagr": pass_base_cagr,
                "pass_base_pf": pass_base_pf,
                "pass_base_max_dd": pass_base_max_dd,
                "pass_stress_x2_geo": pass_stress_x2_geo,
                "pass_stress_x2_pf": pass_stress_x2_pf,
                "pass_stress_x3_geo": pass_stress_x3_geo,
                "pass_stress_x3_pf": pass_stress_x3_pf,
                "pass_long_guard": pass_long_guard,
                "pass_top3": pass_top3,
                "pass_best_year": pass_best_year,
                "pass_oos_geo_non_negative": pass_oos_geo,
                "pass_target_cagr_22": pass_target_cagr_22,
                "pass_btc_diagnostic_non_regression": pass_btc_non_regression,
                "all_pass_m1": all_pass_m1,
                "all_pass": promoted,
            }
        )

    ranking = sorted(
        candidate_rows,
        key=lambda row: (
            bool(row["all_pass"]),
            bool(row["all_pass_m1"]),
            float(row["base_geometric_return_pct"]),
            float(row["base_cagr_pct"]),
            float(row["base_profit_factor"]),
        ),
        reverse=True,
    )
    chosen = ranking[0]
    status = "promoted_management_overlay_candidate" if bool(chosen["all_pass"]) else "rejected_management_overlay"

    decision = {
        "baseline_preset": BASELINE_PRESET,
        "chosen_candidate_preset": chosen["preset"],
        "base_full_2020_flat_geometric_return_pct": round(float(flat_base_primary["geometric_return_pct"]), 4),
        "base_full_2020_candidate_geometric_return_pct": round(float(chosen["base_geometric_return_pct"]), 4),
        "base_full_2020_flat_cagr_pct": round(float(flat_base_primary["cagr_pct"]), 4),
        "base_full_2020_candidate_cagr_pct": round(float(chosen["base_cagr_pct"]), 4),
        "base_full_2020_flat_profit_factor": round(float(flat_base_primary["profit_factor"]), 4),
        "base_full_2020_candidate_profit_factor": round(float(chosen["base_profit_factor"]), 4),
        "base_full_2020_flat_max_dd_r": round(float(flat_base_primary["max_dd_r"]), 4),
        "base_full_2020_candidate_max_dd_r": round(float(chosen["base_max_dd_r"]), 4),
        "stress_x2_candidate_geometric_return_pct": round(float(chosen["stress_x2_geometric_return_pct"]), 4),
        "stress_x3_candidate_geometric_return_pct": round(float(chosen["stress_x3_geometric_return_pct"]), 4),
        "two_year_long_delta_r": round(float(chosen["two_year_long_cum_r"] - float(flat_two_year_long["cum_r"])), 4),
        "candidate_top3_trades_pnl_share_pct": round(float(chosen["top3_trades_pnl_share_pct"]), 4),
        "candidate_best_year_geometric_pct_share": round(float(chosen["best_year_geometric_pct_share"]), 4),
        "candidate_oos_geometric_return_pct": round(float(chosen["oos_geometric_return_pct"]), 4),
        "candidate_oos_delta_vs_flat_geometric_return_pct": round(
            float(chosen["oos_delta_vs_flat_geometric_return_pct"]), 4
        ),
        "candidate_btc_outperform_years": int(chosen["candidate_btc_outperform_years"]),
        "flat_btc_outperform_years": int(flat_btc_years),
        "pass_base_geo": bool(chosen["pass_base_geo"]),
        "pass_base_cagr": bool(chosen["pass_base_cagr"]),
        "pass_base_pf": bool(chosen["pass_base_pf"]),
        "pass_base_max_dd": bool(chosen["pass_base_max_dd"]),
        "pass_stress_x2_geo": bool(chosen["pass_stress_x2_geo"]),
        "pass_stress_x2_pf": bool(chosen["pass_stress_x2_pf"]),
        "pass_stress_x3_geo": bool(chosen["pass_stress_x3_geo"]),
        "pass_stress_x3_pf": bool(chosen["pass_stress_x3_pf"]),
        "pass_long_guard": bool(chosen["pass_long_guard"]),
        "pass_top3_concentration": bool(chosen["pass_top3"]),
        "pass_best_year_concentration": bool(chosen["pass_best_year"]),
        "pass_oos_geo_non_negative": bool(chosen["pass_oos_geo_non_negative"]),
        "pass_target_cagr_22": bool(chosen["pass_target_cagr_22"]),
        "pass_btc_diagnostic_non_regression": bool(chosen["pass_btc_diagnostic_non_regression"]),
        "all_pass_m1": bool(chosen["all_pass_m1"]),
        "status": status,
        "next_route": (
            "promote_as_portfolio_blend_baseline_candidate"
            if status == "promoted_management_overlay_candidate"
            else "freeze_portfolio_blend_one_rule_after_single_round"
        ),
    }
    return decision, candidate_rows


def build_oos_selected_vs_flat(
    *,
    fold_rows: list[dict[str, Any]],
    chosen_candidate_preset: str,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(fold_rows)
    if frame.empty:
        return []
    selected = frame[frame["preset"] == chosen_candidate_preset].copy()
    flat = frame[frame["preset"] == BASELINE_PRESET].copy()
    merged = selected.merge(
        flat[
            [
                "fold",
                "geometric_return_pct",
                "cagr_pct",
                "profit_factor",
                "cum_r",
                "max_dd_r",
                "trades",
            ]
        ],
        on="fold",
        suffixes=("_candidate", "_flat"),
        how="inner",
    )
    rows: list[dict[str, Any]] = []
    for _, row in merged.sort_values("fold").iterrows():
        rows.append(
            {
                "fold": int(row["fold"]),
                "candidate_preset": chosen_candidate_preset,
                "candidate_trades": int(row["trades_candidate"]),
                "flat_trades": int(row["trades_flat"]),
                "candidate_geometric_return_pct": round(float(row["geometric_return_pct_candidate"]), 4),
                "flat_geometric_return_pct": round(float(row["geometric_return_pct_flat"]), 4),
                "delta_geometric_return_pct": round(
                    float(row["geometric_return_pct_candidate"] - row["geometric_return_pct_flat"]), 4
                ),
                "candidate_cagr_pct": round(float(row["cagr_pct_candidate"]), 4),
                "flat_cagr_pct": round(float(row["cagr_pct_flat"]), 4),
                "candidate_profit_factor": round(float(row["profit_factor_candidate"]), 4),
                "flat_profit_factor": round(float(row["profit_factor_flat"]), 4),
                "candidate_cum_r": round(float(row["cum_r_candidate"]), 4),
                "flat_cum_r": round(float(row["cum_r_flat"]), 4),
                "candidate_max_dd_r": round(float(row["max_dd_r_candidate"]), 4),
                "flat_max_dd_r": round(float(row["max_dd_r_flat"]), 4),
            }
        )

    base_rows = [row for row in rows if isinstance(row["fold"], int)]
    if base_rows:
        geo_factor_candidate = math.prod(1.0 + float(item["candidate_geometric_return_pct"]) / 100.0 for item in base_rows)
        geo_factor_flat = math.prod(1.0 + float(item["flat_geometric_return_pct"]) / 100.0 for item in base_rows)
        rows.append(
            {
                "fold": "aggregate",
                "candidate_preset": chosen_candidate_preset,
                "candidate_trades": int(sum(int(item["candidate_trades"]) for item in base_rows)),
                "flat_trades": int(sum(int(item["flat_trades"]) for item in base_rows)),
                "candidate_geometric_return_pct": round((geo_factor_candidate - 1.0) * 100.0, 4),
                "flat_geometric_return_pct": round((geo_factor_flat - 1.0) * 100.0, 4),
                "delta_geometric_return_pct": round((geo_factor_candidate - geo_factor_flat) * 100.0, 4),
                "candidate_cagr_pct": "",
                "flat_cagr_pct": "",
                "candidate_profit_factor": "",
                "flat_profit_factor": "",
                "candidate_cum_r": round(float(sum(float(item["candidate_cum_r"]) for item in base_rows)), 4),
                "flat_cum_r": round(float(sum(float(item["flat_cum_r"]) for item in base_rows)), 4),
                "candidate_max_dd_r": "",
                "flat_max_dd_r": "",
            }
        )
    return rows


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
    candidate_eval_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    oos_selected_vs_flat_rows: list[dict[str, Any]],
    decision_row: dict[str, Any],
) -> str:
    base_full = [
        row for row in summary_rows if row["cost_scenario"] == "base" and row["window"] == "full_2020"
    ]
    base_full = sorted(base_full, key=lambda row: float(row["geometric_return_pct"]), reverse=True)
    fold_preview = sorted(fold_rows, key=lambda row: (int(row["fold"]), row["preset"]))[:20]
    conclusion = (
        "候选通过 M1 + CAGR>=22% + BTC 不退步，可作为新的组合层候选 baseline。"
        if decision_row["status"] == "promoted_management_overlay_candidate"
        else "组合层 one-rule 本轮不通过，按纪律冻结该分支，转下一正交家族。"
    )
    return "\n".join(
        [
            "# Switch Portfolio Blend One-Rule Gate",
            "",
            "- baseline source: fixed switch (`simple_candidate_v2 -> challenger_managed`) + fixed management layer (`long_1.15_short_0.79 + long_active_k115`)",
            "- orthogonal sleeve source: `ct_block80`",
            "- one-rule: fixed sleeve weights only, no online switching and no extra cut-point search",
            "- gate level: M1 + satisfaction (`CAGR>=22%`, BTC yearly non-regression)",
            "",
            "## Base Full_2020 Summary",
            "",
            markdown_table(
                base_full,
                [
                    ("preset", "Preset"),
                    ("switch_weight", "Switch W"),
                    ("ct_weight", "CT W"),
                    ("trades", "Trades"),
                    ("profit_factor", "PF"),
                    ("cum_r", "Cum R"),
                    ("max_dd_r", "MaxDD R"),
                    ("geometric_return_pct", "Geom %"),
                    ("cagr_pct", "CAGR %"),
                ],
            ),
            "",
            "## Candidate Evaluation",
            "",
            markdown_table(
                candidate_eval_rows,
                [
                    ("preset", "Preset"),
                    ("base_geometric_return_pct", "Base Geom %"),
                    ("base_cagr_pct", "Base CAGR %"),
                    ("base_profit_factor", "Base PF"),
                    ("base_max_dd_r", "Base MaxDD R"),
                    ("oos_delta_vs_flat_geometric_return_pct", "OOS ΔGeom %"),
                    ("candidate_btc_outperform_years", "BTC Years"),
                    ("pass_target_cagr_22", "CAGR>=22"),
                    ("all_pass_m1", "M1 Pass"),
                    ("all_pass", "All Pass"),
                ],
            ),
            "",
            "## Fold Panel Preview",
            "",
            markdown_table(
                fold_preview,
                [
                    ("fold", "Fold"),
                    ("preset", "Preset"),
                    ("trades", "Trades"),
                    ("geometric_return_pct", "Geom %"),
                    ("delta_vs_flat_geometric_return_pct", "ΔGeom vs Flat"),
                ],
            ),
            "",
            "## OOS Selected vs Flat",
            "",
            markdown_table(
                oos_selected_vs_flat_rows,
                [
                    ("fold", "Fold"),
                    ("candidate_geometric_return_pct", "Candidate Geom %"),
                    ("flat_geometric_return_pct", "Flat Geom %"),
                    ("delta_geometric_return_pct", "ΔGeom %"),
                    ("candidate_cum_r", "Candidate Cum R"),
                    ("flat_cum_r", "Flat Cum R"),
                ],
            ),
            "",
            "## Decision",
            "",
            markdown_table(
                [decision_row],
                [
                    ("chosen_candidate_preset", "Chosen"),
                    ("base_full_2020_candidate_geometric_return_pct", "Candidate Base Geom %"),
                    ("base_full_2020_candidate_cagr_pct", "Candidate Base CAGR %"),
                    ("pass_target_cagr_22", "CAGR>=22"),
                    ("pass_btc_diagnostic_non_regression", "BTC Non-Reg"),
                    ("all_pass_m1", "M1 Pass"),
                    ("status", "Status"),
                ],
            ),
            "",
            conclusion,
        ]
    )


def main() -> None:
    args = parse_args()
    configure_logging()
    validate_preset_weights(PRESETS)

    switch_date = parse_date(args.switch_date)
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
        cost_scenarios=COST_SCENARIOS,
    )
    resolved_end: datetime = case["resolved_end"]
    windows: dict[str, tuple[datetime, datetime]] = case["windows"]

    switch_raw = case["trade_frame"]
    switch_raw = switch_raw[switch_raw["scenario_kind"] == SWITCH_SCENARIO_KIND].copy()
    switch_labeled, switch_label_coverage = attach_axis_band_risk_group(switch_raw)
    switch_managed = build_switch_managed_trades(switch_labeled)

    ct_frames: list[pd.DataFrame] = []
    for cost_scenario, overrides in COST_SCENARIOS.items():
        ct_service = make_ct_service(assumption_overrides=overrides)
        for window in WINDOWS:
            window_start, window_end = windows[window]
            report = run_ct_window_with_retry(
                service=ct_service,
                symbol=args.symbol,
                strategy_profile=args.ct_profile,
                exchange=args.exchange,
                market_type=args.market_type,
                start=window_start,
                end=window_end,
            )
            frame = trades_to_frame(trades=report.trades, cost_scenario=cost_scenario, window=window)
            ct_frames.append(frame)
    ct_trades_all = normalize_trade_frame(pd.concat(ct_frames, ignore_index=True) if ct_frames else pd.DataFrame())

    blended_frames: list[pd.DataFrame] = []
    for preset in PRESETS:
        subset_switch = switch_managed.copy()
        subset_ct = ct_trades_all.copy()
        blended_frames.append(build_blended_trades(switch_managed=subset_switch, ct_trades=subset_ct, preset=preset))
    weighted_trades = normalize_trade_frame(
        pd.concat([frame for frame in blended_frames if not frame.empty], ignore_index=True)
        if blended_frames
        else pd.DataFrame()
    )

    summary_rows: list[dict[str, Any]] = []
    for cost_scenario in COST_SCENARIOS:
        for window in WINDOWS:
            window_start, window_end = windows[window]
            for preset in PRESETS:
                subset = weighted_trades[
                    (weighted_trades["cost_scenario"] == cost_scenario)
                    & (weighted_trades["window"] == window)
                    & (weighted_trades["preset"] == preset)
                ].copy()
                summary_rows.append(
                    summarize_weighted_frame(
                        frame=subset,
                        cost_scenario=cost_scenario,
                        window=window,
                        window_start=window_start,
                        window_end=window_end,
                        preset=preset,
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
    oos_rows = build_oos_summary(fold_rows)

    btc_yearly = fetch_btc_yearly_returns(symbol=args.symbol, start=primary_start, end=resolved_end)
    btc_rows, btc_counts = build_btc_yearly_comparison(yearly_rows=yearly_rows, btc_yearly_returns=btc_yearly)

    decision_row, candidate_eval_rows = build_promotion_decision(
        summary_rows=summary_rows,
        side_rows=side_rows,
        concentration_rows=concentration_rows,
        oos_summary_rows=oos_rows,
        btc_outperform_counts=btc_counts,
    )
    oos_selected_vs_flat_rows = build_oos_selected_vs_flat(
        fold_rows=fold_rows,
        chosen_candidate_preset=str(decision_row["chosen_candidate_preset"]),
    )

    report = build_report(
        summary_rows=summary_rows,
        candidate_eval_rows=candidate_eval_rows,
        fold_rows=fold_rows,
        oos_selected_vs_flat_rows=oos_selected_vs_flat_rows,
        decision_row=decision_row,
    )

    write_csv(output_dir / "full_summary.csv", summary_rows)
    write_csv(output_dir / "side_summary.csv", side_rows)
    write_csv(output_dir / "yearly_geometric_returns.csv", yearly_rows)
    write_csv(output_dir / "btc_yearly_comparison.csv", btc_rows)
    write_csv(output_dir / "cost_sensitivity.csv", cost_rows)
    write_csv(output_dir / "fold_panel.csv", fold_rows)
    write_csv(output_dir / "oos_selected_vs_flat.csv", oos_selected_vs_flat_rows)
    write_csv(output_dir / "trade_concentration.csv", concentration_rows)
    write_csv(output_dir / "promotion_decision.csv", [decision_row])
    (output_dir / "report.md").write_text(report + "\n", encoding="utf-8")

    (output_dir / "meta.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "symbol": args.symbol,
                "switch_date": switch_date.date().isoformat(),
                "primary_start": primary_start.date().isoformat(),
                "secondary_start": secondary_start.date().isoformat(),
                "resolved_end": resolved_end.date().isoformat(),
                "simple_profile": args.simple_profile,
                "challenger_strategy_profile": args.challenger_strategy_profile,
                "challenger_overlay_profile": args.challenger_overlay_profile,
                "ct_profile": args.ct_profile,
                "presets": PRESETS,
                "cost_scenarios": COST_SCENARIOS,
                "switch_label_coverage": round(float(switch_label_coverage), 6),
                "m1_pf_tolerance": M1_PF_TOLERANCE,
                "m1_dd_tolerance_r": M1_DD_TOLERANCE_R,
                "long_guard_tolerance_r": LONG_GUARD_TOLERANCE_R,
                "concentration_top3_max": TOP3_SHARE_MAX,
                "concentration_best_year_max": BEST_YEAR_SHARE_MAX,
                "satisfaction_cagr_target": SATISFACTION_CAGR_TARGET,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"saved switch portfolio blend one-rule gate artifacts to {output_dir}")


if __name__ == "__main__":
    main()
