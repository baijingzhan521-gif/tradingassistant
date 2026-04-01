from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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
from app.schemas.common import Bias
from app.services.strategy_service import StrategyService
from scripts.run_simple_candidate_v2_regime_switch_fixed_calendar import (
    DEFAULT_SWITCH_DATE,
    run_regime_switch_case,
)


SYMBOL = "BTC/USDT:USDT"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "backtests" / "switch_trend_strength_management_one_rule_gate"
SWITCH_SCENARIO_KIND = "switch_simple_candidate_v2_then_challenger_managed"
LABEL_PROFILE = "swing_trend_axis_band_state_note_v1"

WINDOWS = ("full_2020", "two_year")
COST_SCENARIOS = {
    "base": {},
    "stress_x2": {"taker_fee_bps": 10.0, "slippage_bps": 4.0},
    "stress_x3": {"taker_fee_bps": 15.0, "slippage_bps": 6.0},
}

BASE_SIDE_MULTIPLIER = {"LONG": 1.15, "SHORT": 0.79}
BASE_AXIS_ACTIVE_LONG_K = 1.15
TREND_STRENGTH_THRESHOLD = 95
PRESETS: dict[str, dict[str, Any]] = {
    "flat_control": {"trend_long_k": 1.0},
    "long_trend95_k110": {"trend_long_k": 1.10},
    "long_trend95_k115": {"trend_long_k": 1.15},
    "long_trend95_k120": {"trend_long_k": 1.20},
}
BASELINE_PRESET = "flat_control"
BUDGET_MIN = 0.98
BUDGET_MAX = 1.02
M1_PF_TOLERANCE = 0.05
M1_DD_TOLERANCE_R = 1.5
LONG_GUARD_TOLERANCE_R = 2.0
TOP3_SHARE_MAX = 65.0
BEST_YEAR_SHARE_MAX = 80.0
SATISFACTION_CAGR_TARGET = 22.0

LABEL_KEY_COLUMNS = ["symbol", "strategy_profile", "side", "signal_time", "entry_time"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trend-strength management one-rule gate on fixed switch baseline.")
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


def _label_backtest_service() -> BacktestService:
    assumptions = BacktestAssumptions(
        exit_profile="long_scaled1_3_short_fixed1_5",
        take_profit_mode="scaled",
        long_exit={"take_profit_mode": "scaled", "scaled_tp1_r": 1.0, "scaled_tp2_r": 3.0},
        short_exit={"take_profit_mode": "fixed_r", "fixed_take_profit_r": 1.5},
        swing_detection_mode="confirmed",
        cache_dir="artifacts/backtests/cache",
    )
    return BacktestService(
        ohlcv_service=OhlcvService(get_exchange_client_factory()),
        strategy_service=StrategyService(),
        assumptions=assumptions,
    )


def _resolve_index(frame: pd.DataFrame, timestamp: pd.Timestamp) -> int | None:
    left_idx = int(frame["timestamp"].searchsorted(timestamp, side="left"))
    if left_idx < len(frame) and frame.iloc[left_idx]["timestamp"] == timestamp:
        return left_idx
    right_idx = int(frame["timestamp"].searchsorted(timestamp, side="right") - 1)
    if right_idx >= 0 and frame.iloc[right_idx]["timestamp"] <= timestamp:
        return right_idx
    return None


def attach_axis_band_risk_group(trade_frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    if trade_frame.empty:
        return trade_frame.copy(), 0.0

    service = _label_backtest_service()
    strategy = service.strategy_service.build_strategy(LABEL_PROFILE)
    setup_tf = str(strategy.window_config["setup_timeframe"])

    base_reference = trade_frame[
        (trade_frame["cost_scenario"] == "base") & (trade_frame["window"] == "full_2020")
    ].copy()
    if base_reference.empty:
        raise RuntimeError("No base/full_2020 switch trades available for axis-band labeling.")
    base_reference = base_reference.drop_duplicates(subset=LABEL_KEY_COLUMNS).sort_values("signal_time").reset_index(drop=True)

    start = pd.to_datetime(base_reference["signal_time"], utc=True).min().to_pydatetime()
    end = (pd.to_datetime(base_reference["entry_time"], utc=True).max() + pd.Timedelta(hours=24)).to_pydatetime()

    history = service.prepare_history(
        exchange="binance",
        market_type="perpetual",
        symbol=SYMBOL,
        strategy_profile=LABEL_PROFILE,
        start=start,
        end=end,
    )
    enriched = service._prepare_enriched_frames(LABEL_PROFILE, history)

    labels: list[dict[str, Any]] = []
    for trade in base_reference.itertuples(index=False):
        signal_ts = pd.Timestamp(trade.signal_time)
        indices: dict[str, int] = {}
        missing = False
        for timeframe in strategy.required_timeframes:
            idx = _resolve_index(enriched[timeframe], signal_ts)
            if idx is None:
                missing = True
                break
            indices[timeframe] = idx
        if missing:
            continue

        prepared = {
            timeframe: service._build_snapshot(strategy, timeframe, enriched[timeframe], indices[timeframe])
            for timeframe in indices
        }
        try:
            higher_bias = Bias(str(trade.higher_bias))
        except ValueError:
            higher_bias = Bias.BULLISH if str(trade.side) == "LONG" else Bias.BEARISH

        state_note = strategy._compute_axis_band_state_note(higher_bias=higher_bias, setup_ctx=prepared[setup_tf])
        labels.append(
            {
                "symbol": str(trade.symbol),
                "strategy_profile": str(trade.strategy_profile),
                "side": str(trade.side),
                "signal_time": pd.to_datetime(trade.signal_time, utc=True),
                "entry_time": pd.to_datetime(trade.entry_time, utc=True),
                "risk_group": "active" if bool(state_note.get("active")) else "inactive",
            }
        )

    if not labels:
        raise RuntimeError("Axis-band risk labeling produced no labeled trades.")

    labels_frame = pd.DataFrame(labels).drop_duplicates(subset=LABEL_KEY_COLUMNS)
    merged = trade_frame.merge(labels_frame, on=LABEL_KEY_COLUMNS, how="left")
    coverage = 1.0 - float(merged["risk_group"].isna().mean())
    merged = merged.dropna(subset=["risk_group"]).copy()
    if coverage < 0.95:
        raise RuntimeError(f"Axis-band label coverage too low: {coverage:.2%}")
    return merged.reset_index(drop=True), coverage


def build_weighted_trades(trade_frame: pd.DataFrame, *, preset: str) -> pd.DataFrame:
    if trade_frame.empty:
        result = trade_frame.copy()
        result["preset"] = preset
        result["size_multiplier"] = 0.0
        result["weighted_pnl_r"] = 0.0
        result["weighted_pnl_pct"] = 0.0
        return result

    cfg = PRESETS[preset]
    result = trade_frame.copy()
    raw_size = result["side"].map(BASE_SIDE_MULTIPLIER).astype(float)

    # Fixed promoted baseline layer: LONG-active gets 1.15x.
    long_active = (result["side"] == "LONG") & (result["risk_group"] == "active")
    raw_size = raw_size * long_active.map({True: float(BASE_AXIS_ACTIVE_LONG_K), False: 1.0}).astype(float)

    trend_long_k = float(cfg["trend_long_k"])
    if trend_long_k > 1.0:
        trend_active = (result["side"] == "LONG") & (result["trend_strength"].astype(float) >= TREND_STRENGTH_THRESHOLD)
        raw_size = raw_size * trend_active.map({True: trend_long_k, False: 1.0}).astype(float)

    norm_scope = result[(result["cost_scenario"] == "base") & (result["window"] == "full_2020")].index
    raw_avg = float(raw_size.loc[norm_scope].mean()) if len(norm_scope) else 1.0
    normalize_scale = (1.0 / raw_avg) if raw_avg > 0.0 else 1.0

    result["preset"] = preset
    result["size_multiplier"] = raw_size * normalize_scale
    result["weighted_pnl_r"] = result["pnl_r"].astype(float) * result["size_multiplier"]
    result["weighted_pnl_pct"] = result["pnl_pct"].astype(float) * result["size_multiplier"]
    return result


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


def fetch_btc_yearly_returns(*, start: datetime, end: datetime) -> dict[int, float]:
    service = OhlcvService(get_exchange_client_factory())
    frame = service.fetch_ohlcv_range(
        exchange="binance",
        market_type="perpetual",
        symbol=SYMBOL,
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
    *, yearly_rows: list[dict[str, Any]], btc_yearly_returns: dict[int, float]
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

        pass_budget_neutral = BUDGET_MIN <= float(base_primary["avg_size"]) <= BUDGET_MAX
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

        all_pass = (
            pass_budget_neutral
            and pass_base_geo
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

        candidate_rows.append(
            {
                "preset": preset,
                "base_geometric_return_pct": round(float(base_primary["geometric_return_pct"]), 4),
                "base_cagr_pct": round(float(base_primary["cagr_pct"]), 4),
                "base_profit_factor": round(float(base_primary["profit_factor"]), 4),
                "base_max_dd_r": round(float(base_primary["max_dd_r"]), 4),
                "base_avg_size": round(float(base_primary["avg_size"]), 4),
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
                "oos_available": bool(oos.get("oos_available", False)),
                "candidate_btc_outperform_years": candidate_btc_years,
                "flat_btc_outperform_years": flat_btc_years,
                "pass_budget_neutral": pass_budget_neutral,
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
                "all_pass": all_pass,
                "satisfied_result": all_pass and pass_target_cagr_22 and pass_btc_non_regression,
            }
        )

    ranking = sorted(
        candidate_rows,
        key=lambda row: (
            bool(row["all_pass"]),
            bool(row["pass_target_cagr_22"]),
            bool(row["pass_btc_diagnostic_non_regression"]),
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
        "base_full_2020_candidate_avg_size": round(float(chosen["base_avg_size"]), 4),
        "stress_x2_candidate_geometric_return_pct": round(float(chosen["stress_x2_geometric_return_pct"]), 4),
        "stress_x3_candidate_geometric_return_pct": round(float(chosen["stress_x3_geometric_return_pct"]), 4),
        "two_year_long_delta_r": round(float(chosen["two_year_long_cum_r"] - float(flat_two_year_long["cum_r"])), 4),
        "candidate_top3_trades_pnl_share_pct": round(float(chosen["top3_trades_pnl_share_pct"]), 4),
        "candidate_best_year_geometric_pct_share": round(float(chosen["best_year_geometric_pct_share"]), 4),
        "candidate_oos_geometric_return_pct": round(float(chosen["oos_geometric_return_pct"]), 4),
        "candidate_oos_delta_vs_flat_geometric_return_pct": round(
            float(chosen["oos_delta_vs_flat_geometric_return_pct"]), 4
        ),
        "candidate_oos_available": bool(chosen["oos_available"]),
        "candidate_btc_outperform_years": int(chosen["candidate_btc_outperform_years"]),
        "flat_btc_outperform_years": int(flat_btc_years),
        "pass_target_cagr_22": bool(chosen["pass_target_cagr_22"]),
        "pass_btc_diagnostic_non_regression": bool(chosen["pass_btc_diagnostic_non_regression"]),
        "satisfied_result": bool(chosen["satisfied_result"]),
        "pass_budget_neutral": bool(chosen["pass_budget_neutral"]),
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
        "status": status,
        "next_route": (
            "promote_as_management_overlay_baseline_candidate"
            if status == "promoted_management_overlay_candidate" and bool(chosen["satisfied_result"])
            else "freeze_trend_strength_management_one_rule_after_single_round"
        ),
    }
    return decision, candidate_rows


def build_oos_selected_vs_flat(*, fold_rows: list[dict[str, Any]], chosen_candidate_preset: str) -> list[dict[str, Any]]:
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
    if rows:
        geo_factor_candidate = math.prod(1.0 + float(item["candidate_geometric_return_pct"]) / 100.0 for item in rows)
        geo_factor_flat = math.prod(1.0 + float(item["flat_geometric_return_pct"]) / 100.0 for item in rows)
        rows.append(
            {
                "fold": "aggregate",
                "candidate_preset": chosen_candidate_preset,
                "candidate_trades": int(sum(int(item["candidate_trades"]) for item in rows if isinstance(item["fold"], int))),
                "flat_trades": int(sum(int(item["flat_trades"]) for item in rows if isinstance(item["fold"], int))),
                "candidate_geometric_return_pct": round((geo_factor_candidate - 1.0) * 100.0, 4),
                "flat_geometric_return_pct": round((geo_factor_flat - 1.0) * 100.0, 4),
                "delta_geometric_return_pct": round((geo_factor_candidate - geo_factor_flat) * 100.0, 4),
                "candidate_cagr_pct": "",
                "flat_cagr_pct": "",
                "candidate_profit_factor": "",
                "flat_profit_factor": "",
                "candidate_cum_r": round(
                    float(sum(float(item["candidate_cum_r"]) for item in rows if isinstance(item["fold"], int))), 4
                ),
                "flat_cum_r": round(float(sum(float(item["flat_cum_r"]) for item in rows if isinstance(item["fold"], int))), 4),
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
    btc_rows: list[dict[str, Any]],
    decision_row: dict[str, Any],
) -> str:
    base_full = [row for row in summary_rows if row["cost_scenario"] == "base" and row["window"] == "full_2020"]
    base_full = sorted(base_full, key=lambda row: float(row["geometric_return_pct"]), reverse=True)
    btc_agg = [row for row in btc_rows if row["row_type"] == "aggregate"]

    conclusion = (
        "候选通过 M1，且满足 CAGR>=22% 与 BTC 年度不退步，达到本轮满意结果。"
        if decision_row["status"] == "promoted_management_overlay_candidate" and bool(decision_row["satisfied_result"])
        else "候选未达到本轮满意结果（可能是 M1 未过，或 CAGR>=22 / BTC 年度不退步未达标）。"
    )

    return "\n".join(
        [
            "# Switch Trend-Strength Management One-Rule Gate",
            "",
            "- baseline trades source: fixed switch (`simple_candidate_v2 -> challenger_managed`) with fixed cutover date",
            "- one-rule semantics: fixed baseline includes LONG-active 1.15x, then only LONG with trend_strength>=95 receives extra multiplier",
            "- BTC benchmark is diagnostic (non-hard-reject) with non-regression requirement for satisfaction",
            "",
            "## Base Full_2020 Summary",
            "",
            markdown_table(
                base_full,
                [
                    ("preset", "Preset"),
                    ("trades", "Trades"),
                    ("avg_size", "Avg Size"),
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
                    ("pass_target_cagr_22", "CAGR>=22"),
                    ("candidate_btc_outperform_years", "BTC Outperform Years"),
                    ("all_pass", "M1 Pass"),
                    ("satisfied_result", "Satisfied"),
                ],
            ),
            "",
            "## BTC Diagnostic",
            "",
            markdown_table(
                btc_agg,
                [
                    ("preset", "Preset"),
                    ("outperform_years", "Outperform Years"),
                    ("total_years", "Total Years"),
                    ("outperform_rate_pct", "Outperform %"),
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
                    ("base_full_2020_flat_geometric_return_pct", "Flat Base Geom %"),
                    ("pass_target_cagr_22", "CAGR>=22"),
                    ("candidate_btc_outperform_years", "Candidate BTC Years"),
                    ("flat_btc_outperform_years", "Flat BTC Years"),
                    ("pass_btc_diagnostic_non_regression", "BTC Non-Regression"),
                    ("status", "Status"),
                    ("satisfied_result", "Satisfied"),
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

    primary_start = parse_date(args.primary_start)
    secondary_start = parse_date(args.secondary_start)
    switch_date = parse_date(DEFAULT_SWITCH_DATE)
    if secondary_start != switch_date:
        raise ValueError("secondary-start must equal fixed switch date 2024-03-19")
    if secondary_start <= primary_start:
        raise ValueError("secondary-start must be later than primary-start")
    requested_end = parse_date(args.end) if args.end else datetime.now(timezone.utc)

    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    case = run_regime_switch_case(
        symbol=SYMBOL,
        exchange="binance",
        market_type="perpetual",
        simple_profile="swing_trend_simple_candidate_v2",
        challenger_strategy_profile="swing_trend_long_regime_short_no_reversal_no_aux_v1",
        challenger_overlay_profile="be_if_no_extension_within_3bars_after_tp1_and_veto_hold_at_trend_96_98",
        switch_date=switch_date,
        primary_start=primary_start,
        requested_end=requested_end,
        cost_scenarios=COST_SCENARIOS,
    )

    trade_frame = case["trade_frame"].copy()
    trade_frame = trade_frame[trade_frame["scenario_kind"] == SWITCH_SCENARIO_KIND].copy()
    if trade_frame.empty:
        raise RuntimeError("No switch trades generated for the configured window.")
    for column in ("signal_time", "entry_time", "exit_time"):
        trade_frame[column] = pd.to_datetime(trade_frame[column], utc=True)
    trade_frame["year"] = trade_frame["signal_time"].dt.year.astype(int)
    trade_frame = trade_frame.sort_values(["cost_scenario", "window", "entry_time"]).reset_index(drop=True)

    labeled_trade_frame, label_coverage = attach_axis_band_risk_group(trade_frame)
    labeled_trade_frame["year"] = labeled_trade_frame["signal_time"].dt.year.astype(int)

    weighted_frames: list[pd.DataFrame] = []
    for preset in PRESETS:
        weighted_frames.append(build_weighted_trades(labeled_trade_frame, preset=preset))
    weighted_trades = pd.concat(weighted_frames, ignore_index=True) if weighted_frames else pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    for cost_scenario in COST_SCENARIOS:
        for window in WINDOWS:
            window_start, window_end = case["windows"][window]
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
        end=case["resolved_end"],
        train_days=int(args.train_days),
        test_days=int(args.test_days),
        step_days=int(args.step_days),
    )
    fold_rows = build_fold_panel(weighted_trades=weighted_trades, folds=folds)
    oos_summary_rows = build_oos_summary(fold_rows)

    btc_yearly = fetch_btc_yearly_returns(start=primary_start, end=case["resolved_end"])
    btc_rows, btc_counts = build_btc_yearly_comparison(yearly_rows=yearly_rows, btc_yearly_returns=btc_yearly)

    decision_row, candidate_eval_rows = build_promotion_decision(
        summary_rows=summary_rows,
        side_rows=side_rows,
        concentration_rows=concentration_rows,
        oos_summary_rows=oos_summary_rows,
        btc_outperform_counts=btc_counts,
    )
    oos_selected_vs_flat_rows = build_oos_selected_vs_flat(
        fold_rows=fold_rows,
        chosen_candidate_preset=str(decision_row["chosen_candidate_preset"]),
    )

    report = build_report(
        summary_rows=summary_rows,
        candidate_eval_rows=candidate_eval_rows,
        btc_rows=btc_rows,
        decision_row=decision_row,
    )

    write_csv(output_dir / "full_summary.csv", summary_rows)
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
                "symbol": SYMBOL,
                "switch_date": switch_date.date().isoformat(),
                "resolved_end": case["resolved_end"].isoformat(),
                "primary_start": primary_start.date().isoformat(),
                "secondary_start": secondary_start.date().isoformat(),
                "train_days": int(args.train_days),
                "test_days": int(args.test_days),
                "step_days": int(args.step_days),
                "presets": list(PRESETS.keys()),
                "cost_scenarios": COST_SCENARIOS,
                "label_profile": LABEL_PROFILE,
                "label_coverage": round(float(label_coverage), 6),
                "base_axis_active_long_k": BASE_AXIS_ACTIVE_LONG_K,
                "trend_strength_threshold": TREND_STRENGTH_THRESHOLD,
                "satisfaction_cagr_target": SATISFACTION_CAGR_TARGET,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved switch trend-strength management one-rule gate artifacts to {output_dir}")


if __name__ == "__main__":
    main()
