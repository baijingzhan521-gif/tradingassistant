from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.service import BacktestAssumptions, BacktestReport, BacktestService
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.schemas.common import Bias
from app.services.position_confluence_factor_study_service import (
    FACTOR_SPECS,
    HORIZONS,
    PositionConfluenceFactorStudyService,
)
from app.services.strategy_service import StrategyService
from app.utils.timeframes import get_strategy_required_timeframes


SYMBOL = "BTC/USDT:USDT"
FACTOR_PROFILE = "swing_trend_confluence_setup_v1"
TRADE_PROFILE = "swing_trend_long_regime_gate_v1"
CALIBRATION_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
CALIBRATION_END = datetime(2024, 3, 19, tzinfo=timezone.utc)
EVAL_START = CALIBRATION_END
EVAL_END = datetime(2026, 3, 19, tzinfo=timezone.utc)
OUTPUT_DIR = ROOT / "artifacts" / "backtests" / "position_confluence_factor_study"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_None_"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [header, divider]
    for row in rows:
        values: list[str] = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join(body)


def make_backtest_service() -> BacktestService:
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


def run_mainline() -> BacktestReport:
    return make_backtest_service().run(
        exchange="binance",
        market_type="perpetual",
        symbols=[SYMBOL],
        strategy_profiles=[TRADE_PROFILE],
        start=EVAL_START,
        end=EVAL_END,
    )


def build_trade_frame(report: BacktestReport) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(item) for item in report.trades])
    if frame.empty:
        return frame
    frame["signal_time"] = pd.to_datetime(frame["signal_time"], utc=True)
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True)
    frame["signal_month"] = frame["signal_time"].dt.strftime("%Y-%m")
    return frame.sort_values("signal_time").reset_index(drop=True)


def build_factor_frame(service: BacktestService) -> pd.DataFrame:
    strategy = service.strategy_service.build_strategy(FACTOR_PROFILE)
    frames = service.prepare_history(
        exchange="binance",
        market_type="perpetual",
        symbol=SYMBOL,
        strategy_profile=FACTOR_PROFILE,
        start=CALIBRATION_START,
        end=EVAL_END,
    )
    enriched = service._prepare_enriched_frames(FACTOR_PROFILE, frames)

    trigger_tf = str(strategy.window_config["trigger_timeframe"])
    setup_tf = str(strategy.window_config["setup_timeframe"])
    reference_tf = str(strategy.window_config.get("reference_timeframe", setup_tf))
    required = tuple(get_strategy_required_timeframes(FACTOR_PROFILE))
    trigger_frame = enriched[trigger_tf]
    trigger_end_idx = int(trigger_frame["timestamp"].searchsorted(pd.Timestamp(EVAL_END), side="left"))
    indices = {timeframe: 0 for timeframe in required if timeframe != trigger_tf}
    pullback_threshold = float(strategy.config["execution"]["pullback_distance_atr"])

    rows: list[dict[str, Any]] = []
    for trigger_idx in range(trigger_end_idx):
        candle = trigger_frame.iloc[trigger_idx]
        ts = candle["timestamp"].to_pydatetime()
        if ts < CALIBRATION_START:
            continue

        current_indices: dict[str, int] = {trigger_tf: trigger_idx}
        ready = True
        for timeframe in required:
            if timeframe == trigger_tf:
                continue
            frame = enriched[timeframe]
            pointer = indices[timeframe]
            while pointer + 1 < len(frame) and frame.iloc[pointer + 1]["timestamp"] <= candle["timestamp"]:
                pointer += 1
            indices[timeframe] = pointer
            if frame.iloc[pointer]["timestamp"] > candle["timestamp"]:
                ready = False
                break
            current_indices[timeframe] = pointer
        if not ready:
            continue

        min_required = max(int(service.assumptions.lookback // 3), 20)
        if any(index < min_required for index in current_indices.values()):
            continue

        prepared = {
            timeframe: service._build_snapshot(strategy, timeframe, enriched[timeframe], current_indices[timeframe])
            for timeframe in current_indices
        }
        higher_bias, trend_strength = strategy._derive_higher_timeframe_bias(prepared)
        if higher_bias == Bias.NEUTRAL:
            continue

        setup_ctx = prepared[setup_tf]
        reference_ctx = prepared[reference_tf]
        trigger_ctx = prepared[trigger_tf]
        volatility_state = strategy._derive_volatility_state(setup_ctx)
        trend_friendly = strategy._is_trend_friendly(
            higher_bias=higher_bias,
            trend_strength=trend_strength,
            volatility_state=volatility_state,
        )
        setup_assessment = strategy._assess_setup(
            higher_bias,
            setup_ctx,
            setup_tf,
            reference_ctx=reference_ctx,
            current_price=trigger_ctx.model.close,
        )
        if not setup_assessment["aligned"]:
            continue
        if not trend_friendly:
            continue
        if setup_ctx.distance_to_execution_atr > pullback_threshold:
            continue

        confluence = setup_assessment["confluence"]
        components = confluence["components"]
        close = float(trigger_ctx.model.close)
        direction = 1.0 if higher_bias == Bias.BULLISH else -1.0

        row: dict[str, Any] = {
            "timestamp": pd.Timestamp(ts),
            "higher_bias": higher_bias.value,
            "trend_strength": int(trend_strength),
            "volatility_state": volatility_state.value,
            "distance_to_execution_atr": round(float(setup_ctx.distance_to_execution_atr), 4),
            "setup_score": int(setup_assessment["score"]),
            "confluence_hits": int(confluence["hits"]),
            "confluence_ready": bool(confluence["ready"]),
            "band_position": round(float(setup_ctx.band_position), 4),
            "axis_distance_vol": round(float(setup_ctx.axis_distance_vol), 4),
            "ema55_distance_vol": round(float(setup_ctx.ema55_distance_vol), 4),
            "ema55_anchor_gap_atr": components["ema55"]["distance_atr"],
            "pivot_anchor_gap_atr": components["pivot_anchor"]["distance_atr"],
            "band_anchor_gap_atr": components["band_edge"]["distance_atr"],
            "confluence_spread_atr": confluence["spread_atr"],
        }
        for horizon in HORIZONS:
            if trigger_idx + horizon >= len(trigger_frame):
                row[f"raw_forward_return_bps_{horizon}h"] = None
                row[f"aligned_forward_return_bps_{horizon}h"] = None
                row[f"forward_abs_return_bps_{horizon}h"] = None
                continue
            future_close = float(trigger_frame.iloc[trigger_idx + horizon]["close"])
            raw_forward_return_bps = ((future_close / close) - 1.0) * 10000.0 if close else 0.0
            row[f"raw_forward_return_bps_{horizon}h"] = round(raw_forward_return_bps, 4)
            row[f"aligned_forward_return_bps_{horizon}h"] = round(raw_forward_return_bps * direction, 4)
            row[f"forward_abs_return_bps_{horizon}h"] = round(abs(raw_forward_return_bps), 4)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def build_dataset_summary(calibration: pd.DataFrame, evaluation: pd.DataFrame, trades: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "calibration_start": CALIBRATION_START.isoformat(),
            "calibration_end": CALIBRATION_END.isoformat(),
            "eval_start": EVAL_START.isoformat(),
            "eval_end": EVAL_END.isoformat(),
            "calibration_candidates": int(len(calibration)),
            "eval_candidates": int(len(evaluation)),
            "mainline_trades": int(len(trades)),
            "trade_join_rate_pct": round(float(trades["has_factor"].mean() * 100.0), 2) if not trades.empty else 0.0,
        }
    ]


def select_highlights(
    return_edges: list[dict[str, Any]],
    volatility_edges: list[dict[str, Any]],
    trade_edges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return_highlights = (
        pd.DataFrame(return_edges)
        .assign(abs_edge=lambda df: df["q1_minus_q5_aligned_bps"].abs())
        .sort_values(["abs_edge", "same_sign_month_rate_pct", "monotonic_pair_rate_pct"], ascending=[False, False, False])
        .head(10)
        .drop(columns=["abs_edge"])
        .to_dict("records")
        if return_edges
        else []
    )
    volatility_highlights = (
        pd.DataFrame(volatility_edges)
        .sort_values("extreme_to_q3_abs_vol_ratio", ascending=False)
        .head(10)
        .to_dict("records")
        if volatility_edges
        else []
    )
    trade_highlights = []
    if trade_edges:
        trade_frame = pd.DataFrame(trade_edges)
        trade_frame["impact_score"] = (
            trade_frame["q1_minus_q5_expectancy_r"].abs() * 10.0
            + trade_frame["q1_minus_q5_win_rate_pct"].abs() / 10.0
            + trade_frame["q1_minus_q5_avg_hold_bars"].abs() / 10.0
        )
        trade_highlights = (
            trade_frame.sort_values("impact_score", ascending=False)
            .head(10)
            .drop(columns=["impact_score"])
            .to_dict("records")
        )
    return return_highlights, volatility_highlights, trade_highlights


def derive_conclusion(return_edges: list[dict[str, Any]], trade_edges: list[dict[str, Any]]) -> dict[str, Any]:
    return_frame = pd.DataFrame(return_edges)
    trade_frame = pd.DataFrame(trade_edges)

    strong_directional = 0
    if not return_frame.empty:
        strong_directional = int(
            (
                return_frame["q1_minus_q5_aligned_bps"].abs().ge(8.0)
                & return_frame["same_sign_month_rate_pct"].ge(55.0)
                & return_frame["monotonic_pair_rate_pct"].ge(50.0)
            ).sum()
        )

    strong_trade_impact = 0
    if not trade_frame.empty:
        strong_trade_impact = int(
            (
                trade_frame["q1_minus_q5_expectancy_r"].abs().ge(0.2)
                & trade_frame["q1_minus_q5_win_rate_pct"].abs().ge(10.0)
            ).sum()
        )

    if strong_directional == 0 and strong_trade_impact == 0:
        interpretation = "factors_are_descriptive_but_not_gate_ready"
        recommendation = "do_not_replace_setup_with_hard_confluence"
    elif strong_directional == 0 and strong_trade_impact > 0:
        interpretation = "trade_selection_shift_exists_but_forward_monotonicity_is_weak"
        recommendation = "prefer_soft_diagnostics_over_hard_gate"
    else:
        interpretation = "some_continuous_signal_content_detected"
        recommendation = "consider_soft_factor_usage_before_any_hard_gate"

    return {
        "strong_directional_factor_horizons": strong_directional,
        "strong_trade_factor_edges": strong_trade_impact,
        "interpretation": interpretation,
        "recommendation": recommendation,
    }


def build_report(
    dataset_rows: list[dict[str, Any]],
    conclusion_row: dict[str, Any],
    return_highlights: list[dict[str, Any]],
    volatility_highlights: list[dict[str, Any]],
    trade_highlights: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Position Confluence Factor Study",
            "",
            "- 这轮不是继续调 gate，而是把 `pivot / band / EMA55` 从离散条件拆成连续因子，先看它们有没有单调性和分桶信号内容。",
            "- `pivot` 这里仍然只是 confirmed pivot anchor proxy，不是完整 zone。这个限制必须明确，否则会高估结构层的表达能力。",
            "- 样本只取 `higher_bias` 明确、`trend_friendly` 成立、且 `1H` 已经回到 setup 语境的 bar：`aligned=True` 且 `distance_to_execution_atr <= pullback_threshold`。",
            "",
            "## Dataset",
            "",
            markdown_table(
                dataset_rows,
                [
                    ("calibration_start", "Calibration Start"),
                    ("calibration_end", "Calibration End"),
                    ("eval_start", "Eval Start"),
                    ("eval_end", "Eval End"),
                    ("calibration_candidates", "Calibration Candidates"),
                    ("eval_candidates", "Eval Candidates"),
                    ("mainline_trades", "Mainline Trades"),
                    ("trade_join_rate_pct", "Trade Join %"),
                ],
            ),
            "",
            "## Return Conditionality Highlights",
            "",
            markdown_table(
                return_highlights,
                [
                    ("factor", "Factor"),
                    ("horizon_hours", "Horizon h"),
                    ("q1_minus_q5_aligned_bps", "Q1-Q5 Aligned bps"),
                    ("q1_minus_q3_aligned_bps", "Q1-Q3"),
                    ("q3_minus_q5_aligned_bps", "Q3-Q5"),
                    ("same_sign_month_rate_pct", "Same-Sign Month %"),
                    ("monotonic_pair_rate_pct", "Monotonic Pair %"),
                ],
            ),
            "",
            "## Volatility Conditionality Highlights",
            "",
            markdown_table(
                volatility_highlights,
                [
                    ("factor", "Factor"),
                    ("horizon_hours", "Horizon h"),
                    ("q1_minus_q5_abs_forward_bps", "Q1-Q5 |Fwd| bps"),
                    ("q1_to_q3_abs_vol_ratio", "Q1/Q3 |Vol|"),
                    ("q5_to_q3_abs_vol_ratio", "Q5/Q3 |Vol|"),
                    ("extreme_to_q3_abs_vol_ratio", "Extreme/Q3 |Vol|"),
                ],
            ),
            "",
            "## Mainline Trade Impact Highlights",
            "",
            markdown_table(
                trade_highlights,
                [
                    ("factor", "Factor"),
                    ("side", "Side"),
                    ("q1_minus_q5_win_rate_pct", "Q1-Q5 Win %"),
                    ("q1_minus_q5_expectancy_r", "Q1-Q5 Exp R"),
                    ("q1_minus_q5_max_dd_r", "Q1-Q5 MaxDD R"),
                    ("q1_minus_q5_avg_hold_bars", "Q1-Q5 Hold Bars"),
                    ("q1_minus_baseline_expectancy_r", "Q1-Baseline Exp R"),
                    ("q5_minus_baseline_expectancy_r", "Q5-Baseline Exp R"),
                ],
            ),
            "",
            "## Decision",
            "",
            markdown_table(
                [conclusion_row],
                [
                    ("strong_directional_factor_horizons", "Directional FH"),
                    ("strong_trade_factor_edges", "Trade Edges"),
                    ("interpretation", "Interpretation"),
                    ("recommendation", "Recommendation"),
                ],
            ),
            "",
        ]
    )


def main() -> None:
    configure_logging()
    ensure_output_dir()

    service = make_backtest_service()
    study_service = PositionConfluenceFactorStudyService()

    factor_frame = build_factor_frame(service)
    calibration = factor_frame[
        (factor_frame["timestamp"] >= pd.Timestamp(CALIBRATION_START))
        & (factor_frame["timestamp"] < pd.Timestamp(CALIBRATION_END))
    ].copy()
    evaluation = factor_frame[
        (factor_frame["timestamp"] >= pd.Timestamp(EVAL_START))
        & (factor_frame["timestamp"] < pd.Timestamp(EVAL_END))
    ].copy()

    edges = study_service.build_calibration_edges(calibration)
    evaluation = study_service.assign_factor_buckets(evaluation, edges)

    return_bucket_rows, return_edge_rows, return_monthly_rows = study_service.summarize_return_conditionality(evaluation)
    volatility_bucket_rows, volatility_edge_rows = study_service.summarize_volatility_conditionality(evaluation)

    report = run_mainline()
    trades = build_trade_frame(report)
    bucket_columns = [f"{spec.label}_bucket" for spec in FACTOR_SPECS]
    trades = trades.merge(
        evaluation[["timestamp"] + bucket_columns + [spec.value_col for spec in FACTOR_SPECS]],
        left_on="signal_time",
        right_on="timestamp",
        how="left",
    )
    trades["has_factor"] = trades[bucket_columns].notna().all(axis=1)
    trade_bucket_rows, trade_edge_rows, trade_exit_rows = study_service.summarize_mainline_trade_factors(
        trades[trades["has_factor"]].copy()
    )

    dataset_rows = build_dataset_summary(calibration, evaluation, trades)
    conclusion_row = derive_conclusion(return_edge_rows, trade_edge_rows)
    return_highlights, volatility_highlights, trade_highlights = select_highlights(
        return_edge_rows,
        volatility_edge_rows,
        trade_edge_rows,
    )

    (OUTPUT_DIR / "factor_edges_pre_deploy.json").write_text(json.dumps(edges, indent=2), encoding="utf-8")
    factor_frame.to_csv(OUTPUT_DIR / "factor_frame.csv", index=False)
    evaluation.to_csv(OUTPUT_DIR / "factor_frame_evaluation.csv", index=False)
    write_csv(OUTPUT_DIR / "dataset_summary.csv", dataset_rows)
    write_csv(OUTPUT_DIR / "return_bucket_summary.csv", return_bucket_rows)
    write_csv(OUTPUT_DIR / "return_edge_summary.csv", return_edge_rows)
    write_csv(OUTPUT_DIR / "return_monthly_stability.csv", return_monthly_rows)
    write_csv(OUTPUT_DIR / "volatility_bucket_summary.csv", volatility_bucket_rows)
    write_csv(OUTPUT_DIR / "volatility_edge_summary.csv", volatility_edge_rows)
    write_csv(OUTPUT_DIR / "mainline_trade_bucket_summary.csv", trade_bucket_rows)
    write_csv(OUTPUT_DIR / "mainline_trade_edge_summary.csv", trade_edge_rows)
    write_csv(OUTPUT_DIR / "mainline_trade_exit_summary.csv", trade_exit_rows)
    trades.to_csv(OUTPUT_DIR / "mainline_trades_with_factors.csv", index=False)
    (OUTPUT_DIR / "report.md").write_text(
        build_report(dataset_rows, conclusion_row, return_highlights, volatility_highlights, trade_highlights),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
