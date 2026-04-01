from __future__ import annotations

import csv
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.service import BacktestAssumptions, BacktestService, BacktestReport
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.services.derivatives_state_study_service import DerivativesStateStudyService, FEATURE_SPECS
from app.services.strategy_service import StrategyService


SYMBOL = "BTC/USDT:USDT"
PROFILE = "swing_trend_long_regime_gate_v1"
CALIBRATION_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
CALIBRATION_END = datetime(2024, 3, 19, tzinfo=timezone.utc)
EVAL_START = CALIBRATION_END
EVAL_END = datetime(2026, 3, 19, tzinfo=timezone.utc)
INPUT_PATH = ROOT / "artifacts" / "derivatives" / "bybit_btcusdt_hourly_state" / "research_table.csv"
OUTPUT_DIR = ROOT / "artifacts" / "derivatives" / "bybit_btcusdt_strict_state_study"


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
        strategy_profiles=[PROFILE],
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


def load_research_table() -> pd.DataFrame:
    frame = pd.read_csv(INPUT_PATH)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def build_dataset_summary(calibration: pd.DataFrame, evaluation: pd.DataFrame, trades: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "calibration_start": CALIBRATION_START.isoformat(),
            "calibration_end": CALIBRATION_END.isoformat(),
            "eval_start": EVAL_START.isoformat(),
            "eval_end": EVAL_END.isoformat(),
            "calibration_hours": int(len(calibration)),
            "eval_hours": int(len(evaluation)),
            "mainline_trades": int(len(trades)),
            "trade_join_rate_pct": round(float(trades["has_state"].mean() * 100.0), 2) if not trades.empty else 0.0,
        }
    ]


def select_highlights(
    return_edges: list[dict[str, Any]],
    volatility_edges: list[dict[str, Any]],
    mainline_edges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return_highlights = (
        pd.DataFrame(return_edges)
        .assign(abs_q5_q1=lambda df: df["q5_minus_q1_forward_bps"].abs())
        .sort_values(["abs_q5_q1", "same_sign_month_rate_pct"], ascending=[False, False])
        .head(8)
        .drop(columns=["abs_q5_q1"])
        .to_dict("records")
        if return_edges
        else []
    )
    volatility_highlights = (
        pd.DataFrame(volatility_edges)
        .sort_values("extreme_to_q3_abs_vol_ratio", ascending=False)
        .head(8)
        .to_dict("records")
        if volatility_edges
        else []
    )
    mainline_highlights = []
    if mainline_edges:
        mainline_frame = pd.DataFrame(mainline_edges)
        mainline_frame["impact_score"] = (
            mainline_frame["q5_minus_q1_expectancy_r"].abs() * 10.0
            + mainline_frame["q5_minus_q1_win_rate_pct"].abs() / 10.0
            + mainline_frame["q5_minus_q1_avg_hold_bars"].abs() / 10.0
        )
        mainline_highlights = (
            mainline_frame.sort_values("impact_score", ascending=False)
            .head(10)
            .drop(columns=["impact_score"])
            .to_dict("records")
        )
    return return_highlights, volatility_highlights, mainline_highlights


def derive_conclusion(return_edges: list[dict[str, Any]], volatility_edges: list[dict[str, Any]]) -> dict[str, Any]:
    directional_frame = pd.DataFrame(return_edges)
    volatility_frame = pd.DataFrame(volatility_edges)

    strong_directional = 0
    if not directional_frame.empty:
        strong_directional = int(
            (
                directional_frame["q5_minus_q1_forward_bps"].abs().ge(15.0)
                & directional_frame["same_sign_month_rate_pct"].ge(55.0)
            ).sum()
        )

    strong_volatility = 0
    if not volatility_frame.empty:
        strong_volatility = int(volatility_frame["extreme_to_q3_abs_vol_ratio"].ge(1.12).sum())

    if strong_directional == 0 and strong_volatility > 0:
        interpretation = "state_layer_not_directional_alpha"
        recommendation = "pivot_to_carry_basis"
    elif strong_directional > 0 and strong_volatility >= strong_directional:
        interpretation = "mixed_but_state_dominant"
        recommendation = "prefer_carry_basis_or_state_filter"
    else:
        interpretation = "directional_signal_worthy_of_further_testing"
        recommendation = "do_not_pivot_yet"

    return {
        "strong_directional_feature_horizons": strong_directional,
        "strong_volatility_feature_horizons": strong_volatility,
        "interpretation": interpretation,
        "recommendation": recommendation,
    }


def build_report(
    dataset_rows: list[dict[str, Any]],
    conclusion_row: dict[str, Any],
    return_highlights: list[dict[str, Any]],
    volatility_highlights: list[dict[str, Any]],
    mainline_highlights: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# BTC Derivatives Strict State Study",
            "",
            "- 这轮不是再发明 directional strategy，而是把现有 4 个衍生品特征当作 state layer，严格回答三个问题：未来收益条件性、未来波动解释力、以及它们如何改变当前 mainline 的交易分布。",
            "- 口径固定：`2024-01-01 -> 2024-03-19` 只用于校准 quintile 边界；`2024-03-19 -> 2026-03-19` 才是评估窗。",
            "- 这版故意不纳入 liquidation，因为当前没有同口径长历史；也不再把这些特征硬塞成新的 directional rule。",
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
                    ("calibration_hours", "Calibration Hours"),
                    ("eval_hours", "Eval Hours"),
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
                    ("feature", "Feature"),
                    ("horizon_hours", "Horizon h"),
                    ("q5_minus_q1_forward_bps", "Q5-Q1 Fwd bps"),
                    ("q5_minus_q3_forward_bps", "Q5-Q3 Fwd bps"),
                    ("q1_minus_q3_forward_bps", "Q1-Q3 Fwd bps"),
                    ("same_sign_month_rate_pct", "Same-Sign Month %"),
                    ("stable_months", "Stable Months"),
                ],
            ),
            "",
            "## Volatility Conditionality Highlights",
            "",
            markdown_table(
                volatility_highlights,
                [
                    ("feature", "Feature"),
                    ("horizon_hours", "Horizon h"),
                    ("q5_minus_q1_abs_forward_bps", "Q5-Q1 |Fwd| bps"),
                    ("q1_to_q3_abs_vol_ratio", "Q1/Q3 |Vol|"),
                    ("q5_to_q3_abs_vol_ratio", "Q5/Q3 |Vol|"),
                    ("extreme_to_q3_abs_vol_ratio", "Extreme/Q3 |Vol|"),
                ],
            ),
            "",
            "## Mainline State Impact Highlights",
            "",
            markdown_table(
                mainline_highlights,
                [
                    ("feature", "Feature"),
                    ("side", "Side"),
                    ("q5_minus_q1_win_rate_pct", "Q5-Q1 Win %"),
                    ("q5_minus_q1_expectancy_r", "Q5-Q1 Exp R"),
                    ("q5_minus_q1_max_dd_r", "Q5-Q1 MaxDD R"),
                    ("q5_minus_q1_avg_hold_bars", "Q5-Q1 Hold Bars"),
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
                    ("strong_directional_feature_horizons", "Directional FH"),
                    ("strong_volatility_feature_horizons", "Volatility FH"),
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

    frame = load_research_table()
    service = DerivativesStateStudyService()

    calibration = frame[(frame["timestamp"] >= pd.Timestamp(CALIBRATION_START)) & (frame["timestamp"] < pd.Timestamp(CALIBRATION_END))].copy()
    evaluation = frame[(frame["timestamp"] >= pd.Timestamp(EVAL_START)) & (frame["timestamp"] < pd.Timestamp(EVAL_END))].copy()

    edges = service.build_calibration_edges(calibration)
    evaluation = service.assign_feature_buckets(evaluation, edges)

    return_bucket_rows, return_edge_rows, return_monthly_rows = service.summarize_return_conditionality(evaluation)
    volatility_bucket_rows, volatility_edge_rows = service.summarize_volatility_conditionality(evaluation)

    report = run_mainline()
    trades = build_trade_frame(report)
    trades = trades.merge(
        evaluation[
            ["timestamp"]
            + [f"{spec.label}_bucket" for spec in FEATURE_SPECS]
        ],
        left_on="signal_time",
        right_on="timestamp",
        how="left",
    )
    trades["has_state"] = trades[[f"{spec.label}_bucket" for spec in FEATURE_SPECS]].notna().all(axis=1)
    mainline_bucket_rows, mainline_edge_rows, mainline_exit_rows = service.summarize_mainline_trade_state(trades[trades["has_state"]].copy())

    dataset_rows = build_dataset_summary(calibration, evaluation, trades)
    conclusion_row = derive_conclusion(return_edge_rows, volatility_edge_rows)
    return_highlights, volatility_highlights, mainline_highlights = select_highlights(
        return_edge_rows,
        volatility_edge_rows,
        mainline_edge_rows,
    )

    write_csv(OUTPUT_DIR / "dataset_summary.csv", dataset_rows)
    write_csv(OUTPUT_DIR / "return_bucket_summary.csv", return_bucket_rows)
    write_csv(OUTPUT_DIR / "return_edge_summary.csv", return_edge_rows)
    write_csv(OUTPUT_DIR / "return_monthly_stability.csv", return_monthly_rows)
    write_csv(OUTPUT_DIR / "volatility_bucket_summary.csv", volatility_bucket_rows)
    write_csv(OUTPUT_DIR / "volatility_edge_summary.csv", volatility_edge_rows)
    write_csv(OUTPUT_DIR / "mainline_state_bucket_summary.csv", mainline_bucket_rows)
    write_csv(OUTPUT_DIR / "mainline_state_edge_summary.csv", mainline_edge_rows)
    write_csv(OUTPUT_DIR / "mainline_state_exit_summary.csv", mainline_exit_rows)
    trades.to_csv(OUTPUT_DIR / "mainline_trades_with_state.csv", index=False)

    (OUTPUT_DIR / "report.md").write_text(
        build_report(dataset_rows, conclusion_row, return_highlights, volatility_highlights, mainline_highlights),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
