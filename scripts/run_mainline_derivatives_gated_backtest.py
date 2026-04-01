from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.service import (
    BacktestAssumptions,
    BacktestService,
    BacktestSummary,
    BacktestTrade,
    _PendingEntry,
    _SignalSnapshot,
    _OpenPosition,
)
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.schemas.common import Action, RecommendedTiming
from app.services.strategy_service import StrategyService
from app.utils.timeframes import get_strategy_required_timeframes


SYMBOL = "BTC/USDT:USDT"
PROFILE = "swing_trend_long_regime_gate_v1"
OUTPUT_DIR = ROOT / "artifacts" / "backtests" / "mainline_derivatives_gated_backtest"
DERIVATIVES_RESEARCH_PATH = ROOT / "artifacts" / "derivatives" / "bybit_btcusdt_hourly_state" / "research_table.csv"
CALIBRATION_END = datetime(2024, 3, 19, tzinfo=timezone.utc)
WINDOWS = {
    "two_year": (
        datetime(2024, 3, 19, tzinfo=timezone.utc),
        datetime(2026, 3, 19, tzinfo=timezone.utc),
    ),
    "full_2020": (
        datetime(2020, 3, 19, tzinfo=timezone.utc),
        datetime(2026, 3, 19, tzinfo=timezone.utc),
    ),
}
FEATURES = [
    "funding_rate_z_7d",
    "basis_proxy_bps_z_7d",
    "mark_index_spread_bps_z_7d",
]


@dataclass
class GateContext:
    signal_time: datetime
    signal: _SignalSnapshot
    derivatives: dict[str, Any] | None


@dataclass
class GateStats:
    signals_now: int = 0
    signals_with_state: int = 0
    vetoed_signals: int = 0


@dataclass
class GateVariantResult:
    name: str
    description: str
    summary: BacktestSummary
    trades: list[BacktestTrade]
    gate_stats: GateStats


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
    lines = [header, divider]
    for row in rows:
        values: list[str] = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def make_service() -> BacktestService:
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


def load_research_table() -> pd.DataFrame:
    frame = pd.read_csv(DERIVATIVES_RESEARCH_PATH)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def build_quantile_edges(research: pd.DataFrame) -> dict[str, list[float]]:
    calibration = research.loc[research["timestamp"] < pd.Timestamp(CALIBRATION_END)].copy()
    edges: dict[str, list[float]] = {}
    for feature in FEATURES:
        values = calibration[feature].dropna()
        if values.empty:
            raise ValueError(f"No calibration values for feature {feature}")
        edges[feature] = values.quantile([0.2, 0.4, 0.6, 0.8]).tolist()
    return edges


def assign_bucket(series: pd.Series, edges: list[float]) -> pd.Series:
    return pd.cut(
        series,
        bins=[float("-inf"), edges[0], edges[1], edges[2], edges[3], float("inf")],
        labels=[1, 2, 3, 4, 5],
        include_lowest=True,
    ).astype("float64")


def enrich_derivatives_for_gate(research: pd.DataFrame, edges: dict[str, list[float]]) -> pd.DataFrame:
    frame = research.copy()
    for feature in FEATURES:
        frame[f"{feature}_bucket"] = assign_bucket(frame[feature], edges[feature])
    return frame


def summarize_trade_frame(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {"trades": 0, "win_rate_pct": 0.0, "expectancy_r": 0.0, "cum_r": 0.0}
    return {
        "trades": int(len(frame)),
        "win_rate_pct": round(float((frame["pnl_r"] > 0).mean() * 100.0), 2),
        "expectancy_r": round(float(frame["pnl_r"].mean()), 4),
        "cum_r": round(float(frame["pnl_r"].sum()), 4),
    }


def trades_to_frame(trades: list[BacktestTrade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["signal_time", "side", "pnl_r"])
    frame = pd.DataFrame([asdict(item) for item in trades])
    frame["signal_time"] = pd.to_datetime(frame["signal_time"], utc=True)
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True)
    return frame


def run_with_gate(
    *,
    service: BacktestService,
    derivatives_lookup: dict[datetime, dict[str, Any]],
    symbol: str,
    strategy_profile: str,
    start: datetime,
    end: datetime,
    enriched: dict[str, pd.DataFrame],
    variant_name: str,
    variant_description: str,
    gate_rule: Callable[[GateContext], bool] | None,
) -> GateVariantResult:
    strategy = service.strategy_service.build_strategy(strategy_profile)
    trigger_tf = str(strategy.window_config["trigger_timeframe"])
    required = tuple(get_strategy_required_timeframes(strategy_profile))
    trigger_frame = enriched[trigger_tf]
    trigger_end_idx = int(trigger_frame["timestamp"].searchsorted(pd.Timestamp(end), side="left"))
    indices = {timeframe: 0 for timeframe in required if timeframe != trigger_tf}

    trades: list[BacktestTrade] = []
    pending_entry: _PendingEntry | None = None
    position: _OpenPosition | None = None
    skipped_entries = 0
    cooldown_remaining = 0
    cooldown_bars_after_exit = int(strategy.config.get("backtest", {}).get("cooldown_bars_after_exit", 0))
    gate_stats = GateStats()

    for trigger_idx in range(trigger_end_idx):
        candle = trigger_frame.iloc[trigger_idx]
        ts = candle["timestamp"].to_pydatetime()

        if pending_entry is not None:
            maybe_position = service._open_pending_entry(
                symbol=symbol,
                strategy_profile=strategy_profile,
                pending_entry=pending_entry,
                candle=candle,
            )
            if maybe_position is None:
                skipped_entries += 1
            else:
                position = maybe_position
            pending_entry = None

        if position is not None:
            trade = service._update_open_position(
                position=position,
                candle=candle,
                max_hold_bars=service._max_hold_bars(strategy_profile),
            )
            if trade is not None:
                trades.append(trade)
                position = None
                cooldown_remaining = cooldown_bars_after_exit

        if ts < start:
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
        if position is not None or pending_entry is not None:
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        signal = service._evaluate_signal(
            strategy=strategy,
            strategy_profile=strategy_profile,
            enriched=enriched,
            indices=current_indices,
            timestamp=ts,
        )
        if signal.action not in {Action.LONG, Action.SHORT} or signal.recommended_timing != RecommendedTiming.NOW:
            continue

        gate_stats.signals_now += 1
        state_row = derivatives_lookup.get(ts)
        if state_row is not None:
            gate_stats.signals_with_state += 1
        if gate_rule is not None and gate_rule(GateContext(signal_time=ts, signal=signal, derivatives=state_row)):
            gate_stats.vetoed_signals += 1
            continue

        pending_entry = _PendingEntry(signal=signal)

    if position is not None and trigger_end_idx > 0:
        final_candle = trigger_frame.iloc[trigger_end_idx - 1]
        trades.append(service._close_position(position, final_candle, exit_reason="end_of_test", fill_price=float(final_candle["close"])))

    summary = service._summarize_trades(
        trades=trades,
        strategy_profile=strategy_profile,
        symbol=symbol,
        signals_now=gate_stats.signals_now,
        skipped_entries=skipped_entries,
    )
    return GateVariantResult(
        name=variant_name,
        description=variant_description,
        summary=summary,
        trades=trades,
        gate_stats=gate_stats,
    )


def build_gate_variants() -> list[tuple[str, str, Callable[[GateContext], bool] | None]]:
    def crowded_long(ctx: GateContext) -> bool:
        if ctx.derivatives is None or ctx.signal.action != Action.LONG:
            return False
        return (
            ctx.derivatives.get("funding_rate_z_7d_bucket") == 5.0
            and ctx.derivatives.get("basis_proxy_bps_z_7d_bucket") == 5.0
        )

    def short_discount(ctx: GateContext) -> bool:
        if ctx.derivatives is None or ctx.signal.action != Action.SHORT:
            return False
        return ctx.derivatives.get("mark_index_spread_bps_z_7d_bucket") == 1.0

    def combined(ctx: GateContext) -> bool:
        return crowded_long(ctx) or short_discount(ctx)

    return [
        ("baseline_no_gate", "No derivatives gate.", None),
        ("gate_long_crowded_q5q5", "Veto LONG when funding_z and basis_z are both in Q5.", crowded_long),
        ("gate_short_mark_spread_q1", "Veto SHORT when mark-index spread z-score is in Q1.", short_discount),
        (
            "gate_long_crowded_q5q5_or_short_mark_spread_q1",
            "Apply both selected rules together.",
            combined,
        ),
    ]


def build_report_rows(results: list[GateVariantResult], derivatives_lookup: dict[datetime, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    for result in results:
        summary_rows.append(
            {
                "variant": result.name,
                "description": result.description,
                "trades": result.summary.total_trades,
                "win_rate_pct": result.summary.win_rate,
                "profit_factor": result.summary.profit_factor,
                "expectancy_r": result.summary.expectancy_r,
                "cum_r": result.summary.cumulative_r,
                "max_dd_r": result.summary.max_drawdown_r,
                "signals_now": result.gate_stats.signals_now,
                "signals_with_state": result.gate_stats.signals_with_state,
                "gate_coverage_pct": round(
                    (result.gate_stats.signals_with_state / result.gate_stats.signals_now) * 100.0,
                    2,
                )
                if result.gate_stats.signals_now
                else 0.0,
                "vetoed_signals": result.gate_stats.vetoed_signals,
            }
        )

        frame = trades_to_frame(result.trades)
        if frame.empty:
            cohort_rows.append(
                {
                    "variant": result.name,
                    "cohort": "all_trades",
                    "trades": 0,
                    "win_rate_pct": 0.0,
                    "expectancy_r": 0.0,
                    "cum_r": 0.0,
                }
            )
            cohort_rows.append(
                {
                    "variant": result.name,
                    "cohort": "covered_only",
                    "trades": 0,
                    "win_rate_pct": 0.0,
                    "expectancy_r": 0.0,
                    "cum_r": 0.0,
                }
            )
            continue

        frame["derivatives_covered"] = frame["signal_time"].apply(lambda value: value.to_pydatetime() in derivatives_lookup)
        for cohort_name, cohort_frame in [("all_trades", frame), ("covered_only", frame[frame["derivatives_covered"]])]:
            cohort_rows.append(
                {
                    "variant": result.name,
                    "cohort": cohort_name,
                    **summarize_trade_frame(cohort_frame),
                }
            )
    return summary_rows, cohort_rows


def build_report(window_rows: list[dict[str, Any]], cohort_rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# Mainline Derivatives Gated Backtest",
            "",
            "- 这轮不是 trade-filter 诊断，而是把选定 gate 真正接到 sequence-aware 撮合路径里。",
            "- 没有引入新的搜索空间，只验证两条已选规则，以及它们同时启用的组合。",
            "- 衍生品状态阈值不再用全样本分位数，而是固定用 `2024-01-01 -> 2024-03-19` 的预部署校准窗口，降低前视偏差。",
            "- `2024-01-01` 之前没有衍生品状态覆盖，因此 `full_2020` 里那部分交易默认不 gating。",
            "",
            "## All-Trade Results",
            "",
            markdown_table(
                window_rows,
                [
                    ("window", "Window"),
                    ("variant", "Variant"),
                    ("trades", "Trades"),
                    ("win_rate_pct", "Win Rate %"),
                    ("profit_factor", "PF"),
                    ("expectancy_r", "Exp R"),
                    ("cum_r", "Cum R"),
                    ("max_dd_r", "DD R"),
                    ("signals_now", "Signals NOW"),
                    ("signals_with_state", "Signals With State"),
                    ("gate_coverage_pct", "Gate Coverage %"),
                    ("vetoed_signals", "Vetoed Signals"),
                ],
            ),
            "",
            "## Trade Cohorts",
            "",
            markdown_table(
                cohort_rows,
                [
                    ("window", "Window"),
                    ("variant", "Variant"),
                    ("cohort", "Cohort"),
                    ("trades", "Trades"),
                    ("win_rate_pct", "Win Rate %"),
                    ("expectancy_r", "Exp R"),
                    ("cum_r", "Cum R"),
                ],
            ),
            "",
        ]
    )


def main() -> None:
    configure_logging()
    ensure_output_dir()

    research = load_research_table()
    quantile_edges = build_quantile_edges(research)
    (OUTPUT_DIR / "quantile_edges_pre_deploy.json").write_text(json.dumps(quantile_edges, indent=2), encoding="utf-8")
    derivatives = enrich_derivatives_for_gate(research, quantile_edges)
    derivatives_lookup = {
        row["timestamp"].to_pydatetime(): row
        for row in derivatives.to_dict("records")
    }

    service = make_service()
    variants = build_gate_variants()
    all_window_rows: list[dict[str, Any]] = []
    all_cohort_rows: list[dict[str, Any]] = []

    for window_name, (start, end) in WINDOWS.items():
        enriched = service.prepare_enriched_history(
            exchange="binance",
            market_type="perpetual",
            symbol=SYMBOL,
            strategy_profile=PROFILE,
            start=start,
            end=end,
        )
        results: list[GateVariantResult] = []
        for variant_name, variant_description, gate_rule in variants:
            result = run_with_gate(
                service=service,
                derivatives_lookup=derivatives_lookup,
                symbol=SYMBOL,
                strategy_profile=PROFILE,
                start=start,
                end=end,
                enriched=enriched,
                variant_name=variant_name,
                variant_description=variant_description,
                gate_rule=gate_rule,
            )
            results.append(result)
            trade_rows = [asdict(item) for item in result.trades]
            write_csv(OUTPUT_DIR / f"{window_name}_{variant_name}_trades.csv", trade_rows)
            (OUTPUT_DIR / f"{window_name}_{variant_name}_summary.json").write_text(
                json.dumps(
                    {
                        "variant": result.name,
                        "description": result.description,
                        "summary": asdict(result.summary),
                        "gate_stats": asdict(result.gate_stats),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        window_rows, cohort_rows = build_report_rows(results, derivatives_lookup)
        for row in window_rows:
            row["window"] = window_name
        for row in cohort_rows:
            row["window"] = window_name
        all_window_rows.extend(window_rows)
        all_cohort_rows.extend(cohort_rows)

    write_csv(OUTPUT_DIR / "summary_all.csv", all_window_rows)
    write_csv(OUTPUT_DIR / "cohort_summary_all.csv", all_cohort_rows)
    (OUTPUT_DIR / "report.md").write_text(build_report(all_window_rows, all_cohort_rows), encoding="utf-8")


if __name__ == "__main__":
    main()
