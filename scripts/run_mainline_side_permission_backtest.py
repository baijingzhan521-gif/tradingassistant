from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.service import (  # noqa: E402
    BacktestAssumptions,
    BacktestService,
    BacktestSummary,
    BacktestTrade,
    _OpenPosition,
    _PendingEntry,
)
from app.core.logging import configure_logging  # noqa: E402
from app.data.exchange_client import get_exchange_client_factory  # noqa: E402
from app.data.ohlcv_service import OhlcvService  # noqa: E402
from app.schemas.common import Action, RecommendedTiming  # noqa: E402
from app.services.side_permission_research_service import (  # noqa: E402
    SidePermissionResearchService,
    SidePermissionState,
)
from app.services.strategy_service import StrategyService  # noqa: E402
from app.utils.timeframes import get_strategy_required_timeframes  # noqa: E402


SYMBOL = "BTC/USDT:USDT"
PROFILE = "swing_trend_long_regime_gate_v1"
EXCHANGE = "binance"
MARKET_TYPE = "perpetual"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "backtests" / "mainline_side_permission_backtest"
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
VARIANTS = (
    ("baseline_no_permission", "No permission model applied."),
    ("permission_long_only", "Veto LONG only when model disallows long."),
    ("permission_short_only", "Veto SHORT only when model disallows short."),
    ("permission_full_side_control", "Apply model side permission to both LONG and SHORT."),
)


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
    signal_rows: list[dict[str, Any]]
    vetoed_rows: list[dict[str, Any]]
    gate_stats: GateStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permissions-csv", required=True, help="Hourly permission CSV path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
        rendered: list[str] = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
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


def trades_to_frame(trades: list[BacktestTrade], *, variant: str, description: str) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "variant",
                "description",
                "symbol",
                "strategy_profile",
                "side",
                "higher_bias",
                "trend_strength",
                "signal_time",
                "entry_time",
                "exit_time",
                "bars_held",
                "exit_reason",
                "tp1_hit",
                "tp2_hit",
                "pnl_r",
            ]
        )
    frame = pd.DataFrame([asdict(item) for item in trades])
    frame["variant"] = variant
    frame["description"] = description
    frame["signal_time"] = pd.to_datetime(frame["signal_time"], utc=True)
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True)
    return frame


def summarize_trade_frame(frame: pd.DataFrame) -> dict[str, float | int | None]:
    if frame.empty:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "expectancy_r": 0.0,
            "cum_r": 0.0,
            "profit_factor": None,
        }
    wins = frame.loc[frame["pnl_r"] > 0, "pnl_r"].sum()
    losses = -frame.loc[frame["pnl_r"] < 0, "pnl_r"].sum()
    profit_factor = float(wins / losses) if losses > 0 else None
    return {
        "trades": int(len(frame)),
        "win_rate_pct": round(float((frame["pnl_r"] > 0).mean() * 100.0), 2),
        "expectancy_r": round(float(frame["pnl_r"].mean()), 4),
        "cum_r": round(float(frame["pnl_r"].sum()), 4),
        "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
    }


def attach_permission_coverage(
    *,
    frame: pd.DataFrame,
    permission_service: SidePermissionResearchService,
    permissions: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        result = frame.copy()
        result["permission_covered"] = False
        result["permission_label"] = None
        result["model_version"] = None
        return result

    rows: list[dict[str, Any]] = []
    for trade in frame.to_dict("records"):
        state = permission_service.resolve_permission(permissions, pd.Timestamp(trade["signal_time"]))
        trade["permission_covered"] = state is not None
        trade["permission_label"] = state.permission_label if state is not None else None
        trade["model_version"] = state.model_version if state is not None else None
        rows.append(trade)
    return pd.DataFrame(rows)


def build_summary_rows(results: list[GateVariantResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
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
                "veto_rate_pct": round(
                    (result.gate_stats.vetoed_signals / result.gate_stats.signals_now) * 100.0,
                    2,
                )
                if result.gate_stats.signals_now
                else 0.0,
            }
        )
    return rows


def summarize_veto_audit(vetoed_frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if vetoed_frame.empty:
        return [], []

    side_rows: list[dict[str, Any]] = []
    grouped_side = vetoed_frame.groupby(["variant", "side"], sort=True, dropna=False, observed=True)
    for (variant, side), group in grouped_side:
        baseline_known = group["baseline_trade_pnl_r"].notna()
        baseline_group = group.loc[baseline_known]
        side_rows.append(
            {
                "variant": variant,
                "side": side,
                "vetoed_signals": int(len(group)),
                "covered_baseline_trades": int(len(baseline_group)),
                "baseline_expectancy_r": round(float(baseline_group["baseline_trade_pnl_r"].mean()), 4)
                if not baseline_group.empty
                else None,
                "baseline_cum_r": round(float(baseline_group["baseline_trade_pnl_r"].sum()), 4)
                if not baseline_group.empty
                else None,
                "baseline_stop_loss_rate_pct": round(float((baseline_group["baseline_trade_pnl_r"] < 0).mean() * 100.0), 2)
                if not baseline_group.empty
                else None,
            }
        )

    year_rows: list[dict[str, Any]] = []
    grouped_year = vetoed_frame.groupby(["variant", "year"], sort=True, dropna=False, observed=True)
    for (variant, year), group in grouped_year:
        baseline_known = group["baseline_trade_pnl_r"].notna()
        baseline_group = group.loc[baseline_known]
        year_rows.append(
            {
                "variant": variant,
                "year": int(year),
                "vetoed_signals": int(len(group)),
                "covered_baseline_trades": int(len(baseline_group)),
                "baseline_expectancy_r": round(float(baseline_group["baseline_trade_pnl_r"].mean()), 4)
                if not baseline_group.empty
                else None,
                "baseline_cum_r": round(float(baseline_group["baseline_trade_pnl_r"].sum()), 4)
                if not baseline_group.empty
                else None,
            }
        )
    return side_rows, year_rows


def run_with_permission(
    *,
    service: BacktestService,
    permission_service: SidePermissionResearchService,
    permissions: pd.DataFrame,
    symbol: str,
    strategy_profile: str,
    start: datetime,
    end: datetime,
    enriched: dict[str, pd.DataFrame],
    variant_name: str,
    variant_description: str,
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
    signal_rows: list[dict[str, Any]] = []
    vetoed_rows: list[dict[str, Any]] = []

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
        permission = permission_service.resolve_permission(permissions, pd.Timestamp(ts))
        if permission is not None:
            gate_stats.signals_with_state += 1
        vetoed = permission_service.should_veto(
            variant=variant_name,
            action=signal.action,
            permission=permission,
        )
        signal_rows.append(
            {
                "variant": variant_name,
                "signal_time": pd.Timestamp(ts),
                "side": signal.action.value,
                "higher_bias": signal.bias.value,
                "trend_strength": int(signal.trend_strength),
                "confidence": int(signal.confidence),
                "permission_covered": permission is not None,
                "permission_label": permission.permission_label if permission is not None else None,
                "allow_long": permission.allow_long if permission is not None else None,
                "allow_short": permission.allow_short if permission is not None else None,
                "model_version": permission.model_version if permission is not None else None,
                "vetoed": vetoed,
            }
        )
        if vetoed:
            gate_stats.vetoed_signals += 1
            vetoed_rows.append(
                {
                    "variant": variant_name,
                    "signal_time": pd.Timestamp(ts),
                    "side": signal.action.value,
                    "higher_bias": signal.bias.value,
                    "trend_strength": int(signal.trend_strength),
                    "confidence": int(signal.confidence),
                    "permission_covered": permission is not None,
                    "permission_label": permission.permission_label if permission is not None else None,
                    "allow_long": permission.allow_long if permission is not None else None,
                    "allow_short": permission.allow_short if permission is not None else None,
                    "model_version": permission.model_version if permission is not None else None,
                    "long_score": permission.long_score if permission is not None else None,
                    "short_score": permission.short_score if permission is not None else None,
                    "meta_regime": permission.meta_regime if permission is not None else None,
                    "year": pd.Timestamp(ts).year,
                }
            )
            continue

        pending_entry = _PendingEntry(signal=signal)

    if position is not None and trigger_end_idx > 0:
        final_candle = trigger_frame.iloc[trigger_end_idx - 1]
        trades.append(
            service._close_position(position, final_candle, exit_reason="end_of_test", fill_price=float(final_candle["close"]))
        )

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
        signal_rows=signal_rows,
        vetoed_rows=vetoed_rows,
        gate_stats=gate_stats,
    )


def build_window_report(
    *,
    window_name: str,
    summary_rows: list[dict[str, Any]],
    cohort_rows: list[dict[str, Any]],
    side_rows: list[dict[str, Any]],
    year_rows: list[dict[str, Any]],
    cluster_rows: list[dict[str, Any]],
    veto_side_rows: list[dict[str, Any]],
    veto_year_rows: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            f"# Mainline Side Permission Backtest: {window_name}",
            "",
            "- 这轮只验证外部 side permission model 能否作为研究型状态层提供增量信息。",
            "- 主策略 `swing_trend_long_regime_gate_v1` 的 entry / exit / sizing / confirmation 全冻结。",
            "- gate 只发生在 `signal.recommended_timing == NOW` 之后、挂单之前。",
            "- missing model state 一律 pass-through，并单独统计 coverage。",
            "",
            "## Headline Summary",
            markdown_table(
                summary_rows,
                [
                    ("variant", "Variant"),
                    ("trades", "Trades"),
                    ("win_rate_pct", "Win Rate %"),
                    ("profit_factor", "PF"),
                    ("expectancy_r", "Exp R"),
                    ("cum_r", "Cum R"),
                    ("max_dd_r", "MaxDD R"),
                    ("gate_coverage_pct", "Coverage %"),
                    ("vetoed_signals", "Vetoed"),
                ],
            ),
            "",
            "## Coverage / Cohorts",
            markdown_table(
                cohort_rows,
                [
                    ("variant", "Variant"),
                    ("cohort", "Cohort"),
                    ("trades", "Trades"),
                    ("win_rate_pct", "Win Rate %"),
                    ("profit_factor", "PF"),
                    ("expectancy_r", "Exp R"),
                    ("cum_r", "Cum R"),
                ],
            ),
            "",
            "## Side Attribution",
            markdown_table(
                side_rows,
                [
                    ("variant", "Variant"),
                    ("side", "Side"),
                    ("trades", "Trades"),
                    ("win_rate_pct", "Win Rate %"),
                    ("expectancy_r", "Exp R"),
                    ("cumulative_r", "Cum R"),
                    ("profit_factor", "PF"),
                ],
            ),
            "",
            "## Annual Distribution",
            markdown_table(
                year_rows,
                [
                    ("variant", "Variant"),
                    ("year", "Year"),
                    ("trades", "Trades"),
                    ("win_rate_pct", "Win Rate %"),
                    ("expectancy_r", "Exp R"),
                    ("cumulative_r", "Cum R"),
                    ("profit_factor", "PF"),
                ],
            ),
            "",
            "## Loss Clusters",
            markdown_table(
                cluster_rows,
                [
                    ("variant", "Variant"),
                    ("clusters", "Clusters"),
                    ("loss_trades_in_clusters", "Loss Trades"),
                    ("avg_cluster_length", "Avg Len"),
                    ("max_cluster_length", "Max Len"),
                    ("avg_cluster_cumulative_r", "Avg Cluster R"),
                    ("worst_cluster_cumulative_r", "Worst Cluster R"),
                    ("worst_cluster_length", "Worst Len"),
                ],
            ),
            "",
            "## Veto Audit By Side",
            markdown_table(
                veto_side_rows,
                [
                    ("variant", "Variant"),
                    ("side", "Side"),
                    ("vetoed_signals", "Vetoed"),
                    ("covered_baseline_trades", "Matched Baseline"),
                    ("baseline_expectancy_r", "Baseline Exp R"),
                    ("baseline_cum_r", "Baseline Cum R"),
                    ("baseline_stop_loss_rate_pct", "Baseline Stop %"),
                ],
            ),
            "",
            "## Veto Audit By Year",
            markdown_table(
                veto_year_rows,
                [
                    ("variant", "Variant"),
                    ("year", "Year"),
                    ("vetoed_signals", "Vetoed"),
                    ("covered_baseline_trades", "Matched Baseline"),
                    ("baseline_expectancy_r", "Baseline Exp R"),
                    ("baseline_cum_r", "Baseline Cum R"),
                ],
            ),
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    configure_logging()

    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    permission_service = SidePermissionResearchService()
    permissions = permission_service.load_permission_csv(args.permissions_csv)
    service = make_service()

    all_summary_rows: list[dict[str, Any]] = []
    all_cohort_rows: list[dict[str, Any]] = []
    all_year_rows: list[dict[str, Any]] = []
    all_side_rows: list[dict[str, Any]] = []
    all_cluster_summary_rows: list[dict[str, Any]] = []
    all_vetoed_rows: list[dict[str, Any]] = []
    all_trades_rows: list[dict[str, Any]] = []

    for window_name, (start, end) in WINDOWS.items():
        window_dir = output_dir / window_name
        ensure_output_dir(window_dir)

        enriched = service.prepare_enriched_history(
            exchange=EXCHANGE,
            market_type=MARKET_TYPE,
            symbol=SYMBOL,
            strategy_profile=PROFILE,
            start=start,
            end=end,
        )
        results: list[GateVariantResult] = []
        for variant_name, variant_description in VARIANTS:
            results.append(
                run_with_permission(
                    service=service,
                    permission_service=permission_service,
                    permissions=permissions,
                    symbol=SYMBOL,
                    strategy_profile=PROFILE,
                    start=start,
                    end=end,
                    enriched=enriched,
                    variant_name=variant_name,
                    variant_description=variant_description,
                )
            )

        summary_rows = build_summary_rows(results)
        summary_by_variant = {row["variant"]: row for row in summary_rows}

        trades_frames: list[pd.DataFrame] = []
        for result in results:
            frame = trades_to_frame(result.trades, variant=result.name, description=result.description)
            frame = attach_permission_coverage(frame=frame, permission_service=permission_service, permissions=permissions)
            if not frame.empty:
                frame["window"] = window_name
                frame["year"] = frame["signal_time"].dt.year.astype(int)
            trades_frames.append(frame)

        trades_frame = pd.concat(trades_frames, ignore_index=True) if trades_frames else pd.DataFrame()
        if not trades_frame.empty:
            write_csv(window_dir / "trades.csv", trades_frame.to_dict("records"))
            all_trades_rows.extend(trades_frame.to_dict("records"))

        cohort_rows: list[dict[str, Any]] = []
        year_rows: list[dict[str, Any]] = []
        side_rows: list[dict[str, Any]] = []
        cluster_summary_rows: list[dict[str, Any]] = []
        vetoed_rows: list[dict[str, Any]] = []

        baseline_frame = trades_frame[trades_frame["variant"] == "baseline_no_permission"].copy() if not trades_frame.empty else pd.DataFrame()
        baseline_signal_map = {}
        if not baseline_frame.empty:
            baseline_signal_map = (
                baseline_frame.sort_values("signal_time")
                .drop_duplicates(subset=["signal_time"], keep="last")
                .set_index("signal_time")[
                    ["pnl_r", "exit_reason", "bars_held", "side"]
                ]
                .to_dict("index")
            )

        for result in results:
            variant_frame = trades_frame[trades_frame["variant"] == result.name].copy() if not trades_frame.empty else pd.DataFrame()
            for cohort_name, cohort_frame in [
                ("all_trades", variant_frame),
                ("covered_only", variant_frame[variant_frame["permission_covered"]]),
            ]:
                cohort_rows.append(
                    {
                        "variant": result.name,
                        "cohort": cohort_name,
                        **summarize_trade_frame(cohort_frame),
                    }
                )

            year_rows.extend(permission_service.summarize_trade_distribution(variant_frame, group_cols=["variant", "year"]))
            side_rows.extend(permission_service.summarize_trade_distribution(variant_frame, group_cols=["variant", "side"]))

            clusters = permission_service.identify_loss_clusters(
                variant_frame,
                group_cols=["window", "variant"],
            )
            cluster_summary = permission_service.summarize_loss_clusters(
                clusters,
                group_cols=["window", "variant"],
            )
            for row in cluster_summary:
                row["variant"] = result.name
            cluster_summary_rows.extend(cluster_summary)

            for row in result.vetoed_rows:
                signal_time = pd.Timestamp(row["signal_time"])
                baseline_row = baseline_signal_map.get(signal_time)
                row["window"] = window_name
                row["baseline_trade_found"] = baseline_row is not None
                row["baseline_trade_pnl_r"] = baseline_row["pnl_r"] if baseline_row is not None else None
                row["baseline_trade_exit_reason"] = baseline_row["exit_reason"] if baseline_row is not None else None
                row["baseline_trade_bars_held"] = baseline_row["bars_held"] if baseline_row is not None else None
                row["baseline_trade_side"] = baseline_row["side"] if baseline_row is not None else None
                vetoed_rows.append(row)

        veto_side_rows, veto_year_rows = summarize_veto_audit(pd.DataFrame(vetoed_rows))

        for row in summary_rows:
            row["window"] = window_name
        for row in cohort_rows:
            row["window"] = window_name
        for row in year_rows:
            row["window"] = window_name
        for row in side_rows:
            row["window"] = window_name
        for row in cluster_summary_rows:
            row["window"] = window_name

        write_csv(window_dir / "summary_all.csv", summary_rows)
        write_csv(window_dir / "cohort_summary_all.csv", cohort_rows)
        write_csv(window_dir / "year_summary.csv", year_rows)
        write_csv(window_dir / "side_summary.csv", side_rows)
        write_csv(window_dir / "loss_cluster_summary.csv", cluster_summary_rows)
        write_csv(window_dir / "vetoed_signals.csv", vetoed_rows)
        (window_dir / "report.md").write_text(
            build_window_report(
                window_name=window_name,
                summary_rows=summary_rows,
                cohort_rows=cohort_rows,
                side_rows=side_rows,
                year_rows=year_rows,
                cluster_rows=cluster_summary_rows,
                veto_side_rows=veto_side_rows,
                veto_year_rows=veto_year_rows,
            ),
            encoding="utf-8",
        )

        all_summary_rows.extend(summary_rows)
        all_cohort_rows.extend(cohort_rows)
        all_year_rows.extend(year_rows)
        all_side_rows.extend(side_rows)
        all_cluster_summary_rows.extend(cluster_summary_rows)
        all_vetoed_rows.extend(vetoed_rows)

        (window_dir / "model_contract.json").write_text(
            json.dumps(
                {
                    "required_columns": ["timestamp", "permission_label", "allow_long", "allow_short", "model_version"],
                    "optional_columns": ["long_score", "short_score", "meta_regime"],
                    "model_version_values": sorted(set(permissions["model_version"].astype(str))),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    write_csv(output_dir / "summary_all.csv", all_summary_rows)
    write_csv(output_dir / "cohort_summary_all.csv", all_cohort_rows)
    write_csv(output_dir / "year_summary.csv", all_year_rows)
    write_csv(output_dir / "side_summary.csv", all_side_rows)
    write_csv(output_dir / "loss_cluster_summary.csv", all_cluster_summary_rows)
    write_csv(output_dir / "vetoed_signals.csv", all_vetoed_rows)
    if all_trades_rows:
        write_csv(output_dir / "trades.csv", all_trades_rows)
    print(f"saved side permission research artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
