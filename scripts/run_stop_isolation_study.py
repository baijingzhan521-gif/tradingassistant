from __future__ import annotations

import argparse
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

from app.backtesting.service import BacktestAssumptions, BacktestService, BacktestTrade, _OpenPosition
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.schemas.common import Action, Bias
from app.services.strategy_service import StrategyService


EXIT_ASSUMPTIONS = {
    "exit_profile": "stop_isolation_fixed_targets",
    "take_profit_mode": "scaled",
    "scaled_tp1_r": 1.0,
    "scaled_tp2_r": 3.0,
    "long_exit": {"take_profit_mode": "scaled", "scaled_tp1_r": 1.0, "scaled_tp2_r": 3.0},
    "short_exit": {"take_profit_mode": "fixed_r", "fixed_take_profit_r": 1.5},
    "swing_detection_mode": "confirmed",
}

STOP_MODES = (
    ("structure", "Structure + ATR"),
    ("ema21", "EMA21 Initial Stop"),
    ("ema55", "EMA55 Initial Stop"),
    ("hybrid_ema55_cap", "Hybrid: Cap At EMA55"),
    ("structure_ema21_trail_after_tp1", "Structure + EMA21 Trail After TP1"),
)

WINDOW_PRESETS = {
    "two_year": ("2024-03-19", "2026-03-19"),
    "full_2020": ("2020-03-19", "2026-03-19"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stop-only isolation study with mainline entries and absolute price targets fixed."
    )
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument(
        "--windows",
        default="two_year,full_2020",
        help="Comma-separated presets: two_year,full_2020",
    )
    parser.add_argument(
        "--stop-modes",
        default=",".join(mode for mode, _ in STOP_MODES),
        help="Comma-separated stop modes",
    )
    parser.add_argument(
        "--baseline-dir",
        default="artifacts/backtests/stop_ablation_mainline",
        help="Directory containing *_structure_trades.csv from the full backtest run.",
    )
    parser.add_argument("--output-dir", default="artifacts/backtests/stop_isolation_mainline")
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def build_service() -> BacktestService:
    return BacktestService(
        ohlcv_service=OhlcvService(get_exchange_client_factory()),
        strategy_service=StrategyService(),
        assumptions=BacktestAssumptions(**EXIT_ASSUMPTIONS),
    )


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
    body = []
    for row in rows:
        parts = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, float):
                parts.append(f"{value:.4f}")
            else:
                parts.append(str(value))
        body.append("| " + " | ".join(parts) + " |")
    return "\n".join([header, divider, *body])


def load_baseline_trades(path: Path) -> pd.DataFrame:
    trades = pd.read_csv(path)
    if trades.empty:
        raise ValueError(f"Baseline trades CSV is empty: {path}")
    for column in ("signal_time", "entry_time", "exit_time"):
        trades[column] = pd.to_datetime(trades[column], utc=True)
    return trades.sort_values("entry_time").reset_index(drop=True)


def build_timestamp_index(frame: pd.DataFrame) -> dict[str, int]:
    result: dict[str, int] = {}
    for idx, value in enumerate(frame["timestamp"]):
        ts = pd.Timestamp(value)
        result[ts.isoformat()] = idx
    return result


def resolve_initial_stop(
    *,
    stop_mode: str,
    trade_row: pd.Series,
    signal_candle: pd.Series,
) -> float:
    base_stop = float(trade_row["stop_price"])
    entry_price = float(trade_row["entry_price"])
    side = str(trade_row["side"])
    atr14 = float(signal_candle["atr14"])
    ema21 = float(signal_candle["ema21"])
    ema55 = float(signal_candle["ema55"])
    epsilon = max(atr14 * 0.05, 1e-6)

    if stop_mode in {"structure", "structure_ema21_trail_after_tp1"}:
        stop_price = base_stop
    elif stop_mode == "ema21":
        stop_price = ema21
    elif stop_mode == "ema55":
        stop_price = ema55
    elif stop_mode == "hybrid_ema55_cap":
        if side == "LONG":
            stop_price = max(base_stop, ema55)
        else:
            stop_price = min(base_stop, ema55)
    else:
        raise ValueError(f"Unsupported stop mode: {stop_mode}")

    if side == "LONG" and stop_price >= entry_price - epsilon:
        return min(base_stop, entry_price - epsilon)
    if side == "SHORT" and stop_price <= entry_price + epsilon:
        return max(base_stop, entry_price + epsilon)
    return float(stop_price)


def build_position(
    *,
    trade_row: pd.Series,
    strategy_profile: str,
    stop_price: float,
    service: BacktestService,
) -> _OpenPosition:
    side = Action(str(trade_row["side"]))
    if side == Action.LONG:
        take_profit_mode = "scaled"
        fixed_take_profit_r = None
    else:
        take_profit_mode = "fixed_r"
        fixed_take_profit_r = 1.5

    higher_bias_raw = str(trade_row["higher_bias"]).lower()
    if higher_bias_raw == "bullish":
        higher_bias = Bias.BULLISH
    elif higher_bias_raw == "bearish":
        higher_bias = Bias.BEARISH
    else:
        higher_bias = Bias.NEUTRAL

    position = _OpenPosition(
        symbol=str(trade_row["symbol"]),
        strategy_profile=strategy_profile,
        side=side,
        higher_bias=higher_bias,
        trend_strength=int(trade_row["trend_strength"]),
        signal_time=pd.Timestamp(trade_row["signal_time"]).to_pydatetime(),
        entry_time=pd.Timestamp(trade_row["entry_time"]).to_pydatetime(),
        entry_price=float(trade_row["entry_price"]),
        initial_stop_price=float(stop_price),
        current_stop_price=float(stop_price),
        tp1_price=float(trade_row["tp1_price"]),
        tp2_price=float(trade_row["tp2_price"]),
        take_profit_mode=take_profit_mode,
        fixed_take_profit_r=fixed_take_profit_r,
        confidence=int(trade_row["confidence"]),
        tp1_scale_out=float(service.assumptions.tp1_scale_out),
        move_stop_to_entry_after_tp1=bool(service.assumptions.move_stop_to_entry_after_tp1),
        max_hold_bars=int(service._default_max_hold_bars("swing_trend_long_regime_gate_v1")),
    )
    entry_fee = position.entry_price * (service.assumptions.taker_fee_bps / 10000)
    position.fees_quote += entry_fee
    position.realized_pnl_quote -= entry_fee
    position.last_fill_price = position.entry_price
    return position


def simulate_trade(
    *,
    service: BacktestService,
    trade_row: pd.Series,
    trigger_frame: pd.DataFrame,
    signal_idx: int,
    entry_idx: int,
    stop_mode: str,
    strategy_profile: str,
) -> BacktestTrade:
    signal_candle = trigger_frame.iloc[signal_idx]
    stop_price = resolve_initial_stop(stop_mode=stop_mode, trade_row=trade_row, signal_candle=signal_candle)
    position = build_position(
        trade_row=trade_row,
        strategy_profile=strategy_profile,
        stop_price=stop_price,
        service=service,
    )
    max_hold_bars = int(position.max_hold_bars or service._default_max_hold_bars("swing_trend_long_regime_gate_v1"))

    for idx in range(entry_idx, len(trigger_frame)):
        candle = trigger_frame.iloc[idx]

        if stop_mode == "structure_ema21_trail_after_tp1" and position.tp1_hit and idx > 0:
            prev_candle = trigger_frame.iloc[idx - 1]
            ema21_prev = float(prev_candle["ema21"])
            if position.side == Action.LONG:
                position.current_stop_price = max(position.current_stop_price, ema21_prev)
            else:
                position.current_stop_price = min(position.current_stop_price, ema21_prev)

        closed = service._update_open_position(
            position=position,
            candle=candle,
            max_hold_bars=max_hold_bars,
        )
        if closed is not None:
            return closed

    final_candle = trigger_frame.iloc[-1]
    return service._close_position(position, final_candle, exit_reason="end_of_test", fill_price=float(final_candle["close"]))


def validate_structure_replay(baseline: pd.DataFrame, simulated: pd.DataFrame) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    simulated = simulated[
        [
            "signal_time",
            "entry_time",
            "exit_time",
            "side",
            "exit_reason",
            "exit_price",
            "tp1_hit",
            "tp2_hit",
            "pnl_r",
        ]
    ].copy()
    for column in ("signal_time", "entry_time", "exit_time"):
        simulated[column] = pd.to_datetime(simulated[column], utc=True)
    merged = baseline.merge(
        simulated,
        on=["signal_time", "entry_time", "side"],
        suffixes=("_baseline", "_simulated"),
        how="inner",
    )
    if len(merged) != len(baseline):
        raise ValueError("Structure replay validation failed: simulated trade count does not match baseline count.")

    for _, row in merged.iterrows():
        same_reason = str(row["exit_reason_baseline"]) == str(row["exit_reason_simulated"])
        same_tp1 = bool(row["tp1_hit_baseline"]) == bool(row["tp1_hit_simulated"])
        same_tp2 = bool(row["tp2_hit_baseline"]) == bool(row["tp2_hit_simulated"])
        same_exit = abs(float(row["exit_price_baseline"]) - float(row["exit_price_simulated"])) <= 1e-4
        same_r = abs(float(row["pnl_r_baseline"]) - float(row["pnl_r_simulated"])) <= 1e-4
        if not (same_reason and same_tp1 and same_tp2 and same_exit and same_r):
            mismatches.append(
                {
                    "signal_time": pd.Timestamp(row["signal_time"]).isoformat(),
                    "side": row["side"],
                    "baseline_exit_reason": row["exit_reason_baseline"],
                    "simulated_exit_reason": row["exit_reason_simulated"],
                    "baseline_exit_price": round(float(row["exit_price_baseline"]), 6),
                    "simulated_exit_price": round(float(row["exit_price_simulated"]), 6),
                    "baseline_pnl_r": round(float(row["pnl_r_baseline"]), 6),
                    "simulated_pnl_r": round(float(row["pnl_r_simulated"]), 6),
                }
            )
    return mismatches


def summarize_results(rows: pd.DataFrame) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for (window, stop_mode, label), group in rows.groupby(["window", "stop_mode", "label"], sort=False):
        wins = float(group.loc[group["pnl_r_baseline"] > 0, "pnl_r_baseline"].sum())
        losses = abs(float(group.loc[group["pnl_r_baseline"] < 0, "pnl_r_baseline"].sum()))
        profit_factor = wins / losses if losses > 0 else (999.0 if wins > 0 else 0.0)
        baseline_tp1_mask = group["baseline_tp1_hit"].astype(bool)
        baseline_tp2_mask = group["baseline_tp2_hit"].astype(bool)
        baseline_stop_mask = group["baseline_exit_reason"] == "stop_loss"

        summaries.append(
            {
                "window": window,
                "stop_mode": stop_mode,
                "label": label,
                "trades": int(len(group)),
                "win_rate_pct": round(float((group["pnl_r_baseline"] > 0).mean() * 100), 2),
                "profit_factor_baseline_r": round(float(profit_factor), 4),
                "cum_baseline_r": round(float(group["pnl_r_baseline"].sum()), 4),
                "avg_baseline_r": round(float(group["pnl_r_baseline"].mean()), 4),
                "cum_pnl_pct": round(float(group["pnl_pct"].sum()), 4),
                "avg_pnl_pct": round(float(group["pnl_pct"].mean()), 4),
                "avg_stop_distance_pct": round(float(group["stop_distance_pct"].mean()), 4),
                "median_risk_vs_baseline": round(float(group["risk_vs_baseline"].median()), 4),
                "tp1_hit_rate_pct": round(float(group["tp1_hit"].mean() * 100), 2),
                "tp2_hit_rate_pct": round(float(group["tp2_hit"].mean() * 100), 2),
                "stop_loss_rate_pct": round(float((group["exit_reason"] == "stop_loss").mean() * 100), 2),
                "breakeven_after_tp1_rate_pct": round(
                    float((group["exit_reason"] == "breakeven_after_tp1").mean() * 100), 2
                ),
                "time_stop_rate_pct": round(float((group["exit_reason"] == "time_stop").mean() * 100), 2),
                "premature_stop_vs_baseline_tp1_pct": round(
                    float(((group["exit_reason"] == "stop_loss") & baseline_tp1_mask).mean() * 100), 2
                ),
                "premature_stop_vs_baseline_tp2_pct": round(
                    float(((group["exit_reason"] == "stop_loss") & baseline_tp2_mask).mean() * 100), 2
                ),
                "preserve_baseline_tp2_pct": (
                    round(float(group.loc[baseline_tp2_mask, "tp2_hit"].mean() * 100), 2)
                    if baseline_tp2_mask.any()
                    else 0.0
                ),
                "upgrade_baseline_stop_pct": (
                    round(float(group.loc[baseline_stop_mask, "pnl_r_baseline"].gt(0).mean() * 100), 2)
                    if baseline_stop_mask.any()
                    else 0.0
                ),
            }
        )
    return summaries


def summarize_side_results(rows: pd.DataFrame) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for (window, stop_mode, label, side), group in rows.groupby(["window", "stop_mode", "label", "side"], sort=False):
        wins = float(group.loc[group["pnl_r_baseline"] > 0, "pnl_r_baseline"].sum())
        losses = abs(float(group.loc[group["pnl_r_baseline"] < 0, "pnl_r_baseline"].sum()))
        profit_factor = wins / losses if losses > 0 else (999.0 if wins > 0 else 0.0)
        summaries.append(
            {
                "window": window,
                "stop_mode": stop_mode,
                "label": label,
                "side": side,
                "trades": int(len(group)),
                "cum_baseline_r": round(float(group["pnl_r_baseline"].sum()), 4),
                "avg_baseline_r": round(float(group["pnl_r_baseline"].mean()), 4),
                "profit_factor_baseline_r": round(float(profit_factor), 4),
                "tp2_hit_rate_pct": round(float(group["tp2_hit"].mean() * 100), 2),
                "stop_loss_rate_pct": round(float((group["exit_reason"] == "stop_loss").mean() * 100), 2),
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    configure_logging()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = ROOT / args.baseline_dir
    selected_windows = [item.strip() for item in args.windows.split(",") if item.strip()]
    stop_mode_labels = dict(STOP_MODES)
    selected_modes = [item.strip() for item in args.stop_modes.split(",") if item.strip()]

    unknown_windows = sorted(set(selected_windows) - set(WINDOW_PRESETS))
    if unknown_windows:
        raise ValueError(f"Unsupported windows: {', '.join(unknown_windows)}")
    unknown_modes = sorted(set(selected_modes) - set(stop_mode_labels))
    if unknown_modes:
        raise ValueError(f"Unsupported stop modes: {', '.join(unknown_modes)}")

    service = build_service()
    base_strategy = service.strategy_service.build_strategy("swing_trend_long_regime_gate_v1")
    all_trade_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []
    all_side_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    report_sections = [
        "# Stop Isolation Study",
        "",
        "- 这次不是完整可部署 backtest，而是固定 `structure` 主线已经发生的 entry 与绝对价格目标，只让 stop 变化。",
        "- 固定项：`signal_time / entry_time / entry_price / tp1_price / tp2_price`。",
        "- 可变项：初始 stop，和 `TP1` 之后是否用 `EMA21` trailing。",
        "- 这里的 `cum_baseline_r` 用的是基线结构止损的原始 `R` 尺度，不是替代 stop 自己的新 `R`。这一步就是为了隔离 stop 本身，不让风险刻度一起跟着变。",
        "",
    ]

    for window_name in selected_windows:
        start_raw, end_raw = WINDOW_PRESETS[window_name]
        start = parse_date(start_raw)
        end = parse_date(end_raw)
        baseline_path = baseline_dir / f"{window_name}_structure_trades.csv"
        baseline_trades = load_baseline_trades(baseline_path)

        base_frames = service.prepare_history(
            exchange=args.exchange,
            market_type=args.market_type,
            symbol=args.symbol,
            strategy_profile="swing_trend_long_regime_gate_v1",
            start=start,
            end=end,
        )
        trigger_frame = service._enrich_frame(base_strategy, "1h", base_frames["1h"]).reset_index(drop=True)
        trigger_index = build_timestamp_index(trigger_frame)

        window_results: list[dict[str, Any]] = []
        structure_simulated: pd.DataFrame | None = None

        for stop_mode in selected_modes:
            strategy_profile = f"stop_isolation_{stop_mode}"
            simulated_trades: list[dict[str, Any]] = []

            for _, trade_row in baseline_trades.iterrows():
                signal_key = pd.Timestamp(trade_row["signal_time"]).isoformat()
                entry_key = pd.Timestamp(trade_row["entry_time"]).isoformat()
                if signal_key not in trigger_index:
                    raise ValueError(f"Signal timestamp not found in trigger frame: {signal_key}")
                if entry_key not in trigger_index:
                    raise ValueError(f"Entry timestamp not found in trigger frame: {entry_key}")

                signal_idx = trigger_index[signal_key]
                entry_idx = trigger_index[entry_key]
                simulated = simulate_trade(
                    service=service,
                    trade_row=trade_row,
                    trigger_frame=trigger_frame,
                    signal_idx=signal_idx,
                    entry_idx=entry_idx,
                    stop_mode=stop_mode,
                    strategy_profile=strategy_profile,
                )
                trade_dict = asdict(simulated)
                baseline_risk = abs(float(trade_row["entry_price"]) - float(trade_row["stop_price"]))
                alt_risk = abs(float(trade_dict["entry_price"]) - float(trade_dict["stop_price"]))
                baseline_pnl_r = float(trade_dict["gross_pnl_quote"]) / baseline_risk if baseline_risk else 0.0
                stop_distance_pct = abs(float(trade_dict["entry_price"]) - float(trade_dict["stop_price"])) / float(
                    trade_dict["entry_price"]
                ) * 100

                merged_row = {
                    "window": window_name,
                    "stop_mode": stop_mode,
                    "label": stop_mode_labels[stop_mode],
                    **trade_dict,
                    "baseline_exit_reason": str(trade_row["exit_reason"]),
                    "baseline_tp1_hit": bool(trade_row["tp1_hit"]),
                    "baseline_tp2_hit": bool(trade_row["tp2_hit"]),
                    "baseline_pnl_r": round(float(trade_row["pnl_r"]), 4),
                    "baseline_risk_abs": round(float(baseline_risk), 6),
                    "pnl_r_baseline": round(float(baseline_pnl_r), 4),
                    "risk_vs_baseline": round(float(alt_risk / baseline_risk), 4) if baseline_risk else 0.0,
                    "stop_distance_pct": round(float(stop_distance_pct), 4),
                }
                simulated_trades.append(merged_row)
                window_results.append(merged_row)
                all_trade_rows.append(merged_row)

            mode_df = pd.DataFrame(simulated_trades)
            write_csv(output_dir / f"{window_name}_{stop_mode}_trades.csv", mode_df.to_dict(orient="records"))
            if stop_mode == "structure":
                structure_simulated = mode_df

        if structure_simulated is None:
            raise ValueError("Structure mode must be included for validation.")

        mismatches = validate_structure_replay(baseline_trades, structure_simulated)
        validation_rows.append(
            {
                "window": window_name,
                "baseline_trades": int(len(baseline_trades)),
                "structure_replay_mismatches": int(len(mismatches)),
            }
        )
        if mismatches:
            mismatch_path = output_dir / f"{window_name}_structure_validation_mismatches.csv"
            write_csv(mismatch_path, mismatches)
            raise ValueError(f"Structure replay validation failed for {window_name}: see {mismatch_path}")

        window_df = pd.DataFrame(window_results)
        summary_rows = summarize_results(window_df)
        side_rows = summarize_side_results(window_df)
        summary_rows.sort(key=lambda item: (item["cum_baseline_r"], item["profit_factor_baseline_r"]), reverse=True)
        side_rows.sort(key=lambda item: (item["label"], item["side"]))

        all_summary_rows.extend(summary_rows)
        all_side_rows.extend(side_rows)
        write_csv(output_dir / f"{window_name}_summary.csv", summary_rows)
        write_csv(output_dir / f"{window_name}_side_summary.csv", side_rows)

        report_sections.extend(
            [
                f"## {window_name}",
                "",
                markdown_table(
                    summary_rows,
                    [
                        ("label", "Stop Mode"),
                        ("trades", "Trades"),
                        ("profit_factor_baseline_r", "PF Baseline R"),
                        ("cum_baseline_r", "Cum Baseline R"),
                        ("avg_baseline_r", "Avg Baseline R"),
                        ("cum_pnl_pct", "Cum PnL %"),
                        ("avg_stop_distance_pct", "Avg Stop Dist %"),
                        ("premature_stop_vs_baseline_tp1_pct", "Premature Stop vs Base TP1 %"),
                        ("preserve_baseline_tp2_pct", "Preserve Base TP2 %"),
                    ],
                ),
                "",
                "按方向拆开：",
                "",
                markdown_table(
                    side_rows,
                    [
                        ("label", "Stop Mode"),
                        ("side", "Side"),
                        ("trades", "Trades"),
                        ("cum_baseline_r", "Cum Baseline R"),
                        ("avg_baseline_r", "Avg Baseline R"),
                        ("profit_factor_baseline_r", "PF Baseline R"),
                        ("tp2_hit_rate_pct", "TP2 Hit %"),
                        ("stop_loss_rate_pct", "Stop Loss %"),
                    ],
                ),
                "",
            ]
        )

    write_csv(output_dir / "summary_all.csv", all_summary_rows)
    write_csv(output_dir / "side_summary_all.csv", all_side_rows)
    write_csv(output_dir / "validation.csv", validation_rows)
    write_csv(output_dir / "trades_all.csv", all_trade_rows)
    (output_dir / "report.md").write_text("\n".join(report_sections).strip() + "\n", encoding="utf-8")

    print(f"Saved report: {output_dir / 'report.md'}")
    print(f"Saved summary CSV: {output_dir / 'summary_all.csv'}")
    print(f"Saved side summary CSV: {output_dir / 'side_summary_all.csv'}")
    print(f"Saved validation CSV: {output_dir / 'validation.csv'}")


if __name__ == "__main__":
    main()
