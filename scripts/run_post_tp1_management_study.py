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
    "exit_profile": "post_tp1_management_study",
    "take_profit_mode": "scaled",
    "scaled_tp1_r": 1.0,
    "scaled_tp2_r": 3.0,
    "long_exit": {"take_profit_mode": "scaled", "scaled_tp1_r": 1.0, "scaled_tp2_r": 3.0},
    "short_exit": {"take_profit_mode": "fixed_r", "fixed_take_profit_r": 1.5},
    "swing_detection_mode": "confirmed",
}

MANAGEMENT_PROFILES = (
    ("baseline_be_after_tp1", "Baseline: BE After TP1", True, "none"),
    ("hold_structure_after_tp1", "Hold Structure Stop After TP1", False, "none"),
    ("be_plus_ema21_trail_after_tp1", "BE + EMA21 Trail After TP1", True, "ema21"),
    ("be_plus_ema55_trail_after_tp1", "BE + EMA55 Trail After TP1", True, "ema55"),
    ("be_plus_atr1_trail_after_tp1", "BE + ATR(1x) Trail After TP1", True, "atr1"),
)

WINDOW_PRESETS = {
    "two_year": ("2024-03-19", "2026-03-19"),
    "full_2020": ("2020-03-19", "2026-03-19"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study post-TP1 management with entries and initial stop fixed.")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument(
        "--windows",
        default="two_year,full_2020",
        help="Comma-separated presets: two_year,full_2020",
    )
    parser.add_argument(
        "--profiles",
        default=",".join(name for name, *_ in MANAGEMENT_PROFILES),
        help="Comma-separated management profiles",
    )
    parser.add_argument(
        "--baseline-dir",
        default="artifacts/backtests/stop_ablation_mainline",
        help="Directory containing *_structure_trades.csv from the full backtest run.",
    )
    parser.add_argument("--output-dir", default="artifacts/backtests/post_tp1_management_mainline")
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


def build_position(
    *,
    trade_row: pd.Series,
    strategy_profile: str,
    move_stop_to_entry_after_tp1: bool,
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
        initial_stop_price=float(trade_row["stop_price"]),
        current_stop_price=float(trade_row["stop_price"]),
        tp1_price=float(trade_row["tp1_price"]),
        tp2_price=float(trade_row["tp2_price"]),
        take_profit_mode=take_profit_mode,
        fixed_take_profit_r=fixed_take_profit_r,
        confidence=int(trade_row["confidence"]),
        tp1_scale_out=float(service.assumptions.tp1_scale_out),
        move_stop_to_entry_after_tp1=move_stop_to_entry_after_tp1,
        max_hold_bars=int(service._default_max_hold_bars("swing_trend_long_regime_gate_v1")),
    )
    entry_fee = position.entry_price * (service.assumptions.taker_fee_bps / 10000)
    position.fees_quote += entry_fee
    position.realized_pnl_quote -= entry_fee
    position.last_fill_price = position.entry_price
    return position


def apply_trailing_stop(
    *,
    position: _OpenPosition,
    trail_mode: str,
    prev_candle: pd.Series,
) -> None:
    if trail_mode == "none":
        return
    close_prev = float(prev_candle["close"])
    ema21_prev = float(prev_candle["ema21"])
    ema55_prev = float(prev_candle["ema55"])
    atr14_prev = float(prev_candle["atr14"])

    if trail_mode == "ema21":
        candidate = ema21_prev
    elif trail_mode == "ema55":
        candidate = ema55_prev
    elif trail_mode == "atr1":
        candidate = close_prev - atr14_prev if position.side == Action.LONG else close_prev + atr14_prev
    else:
        raise ValueError(f"Unsupported trail mode: {trail_mode}")

    if position.side == Action.LONG:
        position.current_stop_price = max(position.current_stop_price, candidate)
    else:
        position.current_stop_price = min(position.current_stop_price, candidate)


def simulate_trade(
    *,
    service: BacktestService,
    trade_row: pd.Series,
    trigger_frame: pd.DataFrame,
    entry_idx: int,
    move_stop_to_entry_after_tp1: bool,
    trail_mode: str,
    strategy_profile: str,
) -> BacktestTrade:
    position = build_position(
        trade_row=trade_row,
        strategy_profile=strategy_profile,
        move_stop_to_entry_after_tp1=move_stop_to_entry_after_tp1,
        service=service,
    )
    max_hold_bars = int(position.max_hold_bars or service._default_max_hold_bars("swing_trend_long_regime_gate_v1"))

    for idx in range(entry_idx, len(trigger_frame)):
        candle = trigger_frame.iloc[idx]

        if position.tp1_hit and idx > 0:
            prev_candle = trigger_frame.iloc[idx - 1]
            apply_trailing_stop(position=position, trail_mode=trail_mode, prev_candle=prev_candle)

        closed = service._update_open_position(
            position=position,
            candle=candle,
            max_hold_bars=max_hold_bars,
        )
        if closed is not None:
            return closed

    final_candle = trigger_frame.iloc[-1]
    return service._close_position(position, final_candle, exit_reason="end_of_test", fill_price=float(final_candle["close"]))


def validate_baseline_replay(baseline: pd.DataFrame, simulated: pd.DataFrame) -> list[dict[str, Any]]:
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
        raise ValueError("Baseline replay validation failed: simulated trade count does not match baseline count.")

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
    for (window, profile, label), group in rows.groupby(["window", "profile", "label"], sort=False):
        baseline_tp2_mask = group["baseline_tp2_hit"].astype(bool)
        baseline_be_mask = group["baseline_exit_reason"] == "breakeven_after_tp1"
        delta = group["pnl_r_baseline"] - group["baseline_pnl_r"]
        wins = float(group.loc[group["pnl_r_baseline"] > 0, "pnl_r_baseline"].sum())
        losses = abs(float(group.loc[group["pnl_r_baseline"] < 0, "pnl_r_baseline"].sum()))
        profit_factor = wins / losses if losses > 0 else (999.0 if wins > 0 else 0.0)

        summaries.append(
            {
                "window": window,
                "profile": profile,
                "label": label,
                "trades": int(len(group)),
                "profit_factor_baseline_r": round(float(profit_factor), 4),
                "cum_baseline_r": round(float(group["pnl_r_baseline"].sum()), 4),
                "cum_delta_vs_baseline_r": round(float(delta.sum()), 4),
                "avg_delta_vs_baseline_r": round(float(delta.mean()), 4),
                "cum_pnl_pct": round(float(group["pnl_pct"].sum()), 4),
                "tp2_hit_rate_pct": round(float(group["tp2_hit"].mean() * 100), 2),
                "stop_loss_rate_pct": round(float((group["exit_reason"] == "stop_loss").mean() * 100), 2),
                "breakeven_after_tp1_rate_pct": round(
                    float((group["exit_reason"] == "breakeven_after_tp1").mean() * 100),
                    2,
                ),
                "preserve_baseline_tp2_pct": (
                    round(float(group.loc[baseline_tp2_mask, "tp2_hit"].mean() * 100), 2)
                    if baseline_tp2_mask.any()
                    else 0.0
                ),
                "upgrade_baseline_be_to_tp2_pct": (
                    round(float(group.loc[baseline_be_mask, "tp2_hit"].mean() * 100), 2)
                    if baseline_be_mask.any()
                    else 0.0
                ),
                "degrade_baseline_be_to_stop_pct": (
                    round(float((group.loc[baseline_be_mask, "exit_reason"] == "stop_loss").mean() * 100), 2)
                    if baseline_be_mask.any()
                    else 0.0
                ),
            }
        )
    return summaries


def summarize_side_results(rows: pd.DataFrame) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for (window, profile, label, side), group in rows.groupby(["window", "profile", "label", "side"], sort=False):
        delta = group["pnl_r_baseline"] - group["baseline_pnl_r"]
        wins = float(group.loc[group["pnl_r_baseline"] > 0, "pnl_r_baseline"].sum())
        losses = abs(float(group.loc[group["pnl_r_baseline"] < 0, "pnl_r_baseline"].sum()))
        profit_factor = wins / losses if losses > 0 else (999.0 if wins > 0 else 0.0)
        summaries.append(
            {
                "window": window,
                "profile": profile,
                "label": label,
                "side": side,
                "trades": int(len(group)),
                "cum_baseline_r": round(float(group["pnl_r_baseline"].sum()), 4),
                "cum_delta_vs_baseline_r": round(float(delta.sum()), 4),
                "avg_delta_vs_baseline_r": round(float(delta.mean()), 4),
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
    selected_profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]
    profile_map = {name: (label, move_be, trail_mode) for name, label, move_be, trail_mode in MANAGEMENT_PROFILES}

    unknown_windows = sorted(set(selected_windows) - set(WINDOW_PRESETS))
    if unknown_windows:
        raise ValueError(f"Unsupported windows: {', '.join(unknown_windows)}")
    unknown_profiles = sorted(set(selected_profiles) - set(profile_map))
    if unknown_profiles:
        raise ValueError(f"Unsupported profiles: {', '.join(unknown_profiles)}")

    service = build_service()
    base_strategy = service.strategy_service.build_strategy("swing_trend_long_regime_gate_v1")
    all_trade_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []
    all_side_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    report_sections = [
        "# Post-TP1 Management Study",
        "",
        "- 固定项：`signal_time / entry_time / entry_price / initial structure stop / tp1_price / tp2_price`。",
        "- 可变项只在 `TP1` 之后生效。",
        "- `cum_baseline_r` 与 `delta_vs_baseline_r` 都使用基线结构止损的原始 `R` 尺度，避免把风险刻度变化误读成管理升级。",
        "- `ATR(1x)` 这里定义为：以前一根已完成 1H K 线 `close ± ATR14` 作为 trailing candidate。",
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
        baseline_simulated: pd.DataFrame | None = None

        for profile_name in selected_profiles:
            label, move_be, trail_mode = profile_map[profile_name]
            strategy_profile = f"post_tp1_{profile_name}"
            simulated_rows: list[dict[str, Any]] = []

            for _, trade_row in baseline_trades.iterrows():
                entry_key = pd.Timestamp(trade_row["entry_time"]).isoformat()
                if entry_key not in trigger_index:
                    raise ValueError(f"Entry timestamp not found in trigger frame: {entry_key}")

                entry_idx = trigger_index[entry_key]
                simulated = simulate_trade(
                    service=service,
                    trade_row=trade_row,
                    trigger_frame=trigger_frame,
                    entry_idx=entry_idx,
                    move_stop_to_entry_after_tp1=move_be,
                    trail_mode=trail_mode,
                    strategy_profile=strategy_profile,
                )
                trade_dict = asdict(simulated)
                baseline_risk = abs(float(trade_row["entry_price"]) - float(trade_row["stop_price"]))
                baseline_pnl_r = float(trade_dict["gross_pnl_quote"]) / baseline_risk if baseline_risk else 0.0

                merged_row = {
                    "window": window_name,
                    "profile": profile_name,
                    "label": label,
                    **trade_dict,
                    "baseline_exit_reason": str(trade_row["exit_reason"]),
                    "baseline_tp1_hit": bool(trade_row["tp1_hit"]),
                    "baseline_tp2_hit": bool(trade_row["tp2_hit"]),
                    "baseline_pnl_r": round(float(trade_row["pnl_r"]), 4),
                    "pnl_r_baseline": round(float(baseline_pnl_r), 4),
                }
                simulated_rows.append(merged_row)
                window_results.append(merged_row)
                all_trade_rows.append(merged_row)

            mode_df = pd.DataFrame(simulated_rows)
            write_csv(output_dir / f"{window_name}_{profile_name}_trades.csv", mode_df.to_dict(orient="records"))
            if profile_name == "baseline_be_after_tp1":
                baseline_simulated = mode_df

        if baseline_simulated is None:
            raise ValueError("baseline_be_after_tp1 must be included for validation.")

        mismatches = validate_baseline_replay(baseline_trades, baseline_simulated)
        validation_rows.append(
            {
                "window": window_name,
                "baseline_trades": int(len(baseline_trades)),
                "baseline_replay_mismatches": int(len(mismatches)),
            }
        )
        if mismatches:
            mismatch_path = output_dir / f"{window_name}_baseline_validation_mismatches.csv"
            write_csv(mismatch_path, mismatches)
            raise ValueError(f"Baseline replay validation failed for {window_name}: see {mismatch_path}")

        window_df = pd.DataFrame(window_results)
        summary_rows = summarize_results(window_df)
        side_rows = summarize_side_results(window_df)
        summary_rows.sort(key=lambda item: (item["cum_delta_vs_baseline_r"], item["cum_baseline_r"]), reverse=True)
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
                        ("label", "Profile"),
                        ("trades", "Trades"),
                        ("cum_baseline_r", "Cum Baseline R"),
                        ("cum_delta_vs_baseline_r", "Delta vs Baseline R"),
                        ("profit_factor_baseline_r", "PF Baseline R"),
                        ("tp2_hit_rate_pct", "TP2 Hit %"),
                        ("breakeven_after_tp1_rate_pct", "BE After TP1 %"),
                        ("upgrade_baseline_be_to_tp2_pct", "Upgrade Base BE -> TP2 %"),
                        ("preserve_baseline_tp2_pct", "Preserve Base TP2 %"),
                    ],
                ),
                "",
                "按方向拆开：",
                "",
                markdown_table(
                    side_rows,
                    [
                        ("label", "Profile"),
                        ("side", "Side"),
                        ("trades", "Trades"),
                        ("cum_baseline_r", "Cum Baseline R"),
                        ("cum_delta_vs_baseline_r", "Delta vs Baseline R"),
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
