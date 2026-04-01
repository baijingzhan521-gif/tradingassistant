from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.service import (
    BacktestAssumptions,
    BacktestService,
    BacktestTrade,
    _OpenPosition,
)
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.schemas.common import Action
from app.services.strategy_service import StrategyService
from app.strategies.windowed_mtf import WindowedMTFStrategy
from app.utils.timeframes import get_strategy_required_timeframes
from scripts.event_study_setup import bucket_trend_strength


SURFACE_TARGETS_R = (1.0, 1.5, 2.0, 3.0)

EXIT_PROFILES = (
    {
        "name": "fixed_1R_full_240h",
        "kind": "fixed_r",
        "target_r": 1.0,
        "max_hold_bars": 240,
    },
    {
        "name": "fixed_1_5R_full_240h",
        "kind": "fixed_r",
        "target_r": 1.5,
        "max_hold_bars": 240,
    },
    {
        "name": "fixed_2R_full_240h",
        "kind": "fixed_r",
        "target_r": 2.0,
        "max_hold_bars": 240,
    },
    {
        "name": "fixed_3R_full_240h",
        "kind": "fixed_r",
        "target_r": 3.0,
        "max_hold_bars": 240,
    },
    {
        "name": "scaled_1R_to_2R_be_240h",
        "kind": "scaled",
        "tp1_r": 1.0,
        "tp2_r": 2.0,
        "tp1_scale_out": 0.5,
        "move_stop_to_entry_after_tp1": True,
        "max_hold_bars": 240,
    },
    {
        "name": "scaled_1R_to_2R_no_be_240h",
        "kind": "scaled",
        "tp1_r": 1.0,
        "tp2_r": 2.0,
        "tp1_scale_out": 0.5,
        "move_stop_to_entry_after_tp1": False,
        "max_hold_bars": 240,
    },
    {
        "name": "scaled_1R_to_3R_be_240h",
        "kind": "scaled",
        "tp1_r": 1.0,
        "tp2_r": 3.0,
        "tp1_scale_out": 0.5,
        "move_stop_to_entry_after_tp1": True,
        "max_hold_bars": 240,
    },
    {
        "name": "scaled_1R_to_2R_be_96h",
        "kind": "scaled",
        "tp1_r": 1.0,
        "tp2_r": 2.0,
        "tp1_scale_out": 0.5,
        "move_stop_to_entry_after_tp1": True,
        "max_hold_bars": 96,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study exit surfaces and counterfactual exit profiles.")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--strategy-profile", default="swing_trend_long_regime_gate_v1")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--start", default=None, help="UTC start date, e.g. 2024-03-19")
    parser.add_argument("--end", default=None, help="UTC end date, e.g. 2026-03-19")
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--entry-source-target-r", type=float, default=2.0)
    parser.add_argument(
        "--output-dir",
        default="artifacts/event_studies/exit_surface",
        help="Directory for report outputs.",
    )
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def build_service(*, assumptions: BacktestAssumptions) -> BacktestService:
    return BacktestService(
        ohlcv_service=OhlcvService(get_exchange_client_factory()),
        strategy_service=StrategyService(),
        assumptions=assumptions,
    )


def build_signal_context_map(
    *,
    service: BacktestService,
    strategy: WindowedMTFStrategy,
    symbol: str,
    exchange: str,
    market_type: str,
    start: datetime,
    end: datetime,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    frames = service._load_history(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        strategy_profile=strategy.name,
        start=start,
        end=end,
    )
    enriched = {
        timeframe: service._enrich_frame(strategy, timeframe, frame)
        for timeframe, frame in frames.items()
    }

    trigger_tf = str(strategy.window_config["trigger_timeframe"])
    reference_key = str(strategy.window_config.get("reference_timeframe", trigger_tf))
    required = tuple(get_strategy_required_timeframes(strategy.name))
    trigger_frame = enriched[trigger_tf]
    indices = {timeframe: 0 for timeframe in required if timeframe != trigger_tf}
    min_required = max(int(service.assumptions.lookback // 3), 20)
    contexts: dict[str, dict[str, Any]] = {}

    for trigger_idx in range(len(trigger_frame)):
        candle = trigger_frame.iloc[trigger_idx]
        ts = candle["timestamp"].to_pydatetime()
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
        if not ready or any(index < min_required for index in current_indices.values()):
            continue

        prepared = {
            timeframe: service._build_snapshot(strategy, timeframe, enriched[timeframe], current_indices[timeframe])
            for timeframe in current_indices
        }
        higher_bias, trend_strength = strategy._derive_higher_timeframe_bias(prepared)
        reference_ctx = prepared[reference_key]
        contexts[ts.isoformat()] = {
            "trend_strength": int(trend_strength),
            "trend_strength_bucket": bucket_trend_strength(int(trend_strength)),
            "higher_bias": higher_bias.value,
            "reference_swing_high": reference_ctx.model.swing_high,
            "reference_swing_low": reference_ctx.model.swing_low,
        }

    return trigger_frame, contexts


def build_entry_sample(
    *,
    symbol: str,
    strategy_profile: str,
    exchange: str,
    market_type: str,
    start: datetime,
    end: datetime,
    entry_source_target_r: float,
) -> tuple[list[BacktestTrade], pd.DataFrame, dict[str, dict[str, Any]], WindowedMTFStrategy]:
    source_service = build_service(
        assumptions=BacktestAssumptions(
            exit_profile=f"entry_source_fixed_{entry_source_target_r:g}R",
            take_profit_mode="fixed_r",
            fixed_take_profit_r=entry_source_target_r,
        )
    )
    report = source_service.run(
        exchange=exchange,
        market_type=market_type,
        symbols=[symbol],
        strategy_profiles=[strategy_profile],
        start=start,
        end=end,
    )
    trades = [item for item in report.trades if item.symbol == symbol and item.strategy_profile == strategy_profile]
    strategy = source_service.strategy_service.build_strategy(strategy_profile)
    trigger_frame, contexts = build_signal_context_map(
        service=source_service,
        strategy=strategy,
        symbol=symbol,
        exchange=exchange,
        market_type=market_type,
        start=start,
        end=end,
    )
    return trades, trigger_frame, contexts, strategy


def build_profile_service(profile: dict[str, Any]) -> BacktestService:
    if profile["kind"] == "fixed_r":
        assumptions = BacktestAssumptions(
            exit_profile=profile["name"],
            take_profit_mode="fixed_r",
            fixed_take_profit_r=float(profile["target_r"]),
            swing_max_hold_bars=int(profile["max_hold_bars"]),
        )
    else:
        assumptions = BacktestAssumptions(
            exit_profile=profile["name"],
            take_profit_mode="scaled",
            fixed_take_profit_r=None,
            swing_max_hold_bars=int(profile["max_hold_bars"]),
            tp1_scale_out=float(profile["tp1_scale_out"]),
            move_stop_to_entry_after_tp1=bool(profile["move_stop_to_entry_after_tp1"]),
        )
    return build_service(assumptions=assumptions)


def build_position(
    *,
    trade: BacktestTrade,
    profile: dict[str, Any],
    service: BacktestService,
) -> _OpenPosition:
    side = trade.side.upper()
    entry_price = float(trade.entry_price)
    stop_price = float(trade.stop_price)
    risk = abs(entry_price - stop_price)
    if side == "LONG":
        direction = 1.0
    else:
        direction = -1.0

    if profile["kind"] == "fixed_r":
        tp_price = entry_price + (direction * risk * float(profile["target_r"]))
        tp1_price = tp_price
        tp2_price = tp_price
        take_profit_mode = "fixed_r"
        fixed_take_profit_r = float(profile["target_r"])
    else:
        tp1_price = entry_price + (direction * risk * float(profile["tp1_r"]))
        tp2_price = entry_price + (direction * risk * float(profile["tp2_r"]))
        take_profit_mode = "scaled"
        fixed_take_profit_r = None

    position = _OpenPosition(
        symbol=trade.symbol,
        strategy_profile=trade.strategy_profile,
        side=Action(trade.side),
        signal_time=datetime.fromisoformat(trade.signal_time),
        entry_time=datetime.fromisoformat(trade.entry_time),
        entry_price=entry_price,
        initial_stop_price=stop_price,
        current_stop_price=stop_price,
        tp1_price=float(tp1_price),
        tp2_price=float(tp2_price),
        take_profit_mode=take_profit_mode,
        fixed_take_profit_r=fixed_take_profit_r,
        confidence=int(trade.confidence),
    )
    entry_fee = entry_price * (service.assumptions.taker_fee_bps / 10000)
    position.fees_quote = entry_fee
    position.realized_pnl_quote = -entry_fee
    position.last_fill_price = entry_price
    return position


def simulate_exit_profile(
    *,
    trade: BacktestTrade,
    entry_idx: int,
    trigger_frame: pd.DataFrame,
    profile: dict[str, Any],
    service: BacktestService,
) -> BacktestTrade:
    position = build_position(trade=trade, profile=profile, service=service)
    max_hold_bars = int(profile["max_hold_bars"])
    for idx in range(entry_idx, len(trigger_frame)):
        candle = trigger_frame.iloc[idx]
        closed = service._update_open_position(
            position=position,
            candle=candle,
            max_hold_bars=max_hold_bars,
        )
        if closed is not None:
            return closed

    final_candle = trigger_frame.iloc[-1]
    return service._close_position(position, final_candle, exit_reason="end_of_test", fill_price=float(final_candle["close"]))


def compute_surface(
    *,
    trade: BacktestTrade,
    entry_idx: int,
    trigger_frame: pd.DataFrame,
    max_hold_bars: int,
    free_space_r: Optional[float],
    trend_strength: int,
    trend_strength_bucket: str,
) -> dict[str, Any]:
    side = trade.side.upper()
    entry_price = float(trade.entry_price)
    stop_price = float(trade.stop_price)
    risk = abs(entry_price - stop_price)
    end_idx = min(len(trigger_frame) - 1, entry_idx + max_hold_bars - 1)
    window = trigger_frame.iloc[entry_idx : end_idx + 1]

    if side == "LONG":
        mfe_r = (float(window["high"].max()) - entry_price) / risk
        mae_r = (float(window["low"].min()) - entry_price) / risk
    else:
        mfe_r = (entry_price - float(window["low"].min())) / risk
        mae_r = (entry_price - float(window["high"].max())) / risk

    record = {
        "symbol": trade.symbol,
        "strategy_profile": trade.strategy_profile,
        "signal_time": trade.signal_time,
        "entry_time": trade.entry_time,
        "side": side,
        "trend_strength": int(trend_strength),
        "trend_strength_bucket": trend_strength_bucket,
        "entry_price": round(entry_price, 6),
        "stop_price": round(stop_price, 6),
        "risk_abs": round(risk, 6),
        "mfe_r": round(float(mfe_r), 4),
        "mae_r": round(float(mae_r), 4),
        "free_space_r": round(float(free_space_r), 4) if free_space_r is not None else None,
    }

    for target_r in SURFACE_TARGETS_R:
        plus_level = entry_price + risk * target_r if side == "LONG" else entry_price - risk * target_r
        hit_before_stop = False
        stop_first = False
        for _, bar in window.iterrows():
            high = float(bar["high"])
            low = float(bar["low"])
            if side == "LONG":
                if low <= stop_price:
                    stop_first = True
                    break
                if high >= plus_level:
                    hit_before_stop = True
                    break
            else:
                if high >= stop_price:
                    stop_first = True
                    break
                if low <= plus_level:
                    hit_before_stop = True
                    break
        key = str(target_r).replace(".", "_")
        record[f"reach_{key}R_before_stop"] = hit_before_stop
        record[f"stop_before_{key}R"] = stop_first and not hit_before_stop

    return record


def summarize_surface(records: pd.DataFrame, group_by: list[str]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for keys, group in records.groupby(group_by, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        item = {column: value for column, value in zip(group_by, keys)}
        item["count"] = int(len(group))
        item["mean_mfe_r"] = round(float(group["mfe_r"].mean()), 4)
        item["median_mfe_r"] = round(float(group["mfe_r"].median()), 4)
        item["mean_mae_r"] = round(float(group["mae_r"].mean()), 4)
        item["median_mae_r"] = round(float(group["mae_r"].median()), 4)
        item["mean_free_space_r"] = (
            round(float(group["free_space_r"].dropna().mean()), 4) if not group["free_space_r"].dropna().empty else None
        )
        for target_r in SURFACE_TARGETS_R:
            key = str(target_r).replace(".", "_")
            item[f"reach_{key}R_before_stop_pct"] = round(float(group[f"reach_{key}R_before_stop"].mean() * 100), 2)
        summaries.append(item)
    return summaries


def summarize_exit_records(records: pd.DataFrame, group_by: list[str]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for keys, group in records.groupby(group_by, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        item = {column: value for column, value in zip(group_by, keys)}
        pnl_rs = group["pnl_r"].tolist()
        winners = [value for value in pnl_rs if value > 0]
        losers = [value for value in pnl_rs if value < 0]
        item["count"] = int(len(group))
        item["win_rate_pct"] = round(float((group["pnl_r"] > 0).mean() * 100), 2)
        item["avg_r"] = round(float(mean(pnl_rs)), 4)
        item["median_r"] = round(float(median(pnl_rs)), 4)
        item["cumulative_r"] = round(float(sum(pnl_rs)), 4)
        item["profit_factor"] = round((sum(winners) / abs(sum(losers))) if losers else 0.0, 4)
        item["avg_holding_bars"] = round(float(group["bars_held"].mean()), 2)
        item["tp1_hit_rate_pct"] = round(float(group["tp1_hit"].mean() * 100), 2)
        item["tp2_hit_rate_pct"] = round(float(group["tp2_hit"].mean() * 100), 2)
        item["time_stop_rate_pct"] = round(float((group["exit_reason"] == "time_stop").mean() * 100), 2)
        summaries.append(item)
    return summaries


def render_markdown(
    *,
    symbol: str,
    strategy_profile: str,
    start: datetime,
    end: datetime,
    entry_count: int,
    surface_overall: list[dict[str, Any]],
    surface_by_side_bucket: list[dict[str, Any]],
    exit_overall: list[dict[str, Any]],
    exit_by_side: list[dict[str, Any]],
) -> str:
    lines = [
        "# Exit Surface Study",
        "",
        f"- 标的: {symbol}",
        f"- 入口策略: {strategy_profile}",
        f"- 研究窗口: {start.isoformat()} -> {end.isoformat()}",
        f"- Entry sample: {entry_count} 笔当前实际执行入场",
        "- 注意: 这是一份“同一批 entry 的反事实 exit 对比”，不是每种 exit 的完整再回测。",
        "",
        "## MFE / MAE Surface",
        "",
        "| scope | count | mean_mfe_r | median_mfe_r | mean_mae_r | median_mae_r | mean_free_space_r | reach_1_0R_before_stop_pct | reach_1_5R_before_stop_pct | reach_2_0R_before_stop_pct | reach_3_0R_before_stop_pct |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in surface_overall:
        label = item.get("side", "overall")
        lines.append(
            f"| {label} | {item['count']} | {item['mean_mfe_r']} | {item['median_mfe_r']} | {item['mean_mae_r']} | {item['median_mae_r']} | {item['mean_free_space_r']} | {item['reach_1_0R_before_stop_pct']} | {item['reach_1_5R_before_stop_pct']} | {item['reach_2_0R_before_stop_pct']} | {item['reach_3_0R_before_stop_pct']} |"
        )

    lines.extend(
        [
            "",
            "## MFE / MAE By Regime",
            "",
            "| trend_strength_bucket | side | count | mean_mfe_r | mean_mae_r | mean_free_space_r | reach_1_0R_before_stop_pct | reach_1_5R_before_stop_pct | reach_2_0R_before_stop_pct | reach_3_0R_before_stop_pct |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in surface_by_side_bucket:
        lines.append(
            f"| {item['trend_strength_bucket']} | {item['side']} | {item['count']} | {item['mean_mfe_r']} | {item['mean_mae_r']} | {item['mean_free_space_r']} | {item['reach_1_0R_before_stop_pct']} | {item['reach_1_5R_before_stop_pct']} | {item['reach_2_0R_before_stop_pct']} | {item['reach_3_0R_before_stop_pct']} |"
        )

    lines.extend(
        [
            "",
            "## Exit Profile Comparison",
            "",
            "| exit_profile | count | win_rate_pct | avg_r | cumulative_r | profit_factor | avg_holding_bars | tp1_hit_rate_pct | tp2_hit_rate_pct | time_stop_rate_pct |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in exit_overall:
        lines.append(
            f"| {item['exit_profile']} | {item['count']} | {item['win_rate_pct']} | {item['avg_r']} | {item['cumulative_r']} | {item['profit_factor']} | {item['avg_holding_bars']} | {item['tp1_hit_rate_pct']} | {item['tp2_hit_rate_pct']} | {item['time_stop_rate_pct']} |"
        )

    lines.extend(
        [
            "",
            "## Exit Profile By Side",
            "",
            "| side | exit_profile | count | win_rate_pct | avg_r | cumulative_r | profit_factor | avg_holding_bars | time_stop_rate_pct |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in exit_by_side:
        lines.append(
            f"| {item['side']} | {item['exit_profile']} | {item['count']} | {item['win_rate_pct']} | {item['avg_r']} | {item['cumulative_r']} | {item['profit_factor']} | {item['avg_holding_bars']} | {item['time_stop_rate_pct']} |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    configure_logging()

    now = datetime.now(timezone.utc)
    end = parse_date(args.end) if args.end else now
    start = parse_date(args.start) if args.start else end - timedelta(days=args.years * 365)

    trades, trigger_frame, contexts, _strategy = build_entry_sample(
        symbol=args.symbol,
        strategy_profile=args.strategy_profile,
        exchange=args.exchange,
        market_type=args.market_type,
        start=start,
        end=end,
        entry_source_target_r=float(args.entry_source_target_r),
    )

    entry_index_map = {
        row["timestamp"].to_pydatetime().isoformat(): idx
        for idx, row in trigger_frame.iterrows()
    }

    surface_records: list[dict[str, Any]] = []
    exit_records: list[dict[str, Any]] = []
    services = {profile["name"]: build_profile_service(profile) for profile in EXIT_PROFILES}

    for trade in trades:
        entry_idx = entry_index_map.get(datetime.fromisoformat(trade.entry_time).isoformat())
        if entry_idx is None:
            continue
        signal_ctx = contexts.get(datetime.fromisoformat(trade.signal_time).isoformat(), {})
        risk = abs(float(trade.entry_price) - float(trade.stop_price))
        side = trade.side.upper()
        reference_high = signal_ctx.get("reference_swing_high")
        reference_low = signal_ctx.get("reference_swing_low")
        free_space_r: Optional[float]
        if risk <= 0:
            free_space_r = None
        elif side == "LONG" and reference_high is not None:
            free_space_r = (float(reference_high) - float(trade.entry_price)) / risk
        elif side == "SHORT" and reference_low is not None:
            free_space_r = (float(trade.entry_price) - float(reference_low)) / risk
        else:
            free_space_r = None

        surface_records.append(
            compute_surface(
                trade=trade,
                entry_idx=entry_idx,
                trigger_frame=trigger_frame,
                max_hold_bars=240,
                free_space_r=free_space_r,
                trend_strength=int(signal_ctx.get("trend_strength", 0)),
                trend_strength_bucket=str(signal_ctx.get("trend_strength_bucket", "unknown")),
            )
        )

        for profile in EXIT_PROFILES:
            simulated = simulate_exit_profile(
                trade=trade,
                entry_idx=entry_idx,
                trigger_frame=trigger_frame,
                profile=profile,
                service=services[profile["name"]],
            )
            exit_records.append(
                {
                    "signal_time": simulated.signal_time,
                    "entry_time": simulated.entry_time,
                    "exit_profile": profile["name"],
                    "side": simulated.side.upper(),
                    "pnl_r": simulated.pnl_r,
                    "bars_held": simulated.bars_held,
                    "exit_reason": simulated.exit_reason,
                    "tp1_hit": simulated.tp1_hit,
                    "tp2_hit": simulated.tp2_hit,
                    "trend_strength_bucket": str(signal_ctx.get("trend_strength_bucket", "unknown")),
                }
            )

    surface_df = pd.DataFrame(surface_records)
    exit_df = pd.DataFrame(exit_records)
    surface_overall = summarize_surface(surface_df.assign(side="overall"), ["side"])
    surface_by_side = summarize_surface(surface_df, ["side"])
    surface_overall.extend(surface_by_side)
    surface_by_side_bucket = summarize_surface(surface_df, ["trend_strength_bucket", "side"])
    exit_overall = summarize_exit_records(exit_df, ["exit_profile"])
    exit_by_side = summarize_exit_records(exit_df, ["side", "exit_profile"])

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": args.symbol,
        "strategy_profile": args.strategy_profile,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "entry_source_target_r": float(args.entry_source_target_r),
        "entry_count": int(len(surface_df)),
        "surface_overall": surface_overall,
        "surface_by_side_bucket": surface_by_side_bucket,
        "exit_overall": exit_overall,
        "exit_by_side": exit_by_side,
    }

    output_dir = Path(args.output_dir) / args.strategy_profile
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    surface_csv = output_dir / f"exit_surface_{stamp}_surface.csv"
    exit_csv = output_dir / f"exit_surface_{stamp}_exits.csv"
    json_path = output_dir / f"exit_surface_{stamp}.json"
    md_path = output_dir / f"exit_surface_{stamp}.md"
    surface_df.to_csv(surface_csv, index=False)
    exit_df.to_csv(exit_csv, index=False)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(
        render_markdown(
            symbol=args.symbol,
            strategy_profile=args.strategy_profile,
            start=start,
            end=end,
            entry_count=int(len(surface_df)),
            surface_overall=surface_overall,
            surface_by_side_bucket=surface_by_side_bucket,
            exit_overall=exit_overall,
            exit_by_side=exit_by_side,
        ),
        encoding="utf-8",
    )

    print(f"Saved surface CSV: {surface_csv}")
    print(f"Saved exit CSV: {exit_csv}")
    print(f"Saved report JSON: {json_path}")
    print(f"Saved report Markdown: {md_path}")
    print(md_path.read_text(encoding='utf-8'))


if __name__ == "__main__":
    main()
