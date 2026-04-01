from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.diagnostics import collect_signal_diagnostics
from app.backtesting.service import BacktestAssumptions, BacktestService
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.services.strategy_service import StrategyService


DEFAULT_HORIZONS = (24, 48, 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study whether setup-ready events have edge before trigger filtering.")
    parser.add_argument(
        "--symbols",
        default="BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT",
        help="Comma-separated symbols.",
    )
    parser.add_argument("--strategy-profile", default="swing_trend_v1")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--start", default=None, help="UTC start date, e.g. 2024-03-19")
    parser.add_argument("--end", default=None, help="UTC end date, e.g. 2026-03-19")
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument(
        "--horizons",
        default="24,48,72",
        help="Comma-separated forward horizons in trigger bars.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/event_studies/setup_edge",
        help="Directory for report outputs.",
    )
    parser.add_argument(
        "--event-stage",
        choices=["setup_ready", "trigger_confirmed", "both"],
        default="both",
        help="Which event stages to include in the report.",
    )
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def parse_horizons(raw: str) -> tuple[int, ...]:
    items = tuple(sorted({int(item.strip()) for item in raw.split(",") if item.strip()}))
    if not items:
        raise ValueError("At least one horizon is required")
    if any(item <= 0 for item in items):
        raise ValueError("Horizons must be positive integers")
    return items


def bucket_trend_strength(score: int) -> str:
    if score < 70:
        return "60-69"
    if score < 80:
        return "70-79"
    if score < 90:
        return "80-89"
    return "90+"


def _setup_ready_mask(signals: pd.DataFrame) -> pd.Series:
    return (
        (signals["higher_bias"] != "neutral")
        & signals["trend_friendly"]
        & signals["setup_aligned"]
        & signals["setup_pullback_ready"]
        & signals["setup_reversal_ready"]
        & ((~signals["setup_divergence_required"]) | signals["setup_divergence_ready"])
    )


def _trigger_confirmed_mask(signals: pd.DataFrame) -> pd.Series:
    return _setup_ready_mask(signals) & signals["trigger_state"].isin(
        {"bullish_confirmed", "bearish_confirmed"}
    )


def _transition_events(mask: pd.Series) -> pd.Series:
    shifted = mask.shift(1, fill_value=False)
    return mask & (~shifted)


def _directional_forward_return(entry_price: float, future_close: float, side: str) -> float:
    if side == "LONG":
        return (future_close - entry_price) / entry_price
    return (entry_price - future_close) / entry_price


def _risk_reference(
    *,
    row: pd.Series,
    side: str,
    atr_buffer: float,
    minimum_r_multiple: float,
) -> tuple[float, float]:
    entry_price = float(row["close"])
    execution_low = float(row["execution_zone_low"])
    execution_high = float(row["execution_zone_high"])
    atr14 = float(row["atr14"])

    if side == "LONG":
        anchor = execution_low if pd.isna(row["recent_swing_low"]) else float(row["recent_swing_low"])
        stop_price = min(anchor, execution_low) - (atr14 * atr_buffer)
        risk = max(entry_price - stop_price, atr14 * minimum_r_multiple)
        return stop_price, risk

    anchor = execution_high if pd.isna(row["recent_swing_high"]) else float(row["recent_swing_high"])
    stop_price = max(anchor, execution_high) + (atr14 * atr_buffer)
    risk = max(stop_price - entry_price, atr14 * minimum_r_multiple)
    return stop_price, risk


def _path_metrics(
    *,
    future_window: pd.DataFrame,
    entry_price: float,
    risk: float,
    side: str,
) -> tuple[float, float, str]:
    if future_window.empty:
        return 0.0, 0.0, "insufficient"

    if side == "LONG":
        mfe_r = (float(future_window["high"].max()) - entry_price) / risk
        mae_r = (float(future_window["low"].min()) - entry_price) / risk
        positive_touch = entry_price + risk
        negative_touch = entry_price - risk
    else:
        mfe_r = (entry_price - float(future_window["low"].min())) / risk
        mae_r = (entry_price - float(future_window["high"].max())) / risk
        positive_touch = entry_price - risk
        negative_touch = entry_price + risk

    first_touch = "neither"
    for _, candle in future_window.iterrows():
        high = float(candle["high"])
        low = float(candle["low"])
        if side == "LONG":
            hit_positive = high >= positive_touch
            hit_negative = low <= negative_touch
        else:
            hit_positive = low <= positive_touch
            hit_negative = high >= negative_touch
        if hit_positive and hit_negative:
            first_touch = "same_bar_both"
            break
        if hit_positive:
            first_touch = "plus_1r_first"
            break
        if hit_negative:
            first_touch = "minus_1r_first"
            break

    return mfe_r, mae_r, first_touch


def build_event_records(
    *,
    service: BacktestService,
    exchange: str,
    market_type: str,
    symbol: str,
    strategy_profile: str,
    start: datetime,
    end: datetime,
    horizons: tuple[int, ...],
    event_stage: str,
) -> list[dict[str, Any]]:
    signals = collect_signal_diagnostics(
        service=service,
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        strategy_profile=strategy_profile,
        start=start,
        end=end,
    )
    if signals.empty:
        return []

    strategy = service.strategy_service.build_strategy(strategy_profile)
    frames = service._load_history(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        strategy_profile=strategy_profile,
        start=start,
        end=end,
    )
    trigger_tf = str(strategy.window_config["trigger_timeframe"])
    trigger_frame = service._enrich_frame(strategy, trigger_tf, frames[trigger_tf]).copy()
    trigger_frame["timestamp"] = pd.to_datetime(trigger_frame["timestamp"], utc=True)
    timestamp_to_index = {timestamp.value: idx for idx, timestamp in enumerate(trigger_frame["timestamp"])}

    signals = signals.copy()
    signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True)
    signals = signals.sort_values("timestamp").reset_index(drop=True)

    stage_masks: dict[str, pd.Series] = {}
    if event_stage in {"setup_ready", "both"}:
        stage_masks["setup_ready"] = _transition_events(_setup_ready_mask(signals))
    if event_stage in {"trigger_confirmed", "both"}:
        stage_masks["trigger_confirmed"] = _transition_events(_trigger_confirmed_mask(signals))

    atr_buffer = float(strategy.config["risk"]["atr_buffer"])
    minimum_r_multiple = float(strategy.config["risk"]["minimum_r_multiple"])
    touch_horizon = max(horizons)
    records: list[dict[str, Any]] = []

    for stage_name, stage_mask in stage_masks.items():
        for _, signal in signals.loc[stage_mask].iterrows():
            timestamp = signal["timestamp"]
            trigger_idx = timestamp_to_index.get(timestamp.value)
            if trigger_idx is None:
                continue

            row = trigger_frame.iloc[trigger_idx]
            side = "LONG" if signal["higher_bias"] == "bullish" else "SHORT"
            stop_price, risk = _risk_reference(
                row=row,
                side=side,
                atr_buffer=atr_buffer,
                minimum_r_multiple=minimum_r_multiple,
            )
            entry_price = float(row["close"])

            record: dict[str, Any] = {
                "symbol": symbol,
                "stage": stage_name,
                "timestamp": timestamp.isoformat(),
                "side": side,
                "higher_bias": str(signal["higher_bias"]),
                "trend_strength": int(signal["trend_strength"]),
                "trend_strength_bucket": bucket_trend_strength(int(signal["trend_strength"])),
                "volatility_state": str(signal["volatility_state"]),
                "setup_reversal_ready": bool(signal["setup_reversal_ready"]),
                "setup_divergence_ready": bool(signal["setup_divergence_ready"]),
                "setup_divergence_required": bool(signal["setup_divergence_required"]),
                "setup_structure_ready": bool(signal["setup_structure_ready"]),
                "entry_price": round(entry_price, 4),
                "stop_price": round(stop_price, 4),
                "risk_r_abs": round(risk, 4),
                "setup_distance_to_execution_atr": round(float(signal["setup_distance_to_execution_atr"]), 4),
                "trigger_state": str(signal["trigger_state"]),
            }

            for horizon in horizons:
                future_idx = trigger_idx + horizon
                if future_idx >= len(trigger_frame):
                    record[f"forward_return_{horizon}b_pct"] = None
                    continue
                future_close = float(trigger_frame.iloc[future_idx]["close"])
                forward_return = _directional_forward_return(entry_price, future_close, side) * 100
                record[f"forward_return_{horizon}b_pct"] = round(forward_return, 4)

            future_window = trigger_frame.iloc[trigger_idx + 1 : trigger_idx + touch_horizon + 1].copy()
            mfe_r, mae_r, first_touch = _path_metrics(
                future_window=future_window,
                entry_price=entry_price,
                risk=risk,
                side=side,
            )
            record[f"mfe_r_{touch_horizon}b"] = round(mfe_r, 4)
            record[f"mae_r_{touch_horizon}b"] = round(mae_r, 4)
            record[f"first_touch_{touch_horizon}b"] = first_touch
            records.append(record)

    return records


def summarize_events(
    events: pd.DataFrame,
    *,
    group_by: list[str],
    horizons: tuple[int, ...],
) -> list[dict[str, Any]]:
    if events.empty:
        return []

    touch_horizon = max(horizons)
    summaries: list[dict[str, Any]] = []
    for keys, group in events.groupby(group_by, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        item = {column: value for column, value in zip(group_by, keys)}
        item["count"] = int(len(group))

        for horizon in horizons:
            column = f"forward_return_{horizon}b_pct"
            series = pd.to_numeric(group[column], errors="coerce").dropna()
            item[f"mean_forward_return_{horizon}b_pct"] = round(float(series.mean()), 4) if not series.empty else None
            item[f"median_forward_return_{horizon}b_pct"] = round(float(series.median()), 4) if not series.empty else None
            item[f"positive_return_rate_{horizon}b_pct"] = (
                round(float((series > 0).mean() * 100), 2) if not series.empty else None
            )

        mfe_series = pd.to_numeric(group[f"mfe_r_{touch_horizon}b"], errors="coerce").dropna()
        mae_series = pd.to_numeric(group[f"mae_r_{touch_horizon}b"], errors="coerce").dropna()
        item[f"mean_mfe_r_{touch_horizon}b"] = round(float(mfe_series.mean()), 4) if not mfe_series.empty else None
        item[f"mean_mae_r_{touch_horizon}b"] = round(float(mae_series.mean()), 4) if not mae_series.empty else None
        item[f"median_mfe_r_{touch_horizon}b"] = round(float(mfe_series.median()), 4) if not mfe_series.empty else None
        item[f"median_mae_r_{touch_horizon}b"] = round(float(mae_series.median()), 4) if not mae_series.empty else None

        first_touch_counts = group[f"first_touch_{touch_horizon}b"].value_counts(normalize=True)
        item[f"plus_1r_first_rate_{touch_horizon}b_pct"] = round(float(first_touch_counts.get("plus_1r_first", 0.0) * 100), 2)
        item[f"minus_1r_first_rate_{touch_horizon}b_pct"] = round(float(first_touch_counts.get("minus_1r_first", 0.0) * 100), 2)
        item[f"same_bar_both_rate_{touch_horizon}b_pct"] = round(float(first_touch_counts.get("same_bar_both", 0.0) * 100), 2)
        item[f"neither_rate_{touch_horizon}b_pct"] = round(float(first_touch_counts.get("neither", 0.0) * 100), 2)
        summaries.append(item)

    return summaries


def derive_findings(
    *,
    summary_by_stage_side: list[dict[str, Any]],
    touch_horizon: int,
    longest_horizon: int,
) -> list[str]:
    findings: list[str] = []
    lookup = {(item["stage"], item["side"]): item for item in summary_by_stage_side}

    for side in ("LONG", "SHORT"):
        setup = lookup.get(("setup_ready", side))
        trigger = lookup.get(("trigger_confirmed", side))
        if setup is None:
            continue

        setup_mean = setup.get(f"mean_forward_return_{longest_horizon}b_pct")
        setup_plus = setup.get(f"plus_1r_first_rate_{touch_horizon}b_pct")
        setup_minus = setup.get(f"minus_1r_first_rate_{touch_horizon}b_pct")

        if setup_mean is not None and setup_plus is not None and setup_minus is not None:
            if setup_mean > 0 and setup_plus > setup_minus:
                findings.append(
                    f"{side} setup 本身呈正分布：{longest_horizon}bar 平均 forward return {setup_mean}%，"
                    f"+1R 先到 {setup_plus}% > -1R 先到 {setup_minus}%。"
                )
            elif setup_mean <= 0 and setup_plus <= setup_minus:
                findings.append(
                    f"{side} setup 本身没有看到正 edge：{longest_horizon}bar 平均 forward return {setup_mean}%，"
                    f"+1R 先到 {setup_plus}% <= -1R 先到 {setup_minus}%。"
                )

        if setup and trigger:
            trigger_mean = trigger.get(f"mean_forward_return_{longest_horizon}b_pct")
            if setup_mean is not None and trigger_mean is not None:
                if setup_mean > 0 and trigger_mean < setup_mean:
                    findings.append(
                        f"{side} 从 setup 到 trigger 后，{longest_horizon}bar 平均收益从 {setup_mean}% 下降到 {trigger_mean}%，"
                        "trigger 可能在削弱 edge。"
                    )
                elif setup_mean <= 0 and trigger_mean > setup_mean:
                    findings.append(
                        f"{side} setup 原本偏弱，但 trigger 后分布有改善：{longest_horizon}bar 平均收益从 {setup_mean}% 变到 {trigger_mean}%。"
                    )

    return findings


def render_markdown(
    *,
    headline: dict[str, Any],
    findings: list[str],
    summary_by_stage: list[dict[str, Any]],
    summary_by_stage_symbol_side: list[dict[str, Any]],
    summary_by_stage_regime_side: list[dict[str, Any]],
    horizons: tuple[int, ...],
) -> str:
    longest_horizon = max(horizons)
    touch_horizon = max(horizons)
    lines = [
        "# Setup Edge Event Study",
        "",
        f"- 生成时间: {headline['generated_at']}",
        f"- 策略: {headline['strategy_profile']}",
        f"- 标的: {', '.join(headline['symbols'])}",
        f"- 回测窗口: {headline['start']} -> {headline['end']}",
        f"- 事件定义: setup/trigger 从 `not ready -> ready` 的切换时刻，不重复统计持续态",
        f"- 触发周期: {headline['trigger_timeframe']}",
        f"- Forward horizons: {', '.join(f'{item} bars' for item in horizons)}",
        "",
        "## 关键结论",
        "",
    ]
    for item in findings:
        lines.append(f"- {item}")

    def add_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
        lines.extend(["", f"## {title}", ""])
        if not rows:
            lines.append("无样本。")
            return
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---:" if column == "count" else "---" for column in columns) + " |")
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column)
                values.append("NA" if value is None else str(value))
            lines.append("| " + " | ".join(values) + " |")

    add_table(
        "Stage Summary",
        summary_by_stage,
        [
            "stage",
            "count",
            f"mean_forward_return_{horizons[0]}b_pct",
            f"mean_forward_return_{horizons[1]}b_pct" if len(horizons) > 1 else f"mean_forward_return_{horizons[0]}b_pct",
            f"mean_forward_return_{longest_horizon}b_pct",
            f"plus_1r_first_rate_{touch_horizon}b_pct",
            f"minus_1r_first_rate_{touch_horizon}b_pct",
            f"mean_mfe_r_{touch_horizon}b",
            f"mean_mae_r_{touch_horizon}b",
        ],
    )
    add_table(
        "By Symbol And Side",
        summary_by_stage_symbol_side,
        [
            "stage",
            "symbol",
            "side",
            "count",
            f"mean_forward_return_{longest_horizon}b_pct",
            f"positive_return_rate_{longest_horizon}b_pct",
            f"plus_1r_first_rate_{touch_horizon}b_pct",
            f"minus_1r_first_rate_{touch_horizon}b_pct",
            f"mean_mfe_r_{touch_horizon}b",
            f"mean_mae_r_{touch_horizon}b",
        ],
    )
    add_table(
        "By Trend Regime And Side",
        summary_by_stage_regime_side,
        [
            "stage",
            "trend_strength_bucket",
            "side",
            "count",
            f"mean_forward_return_{longest_horizon}b_pct",
            f"positive_return_rate_{longest_horizon}b_pct",
            f"plus_1r_first_rate_{touch_horizon}b_pct",
            f"minus_1r_first_rate_{touch_horizon}b_pct",
        ],
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    configure_logging()

    now = datetime.now(timezone.utc)
    end = parse_date(args.end) if args.end else now
    start = parse_date(args.start) if args.start else end - timedelta(days=args.years * 365)
    horizons = parse_horizons(args.horizons)
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]

    service = BacktestService(
        ohlcv_service=OhlcvService(get_exchange_client_factory()),
        strategy_service=StrategyService(),
        assumptions=BacktestAssumptions(exit_profile="setup_event_study", take_profit_mode="fixed_r", fixed_take_profit_r=2.0),
    )

    records: list[dict[str, Any]] = []
    for symbol in symbols:
        records.extend(
            build_event_records(
                service=service,
                exchange=args.exchange,
                market_type=args.market_type,
                symbol=symbol,
                strategy_profile=args.strategy_profile,
                start=start,
                end=end,
                horizons=horizons,
                event_stage=args.event_stage,
            )
        )

    events = pd.DataFrame(records)
    summary_by_stage = summarize_events(events, group_by=["stage"], horizons=horizons)
    summary_by_stage_side = summarize_events(events, group_by=["stage", "side"], horizons=horizons)
    summary_by_stage_symbol_side = summarize_events(events, group_by=["stage", "symbol", "side"], horizons=horizons)
    summary_by_stage_regime_side = summarize_events(
        events,
        group_by=["stage", "trend_strength_bucket", "side"],
        horizons=horizons,
    )

    headline = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy_profile": args.strategy_profile,
        "symbols": symbols,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "trigger_timeframe": str(service.strategy_service.build_strategy(args.strategy_profile).window_config["trigger_timeframe"]),
    }
    findings = derive_findings(
        summary_by_stage_side=summary_by_stage_side,
        touch_horizon=max(horizons),
        longest_horizon=max(horizons),
    )
    report = {
        "headline": headline,
        "findings": findings,
        "summary_by_stage": summary_by_stage,
        "summary_by_stage_side": summary_by_stage_side,
        "summary_by_stage_symbol_side": summary_by_stage_symbol_side,
        "summary_by_stage_regime_side": summary_by_stage_regime_side,
        "event_count": int(len(events)),
    }

    output_dir = Path(args.output_dir) / args.strategy_profile
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"setup_event_study_{stamp}.json"
    md_path = output_dir / f"setup_event_study_{stamp}.md"
    csv_path = output_dir / f"setup_event_study_{stamp}_events.csv"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(
        render_markdown(
            headline=headline,
            findings=findings,
            summary_by_stage=summary_by_stage,
            summary_by_stage_symbol_side=summary_by_stage_symbol_side,
            summary_by_stage_regime_side=summary_by_stage_regime_side,
            horizons=horizons,
        ),
        encoding="utf-8",
    )
    events.to_csv(csv_path, index=False)

    print(f"Saved report JSON: {json_path}")
    print(f"Saved report Markdown: {md_path}")
    print(f"Saved event CSV: {csv_path}")
    print(md_path.read_text(encoding='utf-8'))


if __name__ == "__main__":
    main()
