from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.service import BacktestAssumptions, BacktestService
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.schemas.common import Bias, VolatilityState
from app.services.strategy_service import StrategyService
from app.strategies.swing_trend_entry_attribution import (
    SwingTrendEntryAttributionStrategy,
    build_entry_attribution_config,
)


MATCHED_PAIRS_CSV = (
    ROOT / "artifacts/backtests/mainline_vs_no_reversal_no_aux_trade_diff_2022_2024/matched_pairs.csv"
)
TRADES_CSV = ROOT / "artifacts/backtests/entry_attribution_matrix_2020/entry_attribution_oos_trades.csv"
CACHE_DIR = ROOT / "artifacts/backtests/cache"
OUTPUT_DIR = ROOT / "artifacts/backtests/manual_case_review_2022_2024"

MAINLINE = "entry_attr_r1_rf1_hs1_aux1"
NO_REV_NO_AUX = "entry_attr_r0_rf1_hs1_aux0"
SYMBOL = "BTC/USDT:USDT"
TIMEFRAMES = ("1d", "4h", "1h")

EXIT_ASSUMPTIONS = {
    "exit_profile": "entry_attribution_long_scaled1_3_short_fixed1_5",
    "take_profit_mode": "scaled",
    "scaled_tp1_r": 1.0,
    "scaled_tp2_r": 3.0,
    "long_exit": {"take_profit_mode": "scaled", "scaled_tp1_r": 1.0, "scaled_tp2_r": 3.0},
    "short_exit": {"take_profit_mode": "fixed_r", "fixed_take_profit_r": 1.5},
    "swing_detection_mode": "confirmed",
}


@dataclass(frozen=True)
class ReviewCase:
    case_id: str
    year: int
    side: str
    delta_r: float
    gap_hours: float
    main_signal_time: datetime
    alt_signal_time: datetime
    main_entry_time: datetime
    alt_entry_time: datetime
    main_exit_time: datetime
    alt_exit_time: datetime


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_service() -> BacktestService:
    return BacktestService(
        ohlcv_service=OhlcvService(get_exchange_client_factory()),
        strategy_service=StrategyService(),
        assumptions=BacktestAssumptions(**EXIT_ASSUMPTIONS),
    )


def load_cases() -> list[ReviewCase]:
    matched = pd.read_csv(
        MATCHED_PAIRS_CSV,
        parse_dates=[
            "main_signal_time",
            "alt_signal_time",
            "main_entry_time",
            "alt_entry_time",
            "main_exit_time",
            "alt_exit_time",
        ],
    )
    selected = []
    year_2022 = matched[matched["year"] == 2022].sort_values("delta_r", ascending=False).head(5)
    year_2024 = matched[matched["year"] == 2024].sort_values("delta_r", ascending=True).head(5)
    for idx, row in enumerate(year_2022.itertuples(index=False), start=1):
        selected.append(
            ReviewCase(
                case_id=f"2022-S{idx}",
                year=2022,
                side=str(row.side),
                delta_r=float(row.delta_r),
                gap_hours=float(row.gap_hours),
                main_signal_time=row.main_signal_time.to_pydatetime(),
                alt_signal_time=row.alt_signal_time.to_pydatetime(),
                main_entry_time=row.main_entry_time.to_pydatetime(),
                alt_entry_time=row.alt_entry_time.to_pydatetime(),
                main_exit_time=row.main_exit_time.to_pydatetime(),
                alt_exit_time=row.alt_exit_time.to_pydatetime(),
            )
        )
    for idx, row in enumerate(year_2024.itertuples(index=False), start=1):
        selected.append(
            ReviewCase(
                case_id=f"2024-L{idx}",
                year=2024,
                side=str(row.side),
                delta_r=float(row.delta_r),
                gap_hours=float(row.gap_hours),
                main_signal_time=row.main_signal_time.to_pydatetime(),
                alt_signal_time=row.alt_signal_time.to_pydatetime(),
                main_entry_time=row.main_entry_time.to_pydatetime(),
                alt_entry_time=row.alt_entry_time.to_pydatetime(),
                main_exit_time=row.main_exit_time.to_pydatetime(),
                alt_exit_time=row.alt_exit_time.to_pydatetime(),
            )
        )
    return selected


def load_trades() -> pd.DataFrame:
    trades = pd.read_csv(
        TRADES_CSV,
        parse_dates=["signal_time", "entry_time", "exit_time"],
    )
    return trades.set_index(["strategy_profile", "signal_time"]).sort_index()


def pick_cache_file(timeframe: str, *, start: datetime, end: datetime) -> Path:
    pattern = re.compile(rf"btc_usdt_usdt_{timeframe}_(\d{{8}})_(\d{{8}})\.csv$")
    candidates: list[tuple[datetime, datetime, Path]] = []
    for path in CACHE_DIR.glob(f"btc_usdt_usdt_{timeframe}_*.csv"):
        match = pattern.search(path.name)
        if not match:
            continue
        start_dt = datetime.strptime(match.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(match.group(2), "%Y%m%d").replace(tzinfo=timezone.utc)
        candidates.append((start_dt, end_dt, path))
    covering = [item for item in candidates if item[0] <= start and item[1] >= end]
    if not covering:
        raise FileNotFoundError(f"No cache file covers {timeframe} {start.isoformat()} -> {end.isoformat()}")
    covering.sort(key=lambda item: (item[0], -item[1].timestamp()))
    return covering[0][2]


def load_frames(*, start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for timeframe in TIMEFRAMES:
        path = pick_cache_file(timeframe, start=start, end=end)
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        frame = frame[(frame["timestamp"] >= pd.Timestamp(start)) & (frame["timestamp"] <= pd.Timestamp(end))].copy()
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        frames[timeframe] = frame
    return frames


def build_strategies() -> dict[str, SwingTrendEntryAttributionStrategy]:
    return {
        "mainline": SwingTrendEntryAttributionStrategy(
            build_entry_attribution_config(
                include_reversal=True,
                include_regained_fast=True,
                include_held_slow=True,
                include_auxiliary=True,
            ),
            profile_name=MAINLINE,
        ),
        "no_reversal_no_aux": SwingTrendEntryAttributionStrategy(
            build_entry_attribution_config(
                include_reversal=False,
                include_regained_fast=True,
                include_held_slow=True,
                include_auxiliary=False,
            ),
            profile_name=NO_REV_NO_AUX,
        ),
    }


def extract_blockers(
    *,
    setup: dict[str, Any],
    trigger: dict[str, Any],
    trigger_requirements: dict[str, bool],
    trend_friendly: bool,
) -> list[str]:
    blockers: list[str] = []
    if not trend_friendly:
        blockers.append("trend_strength_not_friendly")
    if not setup["aligned"]:
        blockers.append("setup_not_aligned")
    if not setup["pullback_ready"]:
        blockers.append("pullback_not_ready")
    if setup.get("require_reversal_candle") and not setup.get("reversal_ready"):
        blockers.append("reversal_missing")
    if trigger_requirements["require_regained_fast"] and not trigger["regained_fast"]:
        blockers.append("regained_fast_missing")
    if trigger_requirements["require_held_slow"] and not trigger["held_slow"]:
        blockers.append("held_slow_missing")
    if trigger_requirements["require_auxiliary"] and (
        int(trigger["auxiliary_count"]) < int(trigger["min_auxiliary_confirmations"])
    ):
        blockers.append("auxiliary_missing")
    return blockers


def evaluate_at(
    *,
    service: BacktestService,
    strategy: SwingTrendEntryAttributionStrategy,
    enriched: dict[str, pd.DataFrame],
    ts: datetime,
) -> dict[str, Any]:
    prepared = {}
    for timeframe, frame in enriched.items():
        idx = int(frame["timestamp"].searchsorted(pd.Timestamp(ts), side="right")) - 1
        if idx < 0:
            raise ValueError(f"No {timeframe} candle available at {ts.isoformat()}")
        prepared[timeframe] = service._build_snapshot(strategy, timeframe, frame, idx)

    higher_bias, trend_strength = strategy._derive_higher_timeframe_bias(prepared)
    setup_key = str(strategy.window_config["setup_timeframe"])
    trigger_key = str(strategy.window_config["trigger_timeframe"])
    reference_key = str(strategy.window_config.get("reference_timeframe", setup_key))
    volatility_state = strategy._derive_volatility_state(prepared[setup_key])
    trend_friendly = strategy._is_trend_friendly(
        higher_bias=higher_bias,
        trend_strength=trend_strength,
        volatility_state=volatility_state,
    )
    setup = strategy._assess_setup(
        higher_bias,
        prepared[setup_key],
        setup_key,
        reference_ctx=prepared[reference_key],
        current_price=prepared[trigger_key].model.close,
    )
    trigger = strategy._assess_trigger(
        higher_bias,
        prepared[trigger_key],
        trigger_key,
        trend_strength=trend_strength,
    )
    trigger_requirements = strategy._resolve_trigger_requirements(higher_bias, trend_strength)

    confidence = 50
    confidence += 15 if higher_bias != Bias.NEUTRAL else -20
    confidence += int(setup["score"])
    confidence += int(trigger["score"])
    if volatility_state == VolatilityState.HIGH:
        confidence -= 15
    elif volatility_state == VolatilityState.LOW:
        confidence += 3

    action, _, recommended_timing = strategy._decide(
        higher_bias=higher_bias,
        trend_friendly=trend_friendly,
        setup_assessment=setup,
        trigger_assessment=trigger,
        confidence=confidence,
    )
    blockers = extract_blockers(
        setup=setup,
        trigger=trigger,
        trigger_requirements=trigger_requirements,
        trend_friendly=trend_friendly,
    )
    return {
        "timestamp": ts.isoformat(),
        "higher_bias": higher_bias.value,
        "trend_strength": int(trend_strength),
        "volatility_state": volatility_state.value,
        "trend_friendly": bool(trend_friendly),
        "confidence": int(confidence),
        "action": action.value,
        "recommended_timing": recommended_timing.value,
        "setup": setup,
        "trigger": trigger,
        "trigger_requirements": trigger_requirements,
        "blockers": blockers,
    }


def compute_interval_stats(
    *,
    trigger_frame: pd.DataFrame,
    side: str,
    entry_time: datetime,
    later_entry_time: datetime,
    entry_price: float,
    stop_price: float,
    tp1_price: float,
) -> dict[str, Any]:
    window = trigger_frame[
        (trigger_frame["timestamp"] >= pd.Timestamp(entry_time))
        & (trigger_frame["timestamp"] <= pd.Timestamp(later_entry_time))
    ].copy()
    if window.empty:
        return {
            "bars": 0,
            "favorable_r": 0.0,
            "adverse_r": 0.0,
            "tp1_hit_before_later_entry": False,
            "stop_hit_before_later_entry": False,
        }
    risk = abs(entry_price - stop_price)
    if risk <= 0:
        risk = 1.0
    if side == "SHORT":
        favorable = max(0.0, entry_price - float(window["low"].min()))
        adverse = max(0.0, float(window["high"].max()) - entry_price)
        tp1_hit = bool((window["low"] <= tp1_price).any())
        stop_hit = bool((window["high"] >= stop_price).any())
    else:
        favorable = max(0.0, float(window["high"].max()) - entry_price)
        adverse = max(0.0, entry_price - float(window["low"].min()))
        tp1_hit = bool((window["high"] >= tp1_price).any())
        stop_hit = bool((window["low"] <= stop_price).any())
    return {
        "bars": int(len(window)),
        "favorable_r": round(favorable / risk, 4),
        "adverse_r": round(adverse / risk, 4),
        "tp1_hit_before_later_entry": tp1_hit,
        "stop_hit_before_later_entry": stop_hit,
    }


def format_blockers(blockers: list[str]) -> str:
    mapping = {
        "trend_strength_not_friendly": "trend strength 不够友好",
        "setup_not_aligned": "setup 不再顺势",
        "pullback_not_ready": "pullback 还没回到执行区",
        "reversal_missing": "reversal 还没出现",
        "regained_fast_missing": "还没收回/失守 EMA21",
        "held_slow_missing": "还没站稳/失守 EMA55 或结构未修复",
        "auxiliary_missing": "辅助确认还不够",
    }
    return " / ".join(mapping[item] for item in blockers) if blockers else "无硬阻塞"


def classify_case(
    *,
    case: ReviewCase,
    main_at_alt: dict[str, Any],
    main_at_main: dict[str, Any],
    interval_stats: dict[str, Any],
) -> tuple[str, str]:
    blockers = set(main_at_alt["blockers"])
    favorable_r = float(interval_stats["favorable_r"])
    adverse_r = float(interval_stats["adverse_r"])

    if case.year == 2022:
        if "reversal_missing" in blockers and favorable_r >= 0.7:
            return (
                "更像 Mainline 等晚了",
                "Mainline 在 alt signal 时主要卡在 reversal，且等待窗口内市场已经先走出明显顺向利润。",
            )
        if "auxiliary_missing" in blockers and favorable_r >= 0.7:
            return (
                "更像确认等晚了",
                "Mainline 额外等辅助确认，代价是空头延续已经先走出一段。",
            )
        return (
            "更像晚 entry 与后续 stop 联合作用",
            "Mainline 并不只是单个条件错杀，而是确认更晚后把 RR 做坏，随后反抽触发 stop。",
        )

    if interval_stats["tp1_hit_before_later_entry"] and favorable_r >= 1.0 and adverse_r <= 0.3:
        return (
            "更像 Mainline 更善于吃 continuation",
            "No-Reversal-No-Aux 的更早 entry 本身并不坏，甚至在 mainline 入场前已经先拿到过部分顺向利润；Mainline 这里更像是后手 continuation 更强。",
        )
    if adverse_r >= 0.45 and favorable_r < 1.0:
        return (
            "更像 No-Reversal-No-Aux 提前了",
            "更早放行时，价格在 mainline entry 前还没有走出足够顺向空间，反而先承受了明显回撤。",
        )
    if "reversal_missing" in blockers and main_at_main["setup"]["reversal_ready"]:
        return (
            "更像 Mainline 等确认过滤了未完成 pullback",
            "Mainline 不是单纯慢，而是等到 reversal 真正出现后再进，从而避免了更早的未完成回踩。",
        )
    return (
        "更像 Mainline 更善于吃延续",
        "No-Reversal-No-Aux 不一定完全错误，但更早 entry 没拿到延续段，Mainline 的确认更匹配这段顺势行情。",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_case_svg(
    *,
    case: ReviewCase,
    frame: pd.DataFrame,
    main_trade: pd.Series,
    alt_trade: pd.Series,
    output_path: Path,
) -> None:
    window_start = min(case.main_signal_time, case.alt_signal_time) - timedelta(hours=24)
    window_end = min(case.main_exit_time, case.alt_exit_time, min(case.main_signal_time, case.alt_signal_time) + timedelta(hours=120))
    if window_end <= max(case.main_signal_time, case.alt_signal_time):
        window_end = max(case.main_signal_time, case.alt_signal_time) + timedelta(hours=72)

    subset = frame[
        (frame["timestamp"] >= pd.Timestamp(window_start))
        & (frame["timestamp"] <= pd.Timestamp(window_end))
    ].copy()
    subset = subset.reset_index(drop=True)

    width = 1380
    height = 460
    left = 74
    right = 28
    top = 42
    bottom = 58
    inner_w = width - left - right
    inner_h = height - top - bottom
    candle_w = max(inner_w / max(len(subset), 1) * 0.72, 1.0)

    prices = list(subset["low"]) + list(subset["high"]) + list(subset["ema_21"]) + list(subset["ema_55"])
    for value in (
        main_trade["entry_price"],
        alt_trade["entry_price"],
        main_trade["exit_price"],
        alt_trade["exit_price"],
        main_trade["stop_price"],
        alt_trade["stop_price"],
    ):
        prices.append(float(value))
    price_min = min(prices)
    price_max = max(prices)
    pad = (price_max - price_min) * 0.08 if price_max > price_min else price_max * 0.02 or 1.0
    price_min -= pad
    price_max += pad

    def x_at(index: int) -> float:
        if len(subset) <= 1:
            return left + inner_w / 2
        return left + (index / (len(subset) - 1)) * inner_w

    def y_at(price: float) -> float:
        if price_max == price_min:
            return top + inner_h / 2
        return top + inner_h - ((price - price_min) / (price_max - price_min)) * inner_h

    def idx_for(ts: datetime) -> int:
        idx = int(subset["timestamp"].searchsorted(pd.Timestamp(ts), side="left"))
        return max(0, min(idx, len(subset) - 1))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Menlo,Consolas,monospace;font-size:12px;fill:#111827}.title{font-size:18px;font-weight:700}.small{font-size:11px;fill:#4b5563}.axis{stroke:#9ca3af;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.bull{stroke:#0f766e;fill:#0f766e}.bear{stroke:#b91c1c;fill:#b91c1c}.ema21{stroke:#1d4ed8;fill:none;stroke-width:2}.ema55{stroke:#a16207;fill:none;stroke-width:2}.main{stroke:#0f766e;fill:#0f766e}.alt{stroke:#b45309;fill:#b45309}.sig{stroke-dasharray:5 4;stroke-width:1.5}</style>',
        f'<text x="{left}" y="24" class="title">{case.case_id} Focus Chart</text>',
        f'<text x="{left}" y="{height - 12}" class="small">1H candles with EMA21 / EMA55, signal-entry-exit markers</text>',
        f'<line x1="{left}" y1="{top + inner_h}" x2="{left + inner_w}" y2="{top + inner_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + inner_h}" class="axis"/>',
    ]

    for idx in range(5):
        value = price_min + (price_max - price_min) * idx / 4
        y = y_at(value)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + inner_w}" y2="{y:.1f}" class="grid"/>')
        svg.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="small">{value:.0f}</text>')

    ema21_points = []
    ema55_points = []
    for idx, row in subset.iterrows():
        x = x_at(idx)
        color_cls = "bull" if float(row["close"]) >= float(row["open"]) else "bear"
        svg.append(
            f'<line x1="{x:.1f}" y1="{y_at(float(row["high"])):.1f}" x2="{x:.1f}" y2="{y_at(float(row["low"])):.1f}" class="{color_cls}"/>'
        )
        body_top = y_at(max(float(row["open"]), float(row["close"])))
        body_bottom = y_at(min(float(row["open"]), float(row["close"])))
        body_height = max(abs(body_bottom - body_top), 1.2)
        rect_y = min(body_top, body_bottom)
        svg.append(
            f'<rect x="{x - candle_w / 2:.1f}" y="{rect_y:.1f}" width="{candle_w:.1f}" height="{body_height:.1f}" class="{color_cls}" fill-opacity="0.55"/>'
        )
        ema21_points.append(f"{x:.1f},{y_at(float(row['ema_21'])):.1f}")
        ema55_points.append(f"{x:.1f},{y_at(float(row['ema_55'])):.1f}")
    svg.append(f'<polyline class="ema21" points="{" ".join(ema21_points)}"/>')
    svg.append(f'<polyline class="ema55" points="{" ".join(ema55_points)}"/>')

    event_specs = [
        ("Main signal", case.main_signal_time, "main", True),
        ("Alt signal", case.alt_signal_time, "alt", True),
        ("Main entry", case.main_entry_time, "main", False),
        ("Alt entry", case.alt_entry_time, "alt", False),
        ("Main exit", case.main_exit_time, "main", False),
        ("Alt exit", case.alt_exit_time, "alt", False),
    ]
    for label, ts, cls, vertical in event_specs:
        if ts < window_start or ts > window_end:
            continue
        idx = idx_for(ts)
        x = x_at(idx)
        if vertical:
            svg.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + inner_h}" class="{cls} sig" opacity="0.85"/>'
            )
            svg.append(f'<text x="{x + 4:.1f}" y="{top + 14}" class="small">{label}</text>')

    trade_markers = [
        ("Main entry", case.main_entry_time, float(main_trade["entry_price"]), "main"),
        ("Alt entry", case.alt_entry_time, float(alt_trade["entry_price"]), "alt"),
        ("Main exit", case.main_exit_time, float(main_trade["exit_price"]), "main"),
        ("Alt exit", case.alt_exit_time, float(alt_trade["exit_price"]), "alt"),
    ]
    for label, ts, price, cls in trade_markers:
        if ts < window_start or ts > window_end:
            continue
        idx = idx_for(ts)
        x = x_at(idx)
        y = y_at(price)
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" class="{cls}"/>')
        svg.append(f'<text x="{x + 6:.1f}" y="{y - 6:.1f}" class="small">{label}</text>')

    tick_positions = sorted(set([0, len(subset) // 4, len(subset) // 2, (3 * len(subset)) // 4, len(subset) - 1]))
    for idx in tick_positions:
        x = x_at(idx)
        label = subset.iloc[idx]["timestamp"].strftime("%m-%d %H:%M")
        svg.append(f'<text x="{x:.1f}" y="{top + inner_h + 22}" text-anchor="middle" class="small">{label}</text>')

    legend_x = left
    legend_y = height - 34
    legend = [
        ("EMA21", "ema21"),
        ("EMA55", "ema55"),
        ("Mainline", "main"),
        ("No-Rev-No-Aux", "alt"),
    ]
    for idx, (label, cls) in enumerate(legend):
        x = legend_x + idx * 160
        if cls.startswith("ema"):
            svg.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 20}" y2="{legend_y}" class="{cls}"/>')
        else:
            svg.append(f'<circle cx="{x + 10}" cy="{legend_y}" r="5" class="{cls}"/>')
        svg.append(f'<text x="{x + 28}" y="{legend_y + 4}" class="small">{label}</text>')

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def build_case_rows(
    *,
    cases: list[ReviewCase],
    trades: pd.DataFrame,
    enriched: dict[str, pd.DataFrame],
    service: BacktestService,
    strategies: dict[str, SwingTrendEntryAttributionStrategy],
) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    report_lines = [
        "# Manual Case Review",
        "",
        "这份报告不是重新跑回测，而是基于原始 OOS 成交、缓存 OHLCV 和策略内部判定逻辑，对 10 个 case 做人工 review 前置整理。",
        "",
        "方法限制：",
        "- 这里能严格回答“当时哪一边被什么条件挡住”以及“等待窗口里价格先走了多少”。",
        "- 这里不能仅靠代码输出就断言“删掉 reversal 就一定更优”；图表仍然是最终裁决，只是这份报告已经把最关键的验证点预先压缩出来了。",
        "",
    ]

    trigger_frame = enriched["1h"]

    for case in cases:
        main_trade = trades.loc[(MAINLINE, pd.Timestamp(case.main_signal_time))]
        alt_trade = trades.loc[(NO_REV_NO_AUX, pd.Timestamp(case.alt_signal_time))]
        main_at_alt = evaluate_at(
            service=service,
            strategy=strategies["mainline"],
            enriched=enriched,
            ts=case.alt_signal_time,
        )
        alt_at_alt = evaluate_at(
            service=service,
            strategy=strategies["no_reversal_no_aux"],
            enriched=enriched,
            ts=case.alt_signal_time,
        )
        main_at_main = evaluate_at(
            service=service,
            strategy=strategies["mainline"],
            enriched=enriched,
            ts=case.main_signal_time,
        )
        alt_at_main = evaluate_at(
            service=service,
            strategy=strategies["no_reversal_no_aux"],
            enriched=enriched,
            ts=case.main_signal_time,
        )
        interval_stats = compute_interval_stats(
            trigger_frame=trigger_frame,
            side=case.side,
            entry_time=case.alt_entry_time,
            later_entry_time=case.main_entry_time,
            entry_price=float(alt_trade["entry_price"]),
            stop_price=float(alt_trade["stop_price"]),
            tp1_price=float(alt_trade["tp1_price"]),
        )
        verdict, verdict_note = classify_case(
            case=case,
            main_at_alt=main_at_alt,
            main_at_main=main_at_main,
            interval_stats=interval_stats,
        )
        chart_path = OUTPUT_DIR / f"{case.case_id.lower()}_focus.svg"
        render_case_svg(
            case=case,
            frame=trigger_frame,
            main_trade=main_trade,
            alt_trade=alt_trade,
            output_path=chart_path,
        )

        row = {
            "case_id": case.case_id,
            "year": case.year,
            "side": case.side,
            "delta_r": round(case.delta_r, 4),
            "gap_hours": round(case.gap_hours, 2),
            "verdict": verdict,
            "verdict_note": verdict_note,
            "mainline_blockers_at_alt_signal": format_blockers(main_at_alt["blockers"]),
            "mainline_action_at_alt_signal": main_at_alt["action"],
            "mainline_action_at_main_signal": main_at_main["action"],
            "no_rev_action_at_alt_signal": alt_at_alt["action"],
            "no_rev_action_at_main_signal": alt_at_main["action"],
            "wait_window_bars": interval_stats["bars"],
            "wait_window_favorable_r_for_alt": interval_stats["favorable_r"],
            "wait_window_adverse_r_for_alt": interval_stats["adverse_r"],
            "wait_window_tp1_hit_for_alt": interval_stats["tp1_hit_before_later_entry"],
            "wait_window_stop_hit_for_alt": interval_stats["stop_hit_before_later_entry"],
            "chart_svg": chart_path.name,
        }
        rows.append(row)

        report_lines.extend(
            [
                f"## {case.case_id}",
                "",
                f"- 初判: `{verdict}`",
                f"- 说明: {verdict_note}",
                f"- 差值: `{case.delta_r:.4f}R`，信号时间差 `{case.gap_hours:.1f}h`",
                f"- Mainline 在 alt signal 时: `{main_at_alt['action']}` / blockers `{format_blockers(main_at_alt['blockers'])}` / trigger `{main_at_alt['trigger']['state'].value}` / reversal_ready `{main_at_alt['setup']['reversal_ready']}` / aux `{main_at_alt['trigger']['auxiliary_count']}`",
                f"- Mainline 到自己 signal 时: `{main_at_main['action']}` / trigger `{main_at_main['trigger']['state'].value}` / reversal_ready `{main_at_main['setup']['reversal_ready']}` / aux `{main_at_main['trigger']['auxiliary_count']}`",
                f"- No-Reversal-No-Aux 在 alt signal 时: `{alt_at_alt['action']}` / trigger `{alt_at_alt['trigger']['state'].value}`",
                f"- 等待窗口统计: `{interval_stats['bars']}` bars，按 No-Reversal-No-Aux 的先行 entry 计，后者入场前最大顺向 `{interval_stats['favorable_r']:.4f}R`，最大逆向 `{interval_stats['adverse_r']:.4f}R`，`TP1 hit={interval_stats['tp1_hit_before_later_entry']}`，`Stop hit={interval_stats['stop_hit_before_later_entry']}`",
                f"- Mainline setup reasons_against @ alt: {', '.join(main_at_alt['setup']['reasons_against'][:3]) or '无'}",
                f"- Mainline trigger reasons_against @ alt: {', '.join(main_at_alt['trigger']['reasons_against'][:3]) or '无'}",
                f"- Mainline trigger reasons_for @ main: {', '.join(main_at_main['trigger']['reasons_for'][:3]) or '无'}",
                f"- 图: `{chart_path.name}`",
                "",
            ]
        )

    return rows, "\n".join(report_lines).strip() + "\n"


def main() -> None:
    ensure_output_dir()
    cases = load_cases()
    trades = load_trades()
    review_start = min(min(case.main_signal_time, case.alt_signal_time) for case in cases) - timedelta(days=400)
    review_end = max(max(case.main_exit_time, case.alt_exit_time) for case in cases) + timedelta(days=10)
    frames = load_frames(start=review_start, end=review_end)

    service = build_service()
    base_strategy = service.strategy_service.build_strategy("swing_trend_long_regime_gate_v1")
    enriched = {
        timeframe: service._enrich_frame(base_strategy, timeframe, frame)
        for timeframe, frame in frames.items()
    }
    strategies = build_strategies()
    rows, report = build_case_rows(
        cases=cases,
        trades=trades,
        enriched=enriched,
        service=service,
        strategies=strategies,
    )

    write_csv(OUTPUT_DIR / "case_review_summary.csv", rows)
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")
    print(f"Saved report: {OUTPUT_DIR / 'report.md'}")
    print(f"Saved summary CSV: {OUTPUT_DIR / 'case_review_summary.csv'}")


if __name__ == "__main__":
    main()
