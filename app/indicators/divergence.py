from __future__ import annotations

from typing import Any

import pandas as pd

from app.indicators.atr import apply_atr_indicator
from app.indicators.ema import apply_ema_indicators
from app.indicators.rsi import apply_rsi_indicator


def empty_divergence_profile(*, enabled: bool = False) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "leg_state": "neutral",
        "bullish_signal": False,
        "bearish_signal": False,
        "bullish_level": 0,
        "bearish_level": 0,
        "bullish_reversal_score": 0,
        "bearish_reversal_score": 0,
        "bullish_rsi_diff": 0.0,
        "bearish_rsi_diff": 0.0,
        "bullish_price_move_atr": 0.0,
        "bearish_price_move_atr": 0.0,
        "bullish_stretch_atr": 0.0,
        "bearish_stretch_atr": 0.0,
        "bullish_compare_offset": None,
        "bearish_compare_offset": None,
    }


def _bear_level(price_move_atr: float, rsi_diff: float) -> int:
    if price_move_atr >= 1.2 and rsi_diff >= 4.0:
        return 3
    if price_move_atr >= 0.8 and rsi_diff >= 2.8:
        return 2
    if price_move_atr >= 0.4 and rsi_diff >= 1.6:
        return 1
    return 0


def _bull_level(price_move_atr: float, rsi_diff: float) -> int:
    if price_move_atr >= 1.2 and rsi_diff >= 4.0:
        return 3
    if price_move_atr >= 0.8 and rsi_diff >= 2.8:
        return 2
    if price_move_atr >= 0.4 and rsi_diff >= 1.6:
        return 1
    return 0


def apply_divergence_indicator(
    df: pd.DataFrame,
    *,
    rsi_period: int = 10,
    swing_window: int = 14,
    ema_period: int = 34,
    atr_period: int = 14,
    min_rsi_diff: float = 1.6,
    min_move_atr_mult: float = 0.3,
    stretch_atr_mult: float = 0.8,
    wick_ratio_min: float = 0.4,
    min_reversal_score: int = 2,
    cooldown_bars: int = 20,
) -> pd.DataFrame:
    enriched = df.copy().sort_values("timestamp").reset_index(drop=True)
    if enriched.empty:
        return enriched

    if f"rsi_{rsi_period}" not in enriched.columns:
        enriched = apply_rsi_indicator(enriched, period=rsi_period)
    if f"ema_{ema_period}" not in enriched.columns:
        enriched = apply_ema_indicators(enriched, periods=(ema_period,))
    if f"atr_{atr_period}" not in enriched.columns:
        enriched = apply_atr_indicator(enriched, period=atr_period)

    high_roll = enriched["high"].rolling(window=swing_window, min_periods=1).max()
    low_roll = enriched["low"].rolling(window=swing_window, min_periods=1).min()
    is_new_high = enriched["high"] >= high_roll
    is_new_low = enriched["low"] <= low_roll

    bear_signals: list[bool] = []
    bull_signals: list[bool] = []
    bear_levels: list[int] = []
    bull_levels: list[int] = []
    bear_scores: list[int] = []
    bull_scores: list[int] = []
    bear_rsi_diffs: list[float] = []
    bull_rsi_diffs: list[float] = []
    bear_price_moves: list[float] = []
    bull_price_moves: list[float] = []
    bear_stretches: list[float] = []
    bull_stretches: list[float] = []
    bear_offsets: list[int | None] = []
    bull_offsets: list[int | None] = []
    leg_states: list[str] = []

    prior_high_events: list[tuple[int, float, float]] = []
    prior_low_events: list[tuple[int, float, float]] = []

    first_close = float(enriched.iloc[0]["close"])
    first_ema = float(enriched.iloc[0][f"ema_{ema_period}"])
    in_up_leg = first_close > first_ema
    in_down_leg = first_close < first_ema
    used_bear_in_leg = False
    used_bull_in_leg = False
    last_bear_idx: int | None = None
    last_bull_idx: int | None = None

    for idx in range(len(enriched)):
        row = enriched.iloc[idx]
        close = float(row["close"])
        open_price = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        leg_ema = float(row[f"ema_{ema_period}"])
        rsi = float(row[f"rsi_{rsi_period}"])
        atr = float(row[f"atr_{atr_period}"])

        if idx > 0:
            previous = enriched.iloc[idx - 1]
            previous_close = float(previous["close"])
            previous_ema = float(previous[f"ema_{ema_period}"])
            if previous_close <= previous_ema and close > leg_ema:
                in_up_leg = True
                in_down_leg = False
                used_bear_in_leg = False
            elif previous_close >= previous_ema and close < leg_ema:
                in_up_leg = False
                in_down_leg = True
                used_bull_in_leg = False

        leg_states.append("up" if in_up_leg else "down" if in_down_leg else "neutral")

        candle_range = max(high - low, 1e-9)
        upper_wick = max(high - max(open_price, close), 0.0)
        lower_wick = max(min(open_price, close) - low, 0.0)
        upper_ratio = upper_wick / candle_range
        lower_ratio = lower_wick / candle_range

        if idx > 0:
            previous = enriched.iloc[idx - 1]
            previous_open = float(previous["open"])
            previous_high = float(previous["high"])
            previous_low = float(previous["low"])
            previous_close = float(previous["close"])
            bear_engulf = (
                open_price > previous_close
                and close < previous_open
                and high >= previous_high
                and low <= previous_low
            )
            bull_engulf = (
                open_price < previous_close
                and close > previous_open
                and high >= previous_high
                and low <= previous_low
            )
            bos_down = close < previous_low
            bos_up = close > previous_high
            close_weaker = close < previous_close
            close_stronger = close > previous_close
        else:
            bear_engulf = False
            bull_engulf = False
            bos_down = False
            bos_up = False
            close_weaker = False
            close_stronger = False

        bear_reversal_score = (
            int(upper_ratio >= wick_ratio_min)
            + int(bear_engulf)
            + int(bos_down)
            + int(close_weaker)
        )
        bull_reversal_score = (
            int(lower_ratio >= wick_ratio_min)
            + int(bull_engulf)
            + int(bos_up)
            + int(close_stronger)
        )

        bear_signal = False
        bull_signal = False
        bear_level = 0
        bull_level = 0
        bear_rsi_diff = 0.0
        bull_rsi_diff = 0.0
        bear_price_move = 0.0
        bull_price_move = 0.0
        bear_stretch = 0.0
        bull_stretch = 0.0
        bear_offset: int | None = None
        bull_offset: int | None = None

        allow_bear_now = (
            in_up_leg
            and not used_bear_in_leg
            and (last_bear_idx is None or idx - last_bear_idx >= cooldown_bars)
        )
        allow_bull_now = (
            in_down_leg
            and not used_bull_in_leg
            and (last_bull_idx is None or idx - last_bull_idx >= cooldown_bars)
        )

        if allow_bear_now and bool(is_new_high.iloc[idx]) and atr > 0 and bear_reversal_score >= min_reversal_score:
            for offset in (3, 2, 1):
                if len(prior_high_events) < offset:
                    continue
                _, prior_high, prior_rsi = prior_high_events[-offset]
                price_move_atr = (high - prior_high) / atr
                rsi_diff = prior_rsi - rsi
                stretch_atr = (high - leg_ema) / atr
                if (
                    high > prior_high
                    and rsi_diff >= min_rsi_diff
                    and price_move_atr >= min_move_atr_mult
                    and stretch_atr >= stretch_atr_mult
                ):
                    level = _bear_level(price_move_atr, rsi_diff)
                    if level > 0:
                        bear_signal = True
                        bear_level = level
                        bear_rsi_diff = rsi_diff
                        bear_price_move = price_move_atr
                        bear_stretch = stretch_atr
                        bear_offset = offset
                        break

        if allow_bull_now and bool(is_new_low.iloc[idx]) and atr > 0 and bull_reversal_score >= min_reversal_score:
            for offset in (3, 2, 1):
                if len(prior_low_events) < offset:
                    continue
                _, prior_low, prior_rsi = prior_low_events[-offset]
                price_move_atr = (prior_low - low) / atr
                rsi_diff = rsi - prior_rsi
                stretch_atr = (leg_ema - low) / atr
                if (
                    low < prior_low
                    and rsi_diff >= min_rsi_diff
                    and price_move_atr >= min_move_atr_mult
                    and stretch_atr >= stretch_atr_mult
                ):
                    level = _bull_level(price_move_atr, rsi_diff)
                    if level > 0:
                        bull_signal = True
                        bull_level = level
                        bull_rsi_diff = rsi_diff
                        bull_price_move = price_move_atr
                        bull_stretch = stretch_atr
                        bull_offset = offset
                        break

        if bear_signal:
            used_bear_in_leg = True
            last_bear_idx = idx
        if bull_signal:
            used_bull_in_leg = True
            last_bull_idx = idx

        bear_signals.append(bear_signal)
        bull_signals.append(bull_signal)
        bear_levels.append(bear_level)
        bull_levels.append(bull_level)
        bear_scores.append(bear_reversal_score)
        bull_scores.append(bull_reversal_score)
        bear_rsi_diffs.append(bear_rsi_diff)
        bull_rsi_diffs.append(bull_rsi_diff)
        bear_price_moves.append(bear_price_move)
        bull_price_moves.append(bull_price_move)
        bear_stretches.append(bear_stretch)
        bull_stretches.append(bull_stretch)
        bear_offsets.append(bear_offset)
        bull_offsets.append(bull_offset)

        if bool(is_new_high.iloc[idx]):
            prior_high_events.append((idx, high, rsi))
        if bool(is_new_low.iloc[idx]):
            prior_low_events.append((idx, low, rsi))

    enriched["div_leg_state"] = leg_states
    enriched["div_bearish_signal"] = bear_signals
    enriched["div_bullish_signal"] = bull_signals
    enriched["div_bearish_level"] = bear_levels
    enriched["div_bullish_level"] = bull_levels
    enriched["div_bearish_reversal_score"] = bear_scores
    enriched["div_bullish_reversal_score"] = bull_scores
    enriched["div_bearish_rsi_diff"] = bear_rsi_diffs
    enriched["div_bullish_rsi_diff"] = bull_rsi_diffs
    enriched["div_bearish_price_move_atr"] = bear_price_moves
    enriched["div_bullish_price_move_atr"] = bull_price_moves
    enriched["div_bearish_stretch_atr"] = bear_stretches
    enriched["div_bullish_stretch_atr"] = bull_stretches
    enriched["div_bearish_compare_offset"] = bear_offsets
    enriched["div_bullish_compare_offset"] = bull_offsets
    return enriched


def divergence_profile_from_row(row: pd.Series | None, *, enabled: bool) -> dict[str, Any]:
    if row is None:
        return empty_divergence_profile(enabled=enabled)

    profile = empty_divergence_profile(enabled=enabled)
    if not enabled:
        return profile

    profile.update(
        {
            "leg_state": str(row.get("div_leg_state", "neutral")),
            "bullish_signal": bool(row.get("div_bullish_signal", False)),
            "bearish_signal": bool(row.get("div_bearish_signal", False)),
            "bullish_level": int(row.get("div_bullish_level", 0)),
            "bearish_level": int(row.get("div_bearish_level", 0)),
            "bullish_reversal_score": int(row.get("div_bullish_reversal_score", 0)),
            "bearish_reversal_score": int(row.get("div_bearish_reversal_score", 0)),
            "bullish_rsi_diff": round(float(row.get("div_bullish_rsi_diff", 0.0)), 4),
            "bearish_rsi_diff": round(float(row.get("div_bearish_rsi_diff", 0.0)), 4),
            "bullish_price_move_atr": round(float(row.get("div_bullish_price_move_atr", 0.0)), 4),
            "bearish_price_move_atr": round(float(row.get("div_bearish_price_move_atr", 0.0)), 4),
            "bullish_stretch_atr": round(float(row.get("div_bullish_stretch_atr", 0.0)), 4),
            "bearish_stretch_atr": round(float(row.get("div_bearish_stretch_atr", 0.0)), 4),
            "bullish_compare_offset": (
                int(row["div_bullish_compare_offset"])
                if pd.notna(row.get("div_bullish_compare_offset"))
                else None
            ),
            "bearish_compare_offset": (
                int(row["div_bearish_compare_offset"])
                if pd.notna(row.get("div_bearish_compare_offset"))
                else None
            ),
        }
    )
    return profile
