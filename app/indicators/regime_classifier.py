"""Market regime classification for portfolio allocation."""

from __future__ import annotations

from enum import StrEnum

import pandas as pd


class MarketRegimeType(StrEnum):
    """Broad market regime categories used for portfolio weight allocation."""

    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    LOW_VOL_RANGE = "low_vol_range"
    HIGH_VOL_CHOP = "high_vol_chop"
    TRANSITION = "transition"


def classify_market_regime(
    df: pd.DataFrame,
    *,
    atr_period: int = 14,
    ema_fast: int = 20,
    ema_slow: int = 50,
) -> MarketRegimeType:
    """Classify current market regime from OHLCV data.

    Uses trend direction (EMA crossover) and volatility (ATR percentile)
    to bucket the market into one of five regime types.
    """
    if df is None or len(df) < max(atr_period, ema_slow) + 1:
        return MarketRegimeType.TRANSITION

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Trend via EMA crossover
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()
    trend_up = bool(ema_f.iloc[-1] > ema_s.iloc[-1])
    trend_down = bool(ema_f.iloc[-1] < ema_s.iloc[-1])

    # Volatility via ATR percentile
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    atr_pct = atr / close  # normalised ATR
    current_vol = float(atr_pct.iloc[-1]) if not pd.isna(atr_pct.iloc[-1]) else 0.0
    vol_median = float(atr_pct.median()) if not pd.isna(atr_pct.median()) else current_vol
    high_vol = current_vol > vol_median * 1.3

    if high_vol and not (trend_up or trend_down):
        return MarketRegimeType.HIGH_VOL_CHOP
    if not high_vol and not (trend_up or trend_down):
        return MarketRegimeType.LOW_VOL_RANGE
    if trend_up and not high_vol:
        return MarketRegimeType.BULL_TREND
    if trend_down and not high_vol:
        return MarketRegimeType.BEAR_TREND

    return MarketRegimeType.TRANSITION
