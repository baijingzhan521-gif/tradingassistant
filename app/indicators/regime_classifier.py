"""Market regime classification for portfolio allocation."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd


class MarketRegimeType(StrEnum):
    """Broad market regime categories used for portfolio weight allocation."""

    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    LOW_VOL_RANGE = "low_vol_range"
    HIGH_VOL_CHOP = "high_vol_chop"
    TRANSITION = "transition"


def classify_market_regime(
    *,
    adx: float,
    realized_vol: float,
    median_vol: float,
    ema_aligned: int,
    price_vs_ema200_atr: float,
) -> MarketRegimeType:
    """Classify current market regime from multiple factors.

    Args:
        adx: Current ADX value (0-100)
        realized_vol: Current realized volatility (annualized)
        median_vol: Median realized volatility over longer period
        ema_aligned: 1=bullish alignment, -1=bearish, 0=mixed
        price_vs_ema200_atr: (close - ema200) / atr, signed distance
    """
    is_trending = adx > 25
    is_low_vol = realized_vol < median_vol
    is_high_vol = realized_vol > median_vol * 1.5

    if is_trending and ema_aligned == 1 and price_vs_ema200_atr > 1.0:
        return MarketRegimeType.BULL_TREND
    if is_trending and ema_aligned == -1 and price_vs_ema200_atr < -1.0:
        return MarketRegimeType.BEAR_TREND
    if not is_trending and is_low_vol:
        return MarketRegimeType.LOW_VOL_RANGE
    if not is_trending and is_high_vol:
        return MarketRegimeType.HIGH_VOL_CHOP
    return MarketRegimeType.TRANSITION


def classify_regime_from_df(
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
    atr_pct = atr / close
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
    if high_vol and trend_up:
        return MarketRegimeType.BULL_TREND
    if high_vol and trend_down:
        return MarketRegimeType.BEAR_TREND

    return MarketRegimeType.TRANSITION


def apply_regime_classifier(
    df: pd.DataFrame,
    *,
    adx_period: int = 14,
    vol_window: int = 20,
    vol_median_window: int = 100,
) -> pd.DataFrame:
    """Add regime classification column to the dataframe.

    Requires columns: close, high, low, ema_21, ema_55, ema_100, ema_200, atr_{adx_period}.
    Also requires adx_{adx_period}.

    New columns:
        regime              – MarketRegimeType enum value
        realized_vol_{vol_window} – Annualized realized volatility
    """
    enriched = df.copy()
    close = enriched["close"].astype(float)

    # Realized volatility (annualized from log returns)
    log_returns = np.log(close / close.shift(1))
    realized_vol = log_returns.rolling(window=vol_window, min_periods=1).std() * np.sqrt(365)
    median_vol = realized_vol.rolling(window=vol_median_window, min_periods=1).median()

    enriched[f"realized_vol_{vol_window}"] = realized_vol.round(6)

    # EMA alignment
    ema21 = enriched["ema_21"].astype(float) if "ema_21" in enriched.columns else enriched.get("ema21", pd.Series(dtype=float))
    ema55 = enriched["ema_55"].astype(float) if "ema_55" in enriched.columns else enriched.get("ema55", pd.Series(dtype=float))
    ema100 = enriched["ema_100"].astype(float) if "ema_100" in enriched.columns else enriched.get("ema100", pd.Series(dtype=float))
    ema200 = enriched["ema_200"].astype(float) if "ema_200" in enriched.columns else enriched.get("ema200", pd.Series(dtype=float))

    bull_align = (ema21 > ema55) & (ema55 > ema100) & (ema100 > ema200)
    bear_align = (ema21 < ema55) & (ema55 < ema100) & (ema100 < ema200)
    ema_aligned = np.select([bull_align, bear_align], [1, -1], default=0)

    atr_col = f"atr_{adx_period}"
    adx_col = f"adx_{adx_period}"
    atr = enriched[atr_col].astype(float) if atr_col in enriched.columns else pd.Series(1.0, index=df.index)
    adx = enriched[adx_col].astype(float) if adx_col in enriched.columns else pd.Series(0.0, index=df.index)

    safe_atr = atr.replace(0.0, np.nan)
    price_vs_ema200_atr = ((close - ema200) / safe_atr).fillna(0.0)

    regimes = []
    for i in range(len(enriched)):
        rv = float(realized_vol.iloc[i]) if not np.isnan(realized_vol.iloc[i]) else 0.0
        mv = float(median_vol.iloc[i]) if not np.isnan(median_vol.iloc[i]) else 0.0
        regimes.append(
            classify_market_regime(
                adx=float(adx.iloc[i]),
                realized_vol=rv,
                median_vol=mv,
                ema_aligned=int(ema_aligned[i]),
                price_vs_ema200_atr=float(price_vs_ema200_atr.iloc[i]),
            ).value
        )
    enriched["regime"] = regimes
    return enriched
