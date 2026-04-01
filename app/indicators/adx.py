from __future__ import annotations

import numpy as np
import pandas as pd


def apply_adx_indicator(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add ADX, +DI, -DI columns to the dataframe.

    New columns:
        adx_{period}   – Average Directional Index (0-100)
        plus_di_{period} – Positive Directional Indicator
        minus_di_{period} – Negative Directional Indicator
    """
    enriched = df.copy()
    high = enriched["high"].astype(float)
    low = enriched["low"].astype(float)
    close = enriched["close"].astype(float)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # True Range
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Directional Movement
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Smoothed averages using Wilder's smoothing (EWM with alpha=1/period)
    alpha = 1.0 / period
    atr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()

    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / atr_smooth.replace(0, np.nan)).fillna(0.0)
    minus_di = 100 * (minus_dm_smooth / atr_smooth.replace(0, np.nan)).fillna(0.0)

    # DX and ADX
    di_sum = plus_di + minus_di
    dx = 100 * ((plus_di - minus_di).abs() / di_sum.replace(0, np.nan)).fillna(0.0)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    enriched[f"adx_{period}"] = adx.round(4)
    enriched[f"plus_di_{period}"] = plus_di.round(4)
    enriched[f"minus_di_{period}"] = minus_di.round(4)
    return enriched
