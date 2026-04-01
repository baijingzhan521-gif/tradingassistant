from __future__ import annotations

import numpy as np
import pandas as pd


def apply_donchian_channel(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add Donchian Channel columns to the dataframe.

    New columns:
        dc_upper_{period}   – Highest high over period
        dc_lower_{period}   – Lowest low over period
        dc_mid_{period}     – Midpoint of channel
        dc_width_{period}   – Channel width as fraction of mid
        dc_breakout_up_{period}  – True if close > previous bar's upper channel
        dc_breakout_down_{period} – True if close < previous bar's lower channel
    """
    enriched = df.copy()
    high = enriched["high"].astype(float)
    low = enriched["low"].astype(float)
    close = enriched["close"].astype(float)

    dc_upper = high.rolling(window=period, min_periods=1).max()
    dc_lower = low.rolling(window=period, min_periods=1).min()
    dc_mid = (dc_upper + dc_lower) / 2

    dc_width = np.where(dc_mid != 0, (dc_upper - dc_lower) / dc_mid, 0.0)

    # Breakout signals compare current close vs PREVIOUS bar's channel
    prev_upper = dc_upper.shift(1)
    prev_lower = dc_lower.shift(1)
    breakout_up = close > prev_upper
    breakout_down = close < prev_lower

    enriched[f"dc_upper_{period}"] = dc_upper.round(6)
    enriched[f"dc_lower_{period}"] = dc_lower.round(6)
    enriched[f"dc_mid_{period}"] = dc_mid.round(6)
    enriched[f"dc_width_{period}"] = pd.Series(dc_width, index=df.index).round(6)
    enriched[f"dc_breakout_up_{period}"] = breakout_up.fillna(False)
    enriched[f"dc_breakout_down_{period}"] = breakout_down.fillna(False)
    return enriched
