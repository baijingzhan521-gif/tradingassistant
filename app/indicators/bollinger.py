from __future__ import annotations

import numpy as np
import pandas as pd


def apply_bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Add Bollinger Band columns to the dataframe.

    New columns:
        bb_mid_{period}   – Middle band (SMA)
        bb_upper_{period} – Upper band (SMA + num_std * std)
        bb_lower_{period} – Lower band (SMA - num_std * std)
        bb_width_{period} – Band width as fraction of mid
        bb_pctb_{period}  – %B (position within bands, 0=lower, 1=upper)
    """
    enriched = df.copy()
    close = enriched["close"].astype(float)

    mid = close.rolling(window=period, min_periods=1).mean()
    std = close.rolling(window=period, min_periods=1).std(ddof=0)

    upper = mid + num_std * std
    lower = mid - num_std * std

    band_width = np.where(mid != 0, (upper - lower) / mid, 0.0)
    band_range = (upper - lower).replace(0.0, np.nan)
    pctb = ((close - lower) / band_range).fillna(0.5)

    enriched[f"bb_mid_{period}"] = mid.round(6)
    enriched[f"bb_upper_{period}"] = upper.round(6)
    enriched[f"bb_lower_{period}"] = lower.round(6)
    enriched[f"bb_width_{period}"] = pd.Series(band_width, index=df.index).round(6)
    enriched[f"bb_pctb_{period}"] = pctb.round(6)
    return enriched
