from __future__ import annotations

import numpy as np
import pandas as pd


def apply_rsi_indicator(df: pd.DataFrame, *, period: int = 14, source: str = "close") -> pd.DataFrame:
    enriched = df.copy()
    delta = enriched[source].astype(float).diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    rsi = rsi.where(~((avg_gain == 0.0) & (avg_loss == 0.0)), 50.0)

    enriched[f"rsi_{period}"] = rsi
    return enriched
