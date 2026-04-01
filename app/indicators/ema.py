from __future__ import annotations

import pandas as pd


EMA_PERIODS = (21, 55, 100, 200)


def apply_ema_indicators(df: pd.DataFrame, periods: tuple[int, ...] = EMA_PERIODS) -> pd.DataFrame:
    enriched = df.copy()
    for period in periods:
        enriched[f"ema_{period}"] = enriched["close"].ewm(span=period, adjust=False).mean()
    return enriched
