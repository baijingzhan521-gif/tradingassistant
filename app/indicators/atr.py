from __future__ import annotations

import pandas as pd


def apply_atr_indicator(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    enriched = df.copy()
    previous_close = enriched["close"].shift(1)
    true_range = pd.concat(
        [
            enriched["high"] - enriched["low"],
            (enriched["high"] - previous_close).abs(),
            (enriched["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    enriched[f"atr_{period}"] = true_range.ewm(alpha=1 / period, adjust=False).mean()
    return enriched
