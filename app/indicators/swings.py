from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd


def identify_swings(df: pd.DataFrame, window: int = 3, *, mode: str = "centered") -> pd.DataFrame:
    enriched = df.copy()
    highs = enriched["high"].to_numpy()
    lows = enriched["low"].to_numpy()
    swing_highs = np.full(len(enriched), np.nan)
    swing_lows = np.full(len(enriched), np.nan)

    if mode not in {"centered", "confirmed"}:
        raise ValueError(f"Unsupported swing detection mode: {mode}")

    for idx in range(window, len(enriched) - window):
        high_slice = highs[idx - window : idx + window + 1]
        low_slice = lows[idx - window : idx + window + 1]
        marker_idx = idx if mode == "centered" else idx + window

        if highs[idx] == high_slice.max() and np.count_nonzero(high_slice == highs[idx]) == 1:
            swing_highs[marker_idx] = highs[idx]
        if lows[idx] == low_slice.min() and np.count_nonzero(low_slice == lows[idx]) == 1:
            swing_lows[marker_idx] = lows[idx]

    enriched["swing_high_marker"] = swing_highs
    enriched["swing_low_marker"] = swing_lows
    return enriched


def recent_swing_levels(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    swing_highs = df["swing_high_marker"].dropna()
    swing_lows = df["swing_low_marker"].dropna()
    recent_high = float(swing_highs.iloc[-1]) if not swing_highs.empty else None
    recent_low = float(swing_lows.iloc[-1]) if not swing_lows.empty else None
    return recent_high, recent_low


def recent_swing_points(df: pd.DataFrame, count: int = 3) -> dict[str, list[dict[str, Any]]]:
    swing_highs = df.loc[df["swing_high_marker"].notna(), ["timestamp", "swing_high_marker"]].tail(count)
    swing_lows = df.loc[df["swing_low_marker"].notna(), ["timestamp", "swing_low_marker"]].tail(count)
    return {
        "highs": [
            {"timestamp": row.timestamp.isoformat(), "price": float(row.swing_high_marker)}
            for row in swing_highs.itertuples(index=False)
        ],
        "lows": [
            {"timestamp": row.timestamp.isoformat(), "price": float(row.swing_low_marker)}
            for row in swing_lows.itertuples(index=False)
        ],
    }
