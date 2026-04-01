from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_candle_profile(
    df: pd.DataFrame,
    lookback: int = 20,
    *,
    doji_body_ratio_max: float = 0.12,
    reversal_wick_ratio_min: float = 0.2,
) -> dict[str, Any]:
    if df.empty:
        return {
            "lookback": lookback,
            "latest_body_ratio": 0.0,
            "latest_upper_wick_ratio": 0.0,
            "latest_lower_wick_ratio": 0.0,
            "latest_range_ratio": 0.0,
            "volume_ratio": 0.0,
            "quote_volume_ratio": 0.0,
            "is_volume_contracting": False,
            "is_spiky": False,
            "is_doji": False,
            "has_bullish_rejection": False,
            "has_bearish_rejection": False,
            "has_bullish_reversal_candle": False,
            "has_bearish_reversal_candle": False,
        }

    recent = df.tail(max(lookback, 3)).copy()
    latest = recent.iloc[-1]
    history = recent.iloc[:-1]

    def _candle_parts(row: pd.Series) -> tuple[float, float, float, float]:
        high = float(row["high"])
        low = float(row["low"])
        open_price = float(row["open"])
        close = float(row["close"])
        candle_range = max(high - low, 0.0)
        body = abs(close - open_price)
        upper_wick = max(high - max(open_price, close), 0.0)
        lower_wick = max(min(open_price, close) - low, 0.0)
        return candle_range, body, upper_wick, lower_wick

    latest_range, latest_body, latest_upper_wick, latest_lower_wick = _candle_parts(latest)
    history_ranges = []
    history_volumes = []
    history_quote_volumes = []
    for _, row in history.iterrows():
        candle_range, _, _, _ = _candle_parts(row)
        history_ranges.append(candle_range)
        history_volumes.append(float(row["volume"]))
        history_quote_volumes.append(float(row["volume"]) * float(row["close"]))

    history_range_median = float(pd.Series(history_ranges).median()) if history_ranges else latest_range
    history_volume_median = float(pd.Series(history_volumes).median()) if history_volumes else float(latest["volume"])
    history_quote_volume_median = (
        float(pd.Series(history_quote_volumes).median())
        if history_quote_volumes
        else float(latest["volume"]) * float(latest["close"])
    )

    latest_volume = float(latest["volume"])
    latest_quote_volume = latest_volume * float(latest["close"])
    latest_body_ratio = latest_body / latest_range if latest_range else 0.0
    latest_upper_wick_ratio = latest_upper_wick / latest_range if latest_range else 0.0
    latest_lower_wick_ratio = latest_lower_wick / latest_range if latest_range else 0.0
    latest_range_ratio = latest_range / history_range_median if history_range_median else 0.0
    volume_ratio = latest_volume / history_volume_median if history_volume_median else 0.0
    quote_volume_ratio = latest_quote_volume / history_quote_volume_median if history_quote_volume_median else 0.0

    is_volume_contracting = volume_ratio <= 0.9 or quote_volume_ratio <= 0.9
    is_spiky = latest_range_ratio >= 1.6 or max(latest_upper_wick_ratio, latest_lower_wick_ratio) >= 0.45
    is_doji = latest_body_ratio <= doji_body_ratio_max
    has_long_lower_wick_reversal = (
        latest_lower_wick_ratio >= reversal_wick_ratio_min
        and latest_lower_wick_ratio > latest_upper_wick_ratio
    )
    has_long_upper_wick_reversal = (
        latest_upper_wick_ratio >= reversal_wick_ratio_min
        and latest_upper_wick_ratio > latest_lower_wick_ratio
    )
    has_bullish_rejection = (
        float(latest["close"]) >= float(latest["open"])
        and has_long_lower_wick_reversal
    )
    has_bearish_rejection = (
        float(latest["close"]) <= float(latest["open"])
        and has_long_upper_wick_reversal
    )
    has_bullish_reversal_candle = has_bullish_rejection or (is_doji and has_long_lower_wick_reversal)
    has_bearish_reversal_candle = has_bearish_rejection or (is_doji and has_long_upper_wick_reversal)

    return {
        "lookback": lookback,
        "latest_body_ratio": round(latest_body_ratio, 4),
        "latest_upper_wick_ratio": round(latest_upper_wick_ratio, 4),
        "latest_lower_wick_ratio": round(latest_lower_wick_ratio, 4),
        "latest_range_ratio": round(latest_range_ratio, 4),
        "volume_ratio": round(volume_ratio, 4),
        "quote_volume_ratio": round(quote_volume_ratio, 4),
        "is_volume_contracting": is_volume_contracting,
        "is_spiky": is_spiky,
        "is_doji": is_doji,
        "has_bullish_rejection": has_bullish_rejection,
        "has_bearish_rejection": has_bearish_rejection,
        "has_bullish_reversal_candle": has_bullish_reversal_candle,
        "has_bearish_reversal_candle": has_bearish_reversal_candle,
        "latest_quote_volume": round(latest_quote_volume, 4),
        "median_quote_volume": round(history_quote_volume_median, 4),
    }
