from __future__ import annotations

import pandas as pd

from app.indicators.atr import apply_atr_indicator
from app.indicators.candle_profile import summarize_candle_profile
from app.indicators.divergence import apply_divergence_indicator, divergence_profile_from_row
from app.indicators.ema import apply_ema_indicators
from app.indicators.market_structure import classify_structure, determine_ema_alignment, determine_trend_bias
from app.indicators.swings import identify_swings, recent_swing_levels
from app.schemas.common import Bias, EmaAlignment, StructureState


def test_indicator_pipeline_produces_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=12, freq="h", tz="UTC"),
            "open": [100, 101, 102, 103, 102, 104, 105, 104, 106, 107, 106, 108],
            "high": [101, 102, 103, 104, 103, 105, 106, 105, 107, 108, 107, 109],
            "low": [99, 100, 101, 102, 101, 103, 104, 103, 105, 106, 105, 107],
            "close": [100, 101, 102, 103, 102, 104, 105, 104, 106, 107, 106, 108],
            "volume": [1000] * 12,
        }
    )

    enriched = apply_ema_indicators(df)
    enriched = apply_atr_indicator(enriched)
    enriched = identify_swings(enriched, window=1)

    assert {"ema_21", "ema_55", "ema_100", "ema_200", "atr_14"}.issubset(enriched.columns)
    assert enriched["atr_14"].iloc[-1] > 0


def test_structure_and_swings_classification() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=9, freq="h", tz="UTC"),
            "open": [100, 102, 101, 104, 103, 106, 105, 108, 107],
            "high": [101, 103, 102, 105, 104, 107, 106, 109, 108],
            "low": [99, 101, 100, 103, 102, 105, 104, 107, 106],
            "close": [100, 102, 101, 104, 103, 106, 105, 108, 107],
            "volume": [1000] * 9,
        }
    )

    enriched = identify_swings(df, window=1)
    swing_high, swing_low = recent_swing_levels(enriched)

    assert swing_high is not None
    assert swing_low is not None
    assert determine_ema_alignment(10, 9, 8, 7) == EmaAlignment.BULLISH
    assert determine_trend_bias(110, 100, EmaAlignment.BULLISH) == Bias.BULLISH
    assert classify_structure([101, 105], [99, 103]) == StructureState.BULLISH


def test_candle_profile_detects_spikes_and_volume_contraction() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=6, freq="15min", tz="UTC"),
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 120],
            "low": [99, 100, 101, 102, 103, 90],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 104.0],
            "volume": [2000, 1900, 1800, 1700, 1600, 600],
        }
    )

    profile = summarize_candle_profile(df, lookback=5)

    assert profile["is_spiky"] is True
    assert profile["is_volume_contracting"] is True
    assert profile["latest_upper_wick_ratio"] > profile["latest_lower_wick_ratio"]
    assert profile["has_bearish_rejection"] is True


def test_candle_profile_detects_doji_reversal_variants() -> None:
    bullish_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="15min", tz="UTC"),
            "open": [100, 100.2, 100.4, 100.6, 100.0],
            "high": [100.4, 100.5, 100.7, 100.9, 100.2],
            "low": [99.8, 100.0, 100.2, 100.4, 98.8],
            "close": [100.2, 100.4, 100.6, 100.8, 100.02],
            "volume": [1000, 980, 960, 940, 900],
        }
    )
    bearish_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="15min", tz="UTC"),
            "open": [100, 99.8, 99.6, 99.4, 100.0],
            "high": [100.2, 100.0, 99.8, 99.6, 101.2],
            "low": [99.6, 99.4, 99.2, 99.0, 99.8],
            "close": [99.8, 99.6, 99.4, 99.2, 99.98],
            "volume": [1000, 980, 960, 940, 900],
        }
    )

    bullish_profile = summarize_candle_profile(bullish_df, lookback=4)
    bearish_profile = summarize_candle_profile(bearish_df, lookback=4)

    assert bullish_profile["is_doji"] is True
    assert bullish_profile["has_bullish_reversal_candle"] is True
    assert bullish_profile["has_bearish_reversal_candle"] is False
    assert bearish_profile["is_doji"] is True
    assert bearish_profile["has_bearish_reversal_candle"] is True
    assert bearish_profile["has_bullish_reversal_candle"] is False


def test_divergence_indicator_detects_bullish_and_bearish_levels() -> None:
    bullish_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC"),
            "open": [110.0, 108.5, 107.0, 105.5, 103.5, 101.0, 99.0, 97.8],
            "high": [110.6, 109.0, 107.5, 106.0, 104.0, 101.8, 100.0, 100.3],
            "low": [109.2, 107.7, 106.0, 104.4, 102.0, 99.5, 97.5, 97.0],
            "close": [108.8, 107.2, 105.8, 104.0, 102.2, 100.0, 98.0, 99.2],
            "volume": [1000] * 8,
        }
    )
    bearish_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC"),
            "open": [100.0, 101.5, 103.0, 104.5, 106.5, 108.5, 111.0, 112.0],
            "high": [101.0, 102.5, 104.0, 105.5, 107.8, 109.8, 112.0, 114.0],
            "low": [99.5, 100.8, 102.2, 103.8, 105.8, 107.8, 110.5, 110.2],
            "close": [101.2, 102.8, 104.2, 105.8, 107.6, 109.5, 111.8, 110.8],
            "volume": [1000] * 8,
        }
    )

    bullish_enriched = apply_divergence_indicator(
        bullish_df,
        rsi_period=3,
        swing_window=3,
        ema_period=5,
        atr_period=14,
        min_rsi_diff=0.4,
        min_move_atr_mult=0.1,
        stretch_atr_mult=0.1,
        wick_ratio_min=0.25,
        min_reversal_score=2,
        cooldown_bars=0,
    )
    bearish_enriched = apply_divergence_indicator(
        bearish_df,
        rsi_period=3,
        swing_window=3,
        ema_period=5,
        atr_period=14,
        min_rsi_diff=0.4,
        min_move_atr_mult=0.1,
        stretch_atr_mult=0.1,
        wick_ratio_min=0.25,
        min_reversal_score=2,
        cooldown_bars=0,
    )

    bullish_profile = divergence_profile_from_row(bullish_enriched.iloc[-1], enabled=True)
    bearish_profile = divergence_profile_from_row(bearish_enriched.iloc[-1], enabled=True)

    assert bullish_profile["bullish_signal"] is True
    assert bullish_profile["bullish_level"] >= 1
    assert bullish_profile["bullish_reversal_score"] >= 2
    assert bearish_profile["bearish_signal"] is True
    assert bearish_profile["bearish_level"] >= 1
    assert bearish_profile["bearish_reversal_score"] >= 2
