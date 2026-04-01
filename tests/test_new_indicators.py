from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.indicators.adx import apply_adx_indicator
from app.indicators.bollinger import apply_bollinger_bands
from app.indicators.donchian import apply_donchian_channel
from app.indicators.regime_classifier import (
    MarketRegimeType,
    apply_regime_classifier,
    classify_market_regime,
)
from tests.conftest import make_ohlcv_frame


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return make_ohlcv_frame(periods=200, start_price=100.0, up_step=0.5, freq="1h")


class TestADX:
    def test_apply_adx_adds_expected_columns(self, sample_frame: pd.DataFrame) -> None:
        result = apply_adx_indicator(sample_frame, period=14)
        assert "adx_14" in result.columns
        assert "plus_di_14" in result.columns
        assert "minus_di_14" in result.columns

    def test_adx_values_are_bounded(self, sample_frame: pd.DataFrame) -> None:
        result = apply_adx_indicator(sample_frame, period=14)
        adx_vals = result["adx_14"].dropna()
        assert (adx_vals >= 0).all()
        assert (adx_vals <= 100).all()

    def test_adx_detects_strong_trend_in_uptrend(self, sample_frame: pd.DataFrame) -> None:
        result = apply_adx_indicator(sample_frame, period=14)
        last_adx = float(result["adx_14"].iloc[-1])
        assert last_adx > 20


class TestBollingerBands:
    def test_apply_bollinger_adds_expected_columns(self, sample_frame: pd.DataFrame) -> None:
        result = apply_bollinger_bands(sample_frame, period=20, num_std=2.0)
        assert "bb_mid_20" in result.columns
        assert "bb_upper_20" in result.columns
        assert "bb_lower_20" in result.columns
        assert "bb_width_20" in result.columns
        assert "bb_pctb_20" in result.columns

    def test_upper_always_above_lower(self, sample_frame: pd.DataFrame) -> None:
        result = apply_bollinger_bands(sample_frame, period=20)
        assert (result["bb_upper_20"] >= result["bb_lower_20"]).all()

    def test_mid_between_upper_and_lower(self, sample_frame: pd.DataFrame) -> None:
        result = apply_bollinger_bands(sample_frame, period=20)
        assert (result["bb_mid_20"] >= result["bb_lower_20"]).all()
        assert (result["bb_mid_20"] <= result["bb_upper_20"]).all()

    def test_pctb_positive_for_trending_data(self, sample_frame: pd.DataFrame) -> None:
        result = apply_bollinger_bands(sample_frame, period=20)
        last_pctb = float(result["bb_pctb_20"].iloc[-1])
        assert last_pctb > 0.0


class TestDonchianChannel:
    def test_apply_donchian_adds_expected_columns(self, sample_frame: pd.DataFrame) -> None:
        result = apply_donchian_channel(sample_frame, period=20)
        assert "dc_upper_20" in result.columns
        assert "dc_lower_20" in result.columns
        assert "dc_mid_20" in result.columns
        assert "dc_width_20" in result.columns
        assert "dc_breakout_up_20" in result.columns
        assert "dc_breakout_down_20" in result.columns

    def test_upper_always_above_lower(self, sample_frame: pd.DataFrame) -> None:
        result = apply_donchian_channel(sample_frame, period=20)
        assert (result["dc_upper_20"] >= result["dc_lower_20"]).all()

    def test_breakout_detection_in_uptrend(self) -> None:
        # Use larger step to ensure breakouts
        frame = make_ohlcv_frame(periods=200, start_price=100.0, up_step=2.0, freq="1h")
        result = apply_donchian_channel(frame, period=20)
        assert result["dc_breakout_up_20"].any()


class TestRegimeClassifier:
    def test_bull_trend_classification(self) -> None:
        regime = classify_market_regime(
            adx=35.0,
            realized_vol=0.5,
            median_vol=0.5,
            ema_aligned=1,
            price_vs_ema200_atr=2.0,
        )
        assert regime == MarketRegimeType.BULL_TREND

    def test_bear_trend_classification(self) -> None:
        regime = classify_market_regime(
            adx=30.0,
            realized_vol=0.5,
            median_vol=0.5,
            ema_aligned=-1,
            price_vs_ema200_atr=-2.0,
        )
        assert regime == MarketRegimeType.BEAR_TREND

    def test_low_vol_range_classification(self) -> None:
        regime = classify_market_regime(
            adx=15.0,
            realized_vol=0.3,
            median_vol=0.5,
            ema_aligned=0,
            price_vs_ema200_atr=0.5,
        )
        assert regime == MarketRegimeType.LOW_VOL_RANGE

    def test_high_vol_chop_classification(self) -> None:
        regime = classify_market_regime(
            adx=18.0,
            realized_vol=0.9,
            median_vol=0.5,
            ema_aligned=0,
            price_vs_ema200_atr=0.2,
        )
        assert regime == MarketRegimeType.HIGH_VOL_CHOP

    def test_transition_classification(self) -> None:
        regime = classify_market_regime(
            adx=26.0,
            realized_vol=0.5,
            median_vol=0.5,
            ema_aligned=0,
            price_vs_ema200_atr=0.5,
        )
        assert regime == MarketRegimeType.TRANSITION

    def test_apply_regime_classifier_adds_columns(self) -> None:
        from app.indicators.adx import apply_adx_indicator
        from app.indicators.atr import apply_atr_indicator
        from app.indicators.ema import apply_ema_indicators

        frame = make_ohlcv_frame(periods=200, start_price=100.0, up_step=0.5, freq="1d")
        frame = apply_ema_indicators(frame, periods=(21, 55, 100, 200))
        frame = apply_adx_indicator(frame, period=14)
        frame = apply_atr_indicator(frame, period=14)

        result = apply_regime_classifier(frame, adx_period=14)
        assert "regime" in result.columns
        assert "realized_vol_20" in result.columns
        valid_regimes = {r.value for r in MarketRegimeType}
        assert set(result["regime"].unique()).issubset(valid_regimes)
