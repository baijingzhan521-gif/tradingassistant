from __future__ import annotations

from app.services.strategy_service import StrategyService
from app.schemas.common import Bias, VolatilityState


class TestNewStrategyProfiles:
    """Each new strategy is available via StrategyService.build_strategy()."""

    def test_trend_following_v1_available(self) -> None:
        strategy = StrategyService().build_strategy("trend_following_v1")
        assert strategy.name == "trend_following_v1"

    def test_swing_improved_v1_available(self) -> None:
        strategy = StrategyService().build_strategy("swing_improved_v1")
        assert strategy.name == "swing_improved_v1"

    def test_mean_reversion_v1_available(self) -> None:
        strategy = StrategyService().build_strategy("mean_reversion_v1")
        assert strategy.name == "mean_reversion_v1"


class TestTrendFollowingV1Config:
    def test_trend_strength_threshold(self) -> None:
        strategy = StrategyService().build_strategy("trend_following_v1")
        assert strategy.window_config["trend_strength_threshold"] == 40

    def test_confidence_threshold(self) -> None:
        strategy = StrategyService().build_strategy("trend_following_v1")
        assert strategy.config["confidence"]["action_threshold"] == 55

    def test_setup_timeframe(self) -> None:
        strategy = StrategyService().build_strategy("trend_following_v1")
        assert strategy.window_config["setup_timeframe"] == "4h"


class TestSwingImprovedV1Config:
    def test_bullish_trend_strength_threshold(self) -> None:
        strategy = StrategyService().build_strategy("swing_improved_v1")
        assert strategy.window_config["bullish_trend_strength_threshold"] == 72

    def test_bearish_trend_strength_threshold(self) -> None:
        strategy = StrategyService().build_strategy("swing_improved_v1")
        assert strategy.window_config["bearish_trend_strength_threshold"] == 50

    def test_confidence_threshold(self) -> None:
        strategy = StrategyService().build_strategy("swing_improved_v1")
        assert strategy.config["confidence"]["action_threshold"] == 60


class TestMeanReversionV1:
    def test_is_trend_friendly_always_true(self) -> None:
        strategy = StrategyService().build_strategy("mean_reversion_v1")
        assert strategy._is_trend_friendly(
            higher_bias=Bias.NEUTRAL,
            trend_strength=0,
            volatility_state=VolatilityState.HIGH,
        ) is True

    def test_is_trend_friendly_bullish(self) -> None:
        strategy = StrategyService().build_strategy("mean_reversion_v1")
        assert strategy._is_trend_friendly(
            higher_bias=Bias.BULLISH,
            trend_strength=10,
            volatility_state=VolatilityState.NORMAL,
        ) is True

    def test_is_trend_friendly_bearish_high_vol(self) -> None:
        strategy = StrategyService().build_strategy("mean_reversion_v1")
        assert strategy._is_trend_friendly(
            higher_bias=Bias.BEARISH,
            trend_strength=100,
            volatility_state=VolatilityState.HIGH,
        ) is True

    def test_mean_reversion_config(self) -> None:
        strategy = StrategyService().build_strategy("mean_reversion_v1")
        mr = strategy.config["mean_reversion"]
        assert mr["rsi_oversold"] == 35
        assert mr["rsi_overbought"] == 65
        assert mr["bb_pctb_low"] == 0.05
        assert mr["bb_pctb_high"] == 0.95
