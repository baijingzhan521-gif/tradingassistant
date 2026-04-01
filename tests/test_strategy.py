from __future__ import annotations

from copy import deepcopy

from app.services.strategy_service import StrategyService
from app.schemas.common import Action, Bias, RecommendedTiming, VolatilityState
from app.schemas.common import TriggerState
from app.schemas.request import AnalyzeRequest
from app.indicators.divergence import empty_divergence_profile
from app.strategies.intraday_mtf_v1 import (
    DEFAULT_CONFIG as INTRADAY_DEFAULT_CONFIG,
    IntradayMTFV1Strategy,
)
from app.strategies.intraday_mtf_v2 import (
    DEFAULT_CONFIG as INTRADAY_V2_DEFAULT_CONFIG,
    IntradayMTFV2Strategy,
)
from app.strategies.swing_trend_v1 import (
    DEFAULT_CONFIG as SWING_DEFAULT_CONFIG,
    SwingTrendV1Strategy,
)
from app.strategies.swing_trend_divergence_v1 import (
    DEFAULT_CONFIG as SWING_DIVERGENCE_DEFAULT_CONFIG,
    SwingTrendDivergenceV1Strategy,
)
from app.strategies.swing_trend_divergence_min_level3_v1 import (
    DEFAULT_CONFIG as SWING_DIVERGENCE_MIN_LEVEL3_DEFAULT_CONFIG,
    SwingTrendDivergenceMinLevel3V1Strategy,
)
from app.strategies.swing_trend_divergence_no_reversal_v1 import (
    DEFAULT_CONFIG as SWING_DIVERGENCE_NO_REVERSAL_DEFAULT_CONFIG,
    SwingTrendDivergenceNoReversalV1Strategy,
)
from app.strategies.swing_trend_long_divergence_gate_v1 import (
    DEFAULT_CONFIG as SWING_LONG_DIVERGENCE_GATE_DEFAULT_CONFIG,
    SwingTrendLongDivergenceGateV1Strategy,
)
from app.strategies.swing_trend_long_regime_gate_v1 import (
    DEFAULT_CONFIG as SWING_LONG_REGIME_GATE_DEFAULT_CONFIG,
    SwingTrendLongRegimeGateV1Strategy,
)
from app.strategies.swing_trend_axis_band_diagnostic_v1 import (
    DEFAULT_CONFIG as SWING_AXIS_BAND_DIAGNOSTIC_DEFAULT_CONFIG,
    SwingTrendAxisBandDiagnosticV1Strategy,
)
from app.strategies.swing_trend_axis_band_state_note_v1 import (
    DEFAULT_CONFIG as SWING_AXIS_BAND_STATE_NOTE_DEFAULT_CONFIG,
    SwingTrendAxisBandStateNoteV1Strategy,
)
from app.strategies.swing_trend_confluence_setup_v1 import (
    DEFAULT_CONFIG as SWING_CONFLUENCE_SETUP_DEFAULT_CONFIG,
    SwingTrendConfluenceSetupV1Strategy,
)
from app.strategies.swing_trend_confluence_min_hits_3_v1 import (
    DEFAULT_CONFIG as SWING_CONFLUENCE_MIN_HITS_3_DEFAULT_CONFIG,
    SwingTrendConfluenceMinHits3V1Strategy,
)
from app.strategies.swing_trend_confluence_max_spread_10_v1 import (
    DEFAULT_CONFIG as SWING_CONFLUENCE_MAX_SPREAD_10_DEFAULT_CONFIG,
    SwingTrendConfluenceMaxSpread10V1Strategy,
)
from app.strategies.swing_trend_structure_gate_hard_v1 import (
    DEFAULT_CONFIG as SWING_STRUCTURE_GATE_HARD_DEFAULT_CONFIG,
    SwingTrendStructureGateHardV1Strategy,
)
from app.strategies.swing_trend_confluence_structure_gate_hard_v1 import (
    DEFAULT_CONFIG as SWING_CONFLUENCE_STRUCTURE_GATE_HARD_DEFAULT_CONFIG,
    SwingTrendConfluenceStructureGateHardV1Strategy,
)
from app.strategies.swing_trend_level_aware_confirmation_v1 import (
    DEFAULT_CONFIG as SWING_LEVEL_AWARE_CONFIRMATION_DEFAULT_CONFIG,
    SwingTrendLevelAwareConfirmationV1Strategy,
)
from app.strategies.swing_trend_level_aware_confirmation_min_hits_2_v1 import (
    DEFAULT_CONFIG as SWING_LEVEL_AWARE_CONFIRMATION_MIN_HITS_2_DEFAULT_CONFIG,
    SwingTrendLevelAwareConfirmationMinHits2V1Strategy,
)
from app.strategies.swing_trend_level_aware_confirmation_ema55_025_v1 import (
    DEFAULT_CONFIG as SWING_LEVEL_AWARE_CONFIRMATION_EMA55_025_DEFAULT_CONFIG,
    SwingTrendLevelAwareConfirmationEma55025V1Strategy,
)
from app.strategies.swing_trend_level_aware_confirmation_band_touch_035_v1 import (
    DEFAULT_CONFIG as SWING_LEVEL_AWARE_CONFIRMATION_BAND_TOUCH_035_DEFAULT_CONFIG,
    SwingTrendLevelAwareConfirmationBandTouch035V1Strategy,
)
from app.strategies.swing_trend_gate_entry_matrix import (
    GATE_SIMPLE_ENTRY_DEFAULT_CONFIG,
    NO_GATE_CURRENT_ENTRY_DEFAULT_CONFIG,
    NO_GATE_SIMPLE_ENTRY_DEFAULT_CONFIG,
    SwingTrendMatrixGateSimpleEntryV1Strategy,
    SwingTrendMatrixNoGateCurrentEntryV1Strategy,
    SwingTrendMatrixNoGateSimpleEntryV1Strategy,
)
from app.strategies.swing_trend_entry_attribution import (
    SwingTrendEntryAttributionStrategy,
    build_entry_attribution_config,
)
from app.strategies.swing_breakout_v1_btc import (
    DEFAULT_CONFIG as SWING_BREAKOUT_V1_BTC_DEFAULT_CONFIG,
    SwingBreakoutV1BTCStrategy,
)
from app.strategies.swing_breakout_setup_proximity_045_v1_btc import (
    DEFAULT_CONFIG as SWING_BREAKOUT_SETUP_PROXIMITY_045_V1_BTC_DEFAULT_CONFIG,
    SwingBreakoutSetupProximity045V1BTCStrategy,
)
from app.strategies.swing_breakout_trigger_buffer_004_v1_btc import (
    DEFAULT_CONFIG as SWING_BREAKOUT_TRIGGER_BUFFER_004_V1_BTC_DEFAULT_CONFIG,
    SwingBreakoutTriggerBuffer004V1BTCStrategy,
)
from app.strategies.swing_breakout_base_width_45_v1_btc import (
    DEFAULT_CONFIG as SWING_BREAKOUT_BASE_WIDTH_45_V1_BTC_DEFAULT_CONFIG,
    SwingBreakoutBaseWidth45V1BTCStrategy,
)
from app.strategies.swing_range_failure_v1_btc import (
    DEFAULT_CONFIG as SWING_RANGE_FAILURE_V1_BTC_DEFAULT_CONFIG,
    SwingRangeFailureV1BTCStrategy,
)
from app.strategies.swing_range_failure_edge_035_v1_btc import (
    DEFAULT_CONFIG as SWING_RANGE_FAILURE_EDGE_035_V1_BTC_DEFAULT_CONFIG,
    SwingRangeFailureEdge035V1BTCStrategy,
)
from app.strategies.swing_range_failure_sweep_008_v1_btc import (
    DEFAULT_CONFIG as SWING_RANGE_FAILURE_SWEEP_008_V1_BTC_DEFAULT_CONFIG,
    SwingRangeFailureSweep008V1BTCStrategy,
)
from app.strategies.swing_range_failure_max_width_45_v1_btc import (
    DEFAULT_CONFIG as SWING_RANGE_FAILURE_MAX_WIDTH_45_V1_BTC_DEFAULT_CONFIG,
    SwingRangeFailureMaxWidth45V1BTCStrategy,
)
from app.strategies.swing_exhaustion_divergence_v1_btc import (
    DEFAULT_CONFIG as SWING_EXHAUSTION_DIVERGENCE_V1_BTC_DEFAULT_CONFIG,
    SwingExhaustionDivergenceV1BTCStrategy,
)
from app.strategies.swing_exhaustion_divergence_short_only_v1_btc import (
    DEFAULT_CONFIG as SWING_EXHAUSTION_DIVERGENCE_SHORT_ONLY_V1_BTC_DEFAULT_CONFIG,
    SwingExhaustionDivergenceShortOnlyV1BTCStrategy,
)
from app.strategies.swing_exhaustion_divergence_min_level3_v1_btc import (
    DEFAULT_CONFIG as SWING_EXHAUSTION_DIVERGENCE_MIN_LEVEL3_V1_BTC_DEFAULT_CONFIG,
    SwingExhaustionDivergenceMinLevel3V1BTCStrategy,
)
from app.strategies.swing_exhaustion_divergence_ct_block80_v1_btc import (
    DEFAULT_CONFIG as SWING_EXHAUSTION_DIVERGENCE_CT_BLOCK80_V1_BTC_DEFAULT_CONFIG,
    SwingExhaustionDivergenceCTBlock80V1BTCStrategy,
)
from app.strategies.swing_neutral_range_reversion_v1_btc import (
    DEFAULT_CONFIG as SWING_NEUTRAL_RANGE_REVERSION_V1_BTC_DEFAULT_CONFIG,
    SwingNeutralRangeReversionV1BTCStrategy,
)
from app.strategies.swing_neutral_range_reversion_edge_030_v1_btc import (
    DEFAULT_CONFIG as SWING_NEUTRAL_RANGE_REVERSION_EDGE_030_V1_BTC_DEFAULT_CONFIG,
    SwingNeutralRangeReversionEdge030V1BTCStrategy,
)
from app.strategies.swing_neutral_range_reversion_sweep_008_v1_btc import (
    DEFAULT_CONFIG as SWING_NEUTRAL_RANGE_REVERSION_SWEEP_008_V1_BTC_DEFAULT_CONFIG,
    SwingNeutralRangeReversionSweep008V1BTCStrategy,
)
from app.strategies.swing_neutral_range_reversion_opp_r_100_v1_btc import (
    DEFAULT_CONFIG as SWING_NEUTRAL_RANGE_REVERSION_OPP_R_100_V1_BTC_DEFAULT_CONFIG,
    SwingNeutralRangeReversionOppR100V1BTCStrategy,
)
from app.strategies.swing_trend_long_regime_short_relaxed_trigger_v1 import (
    DEFAULT_CONFIG as SWING_LONG_REGIME_SHORT_RELAXED_TRIGGER_DEFAULT_CONFIG,
    SwingTrendLongRegimeShortRelaxedTriggerV1Strategy,
)
from app.strategies.swing_trend_long_regime_short90_free_space_v1 import (
    DEFAULT_CONFIG as SWING_LONG_REGIME_SHORT90_FREE_SPACE_DEFAULT_CONFIG,
    SwingTrendLongRegimeShort90FreeSpaceV1Strategy,
)
from app.strategies.swing_trend_long_regime_short_no_auxiliary_v1 import (
    DEFAULT_CONFIG as SWING_LONG_REGIME_SHORT_NO_AUXILIARY_DEFAULT_CONFIG,
    SwingTrendLongRegimeShortNoAuxiliaryV1Strategy,
)
from app.strategies.swing_trend_long_regime_short_no_reversal_v1 import (
    DEFAULT_CONFIG as SWING_LONG_REGIME_SHORT_NO_REVERSAL_DEFAULT_CONFIG,
    SwingTrendLongRegimeShortNoReversalV1Strategy,
)
from app.strategies.swing_trend_long_regime_short_no_reversal_no_aux_v1 import (
    DEFAULT_CONFIG as SWING_LONG_REGIME_SHORT_NO_REVERSAL_NO_AUX_DEFAULT_CONFIG,
    SwingTrendLongRegimeShortNoReversalNoAuxV1Strategy,
)
from app.strategies.swing_trend_simple_candidate_v1 import (
    DEFAULT_CONFIG as SWING_SIMPLE_CANDIDATE_V1_DEFAULT_CONFIG,
    SwingTrendSimpleCandidateV1Strategy,
)
from app.strategies.swing_trend_simple_candidate_v2 import (
    DEFAULT_CONFIG as SWING_SIMPLE_CANDIDATE_V2_DEFAULT_CONFIG,
    SwingTrendSimpleCandidateV2Strategy,
)
from app.strategies.trend_pullback_v1 import DEFAULT_CONFIG, TrendPullbackV1Strategy
from tests.conftest import make_ohlcv_frame


def test_trend_pullback_strategy_returns_bullish_signal_for_aligned_setup() -> None:
    strategy = TrendPullbackV1Strategy(DEFAULT_CONFIG)
    request = AnalyzeRequest(
        symbol="ETH/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h", "15m"],
        strategy_profile="trend_pullback_v1",
        lookback=300,
    )

    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(
            periods=300,
            start_price=2600,
            up_step=1.1,
            freq="h",
            pullback_step=-1.4,
            pullback_len=18,
            recovery_step=0.9,
            recovery_len=8,
        ),
        "15m": make_ohlcv_frame(
            periods=300,
            start_price=2850,
            up_step=0.25,
            freq="15min",
            pullback_step=-0.35,
            pullback_len=10,
            recovery_step=0.55,
            recovery_len=8,
        ),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.market_regime.higher_timeframe_bias == Bias.BULLISH
    assert result.decision.bias == Bias.BULLISH
    assert result.decision.action in {Action.LONG, Action.WAIT}
    assert result.decision.recommended_timing in {
        RecommendedTiming.NOW,
        RecommendedTiming.WAIT_CONFIRMATION,
        RecommendedTiming.WAIT_PULLBACK,
    }
    assert result.reasoning.reasons_for
    assert result.charts is not None
    assert len(result.charts.day_1.candles) > 0
    assert len(result.charts.hour_1.candles) > 0
    assert len(result.charts.min_15.ema21) == len(result.charts.min_15.candles)
    assert len(result.charts.min_15.ema100) == len(result.charts.min_15.candles)
    assert result.charts.min_3 is None
    assert result.raw_metrics["scorecard"]["total"] == result.decision.confidence


def test_trend_pullback_no_longer_uses_symbol_quality_filter() -> None:
    strategy = TrendPullbackV1Strategy(DEFAULT_CONFIG)
    request = AnalyzeRequest(
        symbol="LTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h", "15m"],
        strategy_profile="trend_pullback_v1",
        lookback=300,
    )

    one_hour = make_ohlcv_frame(
        periods=300,
        start_price=160,
        up_step=0.6,
        freq="h",
        pullback_step=-0.4,
        pullback_len=18,
        recovery_step=0.45,
        recovery_len=8,
    )
    fifteen = make_ohlcv_frame(
        periods=300,
        start_price=170,
        up_step=0.15,
        freq="15min",
        pullback_step=-0.12,
        pullback_len=10,
        recovery_step=0.18,
        recovery_len=8,
    )

    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=150, up_step=2.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=155, up_step=0.8, freq="4h"),
        "1h": one_hour,
        "15m": fifteen,
    }

    result = strategy.analyze(request, ohlcv)

    assert "symbol_quality" not in result.raw_metrics
    assert result.diagnostics.score_breakdown.total == result.decision.confidence


def test_swing_and_intraday_profiles_are_available() -> None:
    request = AnalyzeRequest(
        symbol="ETH/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h", "15m", "3m"],
        strategy_profile="swing_trend_v1",
        lookback=300,
    )
    swing_strategy = SwingTrendV1Strategy(SWING_DEFAULT_CONFIG)
    intraday_strategy = IntradayMTFV1Strategy(INTRADAY_DEFAULT_CONFIG)
    intraday_v2_strategy = IntradayMTFV2Strategy(INTRADAY_V2_DEFAULT_CONFIG)

    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
        "15m": make_ohlcv_frame(periods=300, start_price=2850, up_step=0.25, freq="15min"),
        "3m": make_ohlcv_frame(periods=300, start_price=2900, up_step=0.08, freq="3min"),
    }

    swing_result = swing_strategy.analyze(request, ohlcv)
    intraday_request = request.model_copy(update={"strategy_profile": "intraday_mtf_v1"})
    intraday_result = intraday_strategy.analyze(intraday_request, ohlcv)
    intraday_v2_request = request.model_copy(update={"strategy_profile": "intraday_mtf_v2"})
    intraday_v2_result = intraday_v2_strategy.analyze(intraday_v2_request, ohlcv)

    assert swing_result.strategy_profile == "swing_trend_v1"
    assert intraday_result.strategy_profile == "intraday_mtf_v1"
    assert intraday_v2_result.strategy_profile == "intraday_mtf_v2"
    assert intraday_result.charts is not None
    assert intraday_result.charts.min_3 is not None
    assert intraday_v2_result.charts is not None
    assert intraday_v2_result.charts.min_3 is not None


def test_swing_divergence_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_trend_divergence_v1",
        lookback=300,
    )
    strategy = SwingTrendDivergenceV1Strategy(SWING_DIVERGENCE_DEFAULT_CONFIG)
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_trend_divergence_v1"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_swing_divergence_min_level3_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_trend_divergence_min_level3_v1",
        lookback=300,
    )
    strategy = SwingTrendDivergenceMinLevel3V1Strategy(SWING_DIVERGENCE_MIN_LEVEL3_DEFAULT_CONFIG)
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_trend_divergence_min_level3_v1"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_swing_divergence_no_reversal_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_trend_divergence_no_reversal_v1",
        lookback=300,
    )
    strategy = SwingTrendDivergenceNoReversalV1Strategy(SWING_DIVERGENCE_NO_REVERSAL_DEFAULT_CONFIG)
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_trend_divergence_no_reversal_v1"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_swing_long_divergence_gate_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_trend_long_divergence_gate_v1",
        lookback=300,
    )
    strategy = SwingTrendLongDivergenceGateV1Strategy(SWING_LONG_DIVERGENCE_GATE_DEFAULT_CONFIG)
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_trend_long_divergence_gate_v1"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_swing_long_regime_gate_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_trend_long_regime_gate_v1",
        lookback=300,
    )
    strategy = SwingTrendLongRegimeGateV1Strategy(SWING_LONG_REGIME_GATE_DEFAULT_CONFIG)
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_trend_long_regime_gate_v1"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_swing_axis_band_diagnostic_profile_matches_mainline_decision_and_exposes_position_map() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_trend_long_regime_gate_v1",
        lookback=300,
    )
    mainline = SwingTrendLongRegimeGateV1Strategy(SWING_LONG_REGIME_GATE_DEFAULT_CONFIG)
    diagnostic = SwingTrendAxisBandDiagnosticV1Strategy(SWING_AXIS_BAND_DIAGNOSTIC_DEFAULT_CONFIG)
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    mainline_result = mainline.analyze(request, ohlcv)
    diagnostic_result = diagnostic.analyze(
        request.model_copy(update={"strategy_profile": "swing_trend_axis_band_diagnostic_v1"}),
        ohlcv,
    )

    assert diagnostic_result.strategy_profile == "swing_trend_axis_band_diagnostic_v1"
    assert diagnostic_result.decision.action == mainline_result.decision.action
    assert diagnostic_result.decision.bias == mainline_result.decision.bias
    assert diagnostic_result.decision.confidence == mainline_result.decision.confidence
    position_map = diagnostic_result.raw_metrics["timeframe_debug"]["1h"]["position_map"]
    assert position_map["volatility_mode"] == "atr"
    assert "band_upper" in position_map
    assert "band_lower" in position_map
    assert "axis_distance_vol" in position_map
    assert "band_position" in position_map


def test_swing_axis_band_state_note_profile_matches_mainline_decision_and_emits_pullback_risk_state_note() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_trend_long_regime_gate_v1",
        lookback=300,
    )
    mainline = SwingTrendLongRegimeGateV1Strategy(SWING_LONG_REGIME_GATE_DEFAULT_CONFIG)
    overlay = SwingTrendAxisBandStateNoteV1Strategy(SWING_AXIS_BAND_STATE_NOTE_DEFAULT_CONFIG)
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=5.0, freq="h"),
    }

    mainline_result = mainline.analyze(request, ohlcv)
    overlay_result = overlay.analyze(
        request.model_copy(update={"strategy_profile": "swing_trend_axis_band_state_note_v1"}),
        ohlcv,
    )

    setup_assessment = overlay_result.raw_metrics["setup_assessment"]

    assert overlay_result.strategy_profile == "swing_trend_axis_band_state_note_v1"
    assert overlay_result.decision.action == mainline_result.decision.action
    assert overlay_result.decision.bias == mainline_result.decision.bias
    assert overlay_result.decision.confidence == mainline_result.decision.confidence
    assert setup_assessment["state_note"]["enabled"] is True
    assert setup_assessment["state_note"]["active"] is True
    assert setup_assessment["state_note"]["label"] == "pullback_risk"
    assert any("回撤风险" in note for note in overlay_result.reasoning.state_notes)
    assert any("回撤风险" in note for note in overlay_result.diagnostics.state_notes)
    assert not any("回撤风险" in note for note in overlay_result.reasoning.risk_notes)


def test_swing_confluence_setup_profile_exposes_confluence_assessment() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_trend_confluence_setup_v1",
        lookback=300,
    )
    strategy = SwingTrendConfluenceSetupV1Strategy(SWING_CONFLUENCE_SETUP_DEFAULT_CONFIG)
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)
    setup_assessment = result.raw_metrics["setup_assessment"]

    assert result.strategy_profile == "swing_trend_confluence_setup_v1"
    assert setup_assessment["structure_source"] == "confluence"
    assert setup_assessment["confluence"]["enabled"] is True
    assert "ema55" in setup_assessment["confluence"]["components"]
    assert "pivot_anchor" in setup_assessment["confluence"]["components"]
    assert "band_edge" in setup_assessment["confluence"]["components"]
    assert "位置共振" in setup_assessment["score_note"]


def test_structure_gate_hard_profiles_enable_structure_ready_gate() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_trend_structure_gate_hard_v1",
        lookback=300,
    )
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    swing_distance_strategy = SwingTrendStructureGateHardV1Strategy(SWING_STRUCTURE_GATE_HARD_DEFAULT_CONFIG)
    confluence_strategy = SwingTrendConfluenceStructureGateHardV1Strategy(
        SWING_CONFLUENCE_STRUCTURE_GATE_HARD_DEFAULT_CONFIG
    )

    swing_distance_result = swing_distance_strategy.analyze(request, ohlcv)
    confluence_result = confluence_strategy.analyze(
        request.model_copy(update={"strategy_profile": "swing_trend_confluence_structure_gate_hard_v1"}),
        ohlcv,
    )

    assert swing_distance_result.strategy_profile == "swing_trend_structure_gate_hard_v1"
    assert confluence_result.strategy_profile == "swing_trend_confluence_structure_gate_hard_v1"
    assert swing_distance_result.raw_metrics["setup_assessment"]["structure_source"] == "swing_distance"
    assert confluence_result.raw_metrics["setup_assessment"]["structure_source"] == "confluence"
    assert SWING_STRUCTURE_GATE_HARD_DEFAULT_CONFIG["setup"]["require_structure_ready"] is True
    assert SWING_CONFLUENCE_STRUCTURE_GATE_HARD_DEFAULT_CONFIG["setup"]["require_structure_ready"] is True


def test_confluence_one_rule_profiles_are_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_trend_confluence_min_hits_3_v1",
        lookback=300,
    )
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    min_hits_strategy = SwingTrendConfluenceMinHits3V1Strategy(SWING_CONFLUENCE_MIN_HITS_3_DEFAULT_CONFIG)
    max_spread_strategy = SwingTrendConfluenceMaxSpread10V1Strategy(SWING_CONFLUENCE_MAX_SPREAD_10_DEFAULT_CONFIG)

    min_hits_result = min_hits_strategy.analyze(request, ohlcv)
    max_spread_result = max_spread_strategy.analyze(
        request.model_copy(update={"strategy_profile": "swing_trend_confluence_max_spread_10_v1"}),
        ohlcv,
    )

    assert min_hits_result.strategy_profile == "swing_trend_confluence_min_hits_3_v1"
    assert max_spread_result.strategy_profile == "swing_trend_confluence_max_spread_10_v1"
    assert min_hits_result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}
    assert max_spread_result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_level_aware_confirmation_profile_exposes_level_confirmation_assessment() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_trend_level_aware_confirmation_v1",
        lookback=300,
    )
    strategy = SwingTrendLevelAwareConfirmationV1Strategy(SWING_LEVEL_AWARE_CONFIRMATION_DEFAULT_CONFIG)
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)
    setup_assessment = result.raw_metrics["setup_assessment"]

    assert result.strategy_profile == "swing_trend_level_aware_confirmation_v1"
    assert setup_assessment["level_confirmation"]["enabled"] is True
    assert "ema55" in setup_assessment["level_confirmation"]["components"]
    assert "pivot_anchor" in setup_assessment["level_confirmation"]["components"]
    assert "band_edge" in setup_assessment["level_confirmation"]["components"]
    assert "reclaim/rejection" in setup_assessment["score_note"]


def test_level_aware_confirmation_one_rule_profiles_are_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_trend_level_aware_confirmation_min_hits_2_v1",
        lookback=300,
    )
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    min_hits_strategy = SwingTrendLevelAwareConfirmationMinHits2V1Strategy(
        SWING_LEVEL_AWARE_CONFIRMATION_MIN_HITS_2_DEFAULT_CONFIG
    )
    ema55_strategy = SwingTrendLevelAwareConfirmationEma55025V1Strategy(
        SWING_LEVEL_AWARE_CONFIRMATION_EMA55_025_DEFAULT_CONFIG
    )
    band_touch_strategy = SwingTrendLevelAwareConfirmationBandTouch035V1Strategy(
        SWING_LEVEL_AWARE_CONFIRMATION_BAND_TOUCH_035_DEFAULT_CONFIG
    )

    min_hits_result = min_hits_strategy.analyze(request, ohlcv)
    ema55_result = ema55_strategy.analyze(
        request.model_copy(update={"strategy_profile": "swing_trend_level_aware_confirmation_ema55_025_v1"}),
        ohlcv,
    )
    band_touch_result = band_touch_strategy.analyze(
        request.model_copy(update={"strategy_profile": "swing_trend_level_aware_confirmation_band_touch_035_v1"}),
        ohlcv,
    )

    assert min_hits_result.strategy_profile == "swing_trend_level_aware_confirmation_min_hits_2_v1"
    assert ema55_result.strategy_profile == "swing_trend_level_aware_confirmation_ema55_025_v1"
    assert band_touch_result.strategy_profile == "swing_trend_level_aware_confirmation_band_touch_035_v1"


def test_swing_matrix_no_gate_current_entry_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_trend_matrix_no_gate_current_entry_v1",
        lookback=300,
    )
    strategy = SwingTrendMatrixNoGateCurrentEntryV1Strategy(NO_GATE_CURRENT_ENTRY_DEFAULT_CONFIG)
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_trend_matrix_no_gate_current_entry_v1"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_swing_matrix_gate_simple_entry_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_trend_matrix_gate_simple_entry_v1",
        lookback=300,
    )
    strategy = SwingTrendMatrixGateSimpleEntryV1Strategy(GATE_SIMPLE_ENTRY_DEFAULT_CONFIG)
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_trend_matrix_gate_simple_entry_v1"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_swing_matrix_no_gate_simple_entry_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_trend_matrix_no_gate_simple_entry_v1",
        lookback=300,
    )
    strategy = SwingTrendMatrixNoGateSimpleEntryV1Strategy(NO_GATE_SIMPLE_ENTRY_DEFAULT_CONFIG)
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_trend_matrix_no_gate_simple_entry_v1"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_swing_breakout_v1_btc_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_breakout_v1_btc",
        lookback=300,
    )
    strategy = SwingBreakoutV1BTCStrategy(SWING_BREAKOUT_V1_BTC_DEFAULT_CONFIG)
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_breakout_v1_btc"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_swing_breakout_setup_proximity_045_v1_btc_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_breakout_setup_proximity_045_v1_btc",
        lookback=300,
    )
    strategy = SwingBreakoutSetupProximity045V1BTCStrategy(
        SWING_BREAKOUT_SETUP_PROXIMITY_045_V1_BTC_DEFAULT_CONFIG
    )
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_breakout_setup_proximity_045_v1_btc"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_swing_breakout_trigger_buffer_004_v1_btc_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_breakout_trigger_buffer_004_v1_btc",
        lookback=300,
    )
    strategy = SwingBreakoutTriggerBuffer004V1BTCStrategy(
        SWING_BREAKOUT_TRIGGER_BUFFER_004_V1_BTC_DEFAULT_CONFIG
    )
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_breakout_trigger_buffer_004_v1_btc"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_swing_breakout_base_width_45_v1_btc_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_breakout_base_width_45_v1_btc",
        lookback=300,
    )
    strategy = SwingBreakoutBaseWidth45V1BTCStrategy(
        SWING_BREAKOUT_BASE_WIDTH_45_V1_BTC_DEFAULT_CONFIG
    )
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_breakout_base_width_45_v1_btc"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_swing_range_failure_v1_btc_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_range_failure_v1_btc",
        lookback=300,
    )
    strategy = SwingRangeFailureV1BTCStrategy(SWING_RANGE_FAILURE_V1_BTC_DEFAULT_CONFIG)
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_range_failure_v1_btc"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_swing_range_failure_edge_035_v1_btc_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_range_failure_edge_035_v1_btc",
        lookback=300,
    )
    strategy = SwingRangeFailureEdge035V1BTCStrategy(
        SWING_RANGE_FAILURE_EDGE_035_V1_BTC_DEFAULT_CONFIG
    )
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_range_failure_edge_035_v1_btc"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_swing_range_failure_sweep_008_v1_btc_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_range_failure_sweep_008_v1_btc",
        lookback=300,
    )
    strategy = SwingRangeFailureSweep008V1BTCStrategy(
        SWING_RANGE_FAILURE_SWEEP_008_V1_BTC_DEFAULT_CONFIG
    )
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_range_failure_sweep_008_v1_btc"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_swing_range_failure_max_width_45_v1_btc_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_range_failure_max_width_45_v1_btc",
        lookback=300,
    )
    strategy = SwingRangeFailureMaxWidth45V1BTCStrategy(
        SWING_RANGE_FAILURE_MAX_WIDTH_45_V1_BTC_DEFAULT_CONFIG
    )
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_range_failure_max_width_45_v1_btc"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_swing_exhaustion_divergence_v1_btc_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_exhaustion_divergence_v1_btc",
        lookback=300,
    )
    strategy = SwingExhaustionDivergenceV1BTCStrategy(
        SWING_EXHAUSTION_DIVERGENCE_V1_BTC_DEFAULT_CONFIG
    )
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_exhaustion_divergence_v1_btc"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_swing_exhaustion_divergence_short_only_v1_btc_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_exhaustion_divergence_short_only_v1_btc",
        lookback=300,
    )
    strategy = SwingExhaustionDivergenceShortOnlyV1BTCStrategy(
        SWING_EXHAUSTION_DIVERGENCE_SHORT_ONLY_V1_BTC_DEFAULT_CONFIG
    )
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_exhaustion_divergence_short_only_v1_btc"
    assert result.market_regime.higher_timeframe_bias in {Bias.BEARISH, Bias.NEUTRAL}


def test_swing_exhaustion_divergence_min_level3_v1_btc_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_exhaustion_divergence_min_level3_v1_btc",
        lookback=300,
    )
    strategy = SwingExhaustionDivergenceMinLevel3V1BTCStrategy(
        SWING_EXHAUSTION_DIVERGENCE_MIN_LEVEL3_V1_BTC_DEFAULT_CONFIG
    )
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_exhaustion_divergence_min_level3_v1_btc"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_swing_exhaustion_divergence_ct_block80_v1_btc_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_exhaustion_divergence_ct_block80_v1_btc",
        lookback=300,
    )
    strategy = SwingExhaustionDivergenceCTBlock80V1BTCStrategy(
        SWING_EXHAUSTION_DIVERGENCE_CT_BLOCK80_V1_BTC_DEFAULT_CONFIG
    )
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_exhaustion_divergence_ct_block80_v1_btc"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def _changed_config_paths(base: dict, candidate: dict, *, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    keys = set(base.keys()) | set(candidate.keys())
    for key in keys:
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        base_value = base.get(key)
        candidate_value = candidate.get(key)
        if isinstance(base_value, dict) and isinstance(candidate_value, dict):
            paths |= _changed_config_paths(base_value, candidate_value, prefix=child_prefix)
            continue
        if base_value != candidate_value:
            paths.add(child_prefix)
    return paths


def test_exhaustion_one_rule_profiles_only_change_one_config_key() -> None:
    min_level3_changes = _changed_config_paths(
        SWING_EXHAUSTION_DIVERGENCE_V1_BTC_DEFAULT_CONFIG,
        SWING_EXHAUSTION_DIVERGENCE_MIN_LEVEL3_V1_BTC_DEFAULT_CONFIG,
    )
    ct_block80_changes = _changed_config_paths(
        SWING_EXHAUSTION_DIVERGENCE_V1_BTC_DEFAULT_CONFIG,
        SWING_EXHAUSTION_DIVERGENCE_CT_BLOCK80_V1_BTC_DEFAULT_CONFIG,
    )

    assert min_level3_changes == {"divergence.min_level"}
    assert ct_block80_changes == {"exhaustion.counter_trend_block_threshold"}


def test_trend_divergence_one_rule_profiles_only_change_one_config_key() -> None:
    min_level3_changes = _changed_config_paths(
        SWING_DIVERGENCE_DEFAULT_CONFIG,
        SWING_DIVERGENCE_MIN_LEVEL3_DEFAULT_CONFIG,
    )
    no_reversal_changes = _changed_config_paths(
        SWING_DIVERGENCE_DEFAULT_CONFIG,
        SWING_DIVERGENCE_NO_REVERSAL_DEFAULT_CONFIG,
    )
    long_gate_changes = _changed_config_paths(
        SWING_DIVERGENCE_DEFAULT_CONFIG,
        SWING_LONG_DIVERGENCE_GATE_DEFAULT_CONFIG,
    )

    assert min_level3_changes == {"divergence.min_level"}
    assert no_reversal_changes == {"setup.require_reversal_candle"}
    assert long_gate_changes == {"divergence.gate_biases", "divergence.gate_mode"}


def test_trend_confluence_one_rule_profiles_only_change_one_config_key() -> None:
    min_hits_changes = _changed_config_paths(
        SWING_CONFLUENCE_SETUP_DEFAULT_CONFIG,
        SWING_CONFLUENCE_MIN_HITS_3_DEFAULT_CONFIG,
    )
    max_spread_changes = _changed_config_paths(
        SWING_CONFLUENCE_SETUP_DEFAULT_CONFIG,
        SWING_CONFLUENCE_MAX_SPREAD_10_DEFAULT_CONFIG,
    )
    structure_hard_changes = _changed_config_paths(
        SWING_CONFLUENCE_SETUP_DEFAULT_CONFIG,
        SWING_CONFLUENCE_STRUCTURE_GATE_HARD_DEFAULT_CONFIG,
    )

    assert min_hits_changes == {"setup_confluence.min_hits"}
    assert max_spread_changes == {"setup_confluence.max_spread_atr"}
    assert structure_hard_changes == {"setup.require_structure_ready"}


def test_level_aware_confirmation_one_rule_profiles_only_change_one_config_key() -> None:
    min_hits_changes = _changed_config_paths(
        SWING_LEVEL_AWARE_CONFIRMATION_DEFAULT_CONFIG,
        SWING_LEVEL_AWARE_CONFIRMATION_MIN_HITS_2_DEFAULT_CONFIG,
    )
    ema55_changes = _changed_config_paths(
        SWING_LEVEL_AWARE_CONFIRMATION_DEFAULT_CONFIG,
        SWING_LEVEL_AWARE_CONFIRMATION_EMA55_025_DEFAULT_CONFIG,
    )
    band_touch_changes = _changed_config_paths(
        SWING_LEVEL_AWARE_CONFIRMATION_DEFAULT_CONFIG,
        SWING_LEVEL_AWARE_CONFIRMATION_BAND_TOUCH_035_DEFAULT_CONFIG,
    )

    assert min_hits_changes == {"level_confirmation.min_hits"}
    assert ema55_changes == {"level_confirmation.ema55_touch_proximity_atr"}
    assert band_touch_changes == {"level_confirmation.band_touch_proximity_atr"}


def test_breakout_one_rule_profiles_only_change_one_config_key() -> None:
    setup_proximity_changes = _changed_config_paths(
        SWING_BREAKOUT_V1_BTC_DEFAULT_CONFIG,
        SWING_BREAKOUT_SETUP_PROXIMITY_045_V1_BTC_DEFAULT_CONFIG,
    )
    trigger_buffer_changes = _changed_config_paths(
        SWING_BREAKOUT_V1_BTC_DEFAULT_CONFIG,
        SWING_BREAKOUT_TRIGGER_BUFFER_004_V1_BTC_DEFAULT_CONFIG,
    )
    base_width_changes = _changed_config_paths(
        SWING_BREAKOUT_V1_BTC_DEFAULT_CONFIG,
        SWING_BREAKOUT_BASE_WIDTH_45_V1_BTC_DEFAULT_CONFIG,
    )

    assert setup_proximity_changes == {"breakout.setup_proximity_atr"}
    assert trigger_buffer_changes == {"breakout.trigger_breakout_buffer_atr"}
    assert base_width_changes == {"breakout.base_max_width_atr"}


def test_range_failure_one_rule_profiles_only_change_one_config_key() -> None:
    edge_changes = _changed_config_paths(
        SWING_RANGE_FAILURE_V1_BTC_DEFAULT_CONFIG,
        SWING_RANGE_FAILURE_EDGE_035_V1_BTC_DEFAULT_CONFIG,
    )
    sweep_changes = _changed_config_paths(
        SWING_RANGE_FAILURE_V1_BTC_DEFAULT_CONFIG,
        SWING_RANGE_FAILURE_SWEEP_008_V1_BTC_DEFAULT_CONFIG,
    )
    max_width_changes = _changed_config_paths(
        SWING_RANGE_FAILURE_V1_BTC_DEFAULT_CONFIG,
        SWING_RANGE_FAILURE_MAX_WIDTH_45_V1_BTC_DEFAULT_CONFIG,
    )

    assert edge_changes == {"range_failure.edge_proximity_atr"}
    assert sweep_changes == {"range_failure.sweep_buffer_atr"}
    assert max_width_changes == {"range_failure.max_width_atr"}


def test_swing_neutral_range_reversion_v1_btc_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_neutral_range_reversion_v1_btc",
        lookback=300,
    )
    strategy = SwingNeutralRangeReversionV1BTCStrategy(
        SWING_NEUTRAL_RANGE_REVERSION_V1_BTC_DEFAULT_CONFIG
    )
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_neutral_range_reversion_v1_btc"
    assert result.market_regime.higher_timeframe_bias == Bias.NEUTRAL


def test_swing_neutral_range_reversion_edge_030_v1_btc_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_neutral_range_reversion_edge_030_v1_btc",
        lookback=300,
    )
    strategy = SwingNeutralRangeReversionEdge030V1BTCStrategy(
        SWING_NEUTRAL_RANGE_REVERSION_EDGE_030_V1_BTC_DEFAULT_CONFIG
    )
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_neutral_range_reversion_edge_030_v1_btc"
    assert result.market_regime.higher_timeframe_bias == Bias.NEUTRAL


def test_swing_neutral_range_reversion_sweep_008_v1_btc_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_neutral_range_reversion_sweep_008_v1_btc",
        lookback=300,
    )
    strategy = SwingNeutralRangeReversionSweep008V1BTCStrategy(
        SWING_NEUTRAL_RANGE_REVERSION_SWEEP_008_V1_BTC_DEFAULT_CONFIG
    )
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_neutral_range_reversion_sweep_008_v1_btc"
    assert result.market_regime.higher_timeframe_bias == Bias.NEUTRAL


def test_swing_neutral_range_reversion_opp_r_100_v1_btc_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_neutral_range_reversion_opp_r_100_v1_btc",
        lookback=300,
    )
    strategy = SwingNeutralRangeReversionOppR100V1BTCStrategy(
        SWING_NEUTRAL_RANGE_REVERSION_OPP_R_100_V1_BTC_DEFAULT_CONFIG
    )
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_neutral_range_reversion_opp_r_100_v1_btc"
    assert result.market_regime.higher_timeframe_bias == Bias.NEUTRAL


def test_neutral_range_one_rule_profiles_only_change_one_config_key() -> None:
    edge_changes = _changed_config_paths(
        SWING_NEUTRAL_RANGE_REVERSION_V1_BTC_DEFAULT_CONFIG,
        SWING_NEUTRAL_RANGE_REVERSION_EDGE_030_V1_BTC_DEFAULT_CONFIG,
    )
    sweep_changes = _changed_config_paths(
        SWING_NEUTRAL_RANGE_REVERSION_V1_BTC_DEFAULT_CONFIG,
        SWING_NEUTRAL_RANGE_REVERSION_SWEEP_008_V1_BTC_DEFAULT_CONFIG,
    )
    opp_r_changes = _changed_config_paths(
        SWING_NEUTRAL_RANGE_REVERSION_V1_BTC_DEFAULT_CONFIG,
        SWING_NEUTRAL_RANGE_REVERSION_OPP_R_100_V1_BTC_DEFAULT_CONFIG,
    )

    assert edge_changes == {"neutral_range.edge_proximity_atr"}
    assert sweep_changes == {"neutral_range.sweep_buffer_atr"}
    assert opp_r_changes == {"neutral_range.minimum_opposite_edge_r"}


def test_trend_friendly_threshold_can_differ_by_bias() -> None:
    strategy = SwingTrendLongRegimeGateV1Strategy(SWING_LONG_REGIME_GATE_DEFAULT_CONFIG)

    assert strategy._resolve_trend_strength_threshold(Bias.BULLISH) == 90
    assert strategy._resolve_trend_strength_threshold(Bias.BEARISH) == 60
    assert strategy._resolve_trend_strength_threshold(Bias.NEUTRAL) == 60


def test_no_gate_current_entry_disables_trend_strength_admission_only() -> None:
    strategy = SwingTrendMatrixNoGateCurrentEntryV1Strategy(NO_GATE_CURRENT_ENTRY_DEFAULT_CONFIG)

    assert strategy._is_trend_friendly(
        higher_bias=Bias.BULLISH,
        trend_strength=10,
        volatility_state=VolatilityState.NORMAL,
    ) is True
    assert strategy._is_trend_friendly(
        higher_bias=Bias.NEUTRAL,
        trend_strength=99,
        volatility_state=VolatilityState.NORMAL,
    ) is False


def test_gate_simple_entry_can_fire_without_current_trigger_confirmation() -> None:
    strategy = SwingTrendMatrixGateSimpleEntryV1Strategy(GATE_SIMPLE_ENTRY_DEFAULT_CONFIG)

    action, bias, timing = strategy._decide(
        higher_bias=Bias.BULLISH,
        trend_friendly=True,
        setup_assessment={
            "aligned": True,
            "pullback_ready": True,
            "is_extended": False,
        },
        trigger_assessment={"state": TriggerState.NONE},
        confidence=10,
    )

    assert action == Action.LONG
    assert bias == Bias.BULLISH
    assert timing == RecommendedTiming.NOW


def test_no_gate_simple_entry_still_waits_when_setup_is_extended() -> None:
    strategy = SwingTrendMatrixNoGateSimpleEntryV1Strategy(NO_GATE_SIMPLE_ENTRY_DEFAULT_CONFIG)

    action, bias, timing = strategy._decide(
        higher_bias=Bias.BEARISH,
        trend_friendly=False,
        setup_assessment={
            "aligned": True,
            "pullback_ready": False,
            "is_extended": True,
        },
        trigger_assessment={"state": TriggerState.NONE},
        confidence=10,
    )

    assert action == Action.WAIT
    assert bias == Bias.BEARISH
    assert timing == RecommendedTiming.WAIT_PULLBACK


def test_build_entry_attribution_config_toggles_only_requested_blocks() -> None:
    config = build_entry_attribution_config(
        include_reversal=False,
        include_regained_fast=True,
        include_held_slow=False,
        include_auxiliary=True,
    )

    assert config["setup"]["require_reversal_candle"] is False
    assert config["trigger"]["bullish_require_regained_fast"] is True
    assert config["trigger"]["bearish_require_regained_fast"] is True
    assert config["trigger"]["bullish_require_held_slow"] is False
    assert config["trigger"]["bearish_require_held_slow"] is False
    assert config["trigger"]["bullish_require_auxiliary"] is True
    assert config["trigger"]["bearish_require_auxiliary"] is True


def test_entry_attribution_strategy_bypasses_trigger_when_all_trigger_blocks_disabled() -> None:
    strategy = SwingTrendEntryAttributionStrategy(
        build_entry_attribution_config(
            include_reversal=False,
            include_regained_fast=False,
            include_held_slow=False,
            include_auxiliary=False,
        ),
        profile_name="entry_attr_test_all_off",
    )
    trigger_frame = make_ohlcv_frame(
        periods=300,
        start_price=100.0,
        up_step=0.08,
        freq="h",
    )
    ctx = strategy._prepare_timeframe("1h", trigger_frame)
    assessment = strategy._assess_trigger(Bias.BULLISH, ctx, "1h")

    assert assessment["state"] == TriggerState.BULLISH_CONFIRMED
    assert assessment["score"] == 0
    assert "不启用 current entry trigger 细节过滤" in assessment["score_note"]


def test_entry_attribution_strategy_keeps_trigger_logic_when_any_block_enabled() -> None:
    strategy = SwingTrendEntryAttributionStrategy(
        build_entry_attribution_config(
            include_reversal=True,
            include_regained_fast=True,
            include_held_slow=False,
            include_auxiliary=False,
        ),
        profile_name="entry_attr_test_partial_on",
    )
    trigger_frame = make_ohlcv_frame(
        periods=300,
        start_price=100.0,
        up_step=0.08,
        freq="h",
    )
    ctx = strategy._prepare_timeframe("1h", trigger_frame)
    assessment = strategy._assess_trigger(Bias.BULLISH, ctx, "1h")

    assert assessment["score_note"] != "1h 不启用 current entry trigger 细节过滤"


def test_swing_long_regime_short_relaxed_trigger_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_trend_long_regime_short_relaxed_trigger_v1",
        lookback=300,
    )
    strategy = SwingTrendLongRegimeShortRelaxedTriggerV1Strategy(
        SWING_LONG_REGIME_SHORT_RELAXED_TRIGGER_DEFAULT_CONFIG
    )
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_trend_long_regime_short_relaxed_trigger_v1"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_trigger_requirements_can_relax_bearish_regained_fast_only() -> None:
    strategy = SwingTrendLongRegimeShortRelaxedTriggerV1Strategy(
        SWING_LONG_REGIME_SHORT_RELAXED_TRIGGER_DEFAULT_CONFIG
    )

    bullish_requirements = strategy._resolve_trigger_requirements(Bias.BULLISH)
    bearish_requirements = strategy._resolve_trigger_requirements(Bias.BEARISH)

    assert bullish_requirements["require_regained_fast"] is True
    assert bearish_requirements["require_regained_fast"] is False
    assert bearish_requirements["require_held_slow"] is True
    assert bearish_requirements["require_auxiliary"] is True


def test_setup_reversal_requirement_falls_back_to_legacy_global_flag() -> None:
    strategy = SwingTrendLongRegimeGateV1Strategy(SWING_LONG_REGIME_GATE_DEFAULT_CONFIG)

    assert strategy._resolve_setup_reversal_requirement(Bias.BULLISH) is True
    assert strategy._resolve_setup_reversal_requirement(Bias.BEARISH) is True


def test_setup_reversal_requirement_can_split_by_bias() -> None:
    config = deepcopy(SWING_LONG_REGIME_GATE_DEFAULT_CONFIG)
    config["setup"]["bullish_require_reversal_candle"] = True
    config["setup"]["bearish_require_reversal_candle"] = False
    strategy = SwingTrendLongRegimeGateV1Strategy(config)

    assert strategy._resolve_setup_reversal_requirement(Bias.BULLISH) is True
    assert strategy._resolve_setup_reversal_requirement(Bias.BEARISH) is False


def test_swing_long_regime_short90_free_space_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_trend_long_regime_short90_free_space_v1",
        lookback=300,
    )
    strategy = SwingTrendLongRegimeShort90FreeSpaceV1Strategy(
        SWING_LONG_REGIME_SHORT90_FREE_SPACE_DEFAULT_CONFIG
    )
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_trend_long_regime_short90_free_space_v1"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_swing_simple_candidate_v1_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_trend_simple_candidate_v1",
        lookback=300,
    )
    strategy = SwingTrendSimpleCandidateV1Strategy(SWING_SIMPLE_CANDIDATE_V1_DEFAULT_CONFIG)
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_trend_simple_candidate_v1"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_swing_simple_candidate_v2_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_trend_simple_candidate_v2",
        lookback=300,
    )
    strategy = SwingTrendSimpleCandidateV2Strategy(SWING_SIMPLE_CANDIDATE_V2_DEFAULT_CONFIG)
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_trend_simple_candidate_v2"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_trigger_requirements_can_relax_bearish_regained_fast_in_90_plus_regime_only() -> None:
    strategy = SwingTrendLongRegimeShort90FreeSpaceV1Strategy(
        SWING_LONG_REGIME_SHORT90_FREE_SPACE_DEFAULT_CONFIG
    )

    bearish_weak = strategy._resolve_trigger_requirements(Bias.BEARISH, trend_strength=80)
    bearish_strong = strategy._resolve_trigger_requirements(Bias.BEARISH, trend_strength=90)
    bullish_strong = strategy._resolve_trigger_requirements(Bias.BULLISH, trend_strength=95)

    assert bearish_weak["require_regained_fast"] is True
    assert bearish_strong["require_regained_fast"] is False
    assert bearish_strong["regained_fast_relaxed_by_regime"] is True
    assert bullish_strong["require_regained_fast"] is True


def test_swing_long_regime_short_no_auxiliary_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_trend_long_regime_short_no_auxiliary_v1",
        lookback=300,
    )
    strategy = SwingTrendLongRegimeShortNoAuxiliaryV1Strategy(
        SWING_LONG_REGIME_SHORT_NO_AUXILIARY_DEFAULT_CONFIG
    )
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_trend_long_regime_short_no_auxiliary_v1"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_swing_long_regime_short_no_reversal_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_trend_long_regime_short_no_reversal_v1",
        lookback=300,
    )
    strategy = SwingTrendLongRegimeShortNoReversalV1Strategy(
        SWING_LONG_REGIME_SHORT_NO_REVERSAL_DEFAULT_CONFIG
    )
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_trend_long_regime_short_no_reversal_v1"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_swing_long_regime_short_no_reversal_no_aux_profile_is_available() -> None:
    request = AnalyzeRequest(
        symbol="BTC/USDT:USDT",
        market_type="perpetual",
        exchange="binance",
        timeframes=["1d", "4h", "1h"],
        strategy_profile="swing_trend_long_regime_short_no_reversal_no_aux_v1",
        lookback=300,
    )
    strategy = SwingTrendLongRegimeShortNoReversalNoAuxV1Strategy(
        SWING_LONG_REGIME_SHORT_NO_REVERSAL_NO_AUX_DEFAULT_CONFIG
    )
    ohlcv = {
        "1d": make_ohlcv_frame(periods=300, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=300, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=300, start_price=2600, up_step=1.1, freq="h"),
    }

    result = strategy.analyze(request, ohlcv)

    assert result.strategy_profile == "swing_trend_long_regime_short_no_reversal_no_aux_v1"
    assert result.market_regime.higher_timeframe_bias in {Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL}


def test_short_only_profiles_keep_bullish_reversal_requirement_from_mainline() -> None:
    no_reversal = SwingTrendLongRegimeShortNoReversalV1Strategy(
        SWING_LONG_REGIME_SHORT_NO_REVERSAL_DEFAULT_CONFIG
    )
    no_reversal_no_aux = SwingTrendLongRegimeShortNoReversalNoAuxV1Strategy(
        SWING_LONG_REGIME_SHORT_NO_REVERSAL_NO_AUX_DEFAULT_CONFIG
    )

    assert no_reversal._resolve_setup_reversal_requirement(Bias.BULLISH) is True
    assert no_reversal._resolve_setup_reversal_requirement(Bias.BEARISH) is False
    assert no_reversal_no_aux._resolve_setup_reversal_requirement(Bias.BULLISH) is True
    assert no_reversal_no_aux._resolve_setup_reversal_requirement(Bias.BEARISH) is False


def test_short_only_profiles_change_only_bearish_trigger_requirements() -> None:
    no_aux = SwingTrendLongRegimeShortNoAuxiliaryV1Strategy(
        SWING_LONG_REGIME_SHORT_NO_AUXILIARY_DEFAULT_CONFIG
    )
    no_reversal_no_aux = SwingTrendLongRegimeShortNoReversalNoAuxV1Strategy(
        SWING_LONG_REGIME_SHORT_NO_REVERSAL_NO_AUX_DEFAULT_CONFIG
    )

    no_aux_bullish = no_aux._resolve_trigger_requirements(Bias.BULLISH)
    no_aux_bearish = no_aux._resolve_trigger_requirements(Bias.BEARISH)
    no_reversal_no_aux_bullish = no_reversal_no_aux._resolve_trigger_requirements(Bias.BULLISH)
    no_reversal_no_aux_bearish = no_reversal_no_aux._resolve_trigger_requirements(Bias.BEARISH)

    assert no_aux_bullish["require_auxiliary"] is True
    assert no_aux_bearish["require_auxiliary"] is False
    assert no_reversal_no_aux_bullish["require_auxiliary"] is True
    assert no_reversal_no_aux_bearish["require_auxiliary"] is False


def test_strategy_service_registers_short_only_asymmetric_profiles() -> None:
    service = StrategyService()

    for profile in (
        "swing_trend_long_regime_short_no_auxiliary_v1",
        "swing_trend_long_regime_short_no_reversal_v1",
        "swing_trend_long_regime_short_no_reversal_no_aux_v1",
    ):
        strategy = service.build_strategy(profile)
        assert strategy.name == profile


def test_intraday_v2_requires_stronger_micro_confirmation_than_v1() -> None:
    trigger_frame = make_ohlcv_frame(
        periods=300,
        start_price=100.0,
        up_step=0.08,
        freq="3min",
    )
    v1 = IntradayMTFV1Strategy(INTRADAY_DEFAULT_CONFIG)
    v2 = IntradayMTFV2Strategy(INTRADAY_V2_DEFAULT_CONFIG)

    v1_ctx = v1._prepare_timeframe("3m", trigger_frame)
    v2_ctx = v2._prepare_timeframe("3m", trigger_frame)
    v1_trigger = v1._assess_trigger(Bias.BULLISH, v1_ctx, "3m")
    v2_trigger = v2._assess_trigger(Bias.BULLISH, v2_ctx, "3m")

    assert v1_trigger["state"] != v2_trigger["state"]
    assert v1_trigger["state"].value == "bullish_confirmed"
    assert v2_trigger["state"].value != "bullish_confirmed"
    assert v2_trigger["score"] < v1_trigger["score"]


def test_setup_reversal_gate_blocks_entry_without_required_reversal_candle() -> None:
    strategy = IntradayMTFV1Strategy(INTRADAY_DEFAULT_CONFIG)

    action, bias, timing = strategy._decide(
        higher_bias=Bias.BULLISH,
        trend_friendly=True,
        setup_assessment={
            "aligned": True,
            "pullback_ready": True,
            "reversal_ready": False,
            "require_reversal_candle": True,
            "is_extended": False,
        },
        trigger_assessment={"state": TriggerState.BULLISH_CONFIRMED},
        confidence=90,
    )

    assert action == Action.WAIT
    assert bias == Bias.BULLISH
    assert timing == RecommendedTiming.WAIT_CONFIRMATION


def test_setup_assessment_requires_reversal_candle_on_strategy_setup_timeframe() -> None:
    intraday = IntradayMTFV1Strategy(INTRADAY_DEFAULT_CONFIG)
    setup_frame = make_ohlcv_frame(
        periods=300,
        start_price=100.0,
        up_step=0.15,
        freq="15min",
        pullback_step=-0.1,
        pullback_len=12,
        recovery_step=0.1,
        recovery_len=4,
    )
    latest_idx = setup_frame.index[-1]
    latest_close = float(setup_frame.loc[latest_idx, "close"])
    setup_frame.loc[latest_idx, ["open", "high", "low", "close"]] = [
        latest_close + 0.1,
        latest_close + 0.25,
        latest_close - 1.0,
        latest_close + 0.02,
    ]

    ctx = intraday._prepare_timeframe("15m", setup_frame)
    assessment = intraday._assess_setup(Bias.BULLISH, ctx, "15m")

    assert assessment["require_reversal_candle"] is True
    assert assessment["reversal_ready"] is True
    assert any("止跌 K 线" in reason for reason in assessment["reasons_for"])

    no_reversal_frame = setup_frame.copy()
    no_reversal_frame.loc[latest_idx, ["open", "high", "low", "close"]] = [
        latest_close + 0.1,
        latest_close + 0.2,
        latest_close - 0.2,
        latest_close + 0.05,
    ]
    no_reversal_ctx = intraday._prepare_timeframe("15m", no_reversal_frame)
    no_reversal_assessment = intraday._assess_setup(Bias.BULLISH, no_reversal_ctx, "15m")

    assert no_reversal_assessment["reversal_ready"] is False
    assert any("还没有出现明确止跌 K 线" in reason for reason in no_reversal_assessment["reasons_against"])


def test_swing_divergence_setup_bonus_only_applies_when_same_direction_signal_exists() -> None:
    strategy = SwingTrendDivergenceV1Strategy(SWING_DIVERGENCE_DEFAULT_CONFIG)
    setup_frame = make_ohlcv_frame(
        periods=300,
        start_price=100.0,
        up_step=0.35,
        freq="h",
        pullback_step=-0.2,
        pullback_len=14,
        recovery_step=0.15,
        recovery_len=4,
    )
    ctx = strategy._prepare_timeframe("1h", setup_frame)

    control_ctx = deepcopy(ctx)
    control_ctx.divergence_profile = empty_divergence_profile(enabled=True)
    divergence_ctx = deepcopy(ctx)
    divergence_ctx.divergence_profile = {
        **empty_divergence_profile(enabled=True),
        "bullish_signal": True,
        "bullish_level": 2,
    }

    control_assessment = strategy._assess_setup(Bias.BULLISH, control_ctx, "1h")
    divergence_assessment = strategy._assess_setup(Bias.BULLISH, divergence_ctx, "1h")

    assert divergence_assessment["divergence_ready"] is True
    assert divergence_assessment["divergence_level"] == 2
    assert divergence_assessment["score"] == (
        control_assessment["score"] + int(strategy.config["divergence"]["level_2_bonus"])
    )
    assert any("Bull divergence L2" in reason for reason in divergence_assessment["reasons_for"])


def test_swing_long_divergence_gate_blocks_bullish_entry_without_divergence() -> None:
    strategy = SwingTrendLongDivergenceGateV1Strategy(SWING_LONG_DIVERGENCE_GATE_DEFAULT_CONFIG)

    action, bias, timing = strategy._decide(
        higher_bias=Bias.BULLISH,
        trend_friendly=True,
        setup_assessment={
            "aligned": True,
            "pullback_ready": True,
            "reversal_ready": True,
            "require_reversal_candle": True,
            "require_divergence_gate": True,
            "divergence_ready": False,
            "is_extended": False,
        },
        trigger_assessment={"state": TriggerState.BULLISH_CONFIRMED},
        confidence=90,
    )

    assert action == Action.WAIT
    assert bias == Bias.BULLISH
    assert timing == RecommendedTiming.WAIT_CONFIRMATION


def test_swing_long_divergence_gate_requires_bullish_divergence_in_setup_assessment() -> None:
    strategy = SwingTrendLongDivergenceGateV1Strategy(SWING_LONG_DIVERGENCE_GATE_DEFAULT_CONFIG)
    setup_frame = make_ohlcv_frame(
        periods=300,
        start_price=100.0,
        up_step=0.35,
        freq="h",
        pullback_step=-0.2,
        pullback_len=14,
        recovery_step=0.15,
        recovery_len=4,
    )
    ctx = strategy._prepare_timeframe("1h", setup_frame)

    blocked_ctx = deepcopy(ctx)
    blocked_ctx.divergence_profile = empty_divergence_profile(enabled=True)
    blocked_assessment = strategy._assess_setup(Bias.BULLISH, blocked_ctx, "1h")

    ready_ctx = deepcopy(ctx)
    ready_ctx.divergence_profile = {
        **empty_divergence_profile(enabled=True),
        "bullish_signal": True,
        "bullish_level": 2,
    }
    ready_assessment = strategy._assess_setup(Bias.BULLISH, ready_ctx, "1h")

    assert blocked_assessment["require_divergence_gate"] is True
    assert blocked_assessment["divergence_ready"] is False
    assert any("还没有出现 Bull divergence" in reason for reason in blocked_assessment["reasons_against"])
    assert ready_assessment["divergence_ready"] is True
    assert ready_assessment["score"] > blocked_assessment["score"]


def test_free_space_gate_blocks_long_setup_when_reference_swing_is_too_close() -> None:
    strategy = SwingTrendLongRegimeShort90FreeSpaceV1Strategy(
        SWING_LONG_REGIME_SHORT90_FREE_SPACE_DEFAULT_CONFIG
    )
    setup_frame = make_ohlcv_frame(
        periods=300,
        start_price=100.0,
        up_step=0.35,
        freq="h",
        pullback_step=-0.2,
        pullback_len=14,
        recovery_step=0.15,
        recovery_len=4,
    )
    setup_ctx = strategy._prepare_timeframe("1h", setup_frame)
    reference_ctx = deepcopy(setup_ctx)
    reference_ctx.model.swing_high = setup_ctx.model.close + (setup_ctx.model.atr14 * 0.4)

    assessment = strategy._assess_setup(
        Bias.BULLISH,
        setup_ctx,
        "1h",
        reference_ctx=reference_ctx,
        current_price=setup_ctx.model.close,
    )

    assert assessment["require_free_space_gate"] is True
    assert assessment["execution_ready"] is True
    assert assessment["free_space_ready"] is False
    assert assessment["pullback_ready"] is False
    assert any("free space" in reason for reason in assessment["reasons_against"])
