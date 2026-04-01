from __future__ import annotations

from typing import Optional

import pandas as pd

from app.core.exceptions import TradingAssistantError
from app.schemas.analysis import AnalysisResult
from app.schemas.request import AnalyzeRequest
from app.strategies.config_loader import StrategyConfigLoader
from app.strategies.base import Strategy
from app.strategies.intraday_mtf_v1 import (
    DEFAULT_CONFIG as INTRADAY_DEFAULT_CONFIG,
    IntradayMTFV1Strategy,
)
from app.strategies.intraday_mtf_v2 import (
    DEFAULT_CONFIG as INTRADAY_V2_DEFAULT_CONFIG,
    IntradayMTFV2Strategy,
)
from app.strategies.intraday_mtf_v2_pullback_075_v1 import (
    DEFAULT_CONFIG as INTRADAY_V2_PULLBACK_075_DEFAULT_CONFIG,
    IntradayMTFV2Pullback075V1Strategy,
)
from app.strategies.intraday_mtf_v2_trend70_v1 import (
    DEFAULT_CONFIG as INTRADAY_V2_TREND70_DEFAULT_CONFIG,
    IntradayMTFV2Trend70V1Strategy,
)
from app.strategies.intraday_mtf_v2_cooldown10_v1 import (
    DEFAULT_CONFIG as INTRADAY_V2_COOLDOWN10_DEFAULT_CONFIG,
    IntradayMTFV2Cooldown10V1Strategy,
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
from app.strategies.swing_trend_axis_band_risk_overlay_v1 import (
    DEFAULT_CONFIG as SWING_AXIS_BAND_RISK_OVERLAY_DEFAULT_CONFIG,
    SwingTrendAxisBandRiskOverlayV1Strategy,
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
from app.strategies.swing_exhaustion_divergence_ct_block80_v1_btc import (
    DEFAULT_CONFIG as SWING_EXHAUSTION_DIVERGENCE_CT_BLOCK80_V1_BTC_DEFAULT_CONFIG,
    SwingExhaustionDivergenceCTBlock80V1BTCStrategy,
)
from app.strategies.swing_exhaustion_divergence_min_level3_v1_btc import (
    DEFAULT_CONFIG as SWING_EXHAUSTION_DIVERGENCE_MIN_LEVEL3_V1_BTC_DEFAULT_CONFIG,
    SwingExhaustionDivergenceMinLevel3V1BTCStrategy,
)
from app.strategies.swing_exhaustion_divergence_short_only_v1_btc import (
    DEFAULT_CONFIG as SWING_EXHAUSTION_DIVERGENCE_SHORT_ONLY_V1_BTC_DEFAULT_CONFIG,
    SwingExhaustionDivergenceShortOnlyV1BTCStrategy,
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
from app.strategies.swing_trend_ablation_no_reversal_v1 import (
    DEFAULT_CONFIG as SWING_ABLATION_NO_REVERSAL_DEFAULT_CONFIG,
    SwingTrendAblationNoReversalV1Strategy,
)
from app.strategies.swing_trend_ablation_symmetric_regime_v1 import (
    DEFAULT_CONFIG as SWING_ABLATION_SYMMETRIC_REGIME_DEFAULT_CONFIG,
    SwingTrendAblationSymmetricRegimeV1Strategy,
)
from app.strategies.swing_trend_ablation_no_regained_fast_v1 import (
    DEFAULT_CONFIG as SWING_ABLATION_NO_REGAINED_FAST_DEFAULT_CONFIG,
    SwingTrendAblationNoRegainedFastV1Strategy,
)
from app.strategies.swing_trend_ablation_no_held_slow_v1 import (
    DEFAULT_CONFIG as SWING_ABLATION_NO_HELD_SLOW_DEFAULT_CONFIG,
    SwingTrendAblationNoHeldSlowV1Strategy,
)
from app.strategies.swing_trend_ablation_no_auxiliary_v1 import (
    DEFAULT_CONFIG as SWING_ABLATION_NO_AUXILIARY_DEFAULT_CONFIG,
    SwingTrendAblationNoAuxiliaryV1Strategy,
)
from app.strategies.swing_trend_ablation_minimal_trigger_v1 import (
    DEFAULT_CONFIG as SWING_ABLATION_MINIMAL_TRIGGER_DEFAULT_CONFIG,
    SwingTrendAblationMinimalTriggerV1Strategy,
)
from app.strategies.swing_trend_simple_candidate_v1 import (
    DEFAULT_CONFIG as SWING_SIMPLE_CANDIDATE_V1_DEFAULT_CONFIG,
    SwingTrendSimpleCandidateV1Strategy,
)
from app.strategies.swing_trend_simple_candidate_v2 import (
    DEFAULT_CONFIG as SWING_SIMPLE_CANDIDATE_V2_DEFAULT_CONFIG,
    SwingTrendSimpleCandidateV2Strategy,
)
from app.strategies.trend_following_v1 import (
    DEFAULT_CONFIG as TREND_FOLLOWING_V1_DEFAULT_CONFIG,
    TrendFollowingV1Strategy,
)
from app.strategies.swing_improved_v1 import (
    DEFAULT_CONFIG as SWING_IMPROVED_V1_DEFAULT_CONFIG,
    SwingImprovedV1Strategy,
)
from app.strategies.mean_reversion_v1 import (
    DEFAULT_CONFIG as MEAN_REVERSION_V1_DEFAULT_CONFIG,
    MeanReversionV1Strategy,
)
from app.strategies.trend_pullback_v1 import DEFAULT_CONFIG, TrendPullbackV1Strategy
from app.strategies.trend_pullback_pullback_085_v1 import (
    DEFAULT_CONFIG as TREND_PULLBACK_PULLBACK_085_DEFAULT_CONFIG,
    TrendPullbackPullback085V1Strategy,
)
from app.strategies.trend_pullback_trend70_v1 import (
    DEFAULT_CONFIG as TREND_PULLBACK_TREND70_DEFAULT_CONFIG,
    TrendPullbackTrend70V1Strategy,
)
from app.strategies.trend_pullback_aux2_v1 import (
    DEFAULT_CONFIG as TREND_PULLBACK_AUX2_DEFAULT_CONFIG,
    TrendPullbackAux2V1Strategy,
)


class StrategyService:
    _strategy_registry = {
        "trend_pullback_v1": (DEFAULT_CONFIG, TrendPullbackV1Strategy),
        "trend_pullback_pullback_085_v1": (
            TREND_PULLBACK_PULLBACK_085_DEFAULT_CONFIG,
            TrendPullbackPullback085V1Strategy,
        ),
        "trend_pullback_trend70_v1": (
            TREND_PULLBACK_TREND70_DEFAULT_CONFIG,
            TrendPullbackTrend70V1Strategy,
        ),
        "trend_pullback_aux2_v1": (
            TREND_PULLBACK_AUX2_DEFAULT_CONFIG,
            TrendPullbackAux2V1Strategy,
        ),
        "swing_trend_v1": (SWING_DEFAULT_CONFIG, SwingTrendV1Strategy),
        "swing_trend_divergence_v1": (SWING_DIVERGENCE_DEFAULT_CONFIG, SwingTrendDivergenceV1Strategy),
        "swing_trend_divergence_min_level3_v1": (
            SWING_DIVERGENCE_MIN_LEVEL3_DEFAULT_CONFIG,
            SwingTrendDivergenceMinLevel3V1Strategy,
        ),
        "swing_trend_divergence_no_reversal_v1": (
            SWING_DIVERGENCE_NO_REVERSAL_DEFAULT_CONFIG,
            SwingTrendDivergenceNoReversalV1Strategy,
        ),
        "swing_trend_long_divergence_gate_v1": (
            SWING_LONG_DIVERGENCE_GATE_DEFAULT_CONFIG,
            SwingTrendLongDivergenceGateV1Strategy,
        ),
        "swing_trend_long_regime_gate_v1": (
            SWING_LONG_REGIME_GATE_DEFAULT_CONFIG,
            SwingTrendLongRegimeGateV1Strategy,
        ),
        "swing_trend_axis_band_diagnostic_v1": (
            SWING_AXIS_BAND_DIAGNOSTIC_DEFAULT_CONFIG,
            SwingTrendAxisBandDiagnosticV1Strategy,
        ),
        "swing_trend_axis_band_risk_overlay_v1": (
            SWING_AXIS_BAND_RISK_OVERLAY_DEFAULT_CONFIG,
            SwingTrendAxisBandRiskOverlayV1Strategy,
        ),
        "swing_trend_axis_band_state_note_v1": (
            SWING_AXIS_BAND_STATE_NOTE_DEFAULT_CONFIG,
            SwingTrendAxisBandStateNoteV1Strategy,
        ),
        "swing_trend_confluence_setup_v1": (
            SWING_CONFLUENCE_SETUP_DEFAULT_CONFIG,
            SwingTrendConfluenceSetupV1Strategy,
        ),
        "swing_trend_confluence_min_hits_3_v1": (
            SWING_CONFLUENCE_MIN_HITS_3_DEFAULT_CONFIG,
            SwingTrendConfluenceMinHits3V1Strategy,
        ),
        "swing_trend_confluence_max_spread_10_v1": (
            SWING_CONFLUENCE_MAX_SPREAD_10_DEFAULT_CONFIG,
            SwingTrendConfluenceMaxSpread10V1Strategy,
        ),
        "swing_trend_structure_gate_hard_v1": (
            SWING_STRUCTURE_GATE_HARD_DEFAULT_CONFIG,
            SwingTrendStructureGateHardV1Strategy,
        ),
        "swing_trend_confluence_structure_gate_hard_v1": (
            SWING_CONFLUENCE_STRUCTURE_GATE_HARD_DEFAULT_CONFIG,
            SwingTrendConfluenceStructureGateHardV1Strategy,
        ),
        "swing_trend_level_aware_confirmation_v1": (
            SWING_LEVEL_AWARE_CONFIRMATION_DEFAULT_CONFIG,
            SwingTrendLevelAwareConfirmationV1Strategy,
        ),
        "swing_trend_level_aware_confirmation_min_hits_2_v1": (
            SWING_LEVEL_AWARE_CONFIRMATION_MIN_HITS_2_DEFAULT_CONFIG,
            SwingTrendLevelAwareConfirmationMinHits2V1Strategy,
        ),
        "swing_trend_level_aware_confirmation_ema55_025_v1": (
            SWING_LEVEL_AWARE_CONFIRMATION_EMA55_025_DEFAULT_CONFIG,
            SwingTrendLevelAwareConfirmationEma55025V1Strategy,
        ),
        "swing_trend_level_aware_confirmation_band_touch_035_v1": (
            SWING_LEVEL_AWARE_CONFIRMATION_BAND_TOUCH_035_DEFAULT_CONFIG,
            SwingTrendLevelAwareConfirmationBandTouch035V1Strategy,
        ),
        "swing_trend_matrix_no_gate_current_entry_v1": (
            NO_GATE_CURRENT_ENTRY_DEFAULT_CONFIG,
            SwingTrendMatrixNoGateCurrentEntryV1Strategy,
        ),
        "swing_trend_matrix_gate_simple_entry_v1": (
            GATE_SIMPLE_ENTRY_DEFAULT_CONFIG,
            SwingTrendMatrixGateSimpleEntryV1Strategy,
        ),
        "swing_trend_matrix_no_gate_simple_entry_v1": (
            NO_GATE_SIMPLE_ENTRY_DEFAULT_CONFIG,
            SwingTrendMatrixNoGateSimpleEntryV1Strategy,
        ),
        "swing_breakout_v1_btc": (
            SWING_BREAKOUT_V1_BTC_DEFAULT_CONFIG,
            SwingBreakoutV1BTCStrategy,
        ),
        "swing_breakout_setup_proximity_045_v1_btc": (
            SWING_BREAKOUT_SETUP_PROXIMITY_045_V1_BTC_DEFAULT_CONFIG,
            SwingBreakoutSetupProximity045V1BTCStrategy,
        ),
        "swing_breakout_trigger_buffer_004_v1_btc": (
            SWING_BREAKOUT_TRIGGER_BUFFER_004_V1_BTC_DEFAULT_CONFIG,
            SwingBreakoutTriggerBuffer004V1BTCStrategy,
        ),
        "swing_breakout_base_width_45_v1_btc": (
            SWING_BREAKOUT_BASE_WIDTH_45_V1_BTC_DEFAULT_CONFIG,
            SwingBreakoutBaseWidth45V1BTCStrategy,
        ),
        "swing_range_failure_v1_btc": (
            SWING_RANGE_FAILURE_V1_BTC_DEFAULT_CONFIG,
            SwingRangeFailureV1BTCStrategy,
        ),
        "swing_range_failure_edge_035_v1_btc": (
            SWING_RANGE_FAILURE_EDGE_035_V1_BTC_DEFAULT_CONFIG,
            SwingRangeFailureEdge035V1BTCStrategy,
        ),
        "swing_range_failure_sweep_008_v1_btc": (
            SWING_RANGE_FAILURE_SWEEP_008_V1_BTC_DEFAULT_CONFIG,
            SwingRangeFailureSweep008V1BTCStrategy,
        ),
        "swing_range_failure_max_width_45_v1_btc": (
            SWING_RANGE_FAILURE_MAX_WIDTH_45_V1_BTC_DEFAULT_CONFIG,
            SwingRangeFailureMaxWidth45V1BTCStrategy,
        ),
        "swing_exhaustion_divergence_v1_btc": (
            SWING_EXHAUSTION_DIVERGENCE_V1_BTC_DEFAULT_CONFIG,
            SwingExhaustionDivergenceV1BTCStrategy,
        ),
        "swing_exhaustion_divergence_ct_block80_v1_btc": (
            SWING_EXHAUSTION_DIVERGENCE_CT_BLOCK80_V1_BTC_DEFAULT_CONFIG,
            SwingExhaustionDivergenceCTBlock80V1BTCStrategy,
        ),
        "swing_exhaustion_divergence_min_level3_v1_btc": (
            SWING_EXHAUSTION_DIVERGENCE_MIN_LEVEL3_V1_BTC_DEFAULT_CONFIG,
            SwingExhaustionDivergenceMinLevel3V1BTCStrategy,
        ),
        "swing_exhaustion_divergence_short_only_v1_btc": (
            SWING_EXHAUSTION_DIVERGENCE_SHORT_ONLY_V1_BTC_DEFAULT_CONFIG,
            SwingExhaustionDivergenceShortOnlyV1BTCStrategy,
        ),
        "swing_neutral_range_reversion_v1_btc": (
            SWING_NEUTRAL_RANGE_REVERSION_V1_BTC_DEFAULT_CONFIG,
            SwingNeutralRangeReversionV1BTCStrategy,
        ),
        "swing_neutral_range_reversion_edge_030_v1_btc": (
            SWING_NEUTRAL_RANGE_REVERSION_EDGE_030_V1_BTC_DEFAULT_CONFIG,
            SwingNeutralRangeReversionEdge030V1BTCStrategy,
        ),
        "swing_neutral_range_reversion_sweep_008_v1_btc": (
            SWING_NEUTRAL_RANGE_REVERSION_SWEEP_008_V1_BTC_DEFAULT_CONFIG,
            SwingNeutralRangeReversionSweep008V1BTCStrategy,
        ),
        "swing_neutral_range_reversion_opp_r_100_v1_btc": (
            SWING_NEUTRAL_RANGE_REVERSION_OPP_R_100_V1_BTC_DEFAULT_CONFIG,
            SwingNeutralRangeReversionOppR100V1BTCStrategy,
        ),
        "swing_trend_long_regime_short_relaxed_trigger_v1": (
            SWING_LONG_REGIME_SHORT_RELAXED_TRIGGER_DEFAULT_CONFIG,
            SwingTrendLongRegimeShortRelaxedTriggerV1Strategy,
        ),
        "swing_trend_long_regime_short90_free_space_v1": (
            SWING_LONG_REGIME_SHORT90_FREE_SPACE_DEFAULT_CONFIG,
            SwingTrendLongRegimeShort90FreeSpaceV1Strategy,
        ),
        "swing_trend_long_regime_short_no_auxiliary_v1": (
            SWING_LONG_REGIME_SHORT_NO_AUXILIARY_DEFAULT_CONFIG,
            SwingTrendLongRegimeShortNoAuxiliaryV1Strategy,
        ),
        "swing_trend_long_regime_short_no_reversal_v1": (
            SWING_LONG_REGIME_SHORT_NO_REVERSAL_DEFAULT_CONFIG,
            SwingTrendLongRegimeShortNoReversalV1Strategy,
        ),
        "swing_trend_long_regime_short_no_reversal_no_aux_v1": (
            SWING_LONG_REGIME_SHORT_NO_REVERSAL_NO_AUX_DEFAULT_CONFIG,
            SwingTrendLongRegimeShortNoReversalNoAuxV1Strategy,
        ),
        "swing_trend_ablation_no_reversal_v1": (
            SWING_ABLATION_NO_REVERSAL_DEFAULT_CONFIG,
            SwingTrendAblationNoReversalV1Strategy,
        ),
        "swing_trend_ablation_symmetric_regime_v1": (
            SWING_ABLATION_SYMMETRIC_REGIME_DEFAULT_CONFIG,
            SwingTrendAblationSymmetricRegimeV1Strategy,
        ),
        "swing_trend_ablation_no_regained_fast_v1": (
            SWING_ABLATION_NO_REGAINED_FAST_DEFAULT_CONFIG,
            SwingTrendAblationNoRegainedFastV1Strategy,
        ),
        "swing_trend_ablation_no_held_slow_v1": (
            SWING_ABLATION_NO_HELD_SLOW_DEFAULT_CONFIG,
            SwingTrendAblationNoHeldSlowV1Strategy,
        ),
        "swing_trend_ablation_no_auxiliary_v1": (
            SWING_ABLATION_NO_AUXILIARY_DEFAULT_CONFIG,
            SwingTrendAblationNoAuxiliaryV1Strategy,
        ),
        "swing_trend_ablation_minimal_trigger_v1": (
            SWING_ABLATION_MINIMAL_TRIGGER_DEFAULT_CONFIG,
            SwingTrendAblationMinimalTriggerV1Strategy,
        ),
        "swing_trend_simple_candidate_v1": (
            SWING_SIMPLE_CANDIDATE_V1_DEFAULT_CONFIG,
            SwingTrendSimpleCandidateV1Strategy,
        ),
        "swing_trend_simple_candidate_v2": (
            SWING_SIMPLE_CANDIDATE_V2_DEFAULT_CONFIG,
            SwingTrendSimpleCandidateV2Strategy,
        ),
        "intraday_mtf_v1": (INTRADAY_DEFAULT_CONFIG, IntradayMTFV1Strategy),
        "intraday_mtf_v2": (INTRADAY_V2_DEFAULT_CONFIG, IntradayMTFV2Strategy),
        "intraday_mtf_v2_pullback_075_v1": (
            INTRADAY_V2_PULLBACK_075_DEFAULT_CONFIG,
            IntradayMTFV2Pullback075V1Strategy,
        ),
        "intraday_mtf_v2_trend70_v1": (
            INTRADAY_V2_TREND70_DEFAULT_CONFIG,
            IntradayMTFV2Trend70V1Strategy,
        ),
        "intraday_mtf_v2_cooldown10_v1": (
            INTRADAY_V2_COOLDOWN10_DEFAULT_CONFIG,
            IntradayMTFV2Cooldown10V1Strategy,
        ),
        "trend_following_v1": (TREND_FOLLOWING_V1_DEFAULT_CONFIG, TrendFollowingV1Strategy),
        "swing_improved_v1": (SWING_IMPROVED_V1_DEFAULT_CONFIG, SwingImprovedV1Strategy),
        "mean_reversion_v1": (MEAN_REVERSION_V1_DEFAULT_CONFIG, MeanReversionV1Strategy),
    }

    def __init__(self, config_loader: Optional[StrategyConfigLoader] = None) -> None:
        self.config_loader = config_loader or StrategyConfigLoader()

    def build_strategy(self, strategy_profile: str) -> Strategy:
        if strategy_profile not in self._strategy_registry:
            raise TradingAssistantError(f"Unsupported strategy profile: {strategy_profile}")

        defaults, strategy_cls = self._strategy_registry[strategy_profile]
        strategy_config = self.config_loader.load(strategy_profile, defaults)
        return strategy_cls(strategy_config)

    def run(self, request: AnalyzeRequest, ohlcv_by_timeframe: dict[str, pd.DataFrame]) -> AnalysisResult:
        return self.run_profile(request.strategy_profile, request, ohlcv_by_timeframe)

    def run_profile(
        self,
        strategy_profile: str,
        request: AnalyzeRequest,
        ohlcv_by_timeframe: dict[str, pd.DataFrame],
    ) -> AnalysisResult:
        strategy = self.build_strategy(strategy_profile)
        if request.strategy_profile != strategy_profile:
            request = request.model_copy(update={"strategy_profile": strategy_profile})
        return strategy.analyze(request, ohlcv_by_timeframe)
