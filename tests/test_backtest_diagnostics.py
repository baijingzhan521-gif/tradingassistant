from __future__ import annotations

import pandas as pd

from app.backtesting.diagnostics import (
    _classify_wait_reason,
    build_phase_funnel,
    bucket_confidence,
    bucket_distance,
    summarize_performance,
)
from app.schemas.common import Action, Bias, RecommendedTiming, TriggerState


def test_bucket_helpers_cover_expected_edges() -> None:
    assert bucket_confidence(64) == "<65"
    assert bucket_confidence(65) == "65-74"
    assert bucket_confidence(84) == "75-84"
    assert bucket_confidence(94) == "85-94"
    assert bucket_confidence(95) == "95+"

    assert bucket_distance(0.0) == "inside_execution_zone"
    assert bucket_distance(0.2) == "0-0.25ATR"
    assert bucket_distance(0.4) == "0.25-0.5ATR"
    assert bucket_distance(0.8) == "0.5-1.0ATR"
    assert bucket_distance(1.2) == ">1.0ATR"


def test_classify_wait_reason_prioritizes_primary_blocker() -> None:
    assert (
        _classify_wait_reason(
            higher_bias=Bias.NEUTRAL,
            trend_friendly=False,
            setup_assessment={"aligned": False, "is_extended": False, "pullback_ready": False, "structure_ready": False},
            trigger_assessment={"state": TriggerState.NONE},
            confidence=40,
            action=Action.WAIT,
            recommended_timing=RecommendedTiming.SKIP,
            action_threshold=65,
        )
        == "higher_timeframe_neutral"
    )

    assert (
        _classify_wait_reason(
            higher_bias=Bias.BULLISH,
            trend_friendly=True,
            setup_assessment={"aligned": True, "is_extended": False, "pullback_ready": False, "structure_ready": False},
            trigger_assessment={"state": TriggerState.MIXED},
            confidence=80,
            action=Action.WAIT,
            recommended_timing=RecommendedTiming.WAIT_CONFIRMATION,
            action_threshold=65,
        )
        == "setup_structure_not_ready"
    )

    assert (
        _classify_wait_reason(
            higher_bias=Bias.BEARISH,
            trend_friendly=True,
            setup_assessment={
                "aligned": True,
                "is_extended": False,
                "pullback_ready": True,
                "structure_ready": True,
                "require_reversal_candle": False,
                "reversal_ready": True,
            },
            trigger_assessment={"state": TriggerState.NONE},
            confidence=80,
            action=Action.WAIT,
            recommended_timing=RecommendedTiming.WAIT_CONFIRMATION,
            action_threshold=65,
        )
        == "trigger_none"
    )

    assert (
        _classify_wait_reason(
            higher_bias=Bias.BULLISH,
            trend_friendly=True,
            setup_assessment={
                "aligned": True,
                "is_extended": False,
                "pullback_ready": True,
                "structure_ready": True,
                "require_reversal_candle": True,
                "reversal_ready": False,
            },
            trigger_assessment={"state": TriggerState.BULLISH_CONFIRMED},
            confidence=80,
            action=Action.WAIT,
            recommended_timing=RecommendedTiming.WAIT_CONFIRMATION,
            action_threshold=65,
        )
        == "setup_reversal_not_ready"
    )

    assert (
        _classify_wait_reason(
            higher_bias=Bias.BULLISH,
            trend_friendly=True,
            setup_assessment={
                "aligned": True,
                "is_extended": False,
                "pullback_ready": True,
                "structure_ready": True,
                "require_reversal_candle": True,
                "reversal_ready": True,
                "require_divergence_gate": True,
                "divergence_ready": False,
            },
            trigger_assessment={"state": TriggerState.BULLISH_CONFIRMED},
            confidence=80,
            action=Action.WAIT,
            recommended_timing=RecommendedTiming.WAIT_CONFIRMATION,
            action_threshold=65,
        )
        == "setup_divergence_not_ready"
    )


def test_phase_funnel_and_group_summary() -> None:
    signals = pd.DataFrame(
        [
            {
                "higher_bias": "bullish",
                "trend_friendly": True,
                "setup_aligned": True,
                "setup_pullback_ready": True,
                "setup_reversal_ready": True,
                "setup_divergence_required": True,
                "setup_divergence_ready": True,
                "trigger_state": "bullish_confirmed",
                "confidence": 90,
                "signal_now": True,
            },
            {
                "higher_bias": "bullish",
                "trend_friendly": True,
                "setup_aligned": True,
                "setup_pullback_ready": False,
                "setup_reversal_ready": False,
                "setup_divergence_required": False,
                "setup_divergence_ready": False,
                "trigger_state": "mixed",
                "confidence": 70,
                "signal_now": False,
            },
            {
                "higher_bias": "neutral",
                "trend_friendly": False,
                "setup_aligned": False,
                "setup_pullback_ready": False,
                "setup_reversal_ready": False,
                "setup_divergence_required": False,
                "setup_divergence_ready": False,
                "trigger_state": "none",
                "confidence": 20,
                "signal_now": False,
            },
        ]
    )
    funnel = build_phase_funnel(signals, action_threshold=65)

    assert funnel[0]["stage"] == "evaluated"
    assert funnel[0]["count"] == 3
    assert any(item["stage"] == "setup_divergence_ready" for item in funnel)
    assert funnel[-1]["stage"] == "signal_now"
    assert funnel[-1]["count"] == 1

    trades = pd.DataFrame(
        [
            {"side": "LONG", "pnl_r": 2.0},
            {"side": "LONG", "pnl_r": -1.0},
            {"side": "SHORT", "pnl_r": -1.0},
        ]
    )
    grouped = summarize_performance(trades, group_by="side")
    by_side = {item["side"]: item for item in grouped}

    assert by_side["LONG"]["count"] == 2
    assert by_side["LONG"]["profit_factor"] == 2.0
    assert by_side["SHORT"]["win_rate"] == 0.0
