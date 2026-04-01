from __future__ import annotations

from app.strategies.swing_trend_axis_band_state_note_v1 import (
    DEFAULT_CONFIG as SWING_AXIS_BAND_STATE_NOTE_DEFAULT_CONFIG,
    SwingTrendAxisBandStateNoteV1Strategy,
)


DEFAULT_CONFIG = SWING_AXIS_BAND_STATE_NOTE_DEFAULT_CONFIG


class SwingTrendAxisBandRiskOverlayV1Strategy(SwingTrendAxisBandStateNoteV1Strategy):
    name = "swing_trend_axis_band_risk_overlay_v1"
