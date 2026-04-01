from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.schemas.common import Action, Bias, RecommendedTiming
from app.schemas.request import AnalyzeRequest
from app.strategies.trend_pullback_v1 import DEFAULT_CONFIG, TrendPullbackV1Strategy


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "trend_pullback_v1_baselines.json"
BASELINE_FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
BASELINE_CASES = [case for case in BASELINE_FIXTURE["cases"] if "plan" in case.get("expected", {})]
REGRESSION_CASES = [case for case in BASELINE_FIXTURE["cases"] if "plan" not in case.get("expected", {})]
OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def make_request() -> AnalyzeRequest:
    return AnalyzeRequest(**BASELINE_FIXTURE["request"])


def _build_ohlcv(case: dict[str, Any]) -> dict[str, pd.DataFrame]:
    return {timeframe: _dump_to_frame(rows) for timeframe, rows in case["ohlcv_dump"].items()}


def _case_by_name(name: str) -> dict[str, Any]:
    return next(case for case in BASELINE_FIXTURE["cases"] if case["name"] == name)


def _dump_to_frame(rows: list[list[Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def test_baseline_fixture_contains_raw_ohlcv_dumps() -> None:
    assert BASELINE_FIXTURE["version"] == 2
    for case in BASELINE_FIXTURE["cases"]:
        assert set(case["ohlcv_dump"]) == {"1d", "4h", "1h", "15m"}
        for timeframe, rows in case["ohlcv_dump"].items():
            assert isinstance(rows, list)
            assert rows, timeframe
            assert len(rows[0]) == len(OHLCV_COLUMNS)


@pytest.mark.parametrize("case", BASELINE_CASES, ids=lambda item: item["name"])
def test_trend_pullback_baseline_cases(case: dict[str, Any]) -> None:
    strategy = TrendPullbackV1Strategy(DEFAULT_CONFIG)
    result = strategy.analyze(make_request(), _build_ohlcv(case))
    expected = case["expected"]

    assert result.decision.action == expected["decision"]["action"]
    assert result.decision.bias == expected["decision"]["bias"]
    assert result.decision.recommended_timing == expected["decision"]["recommended_timing"]
    assert 0 <= result.decision.confidence <= 100

    assert result.market_regime.higher_timeframe_bias == expected["market_regime"]["higher_timeframe_bias"]
    assert result.market_regime.trend_strength == expected["market_regime"]["trend_strength"]
    assert result.market_regime.volatility_state == expected["market_regime"]["volatility_state"]
    assert result.market_regime.is_trend_friendly == expected["market_regime"]["is_trend_friendly"]

    assert result.diagnostics.score_breakdown.total == result.decision.confidence
    assert result.diagnostics.trigger_maturity.state == expected["diagnostics"]["trigger_state"]
    assert isinstance(result.diagnostics.trigger_maturity.score, int)
    assert result.diagnostics.setup_quality.higher_timeframe_bias == expected["market_regime"]["higher_timeframe_bias"]
    assert result.diagnostics.setup_quality.trend_friendly == expected["market_regime"]["is_trend_friendly"]
    assert result.diagnostics.setup_quality.setup_timeframe_aligned == expected["diagnostics"]["mid_timeframe_aligned"]
    assert result.diagnostics.setup_quality.setup_timeframe_pullback_ready == expected["diagnostics"]["mid_timeframe_pullback_ready"]
    assert result.diagnostics.setup_quality.setup_timeframe_extended == expected["diagnostics"]["mid_timeframe_extended"]
    assert result.diagnostics.setup_quality.setup_distance_to_value_atr >= 0
    assert isinstance(result.diagnostics.vetoes, list)
    assert isinstance(result.diagnostics.conflict_signals, list)
    assert isinstance(result.diagnostics.uncertainty_notes, list)

    assert result.timeframes.day_1.trend_bias == expected["timeframes"]["1d"]["trend_bias"]
    assert result.timeframes.day_1.ema_alignment == expected["timeframes"]["1d"]["ema_alignment"]
    assert result.timeframes.day_1.structure_state == expected["timeframes"]["1d"]["structure_state"]
    assert result.timeframes.day_1.is_pullback_to_value_area == expected["timeframes"]["1d"]["is_pullback_to_value_area"]
    assert result.timeframes.day_1.is_extended == expected["timeframes"]["1d"]["is_extended"]
    assert result.timeframes.day_1.trigger_state == expected["timeframes"]["1d"]["trigger_state"]
    assert result.timeframes.day_1.trend_score == expected["timeframes"]["1d"]["trend_score"]

    assert result.timeframes.hour_4.trend_bias == expected["timeframes"]["4h"]["trend_bias"]
    assert result.timeframes.hour_4.ema_alignment == expected["timeframes"]["4h"]["ema_alignment"]
    assert result.timeframes.hour_4.structure_state == expected["timeframes"]["4h"]["structure_state"]
    assert result.timeframes.hour_4.is_pullback_to_value_area == expected["timeframes"]["4h"]["is_pullback_to_value_area"]
    assert result.timeframes.hour_4.is_extended == expected["timeframes"]["4h"]["is_extended"]
    assert result.timeframes.hour_4.trigger_state == expected["timeframes"]["4h"]["trigger_state"]
    assert result.timeframes.hour_4.trend_score == expected["timeframes"]["4h"]["trend_score"]

    assert result.timeframes.hour_1.trend_bias == expected["timeframes"]["1h"]["trend_bias"]
    assert result.timeframes.hour_1.ema_alignment == expected["timeframes"]["1h"]["ema_alignment"]
    assert result.timeframes.hour_1.structure_state == expected["timeframes"]["1h"]["structure_state"]
    assert result.timeframes.hour_1.is_pullback_to_value_area == expected["timeframes"]["1h"]["is_pullback_to_value_area"]
    assert result.timeframes.hour_1.is_extended == expected["timeframes"]["1h"]["is_extended"]
    assert result.timeframes.hour_1.trigger_state == expected["timeframes"]["1h"]["trigger_state"]
    assert result.timeframes.hour_1.trend_score == expected["timeframes"]["1h"]["trend_score"]

    assert result.timeframes.min_15.trend_bias == expected["timeframes"]["15m"]["trend_bias"]
    assert result.timeframes.min_15.ema_alignment == expected["timeframes"]["15m"]["ema_alignment"]
    assert result.timeframes.min_15.structure_state == expected["timeframes"]["15m"]["structure_state"]
    assert result.timeframes.min_15.is_pullback_to_value_area == expected["timeframes"]["15m"]["is_pullback_to_value_area"]
    assert result.timeframes.min_15.is_extended == expected["timeframes"]["15m"]["is_extended"]
    assert result.timeframes.min_15.trigger_state == expected["timeframes"]["15m"]["trigger_state"]
    assert result.timeframes.min_15.trend_score == expected["timeframes"]["15m"]["trend_score"]

    assert result.reasoning.summary
    assert isinstance(result.reasoning.reasons_for, list)
    assert isinstance(result.reasoning.reasons_against, list)
    assert isinstance(result.reasoning.risk_notes, list)

    plan = expected["plan"]
    if plan["entry_zone_present"]:
        assert result.decision.entry_zone is not None
        assert result.decision.stop_loss is not None
        assert result.decision.take_profit_hint is not None
        assert result.decision.invalidation_price is not None
        assert "EMA21/EMA55" in result.decision.entry_zone.basis
        assert "ATR" in result.decision.stop_loss.basis
        assert "1R/2R" in result.decision.take_profit_hint.basis
    else:
        assert result.decision.entry_zone is None
        assert result.decision.stop_loss is None
        assert result.decision.take_profit_hint is None
        assert result.decision.invalidation_price is None

    assert result.raw_metrics["scorecard"]["total"] == result.decision.confidence


@pytest.mark.parametrize(
    "case_name",
    [case["name"] for case in REGRESSION_CASES],
)
def test_trend_pullback_regression_cases(case_name: str) -> None:
    case = _case_by_name(case_name)
    strategy = TrendPullbackV1Strategy(DEFAULT_CONFIG)
    result = strategy.analyze(make_request(), _build_ohlcv(case))
    expected = case["expected"]["decision"]

    assert result.decision.action == expected["action"]
    assert result.decision.bias == expected["bias"]
    assert result.decision.recommended_timing == expected["recommended_timing"]
    if "confidence_lt" in expected:
        assert result.decision.confidence < expected["confidence_lt"]
    if "confidence" in expected:
        assert 0 <= result.decision.confidence <= 100
    assert result.decision.entry_zone is None
    assert result.decision.stop_loss is None
    assert result.decision.take_profit_hint is None
