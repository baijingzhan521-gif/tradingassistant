from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

import pandas as pd

from app.indicators.atr import apply_atr_indicator
from app.indicators.candle_profile import summarize_candle_profile
from app.indicators.divergence import apply_divergence_indicator, divergence_profile_from_row, empty_divergence_profile
from app.indicators.ema import apply_ema_indicators
from app.indicators.market_structure import (
    classify_structure,
    compute_trend_strength,
    determine_ema_alignment,
    determine_trend_bias,
)
from app.indicators.swings import identify_swings, recent_swing_levels, recent_swing_points
from app.schemas.analysis import (
    AnalysisCharts,
    AnalysisDiagnostics,
    AnalysisResult,
    ChartCandle,
    ChartLinePoint,
    Decision,
    EntryZone,
    MarketRegime,
    Reasoning,
    ScoreBreakdown,
    ScoreContribution,
    SetupQuality,
    StopLoss,
    TakeProfitHint,
    TimeframeAnalysis,
    TimeframeChart,
    TimeframesAnalysis,
    TriggerMaturity,
)
from app.schemas.common import (
    Action,
    Bias,
    RecommendedTiming,
    StructureState,
    TriggerState,
    VolatilityState,
)
from app.schemas.request import AnalyzeRequest
from app.strategies.base import Strategy
from app.strategies.scoring import ScoreCard
from app.utils.math_utils import midpoint, pct_distance
from app.utils.timeframes import validate_required_timeframes


COMMON_DEFAULT_CONFIG: dict[str, Any] = {
    "atr_period": 14,
    "swing_window": 3,
    "ema": {
        "periods": [21, 55, 100, 200],
        "execution_fast": 21,
        "execution_slow": 55,
    },
    "execution": {
        "pullback_distance_atr": 1.0,
        "extension_distance_atr": 1.55,
        "support_proximity_atr": 0.75,
        "resistance_proximity_atr": 0.75,
    },
    "position_map": {
        "band_volatility_mode": "atr",
        "band_atr_mult": 1.5,
        "band_std_period": 20,
        "band_std_mult": 2.0,
    },
    "micro": {
        "confirmation_lookback": 20,
        "doji_body_ratio_max": 0.12,
        "reversal_wick_ratio_min": 0.2,
    },
    "setup": {
        "require_structure_ready": False,
        "require_reversal_candle": False,
    },
    "setup_confluence": {
        "enabled": False,
        "min_hits": 2,
        "ema55_proximity_atr": 0.6,
        "pivot_proximity_atr": 0.9,
        "band_proximity_atr": 0.9,
        "max_spread_atr": 1.2,
    },
    "level_confirmation": {
        "enabled": False,
        "min_hits": 1,
        "ema55_touch_proximity_atr": 0.35,
        "pivot_touch_proximity_atr": 0.5,
        "band_touch_proximity_atr": 0.5,
    },
    "state_note": {
        "enabled": False,
        "bullish_axis_threshold": 2.0,
        "bullish_band_threshold": 0.9,
        "bullish_extreme_axis_threshold": 3.0,
        "bullish_extreme_band_threshold": 1.0,
        "bearish_axis_threshold": 2.0,
        "bearish_band_threshold": 0.1,
        "bearish_extreme_axis_threshold": 3.0,
        "bearish_extreme_band_threshold": 0.0,
    },
    "divergence": {
        "enabled": False,
        "rsi_period": 10,
        "swing_window": 14,
        "ema_period": 34,
        "min_rsi_diff": 1.6,
        "min_move_atr_mult": 0.3,
        "stretch_atr_mult": 0.8,
        "wick_ratio_min": 0.4,
        "min_reversal_score": 2,
        "cooldown_bars": 20,
        "min_level": 2,
        "level_2_bonus": 4,
        "level_3_bonus": 8,
        "opposing_level_penalty": 4,
        "gate_mode": "score",
        "gate_biases": [],
    },
    "trigger": {
        "min_auxiliary_confirmations": 1,
        "mixed_score": -2,
        "none_score": -12,
        "bullish_require_regained_fast": True,
        "bearish_require_regained_fast": True,
        "bullish_relax_regained_fast_at_trend_strength": None,
        "bearish_relax_regained_fast_at_trend_strength": None,
        "bullish_require_held_slow": True,
        "bearish_require_held_slow": True,
        "bullish_require_auxiliary": True,
        "bearish_require_auxiliary": True,
    },
    "free_space": {
        "enabled": False,
        "long_min_r": 0.0,
        "short_min_r": 0.0,
    },
    "volatility": {"low_atr_pct": 0.6, "high_atr_pct": 2.0},
    "risk": {"atr_buffer": 0.25, "minimum_r_multiple": 1.0},
    "confidence": {"action_threshold": 65},
    "chart": {"candles": 80},
    "backtest": {"cooldown_bars_after_exit": 0},
    "window": {},
}

TIMEFRAME_TO_FIELD = {
    "1d": "day_1",
    "4h": "hour_4",
    "1h": "hour_1",
    "15m": "min_15",
    "3m": "min_3",
}


@dataclass
class PreparedTimeframe:
    model: TimeframeAnalysis
    debug: dict[str, Any]
    df: pd.DataFrame
    value_zone_low: float
    value_zone_high: float
    execution_zone_low: float
    execution_zone_high: float
    distance_to_value_atr: float
    distance_to_execution_atr: float
    band_upper: float
    band_lower: float
    band_volatility_unit: float
    axis_distance_vol: float
    ema55_distance_vol: float
    band_position: float
    swing_support_distance_atr: Optional[float]
    swing_resistance_distance_atr: Optional[float]
    candle_profile: dict[str, Any]
    divergence_profile: dict[str, Any]


class WindowedMTFStrategy(Strategy):
    required_timeframes: tuple[str, ...] = ()

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.window_config = dict(config.get("window", {}))

    def analyze(self, request: AnalyzeRequest, ohlcv_by_timeframe: dict[str, pd.DataFrame]) -> AnalysisResult:
        validate_required_timeframes(
            self.name,
            [item.value if hasattr(item, "value") else str(item) for item in request.timeframes],
        )

        required = set(self.required_timeframes)
        prepared = {
            timeframe: self._prepare_timeframe(timeframe, frame)
            for timeframe, frame in ohlcv_by_timeframe.items()
            if timeframe in required or timeframe in set(self.window_config.get("chart_timeframes", ()))
        }

        higher_bias, trend_strength = self._derive_higher_timeframe_bias(prepared)
        setup_key = str(self.window_config["setup_timeframe"])
        trigger_key = str(self.window_config["trigger_timeframe"])
        reference_key = str(self.window_config.get("reference_timeframe", setup_key))

        volatility_state = self._derive_volatility_state(prepared[setup_key])
        is_trend_friendly = self._is_trend_friendly(
            higher_bias=higher_bias,
            trend_strength=trend_strength,
            volatility_state=volatility_state,
        )

        scorecard = ScoreCard(base=50)
        reasons_for: list[str] = []
        reasons_against: list[str] = []
        risk_notes: list[str] = []

        higher_label = self._format_timeframe_group(tuple(self.window_config["higher_timeframes"]))
        if higher_bias == Bias.BULLISH:
            scorecard.add(15, "higher_bias", f"{higher_label} 同步偏多")
            reasons_for.append(f"{higher_label} 保持在 EMA200 上方，EMA21/55/100/200 排列偏多")
        elif higher_bias == Bias.BEARISH:
            scorecard.add(15, "higher_bias", f"{higher_label} 同步偏空")
            reasons_for.append(f"{higher_label} 保持在 EMA200 下方，EMA21/55/100/200 排列偏空")
        else:
            scorecard.add(-20, "higher_conflict", f"{higher_label} 没有形成同向共振")
            reasons_against.append(f"{higher_label} 没有给出一致方向，环境不够清晰")

        setup_assessment = self._assess_setup(
            higher_bias,
            prepared[setup_key],
            setup_key,
            reference_ctx=prepared[reference_key],
            current_price=prepared[trigger_key].model.close,
        )
        scorecard.add(setup_assessment["score"], f"{setup_key}_setup", setup_assessment["score_note"])
        reasons_for.extend(setup_assessment["reasons_for"])
        reasons_against.extend(setup_assessment["reasons_against"])
        risk_notes.extend(setup_assessment["risk_notes"])
        state_notes = list(setup_assessment.get("state_notes", []))

        trigger_assessment = self._assess_trigger(
            higher_bias,
            prepared[trigger_key],
            trigger_key,
            trend_strength=trend_strength,
        )
        prepared[trigger_key].model.trigger_state = trigger_assessment["state"]
        scorecard.add(trigger_assessment["score"], f"{trigger_key}_trigger", trigger_assessment["score_note"])
        reasons_for.extend(trigger_assessment["reasons_for"])
        reasons_against.extend(trigger_assessment["reasons_against"])
        risk_notes.extend(trigger_assessment["risk_notes"])

        if volatility_state == VolatilityState.HIGH:
            scorecard.add(-15, "volatility", f"{setup_key} ATR 百分比偏高")
            risk_notes.append(f"{setup_key} 波动偏大，止损空间会被迫拉宽")
        elif volatility_state == VolatilityState.LOW:
            scorecard.add(3, "volatility", f"{setup_key} 波动可控")

        confidence = scorecard.total
        action, bias, recommended_timing = self._decide(
            higher_bias=higher_bias,
            trend_friendly=is_trend_friendly,
            setup_assessment=setup_assessment,
            trigger_assessment=trigger_assessment,
            confidence=confidence,
        )
        conflict_signals, uncertainty_notes, vetoes = self._build_diagnostics_notes(
            higher_bias=higher_bias,
            trend_friendly=is_trend_friendly,
            setup_assessment=setup_assessment,
            trigger_assessment=trigger_assessment,
            volatility_state=volatility_state,
            confidence=confidence,
            action=action,
            setup_key=setup_key,
            trigger_key=trigger_key,
        )

        trade_plan = self._build_trade_plan(
            action=action,
            bias=bias,
            setup_ctx=prepared[setup_key],
            reference_ctx=prepared[reference_key],
            current_price=prepared[trigger_key].model.close,
            setup_key=setup_key,
            reference_key=reference_key,
        )

        result = AnalysisResult(
            analysis_id=uuid4().hex,
            timestamp=datetime.now(timezone.utc),
            symbol=request.symbol,
            exchange=request.exchange,
            market_type=request.market_type,
            strategy_profile=request.strategy_profile,
            timeframes=self._build_timeframes_payload(prepared),
            charts=self._build_chart_snapshots(prepared),
            market_regime=MarketRegime(
                higher_timeframe_bias=higher_bias,
                trend_strength=trend_strength,
                volatility_state=volatility_state,
                is_trend_friendly=is_trend_friendly,
            ),
            decision=Decision(
                action=action,
                bias=bias,
                confidence=confidence,
                recommended_timing=recommended_timing,
                entry_zone=trade_plan["entry_zone"],
                stop_loss=trade_plan["stop_loss"],
                invalidation=trade_plan["invalidation"],
                invalidation_price=trade_plan["invalidation_price"],
                take_profit_hint=trade_plan["take_profit_hint"],
            ),
            reasoning=Reasoning(
                reasons_for=list(dict.fromkeys(reasons_for)),
                reasons_against=list(dict.fromkeys(reasons_against)),
                risk_notes=list(dict.fromkeys(risk_notes)),
                state_notes=list(dict.fromkeys(state_notes)),
                summary=self._build_summary(
                    action=action,
                    recommended_timing=recommended_timing,
                    higher_bias=higher_bias,
                    setup_assessment=setup_assessment,
                    trigger_assessment=trigger_assessment,
                    setup_key=setup_key,
                    trigger_key=trigger_key,
                ),
            ),
            diagnostics=AnalysisDiagnostics(
                strategy_config_snapshot=self.config,
                score_breakdown=ScoreBreakdown(
                    base=scorecard.base,
                    total=scorecard.total,
                    contributions=[
                        ScoreContribution(label=item.label, points=item.points, note=item.note)
                        for item in scorecard.contributions
                    ],
                ),
                vetoes=list(dict.fromkeys(vetoes)),
                conflict_signals=list(dict.fromkeys(conflict_signals)),
                uncertainty_notes=list(dict.fromkeys(uncertainty_notes)),
                state_notes=list(dict.fromkeys(state_notes)),
                setup_quality=SetupQuality(
                    setup_timeframe=setup_key,
                    higher_timeframe_bias=higher_bias,
                    trend_friendly=is_trend_friendly,
                    setup_timeframe_aligned=setup_assessment["aligned"],
                    setup_timeframe_pullback_ready=setup_assessment["pullback_ready"],
                    setup_timeframe_extended=setup_assessment["is_extended"],
                    setup_distance_to_value_atr=setup_assessment["distance_to_value_atr"],
                ),
                trigger_maturity=TriggerMaturity(
                    timeframe=trigger_key,
                    state=trigger_assessment["state"],
                    score=trigger_assessment["score"],
                    supporting_signals=trigger_assessment["reasons_for"],
                    blocking_signals=trigger_assessment["reasons_against"],
                ),
            ),
            raw_metrics={
                "scorecard": scorecard.as_dict(),
                "strategy_config_snapshot": self.config,
                "setup_assessment": {
                    key: value
                    for key, value in setup_assessment.items()
                    if key not in {"reasons_for", "reasons_against", "risk_notes"}
                },
                "trigger_assessment": {
                    key: value
                    for key, value in trigger_assessment.items()
                    if key not in {"reasons_for", "reasons_against", "risk_notes"}
                },
                "timeframe_debug": {key: value.debug for key, value in prepared.items()},
            },
        )
        return result

    def _prepare_timeframe(self, timeframe: str, df: pd.DataFrame) -> PreparedTimeframe:
        ema_periods = tuple(sorted(set(int(period) for period in self.config["ema"]["periods"])))
        enriched = apply_ema_indicators(df, periods=ema_periods)
        enriched = apply_atr_indicator(enriched, period=int(self.config["atr_period"]))
        enriched = identify_swings(enriched, window=int(self.config["swing_window"]))
        candle_profile = summarize_candle_profile(
            enriched,
            lookback=int(self.config["micro"]["confirmation_lookback"]),
            doji_body_ratio_max=float(self.config["micro"].get("doji_body_ratio_max", 0.12)),
            reversal_wick_ratio_min=float(self.config["micro"].get("reversal_wick_ratio_min", 0.2)),
        )
        divergence_config = dict(self.config.get("divergence", {}))
        divergence_enabled = bool(divergence_config.get("enabled", False))
        divergence_profile = empty_divergence_profile(enabled=divergence_enabled)
        if divergence_enabled:
            enriched = apply_divergence_indicator(
                enriched,
                rsi_period=int(divergence_config.get("rsi_period", 10)),
                swing_window=int(divergence_config.get("swing_window", 14)),
                ema_period=int(divergence_config.get("ema_period", 34)),
                atr_period=int(self.config["atr_period"]),
                min_rsi_diff=float(divergence_config.get("min_rsi_diff", 1.6)),
                min_move_atr_mult=float(divergence_config.get("min_move_atr_mult", 0.3)),
                stretch_atr_mult=float(divergence_config.get("stretch_atr_mult", 0.8)),
                wick_ratio_min=float(divergence_config.get("wick_ratio_min", 0.4)),
                min_reversal_score=int(divergence_config.get("min_reversal_score", 2)),
                cooldown_bars=int(divergence_config.get("cooldown_bars", 20)),
            )
            divergence_profile = divergence_profile_from_row(enriched.iloc[-1], enabled=True)

        latest = enriched.iloc[-1]
        swing_high, swing_low = recent_swing_levels(enriched)
        swing_points = recent_swing_points(enriched, count=3)
        structure_state = classify_structure(
            [point["price"] for point in swing_points["highs"]],
            [point["price"] for point in swing_points["lows"]],
        )

        ema21 = float(latest["ema_21"])
        ema55 = float(latest["ema_55"])
        ema100 = float(latest["ema_100"])
        ema200 = float(latest["ema_200"])
        close = float(latest["close"])
        atr14 = float(latest[f"atr_{self.config['atr_period']}"])
        atr_pct = (atr14 / close) * 100 if close else 0.0

        alignment = determine_ema_alignment(ema21, ema55, ema100, ema200)
        bias = determine_trend_bias(close, ema200, alignment)
        trend_score = compute_trend_strength(
            enriched,
            close=close,
            ema21=ema21,
            ema55=ema55,
            ema100=ema100,
            ema200=ema200,
            alignment=alignment,
            bias=bias,
            structure_state=structure_state,
        )

        value_zone_low = min(ema21, ema55, ema100)
        value_zone_high = max(ema21, ema55, ema100)
        execution_zone_low = min(ema21, ema55)
        execution_zone_high = max(ema21, ema55)
        band_upper, band_lower, band_volatility_unit = self._derive_position_bands(
            enriched,
            ema100=ema100,
            atr14=atr14,
        )
        axis_distance_vol = self._normalized_distance(close, ema100, band_volatility_unit)
        ema55_distance_vol = self._normalized_distance(close, ema55, band_volatility_unit)
        band_position = self._band_position(close, band_lower, band_upper)

        distance_to_value_atr = self._distance_to_zone_atr(close, value_zone_low, value_zone_high, atr14)
        distance_to_execution_atr = self._distance_to_zone_atr(close, execution_zone_low, execution_zone_high, atr14)
        swing_support_distance_atr = None if swing_low is None or not atr14 else abs(close - swing_low) / atr14
        swing_resistance_distance_atr = None if swing_high is None or not atr14 else abs(swing_high - close) / atr14

        is_pullback = distance_to_execution_atr <= float(self.config["execution"]["pullback_distance_atr"])
        is_extended = distance_to_execution_atr >= float(self.config["execution"]["extension_distance_atr"])

        notes: list[str] = []
        if bias == Bias.BULLISH:
            notes.append("价格站在 EMA200 上方，EMA21/55/100/200 结构仍偏多")
        elif bias == Bias.BEARISH:
            notes.append("价格压在 EMA200 下方，EMA21/55/100/200 结构仍偏空")
        else:
            notes.append("价格与 EMA21/55/100/200 没有形成清晰共振")
        notes.append(f"执行区使用 EMA21/EMA55：{round(execution_zone_low, 4)} - {round(execution_zone_high, 4)}")
        notes.append(
            f"EMA100 中轴 band 为 {round(band_lower, 4)} - {round(band_upper, 4)}，"
            f"当前位置 band_position={round(band_position, 3)}"
        )

        model = TimeframeAnalysis(
            timeframe=timeframe,
            latest_timestamp=latest["timestamp"].to_pydatetime(),
            close=close,
            ema21=ema21,
            ema55=ema55,
            ema100=ema100,
            ema200=ema200,
            atr14=atr14,
            atr_pct=round(atr_pct, 4),
            price_vs_ema21_pct=round(pct_distance(close, ema21), 4),
            price_vs_ema55_pct=round(pct_distance(close, ema55), 4),
            price_vs_ema100_pct=round(pct_distance(close, ema100), 4),
            price_vs_ema200_pct=round(pct_distance(close, ema200), 4),
            ema_alignment=alignment,
            trend_bias=bias,
            trend_score=trend_score,
            structure_state=structure_state,
            swing_high=swing_high,
            swing_low=swing_low,
            is_pullback_to_value_area=is_pullback,
            is_extended=is_extended,
            notes=notes,
        )
        debug = {
            "latest_close": close,
            "value_zone": {"low": round(value_zone_low, 4), "high": round(value_zone_high, 4)},
            "execution_zone": {"low": round(execution_zone_low, 4), "high": round(execution_zone_high, 4)},
            "distance_to_value_atr": round(distance_to_value_atr, 4),
            "distance_to_execution_atr": round(distance_to_execution_atr, 4),
            "position_map": {
                "volatility_mode": str(self.config["position_map"]["band_volatility_mode"]),
                "band_upper": round(band_upper, 4),
                "band_lower": round(band_lower, 4),
                "band_width": round(band_upper - band_lower, 4),
                "band_volatility_unit": round(band_volatility_unit, 4),
                "axis_distance_vol": round(axis_distance_vol, 4),
                "ema55_distance_vol": round(ema55_distance_vol, 4),
                "band_position": round(band_position, 4),
            },
            "swing_support_distance_atr": round(swing_support_distance_atr, 4) if swing_support_distance_atr is not None else None,
            "swing_resistance_distance_atr": round(swing_resistance_distance_atr, 4) if swing_resistance_distance_atr is not None else None,
            "swing_points": swing_points,
            "trend_score": trend_score,
            "candle_profile": candle_profile,
            "divergence_profile": divergence_profile,
            "ema_periods": list(ema_periods),
        }
        return PreparedTimeframe(
            model=model,
            debug=debug,
            df=enriched,
            value_zone_low=value_zone_low,
            value_zone_high=value_zone_high,
            execution_zone_low=execution_zone_low,
            execution_zone_high=execution_zone_high,
            distance_to_value_atr=distance_to_value_atr,
            distance_to_execution_atr=distance_to_execution_atr,
            band_upper=band_upper,
            band_lower=band_lower,
            band_volatility_unit=band_volatility_unit,
            axis_distance_vol=axis_distance_vol,
            ema55_distance_vol=ema55_distance_vol,
            band_position=band_position,
            swing_support_distance_atr=swing_support_distance_atr,
            swing_resistance_distance_atr=swing_resistance_distance_atr,
            candle_profile=candle_profile,
            divergence_profile=divergence_profile,
        )

    def _compute_setup_confluence(self, *, higher_bias: Bias, setup_ctx: PreparedTimeframe) -> dict[str, Any]:
        atr14 = float(setup_ctx.model.atr14)
        if not atr14:
            return {
                "enabled": bool(self.config.get("setup_confluence", {}).get("enabled", False)),
                "ready": False,
                "hits": 0,
                "min_hits": int(self.config.get("setup_confluence", {}).get("min_hits", 2)),
                "max_spread_atr": round(float(self.config.get("setup_confluence", {}).get("max_spread_atr", 1.2)), 4),
                "spread_atr": None,
                "anchor_price": round(midpoint(setup_ctx.execution_zone_low, setup_ctx.execution_zone_high), 4),
                "components": {},
                "source": "ema55_pivot_band_confluence",
            }

        config = dict(self.config.get("setup_confluence", {}))
        anchor_price = midpoint(setup_ctx.execution_zone_low, setup_ctx.execution_zone_high)
        if higher_bias == Bias.BULLISH:
            pivot_price = setup_ctx.model.swing_low
            band_price = setup_ctx.band_lower
        else:
            pivot_price = setup_ctx.model.swing_high
            band_price = setup_ctx.band_upper

        components: dict[str, dict[str, Any]] = {}
        components["ema55"] = {
            "price": round(float(setup_ctx.model.ema55), 4),
            "distance_atr": round(abs(float(setup_ctx.model.ema55) - anchor_price) / atr14, 4),
            "ready": abs(float(setup_ctx.model.ema55) - anchor_price) / atr14 <= float(config.get("ema55_proximity_atr", 0.6)),
        }
        if pivot_price is not None:
            components["pivot_anchor"] = {
                "price": round(float(pivot_price), 4),
                "distance_atr": round(abs(float(pivot_price) - anchor_price) / atr14, 4),
                "ready": abs(float(pivot_price) - anchor_price) / atr14 <= float(config.get("pivot_proximity_atr", 0.9)),
            }
        else:
            components["pivot_anchor"] = {
                "price": None,
                "distance_atr": None,
                "ready": False,
            }
        components["band_edge"] = {
            "price": round(float(band_price), 4),
            "distance_atr": round(abs(float(band_price) - anchor_price) / atr14, 4),
            "ready": abs(float(band_price) - anchor_price) / atr14 <= float(config.get("band_proximity_atr", 0.9)),
        }

        available_prices = [item["price"] for item in components.values() if item["price"] is not None]
        spread_atr = ((max(available_prices) - min(available_prices)) / atr14) if len(available_prices) >= 2 else None
        hits = sum(int(bool(item["ready"])) for item in components.values())
        min_hits = int(config.get("min_hits", 2))
        max_spread_atr = float(config.get("max_spread_atr", 1.2))
        ready = hits >= min_hits and (spread_atr is None or spread_atr <= max_spread_atr)

        return {
            "enabled": bool(config.get("enabled", False)),
            "ready": ready,
            "hits": hits,
            "min_hits": min_hits,
            "max_spread_atr": round(max_spread_atr, 4),
            "spread_atr": round(spread_atr, 4) if spread_atr is not None else None,
            "anchor_price": round(anchor_price, 4),
            "components": components,
            "source": "ema55_pivot_band_confluence",
        }

    def _compute_level_aware_confirmation(self, *, higher_bias: Bias, setup_ctx: PreparedTimeframe) -> dict[str, Any]:
        config = dict(self.config.get("level_confirmation", {}))
        enabled = bool(config.get("enabled", False))
        base_reversal_key = (
            "has_bullish_reversal_candle"
            if higher_bias == Bias.BULLISH
            else "has_bearish_reversal_candle"
        )
        base_reversal_ready = bool(setup_ctx.candle_profile.get(base_reversal_key, False))
        atr14 = float(setup_ctx.model.atr14)
        anchor_price = midpoint(setup_ctx.execution_zone_low, setup_ctx.execution_zone_high)
        latest = setup_ctx.df.iloc[-1]

        def build_component(
            *,
            label: str,
            level_price: Optional[float],
            proximity_atr: float,
        ) -> dict[str, Any]:
            if level_price is None or not atr14:
                return {
                    "price": None,
                    "touch_gap_atr": None,
                    "close_reclaimed": False,
                    "ready": False,
                }

            if higher_bias == Bias.BULLISH:
                touch_gap_atr = max(float(latest["low"]) - float(level_price), 0.0) / atr14
                close_reclaimed = float(latest["close"]) >= float(level_price)
            else:
                touch_gap_atr = max(float(level_price) - float(latest["high"]), 0.0) / atr14
                close_reclaimed = float(latest["close"]) <= float(level_price)

            return {
                "price": round(float(level_price), 4),
                "touch_gap_atr": round(float(touch_gap_atr), 4),
                "close_reclaimed": bool(close_reclaimed),
                "ready": bool(touch_gap_atr <= proximity_atr and close_reclaimed),
            }

        pivot_price = setup_ctx.model.swing_low if higher_bias == Bias.BULLISH else setup_ctx.model.swing_high
        band_price = setup_ctx.band_lower if higher_bias == Bias.BULLISH else setup_ctx.band_upper

        components = {
            "ema55": build_component(
                label="ema55",
                level_price=float(setup_ctx.model.ema55),
                proximity_atr=float(config.get("ema55_touch_proximity_atr", 0.35)),
            ),
            "pivot_anchor": build_component(
                label="pivot_anchor",
                level_price=float(pivot_price) if pivot_price is not None else None,
                proximity_atr=float(config.get("pivot_touch_proximity_atr", 0.5)),
            ),
            "band_edge": build_component(
                label="band_edge",
                level_price=float(band_price),
                proximity_atr=float(config.get("band_touch_proximity_atr", 0.5)),
            ),
        }
        hits = sum(int(bool(item["ready"])) for item in components.values())
        min_hits = int(config.get("min_hits", 1))

        return {
            "enabled": enabled,
            "base_reversal_ready": base_reversal_ready,
            "ready": base_reversal_ready and hits >= min_hits,
            "hits": hits,
            "min_hits": min_hits,
            "anchor_price": round(anchor_price, 4),
            "components": components,
            "source": "level_aware_reclaim_rejection",
        }

    def _compute_axis_band_state_note(self, *, higher_bias: Bias, setup_ctx: PreparedTimeframe) -> dict[str, Any]:
        config = dict(self.config.get("state_note", self.config.get("risk_overlay", {})))
        enabled = bool(config.get("enabled", False))
        axis_distance = float(setup_ctx.axis_distance_vol)
        band_position = float(setup_ctx.band_position)

        if higher_bias == Bias.BULLISH:
            active = (
                axis_distance >= float(config.get("bullish_axis_threshold", 2.0))
                and band_position >= float(config.get("bullish_band_threshold", 0.9))
            )
            extreme = (
                axis_distance >= float(config.get("bullish_extreme_axis_threshold", 3.0))
                and band_position >= float(config.get("bullish_extreme_band_threshold", 1.0))
            )
            label = "pullback_risk" if active else None
            severity = "high" if extreme else "elevated" if active else "none"
            message = (
                f"状态提示：当前相对 EMA100 偏离 {round(axis_distance, 2)} 倍波动单位，且已{'突破' if band_position >= 1.0 else '接近'}上沿 band，短线回撤风险{'偏高' if extreme else '上升'}。"
                if active
                else ""
            )
        elif higher_bias == Bias.BEARISH:
            active = (
                axis_distance <= -float(config.get("bearish_axis_threshold", 2.0))
                and band_position <= float(config.get("bearish_band_threshold", 0.1))
            )
            extreme = (
                axis_distance <= -float(config.get("bearish_extreme_axis_threshold", 3.0))
                and band_position <= float(config.get("bearish_extreme_band_threshold", 0.0))
            )
            label = "rebound_risk" if active else None
            severity = "high" if extreme else "elevated" if active else "none"
            message = (
                f"状态提示：当前相对 EMA100 偏离 {round(abs(axis_distance), 2)} 倍波动单位，且已{'跌破' if band_position <= 0.0 else '接近'}下沿 band，短线反弹风险{'偏高' if extreme else '上升'}。"
                if active
                else ""
            )
        else:
            active = False
            extreme = False
            label = None
            severity = "none"
            message = ""

        return {
            "enabled": enabled,
            "active": bool(active) if enabled else False,
            "label": label if enabled else None,
            "severity": severity if enabled else "none",
            "axis_distance_vol": round(axis_distance, 4),
            "band_position": round(band_position, 4),
            "extreme": bool(extreme) if enabled else False,
            "message": message if enabled else "",
            "thresholds": {
                "bullish_axis_threshold": round(float(config.get("bullish_axis_threshold", 2.0)), 4),
                "bullish_band_threshold": round(float(config.get("bullish_band_threshold", 0.9)), 4),
                "bullish_extreme_axis_threshold": round(float(config.get("bullish_extreme_axis_threshold", 3.0)), 4),
                "bullish_extreme_band_threshold": round(float(config.get("bullish_extreme_band_threshold", 1.0)), 4),
                "bearish_axis_threshold": round(float(config.get("bearish_axis_threshold", 2.0)), 4),
                "bearish_band_threshold": round(float(config.get("bearish_band_threshold", 0.1)), 4),
                "bearish_extreme_axis_threshold": round(float(config.get("bearish_extreme_axis_threshold", 3.0)), 4),
                "bearish_extreme_band_threshold": round(float(config.get("bearish_extreme_band_threshold", 0.0)), 4),
            },
            "source": "axis_distance_band_position_state_note",
        }

    def _compute_axis_band_risk_overlay(self, *, higher_bias: Bias, setup_ctx: PreparedTimeframe) -> dict[str, Any]:
        return self._compute_axis_band_state_note(higher_bias=higher_bias, setup_ctx=setup_ctx)

    def _derive_position_bands(
        self,
        frame: pd.DataFrame,
        *,
        ema100: float,
        atr14: float,
    ) -> tuple[float, float, float]:
        position_config = dict(self.config.get("position_map", {}))
        volatility_mode = str(position_config.get("band_volatility_mode", "atr")).lower()
        if volatility_mode == "std":
            std_period = int(position_config.get("band_std_period", 20))
            std_mult = float(position_config.get("band_std_mult", 2.0))
            rolling_std = frame["close"].rolling(std_period).std(ddof=0)
            std_value = float(rolling_std.iloc[-1]) if not rolling_std.empty and pd.notna(rolling_std.iloc[-1]) else 0.0
            band_volatility_unit = std_value if std_value > 0 else atr14
            band_half_width = band_volatility_unit * std_mult
        else:
            band_volatility_unit = atr14
            band_half_width = atr14 * float(position_config.get("band_atr_mult", 1.5))
        return ema100 + band_half_width, ema100 - band_half_width, band_volatility_unit

    @staticmethod
    def _normalized_distance(price: float, level: float, unit: float) -> float:
        if not unit:
            return 0.0
        return (price - level) / unit

    @staticmethod
    def _band_position(price: float, lower_band: float, upper_band: float) -> float:
        width = upper_band - lower_band
        if width <= 0:
            return 0.5
        return (price - lower_band) / width

    def _derive_higher_timeframe_bias(self, prepared: dict[str, PreparedTimeframe]) -> tuple[Bias, int]:
        higher_timeframes = [prepared[item] for item in self.window_config["higher_timeframes"]]
        trend_strength = int(round(sum(item.model.trend_score for item in higher_timeframes) / len(higher_timeframes)))
        biases = {item.model.trend_bias for item in higher_timeframes}
        if biases == {Bias.BULLISH}:
            return Bias.BULLISH, trend_strength
        if biases == {Bias.BEARISH}:
            return Bias.BEARISH, trend_strength
        return Bias.NEUTRAL, trend_strength

    def _derive_volatility_state(self, setup_ctx: PreparedTimeframe) -> VolatilityState:
        atr_pct = setup_ctx.model.atr_pct
        if atr_pct <= float(self.config["volatility"]["low_atr_pct"]):
            return VolatilityState.LOW
        if atr_pct >= float(self.config["volatility"]["high_atr_pct"]):
            return VolatilityState.HIGH
        return VolatilityState.NORMAL

    def _resolve_trigger_requirements(self, higher_bias: Bias, trend_strength: Optional[int] = None) -> dict[str, bool]:
        trigger_config = dict(self.config.get("trigger", {}))
        prefix = "bullish" if higher_bias == Bias.BULLISH else "bearish"
        requirements = {
            "require_regained_fast": bool(trigger_config.get(f"{prefix}_require_regained_fast", True)),
            "require_held_slow": bool(trigger_config.get(f"{prefix}_require_held_slow", True)),
            "require_auxiliary": bool(trigger_config.get(f"{prefix}_require_auxiliary", True)),
            "regained_fast_relaxed_by_regime": False,
        }
        relax_threshold = trigger_config.get(f"{prefix}_relax_regained_fast_at_trend_strength")
        if relax_threshold is not None and trend_strength is not None and trend_strength >= int(relax_threshold):
            requirements["require_regained_fast"] = False
            requirements["regained_fast_relaxed_by_regime"] = True
        return requirements

    def _resolve_setup_reversal_requirement(self, higher_bias: Bias) -> bool:
        setup_config = dict(self.config.get("setup", {}))
        if higher_bias in {Bias.BULLISH, Bias.BEARISH}:
            prefix = "bullish" if higher_bias == Bias.BULLISH else "bearish"
            key = f"{prefix}_require_reversal_candle"
            if key in setup_config:
                return bool(setup_config[key])
        return bool(setup_config.get("require_reversal_candle", False))

    def _derive_trade_levels(
        self,
        *,
        bias: Bias,
        setup_ctx: PreparedTimeframe,
        reference_ctx: PreparedTimeframe,
        current_price: float,
    ) -> dict[str, Optional[float]]:
        entry_low = float(setup_ctx.execution_zone_low)
        entry_high = float(setup_ctx.execution_zone_high)
        atr_buffer = setup_ctx.model.atr14 * float(self.config["risk"]["atr_buffer"])
        minimum_risk = setup_ctx.model.atr14 * float(self.config["risk"]["minimum_r_multiple"])

        if bias == Bias.BULLISH:
            anchor = float(setup_ctx.model.swing_low or entry_low)
            stop_price = min(anchor, entry_low) - atr_buffer
            risk = max(current_price - stop_price, minimum_risk)
            tp1 = current_price + risk
            tp2 = current_price + (2 * risk)
            reference_target = float(reference_ctx.model.swing_high) if reference_ctx.model.swing_high is not None else None
            if reference_target is not None:
                tp2 = max(tp2, reference_target)
            free_space_r = None if reference_target is None or risk <= 0 else (reference_target - current_price) / risk
            invalidation_price = float(setup_ctx.model.swing_low or entry_low)
            return {
                "entry_low": entry_low,
                "entry_high": entry_high,
                "stop_price": stop_price,
                "risk": risk,
                "tp1": tp1,
                "tp2": tp2,
                "reference_target": reference_target,
                "free_space_r": free_space_r,
                "invalidation_price": invalidation_price,
            }

        anchor = float(setup_ctx.model.swing_high or entry_high)
        stop_price = max(anchor, entry_high) + atr_buffer
        risk = max(stop_price - current_price, minimum_risk)
        tp1 = current_price - risk
        tp2 = current_price - (2 * risk)
        reference_target = float(reference_ctx.model.swing_low) if reference_ctx.model.swing_low is not None else None
        if reference_target is not None:
            tp2 = min(tp2, reference_target)
        free_space_r = None if reference_target is None or risk <= 0 else (current_price - reference_target) / risk
        invalidation_price = float(setup_ctx.model.swing_high or entry_high)
        return {
            "entry_low": entry_low,
            "entry_high": entry_high,
            "stop_price": stop_price,
            "risk": risk,
            "tp1": tp1,
            "tp2": tp2,
            "reference_target": reference_target,
            "free_space_r": free_space_r,
            "invalidation_price": invalidation_price,
        }

    def _build_trigger_score_note(
        self,
        *,
        trigger_key: str,
        higher_bias: Bias,
        requirements: dict[str, bool],
        min_auxiliary_confirmations: int,
    ) -> str:
        parts: list[str] = []
        if requirements["require_regained_fast"]:
            parts.append("EMA21 回收" if higher_bias == Bias.BULLISH else "EMA21 失守")
        if requirements["require_held_slow"]:
            parts.append("EMA55/局部结构修复" if higher_bias == Bias.BULLISH else "EMA55/局部结构转弱")
        if requirements["require_auxiliary"]:
            parts.append(f"至少 {min_auxiliary_confirmations} 个辅助确认")
        if not parts:
            parts.append("局部确认")
        return f"{trigger_key} 触发要求 " + " + ".join(parts)

    def _resolve_trend_strength_threshold(self, higher_bias: Bias) -> int:
        if higher_bias == Bias.BULLISH:
            return int(
                self.window_config.get(
                    "bullish_trend_strength_threshold",
                    self.window_config.get("trend_strength_threshold", 60),
                )
            )
        if higher_bias == Bias.BEARISH:
            return int(
                self.window_config.get(
                    "bearish_trend_strength_threshold",
                    self.window_config.get("trend_strength_threshold", 60),
                )
            )
        return int(self.window_config.get("trend_strength_threshold", 60))

    def _is_trend_friendly(
        self,
        *,
        higher_bias: Bias,
        trend_strength: int,
        volatility_state: VolatilityState,
    ) -> bool:
        if higher_bias == Bias.NEUTRAL or volatility_state == VolatilityState.HIGH:
            return False
        return trend_strength >= self._resolve_trend_strength_threshold(higher_bias)

    def _assess_setup(
        self,
        higher_bias: Bias,
        setup_ctx: PreparedTimeframe,
        setup_key: str,
        *,
        reference_ctx: Optional[PreparedTimeframe] = None,
        current_price: Optional[float] = None,
    ) -> dict[str, Any]:
        reasons_for: list[str] = []
        reasons_against: list[str] = []
        risk_notes: list[str] = []
        reference_ctx = reference_ctx or setup_ctx
        current_price = float(current_price if current_price is not None else setup_ctx.model.close)
        require_structure_ready = bool(self.config.get("setup", {}).get("require_structure_ready", False))
        require_reversal_candle = self._resolve_setup_reversal_requirement(higher_bias)
        level_confirmation_config = dict(self.config.get("level_confirmation", {}))
        level_confirmation_enabled = bool(level_confirmation_config.get("enabled", False))
        confluence_config = dict(self.config.get("setup_confluence", {}))
        confluence_enabled = bool(confluence_config.get("enabled", False))
        free_space_config = dict(self.config.get("free_space", {}))
        free_space_enabled = bool(free_space_config.get("enabled", False))
        divergence_config = dict(self.config.get("divergence", {}))
        divergence_enabled = bool(divergence_config.get("enabled", False))
        state_note = self._compute_axis_band_state_note(higher_bias=higher_bias, setup_ctx=setup_ctx)
        divergence_min_level = int(divergence_config.get("min_level", 2))
        divergence_gate_mode = str(divergence_config.get("gate_mode", "score")).lower()
        divergence_gate_biases = {
            str(item).lower() for item in divergence_config.get("gate_biases", []) if str(item).strip()
        }
        divergence_level = 0
        opposing_divergence_level = 0
        divergence_ready = False
        opposing_divergence_ready = False
        require_divergence_gate = False
        free_space_r: Optional[float] = None
        free_space_min_r = 0.0
        require_free_space_gate = False
        free_space_ready = True

        if higher_bias == Bias.NEUTRAL:
            return {
                "aligned": False,
                "execution_ready": False,
                "pullback_ready": False,
                "reversal_ready": False,
                "require_reversal_candle": require_reversal_candle,
                "divergence_enabled": divergence_enabled,
                "require_divergence_gate": False,
                "require_free_space_gate": False,
                "divergence_ready": False,
                "divergence_level": 0,
                "opposing_divergence_level": 0,
                "free_space_ready": True,
                "free_space_r": None,
                "free_space_min_r": 0.0,
                "is_extended": setup_ctx.model.is_extended,
                "distance_to_value_atr": round(setup_ctx.distance_to_execution_atr, 4),
                "structure_ready": False,
                "structure_source": "confluence" if confluence_enabled else "swing_distance",
                "level_confirmation": {
                    "enabled": level_confirmation_enabled,
                    "base_reversal_ready": False,
                    "ready": False,
                    "hits": 0,
                    "min_hits": int(level_confirmation_config.get("min_hits", 1)),
                    "anchor_price": round(midpoint(setup_ctx.execution_zone_low, setup_ctx.execution_zone_high), 4),
                    "components": {},
                    "source": "level_aware_reclaim_rejection",
                },
                "state_note": state_note,
                "confluence": {
                    "enabled": confluence_enabled,
                    "ready": False,
                    "hits": 0,
                    "min_hits": int(confluence_config.get("min_hits", 2)),
                    "max_spread_atr": round(float(confluence_config.get("max_spread_atr", 1.2)), 4),
                    "spread_atr": None,
                    "anchor_price": round(midpoint(setup_ctx.execution_zone_low, setup_ctx.execution_zone_high), 4),
                    "components": {},
                    "source": "ema55_pivot_band_confluence",
                },
                "score": -10,
                "score_note": f"{setup_key} 在高周期不清晰时不纳入执行判断",
                "reasons_for": reasons_for,
                "reasons_against": [f"高周期未对齐前，{setup_key} 位置不具备可交易性"],
                "risk_notes": risk_notes,
                "state_notes": [],
            }

        if higher_bias == Bias.BULLISH:
            trade_levels = self._derive_trade_levels(
                bias=higher_bias,
                setup_ctx=setup_ctx,
                reference_ctx=reference_ctx,
                current_price=current_price,
            )
            free_space_r = trade_levels["free_space_r"]
            free_space_min_r = float(free_space_config.get("long_min_r", 0.0))
            require_free_space_gate = free_space_enabled and free_space_min_r > 0
            free_space_ready = not require_free_space_gate or (
                free_space_r is not None and free_space_r >= free_space_min_r
            )
            require_divergence_gate = divergence_enabled and divergence_gate_mode == "hard" and "bullish" in divergence_gate_biases
            aligned = (
                setup_ctx.model.close > setup_ctx.model.ema200
                and setup_ctx.model.structure_state != StructureState.BEARISH
                and setup_ctx.model.ema_alignment != "bearish"
            )
            level_confirmation = self._compute_level_aware_confirmation(higher_bias=higher_bias, setup_ctx=setup_ctx)
            reversal_ready = (
                bool(level_confirmation["ready"])
                if level_confirmation_enabled
                else bool(setup_ctx.candle_profile.get("has_bullish_reversal_candle", False))
            )
            confluence = self._compute_setup_confluence(higher_bias=higher_bias, setup_ctx=setup_ctx)
            structure_source = "confluence" if confluence_enabled else "swing_distance"
            structure_ready = (
                bool(confluence["ready"])
                if confluence_enabled
                else setup_ctx.swing_support_distance_atr is not None
                and setup_ctx.swing_support_distance_atr <= float(self.config["execution"]["support_proximity_atr"])
            )
            if aligned:
                reasons_for.append(f"{setup_key} 仍在 EMA200 上方，顺势结构没有被破坏")
            else:
                reasons_against.append(f"{setup_key} 已不再清楚支持偏多环境")
            if setup_ctx.distance_to_execution_atr <= float(self.config["execution"]["pullback_distance_atr"]):
                reasons_for.append(f"{setup_key} 回到 EMA21/EMA55 执行区附近，不是在远离均线的位置追多")
            else:
                reasons_against.append(f"{setup_key} 离 EMA21/EMA55 太远，不适合追多")
            if confluence_enabled:
                if structure_ready:
                    reasons_for.append(
                        f"{setup_key} 的 EMA55 / pivot low / 下沿 band 共振达到 {confluence['hits']}/{len(confluence['components'])}"
                    )
                else:
                    reasons_against.append(
                        f"{setup_key} 的 EMA55 / pivot low / 下沿 band 共振不足，仅 {confluence['hits']}/{len(confluence['components'])}"
                    )
                    if require_structure_ready:
                        risk_notes.append(f"{setup_key} 的支撑共振还不够密集，不把它当成有效缓冲位")
            elif structure_ready:
                reasons_for.append(f"{setup_key} 最近 swing low 靠得够近，止损结构更清晰")
            else:
                reasons_against.append(f"{setup_key} 最近 swing low 离执行区还不够近")
                if require_structure_ready:
                    risk_notes.append(f"{setup_key} 的 swing low 不够贴近，v2 不把它当成有效缓冲位")
            if require_free_space_gate:
                if free_space_ready and free_space_r is not None:
                    reasons_for.append(
                        f"从当前价到 {setup_key} 参考高点大约还有 {round(free_space_r, 2)}R 空间，至少够跑 {round(free_space_min_r, 2)}R"
                    )
                elif free_space_r is None:
                    reasons_against.append("当前还找不到清晰的高周期前高，free space 无法确认")
                else:
                    reasons_against.append(
                        f"free space 只有 {round(free_space_r, 2)}R，不足以支撑至少 {round(free_space_min_r, 2)}R 的多头空间"
                    )
            if require_reversal_candle:
                if reversal_ready:
                    if level_confirmation_enabled:
                        reasons_for.append(
                            f"{setup_key} 出现明确止跌 K 线，且发生在 EMA55 / pivot low / 下沿 band 附近的 reclaim/rejection"
                        )
                    else:
                        reasons_for.append(f"{setup_key} 出现明确止跌 K 线（长下影或偏止跌十字星）")
                else:
                    if level_confirmation_enabled and level_confirmation["base_reversal_ready"]:
                        reasons_against.append(
                            f"{setup_key} 虽有止跌 K 线，但不在 EMA55 / pivot low / 下沿 band 附近"
                        )
                    elif level_confirmation_enabled:
                        reasons_against.append(
                            f"{setup_key} 还没有出现发生在 EMA55 / pivot low / 下沿 band 附近的止跌 reclaim/rejection"
                        )
                    else:
                        reasons_against.append(f"{setup_key} 还没有出现明确止跌 K 线（长下影或偏止跌十字星）")
            if divergence_enabled:
                divergence_level = int(setup_ctx.divergence_profile.get("bullish_level", 0))
                divergence_ready = bool(setup_ctx.divergence_profile.get("bullish_signal", False)) and (
                    divergence_level >= divergence_min_level
                )
                opposing_divergence_level = int(setup_ctx.divergence_profile.get("bearish_level", 0))
                opposing_divergence_ready = bool(setup_ctx.divergence_profile.get("bearish_signal", False)) and (
                    opposing_divergence_level >= divergence_min_level
                )
                if divergence_ready:
                    reasons_for.append(f"{setup_key} 出现 Bull divergence L{divergence_level}，回踩衰竭确认更完整")
                elif require_divergence_gate:
                    reasons_against.append(f"{setup_key} 还没有出现 Bull divergence L{divergence_min_level}+，多头 setup 先不放行")
                elif opposing_divergence_ready:
                    reasons_against.append(f"{setup_key} 反而出现 Bear divergence L{opposing_divergence_level}，继续回撤风险偏高")
            if setup_ctx.model.is_extended:
                risk_notes.append(f"等待价格回到 {setup_key} EMA21/EMA55 再考虑做多")
            elif require_reversal_candle and not reversal_ready:
                if level_confirmation_enabled:
                    risk_notes.append(
                        f"等待 {setup_key} 在 EMA55 / pivot low / 下沿 band 附近出现明确止跌 reclaim/rejection，再考虑做多"
                    )
                else:
                    risk_notes.append(f"等待 {setup_key} 在执行区附近出现明确止跌 K 线，再考虑做多")
            state_notes: list[str] = []
            if state_note["active"]:
                state_notes.append(str(state_note["message"]))
        else:
            trade_levels = self._derive_trade_levels(
                bias=higher_bias,
                setup_ctx=setup_ctx,
                reference_ctx=reference_ctx,
                current_price=current_price,
            )
            free_space_r = trade_levels["free_space_r"]
            free_space_min_r = float(free_space_config.get("short_min_r", 0.0))
            require_free_space_gate = free_space_enabled and free_space_min_r > 0
            free_space_ready = not require_free_space_gate or (
                free_space_r is not None and free_space_r >= free_space_min_r
            )
            require_divergence_gate = divergence_enabled and divergence_gate_mode == "hard" and "bearish" in divergence_gate_biases
            aligned = (
                setup_ctx.model.close < setup_ctx.model.ema200
                and setup_ctx.model.structure_state != StructureState.BULLISH
                and setup_ctx.model.ema_alignment != "bullish"
            )
            level_confirmation = self._compute_level_aware_confirmation(higher_bias=higher_bias, setup_ctx=setup_ctx)
            reversal_ready = (
                bool(level_confirmation["ready"])
                if level_confirmation_enabled
                else bool(setup_ctx.candle_profile.get("has_bearish_reversal_candle", False))
            )
            confluence = self._compute_setup_confluence(higher_bias=higher_bias, setup_ctx=setup_ctx)
            structure_source = "confluence" if confluence_enabled else "swing_distance"
            structure_ready = (
                bool(confluence["ready"])
                if confluence_enabled
                else setup_ctx.swing_resistance_distance_atr is not None
                and setup_ctx.swing_resistance_distance_atr <= float(self.config["execution"]["resistance_proximity_atr"])
            )
            if aligned:
                reasons_for.append(f"{setup_key} 仍在 EMA200 下方，顺势结构没有被破坏")
            else:
                reasons_against.append(f"{setup_key} 已不再清楚支持偏空环境")
            if setup_ctx.distance_to_execution_atr <= float(self.config["execution"]["pullback_distance_atr"]):
                reasons_for.append(f"{setup_key} 回到 EMA21/EMA55 执行区附近，不是在远离均线的位置追空")
            else:
                reasons_against.append(f"{setup_key} 离 EMA21/EMA55 太远，不适合追空")
            if confluence_enabled:
                if structure_ready:
                    reasons_for.append(
                        f"{setup_key} 的 EMA55 / pivot high / 上沿 band 共振达到 {confluence['hits']}/{len(confluence['components'])}"
                    )
                else:
                    reasons_against.append(
                        f"{setup_key} 的 EMA55 / pivot high / 上沿 band 共振不足，仅 {confluence['hits']}/{len(confluence['components'])}"
                    )
                    if require_structure_ready:
                        risk_notes.append(f"{setup_key} 的阻力共振还不够密集，不把它当成有效缓冲位")
            elif structure_ready:
                reasons_for.append(f"{setup_key} 最近 swing high 靠得够近，止损结构更清晰")
            else:
                reasons_against.append(f"{setup_key} 最近 swing high 离执行区还不够近")
                if require_structure_ready:
                    risk_notes.append(f"{setup_key} 的 swing high 不够贴近，v2 不把它当成有效缓冲位")
            if require_free_space_gate:
                if free_space_ready and free_space_r is not None:
                    reasons_for.append(
                        f"从当前价到 {setup_key} 参考低点大约还有 {round(free_space_r, 2)}R 空间，至少够跑 {round(free_space_min_r, 2)}R"
                    )
                elif free_space_r is None:
                    reasons_against.append("当前还找不到清晰的高周期前低，free space 无法确认")
                else:
                    reasons_against.append(
                        f"free space 只有 {round(free_space_r, 2)}R，不足以支撑至少 {round(free_space_min_r, 2)}R 的空头空间"
                    )
            if require_reversal_candle:
                if reversal_ready:
                    if level_confirmation_enabled:
                        reasons_for.append(
                            f"{setup_key} 出现明确见顶 K 线，且发生在 EMA55 / pivot high / 上沿 band 附近的 reclaim/rejection"
                        )
                    else:
                        reasons_for.append(f"{setup_key} 出现明确见顶 K 线（长上影或偏见顶十字星）")
                else:
                    if level_confirmation_enabled and level_confirmation["base_reversal_ready"]:
                        reasons_against.append(
                            f"{setup_key} 虽有见顶 K 线，但不在 EMA55 / pivot high / 上沿 band 附近"
                        )
                    elif level_confirmation_enabled:
                        reasons_against.append(
                            f"{setup_key} 还没有出现发生在 EMA55 / pivot high / 上沿 band 附近的见顶 reclaim/rejection"
                        )
                    else:
                        reasons_against.append(f"{setup_key} 还没有出现明确见顶 K 线（长上影或偏见顶十字星）")
            if divergence_enabled:
                divergence_level = int(setup_ctx.divergence_profile.get("bearish_level", 0))
                divergence_ready = bool(setup_ctx.divergence_profile.get("bearish_signal", False)) and (
                    divergence_level >= divergence_min_level
                )
                opposing_divergence_level = int(setup_ctx.divergence_profile.get("bullish_level", 0))
                opposing_divergence_ready = bool(setup_ctx.divergence_profile.get("bullish_signal", False)) and (
                    opposing_divergence_level >= divergence_min_level
                )
                if divergence_ready:
                    reasons_for.append(f"{setup_key} 出现 Bear divergence L{divergence_level}，反弹衰竭确认更完整")
                elif require_divergence_gate:
                    reasons_against.append(f"{setup_key} 还没有出现 Bear divergence L{divergence_min_level}+，空头 setup 先不放行")
                elif opposing_divergence_ready:
                    reasons_against.append(f"{setup_key} 反而出现 Bull divergence L{opposing_divergence_level}，继续反弹风险偏高")
            if setup_ctx.model.is_extended:
                risk_notes.append(f"等待价格回到 {setup_key} EMA21/EMA55 再考虑做空")
            elif require_reversal_candle and not reversal_ready:
                if level_confirmation_enabled:
                    risk_notes.append(
                        f"等待 {setup_key} 在 EMA55 / pivot high / 上沿 band 附近出现明确见顶 reclaim/rejection，再考虑做空"
                    )
                else:
                    risk_notes.append(f"等待 {setup_key} 在执行区附近出现明确见顶 K 线，再考虑做空")
            state_notes = []
            if state_note["active"]:
                state_notes.append(str(state_note["message"]))

        execution_ready = aligned and setup_ctx.distance_to_execution_atr <= float(self.config["execution"]["pullback_distance_atr"])
        if require_structure_ready:
            execution_ready = execution_ready and structure_ready
        pullback_ready = execution_ready and (not require_free_space_gate or free_space_ready)
        if require_divergence_gate and execution_ready and not divergence_ready:
            if higher_bias == Bias.BULLISH:
                risk_notes.append(f"等待 {setup_key} 给出 Bull divergence L{divergence_min_level}+，再放行多头")
            else:
                risk_notes.append(f"等待 {setup_key} 给出 Bear divergence L{divergence_min_level}+，再放行空头")
        if require_free_space_gate and execution_ready and not free_space_ready:
            if higher_bias == Bias.BULLISH:
                risk_notes.append(
                    f"等待到高周期前高至少留出 {round(free_space_min_r, 2)}R free space，再放行多头"
                )
            else:
                risk_notes.append(
                    f"等待到高周期前低至少留出 {round(free_space_min_r, 2)}R free space，再放行空头"
                )
        score = 15 if pullback_ready else 7 if aligned and not setup_ctx.model.is_extended else 5 if aligned else -12
        if structure_ready:
            score += 4
        if require_reversal_candle:
            score += 4 if reversal_ready else -6
        if require_free_space_gate:
            score += 3 if free_space_ready else -7
        if divergence_enabled:
            if require_divergence_gate:
                score += 4 if divergence_ready else -8
            elif divergence_ready:
                score += int(
                    divergence_config.get(
                        "level_3_bonus" if divergence_level >= 3 else "level_2_bonus",
                        8 if divergence_level >= 3 else 4,
                    )
                )
            elif opposing_divergence_ready:
                score -= int(divergence_config.get("opposing_level_penalty", 4))
        if setup_ctx.model.is_extended:
            score -= 10
        if confluence_enabled:
            score_note = f"{setup_key} 重点看 EMA21/EMA55 回踩与 EMA55 + pivot + band 的位置共振"
        else:
            score_note = f"{setup_key} 重点看 EMA21/EMA55 回踩与 swing 支撑/阻力结构"
        if require_reversal_candle:
            if level_confirmation_enabled:
                score_note += "，并要求 reversal 发生在 EMA55 / pivot / band 附近的 reclaim/rejection"
            else:
                score_note += "，并要求出现明确 reversal K 线"
        if require_free_space_gate:
            score_note += f"，并要求至少保留 {round(free_space_min_r, 2)}R 的高周期 free space"
        if divergence_enabled:
            if require_divergence_gate:
                direction_label = "Bull" if higher_bias == Bias.BULLISH else "Bear"
                score_note += f"，并要求出现 {direction_label} divergence L{divergence_min_level}+"
            else:
                score_note += "，背离只作为 setup 加分项"

        return {
            "aligned": aligned,
            "execution_ready": execution_ready,
            "pullback_ready": pullback_ready,
            "reversal_ready": reversal_ready,
            "require_reversal_candle": require_reversal_candle,
            "divergence_enabled": divergence_enabled,
            "require_divergence_gate": require_divergence_gate,
            "require_free_space_gate": require_free_space_gate,
            "divergence_ready": divergence_ready,
            "divergence_level": divergence_level,
            "opposing_divergence_level": opposing_divergence_level,
            "free_space_ready": free_space_ready,
            "free_space_r": round(free_space_r, 4) if free_space_r is not None else None,
            "free_space_min_r": round(free_space_min_r, 4),
            "is_extended": setup_ctx.model.is_extended,
            "distance_to_value_atr": round(setup_ctx.distance_to_execution_atr, 4),
            "structure_ready": structure_ready,
            "structure_source": structure_source,
            "level_confirmation": level_confirmation,
            "state_note": state_note,
            "confluence": confluence,
            "score": score,
            "score_note": score_note,
            "reasons_for": reasons_for,
            "reasons_against": reasons_against,
            "risk_notes": risk_notes,
            "state_notes": state_notes,
        }

    def _assess_trigger(
        self,
        higher_bias: Bias,
        trigger_ctx: PreparedTimeframe,
        trigger_key: str,
        *,
        trend_strength: Optional[int] = None,
    ) -> dict[str, Any]:
        if higher_bias == Bias.NEUTRAL:
            return {
                "state": TriggerState.NONE,
                "score": -5,
                "score_note": f"高周期中性时，{trigger_key} 触发不单独拍板",
                "reasons_for": [],
                "reasons_against": [f"高周期中性时，{trigger_key} 的局部触发不足以单独改变方向"],
                "risk_notes": [],
            }

        df = trigger_ctx.df
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        recent_low = float(df["low"].tail(3).min())
        prior_low = float(df["low"].iloc[-6:-3].min()) if len(df) >= 6 else recent_low
        recent_high = float(df["high"].tail(3).max())
        prior_high = float(df["high"].iloc[-6:-3].max()) if len(df) >= 6 else recent_high
        ema21_latest = float(latest["ema_21"])
        ema21_prev = float(previous["ema_21"])
        ema55_latest = float(latest["ema_55"])
        candle_profile = trigger_ctx.candle_profile
        min_auxiliary_confirmations = int(self.config.get("trigger", {}).get("min_auxiliary_confirmations", 1))
        mixed_score = int(self.config.get("trigger", {}).get("mixed_score", -2))
        none_score = int(self.config.get("trigger", {}).get("none_score", -12))
        requirements = self._resolve_trigger_requirements(higher_bias, trend_strength)

        reasons_for: list[str] = []
        reasons_against: list[str] = []
        risk_notes: list[str] = []
        auxiliary_count = 0

        if higher_bias == Bias.BULLISH:
            no_new_extreme = recent_low >= prior_low
            if recent_low >= prior_low:
                auxiliary_count += 1
                reasons_for.append(f"{trigger_key} 没有继续创新低")
            else:
                reasons_against.append(f"{trigger_key} 仍在刷新短线低点")

            regained_fast = float(latest["close"]) > ema21_latest and ema21_latest >= ema21_prev
            if regained_fast:
                reasons_for.append(f"{trigger_key} 已回收 EMA21，短线斜率开始改善")
            elif requirements["require_regained_fast"]:
                reasons_against.append(f"{trigger_key} 还没有明确收回 EMA21")
            elif requirements.get("regained_fast_relaxed_by_regime"):
                risk_notes.append(f"{trigger_key} 还没有明确收回 EMA21，但当前强趋势环境下不把它当成硬门槛")
            else:
                risk_notes.append(f"{trigger_key} 还没有明确收回 EMA21，但本实验不把它当成硬门槛")

            held_slow = trigger_ctx.model.structure_state == StructureState.BULLISH or float(latest["close"]) > ema55_latest
            if held_slow:
                reasons_for.append(f"{trigger_key} 站回 EMA55 上方或局部结构开始转强")
            elif requirements["require_held_slow"]:
                reasons_against.append(f"{trigger_key} 仍在 EMA55 下方，局部结构还未修复")
            else:
                risk_notes.append(f"{trigger_key} 仍在 EMA55 下方，局部结构修复还不充分")

            if candle_profile["has_bullish_rejection"]:
                auxiliary_count += 1
                reasons_for.append(f"{trigger_key} 出现带下影的拒绝回落")
            if candle_profile["is_volume_contracting"]:
                auxiliary_count += 1
                reasons_for.append(f"{trigger_key} 抛压正在缩小")

            auxiliary_ready = auxiliary_count >= min_auxiliary_confirmations
            confirm_checks: list[bool] = []
            if requirements["require_regained_fast"]:
                confirm_checks.append(regained_fast)
            if requirements["require_held_slow"]:
                confirm_checks.append(held_slow)
            if requirements["require_auxiliary"]:
                confirm_checks.append(auxiliary_ready)
            supportive_signals = int(regained_fast) + int(held_slow) + int(auxiliary_ready)

            if (confirm_checks and all(confirm_checks)) or (not confirm_checks and supportive_signals >= 1):
                state = TriggerState.BULLISH_CONFIRMED
                score = 12
            elif supportive_signals >= 1:
                state = TriggerState.MIXED
                score = mixed_score
                risk_notes.append(f"{trigger_key} 已有一些建设性信号，但还没有完全收敛")
            else:
                state = TriggerState.NONE
                score = none_score
                risk_notes.append(f"{trigger_key} 缺少确认，时机偏弱")
        else:
            no_new_extreme = recent_high <= prior_high
            if recent_high <= prior_high:
                auxiliary_count += 1
                reasons_for.append(f"{trigger_key} 没有继续创新高")
            else:
                reasons_against.append(f"{trigger_key} 仍在刷新短线高点")

            regained_fast = float(latest["close"]) < ema21_latest and ema21_latest <= ema21_prev
            if regained_fast:
                reasons_for.append(f"{trigger_key} 已跌回 EMA21 下方，短线斜率开始转弱")
            elif requirements["require_regained_fast"]:
                reasons_against.append(f"{trigger_key} 还没有明确跌回 EMA21 下方")
            elif requirements.get("regained_fast_relaxed_by_regime"):
                risk_notes.append(f"{trigger_key} 还没有明确跌回 EMA21 下方，但当前强空环境下不把它当成硬门槛")
            else:
                risk_notes.append(f"{trigger_key} 还没有明确跌回 EMA21 下方，但本实验不把它当成硬门槛")

            held_slow = trigger_ctx.model.structure_state == StructureState.BEARISH or float(latest["close"]) < ema55_latest
            if held_slow:
                reasons_for.append(f"{trigger_key} 站在 EMA55 下方或局部结构开始转弱")
            elif requirements["require_held_slow"]:
                reasons_against.append(f"{trigger_key} 仍在 EMA55 上方，局部结构还不够弱")
            else:
                risk_notes.append(f"{trigger_key} 仍在 EMA55 上方，局部结构转弱还不够充分")

            if candle_profile["has_bearish_rejection"]:
                auxiliary_count += 1
                reasons_for.append(f"{trigger_key} 出现带上影的拒绝拉升")
            if candle_profile["is_volume_contracting"]:
                auxiliary_count += 1
                reasons_for.append(f"{trigger_key} 买盘动能正在缩小")

            auxiliary_ready = auxiliary_count >= min_auxiliary_confirmations
            confirm_checks = []
            if requirements["require_regained_fast"]:
                confirm_checks.append(regained_fast)
            if requirements["require_held_slow"]:
                confirm_checks.append(held_slow)
            if requirements["require_auxiliary"]:
                confirm_checks.append(auxiliary_ready)
            supportive_signals = int(regained_fast) + int(held_slow) + int(auxiliary_ready)

            if (confirm_checks and all(confirm_checks)) or (not confirm_checks and supportive_signals >= 1):
                state = TriggerState.BEARISH_CONFIRMED
                score = 12
            elif supportive_signals >= 1:
                state = TriggerState.MIXED
                score = mixed_score
                risk_notes.append(f"{trigger_key} 已有一些建设性信号，但还没有完全收敛")
            else:
                state = TriggerState.NONE
                score = none_score
                risk_notes.append(f"{trigger_key} 缺少确认，时机偏弱")

        return {
            "state": state,
            "score": score,
            "score_note": self._build_trigger_score_note(
                trigger_key=trigger_key,
                higher_bias=higher_bias,
                requirements=requirements,
                min_auxiliary_confirmations=min_auxiliary_confirmations,
            ),
            "reasons_for": reasons_for,
            "reasons_against": reasons_against,
            "risk_notes": risk_notes,
            "recent_low": round(recent_low, 4),
            "prior_low": round(prior_low, 4),
            "recent_high": round(recent_high, 4),
            "prior_high": round(prior_high, 4),
            "bullish_rejection": candle_profile["has_bullish_rejection"],
            "bearish_rejection": candle_profile["has_bearish_rejection"],
            "volume_contracting": candle_profile["is_volume_contracting"],
            "no_new_extreme": no_new_extreme,
            "regained_fast": regained_fast,
            "held_slow": held_slow,
            "auxiliary_count": auxiliary_count,
            "min_auxiliary_confirmations": min_auxiliary_confirmations,
        }

    def _decide(
        self,
        *,
        higher_bias: Bias,
        trend_friendly: bool,
        setup_assessment: dict[str, Any],
        trigger_assessment: dict[str, Any],
        confidence: int,
    ) -> tuple[Action, Bias, RecommendedTiming]:
        reversal_required = bool(setup_assessment.get("require_reversal_candle", False))
        reversal_ready = bool(setup_assessment.get("reversal_ready", True))
        divergence_required = bool(setup_assessment.get("require_divergence_gate", False))
        divergence_ready = bool(setup_assessment.get("divergence_ready", True))
        if higher_bias == Bias.BULLISH:
            if (
                trend_friendly
                and setup_assessment["aligned"]
                and setup_assessment["pullback_ready"]
                and (not reversal_required or reversal_ready)
                and (not divergence_required or divergence_ready)
                and trigger_assessment["state"] == TriggerState.BULLISH_CONFIRMED
                and confidence >= int(self.config["confidence"]["action_threshold"])
            ):
                return Action.LONG, Bias.BULLISH, RecommendedTiming.NOW
            if setup_assessment["is_extended"]:
                return Action.WAIT, Bias.BULLISH, RecommendedTiming.WAIT_PULLBACK
            if reversal_required and not reversal_ready:
                return Action.WAIT, Bias.BULLISH, RecommendedTiming.WAIT_CONFIRMATION
            if divergence_required and not divergence_ready:
                return Action.WAIT, Bias.BULLISH, RecommendedTiming.WAIT_CONFIRMATION
            if trigger_assessment["state"] != TriggerState.BULLISH_CONFIRMED:
                return Action.WAIT, Bias.BULLISH, RecommendedTiming.WAIT_CONFIRMATION
            return Action.WAIT, Bias.BULLISH, RecommendedTiming.SKIP

        if higher_bias == Bias.BEARISH:
            if (
                trend_friendly
                and setup_assessment["aligned"]
                and setup_assessment["pullback_ready"]
                and (not reversal_required or reversal_ready)
                and (not divergence_required or divergence_ready)
                and trigger_assessment["state"] == TriggerState.BEARISH_CONFIRMED
                and confidence >= int(self.config["confidence"]["action_threshold"])
            ):
                return Action.SHORT, Bias.BEARISH, RecommendedTiming.NOW
            if setup_assessment["is_extended"]:
                return Action.WAIT, Bias.BEARISH, RecommendedTiming.WAIT_PULLBACK
            if reversal_required and not reversal_ready:
                return Action.WAIT, Bias.BEARISH, RecommendedTiming.WAIT_CONFIRMATION
            if divergence_required and not divergence_ready:
                return Action.WAIT, Bias.BEARISH, RecommendedTiming.WAIT_CONFIRMATION
            if trigger_assessment["state"] != TriggerState.BEARISH_CONFIRMED:
                return Action.WAIT, Bias.BEARISH, RecommendedTiming.WAIT_CONFIRMATION
            return Action.WAIT, Bias.BEARISH, RecommendedTiming.SKIP

        return Action.WAIT, Bias.NEUTRAL, RecommendedTiming.SKIP

    def _build_trade_plan(
        self,
        *,
        action: Action,
        bias: Bias,
        setup_ctx: PreparedTimeframe,
        reference_ctx: PreparedTimeframe,
        current_price: float,
        setup_key: str,
        reference_key: str,
    ) -> dict[str, Any]:
        if action == Action.WAIT:
            if bias == Bias.BULLISH:
                return {
                    "entry_zone": None,
                    "stop_loss": None,
                    "take_profit_hint": None,
                    "invalidation": f"当前仍需更清晰的 {setup_key} 回踩和确认，因此先观望。",
                    "invalidation_price": None,
                }
            if bias == Bias.BEARISH:
                return {
                    "entry_zone": None,
                    "stop_loss": None,
                    "take_profit_hint": None,
                    "invalidation": f"当前仍需更清晰的 {setup_key} 反弹和确认，因此先观望。",
                    "invalidation_price": None,
                }
            return {
                "entry_zone": None,
                "stop_loss": None,
                "take_profit_hint": None,
                "invalidation": "当前高周期方向不清晰，因此不提供执行计划。",
                "invalidation_price": None,
            }

        trade_levels = self._derive_trade_levels(
            bias=bias,
            setup_ctx=setup_ctx,
            reference_ctx=reference_ctx,
            current_price=current_price,
        )
        entry_low = round(float(trade_levels["entry_low"] or setup_ctx.execution_zone_low), 4)
        entry_high = round(float(trade_levels["entry_high"] or setup_ctx.execution_zone_high), 4)

        if bias == Bias.BULLISH:
            return {
                "entry_zone": EntryZone(
                    low=entry_low,
                    high=entry_high,
                    basis=f"{setup_key} EMA21/EMA55 回踩区",
                ),
                "stop_loss": StopLoss(
                    price=round(float(trade_levels["stop_price"] or setup_ctx.execution_zone_low), 4),
                    basis=f"最近 {setup_key} swing low 下方加 ATR 缓冲",
                ),
                "take_profit_hint": TakeProfitHint(
                    tp1=round(float(trade_levels["tp1"] or current_price), 4),
                    tp2=round(float(trade_levels["tp2"] or current_price), 4),
                    basis=f"按 1R/2R 推进，优先参考 {reference_key} 前高",
                ),
                "invalidation": f"若 {setup_key} 跌破最近 swing low 并重新失守 EMA21/EMA55，则多头逻辑失效",
                "invalidation_price": round(float(trade_levels["invalidation_price"] or entry_low), 4),
            }

        return {
            "entry_zone": EntryZone(
                low=entry_low,
                high=entry_high,
                basis=f"{setup_key} EMA21/EMA55 反弹区",
            ),
            "stop_loss": StopLoss(
                price=round(float(trade_levels["stop_price"] or setup_ctx.execution_zone_high), 4),
                basis=f"最近 {setup_key} swing high 上方加 ATR 缓冲",
            ),
            "take_profit_hint": TakeProfitHint(
                tp1=round(float(trade_levels["tp1"] or current_price), 4),
                tp2=round(float(trade_levels["tp2"] or current_price), 4),
                basis=f"按 1R/2R 推进，优先参考 {reference_key} 前低",
            ),
            "invalidation": f"若 {setup_key} 突破最近 swing high 并重新站回 EMA21/EMA55，则空头逻辑失效",
            "invalidation_price": round(float(trade_levels["invalidation_price"] or entry_high), 4),
        }

    def _build_summary(
        self,
        *,
        action: Action,
        recommended_timing: RecommendedTiming,
        higher_bias: Bias,
        setup_assessment: dict[str, Any],
        trigger_assessment: dict[str, Any],
        setup_key: str,
        trigger_key: str,
    ) -> str:
        if action == Action.LONG:
            return f"高周期偏多，{setup_key} 回到 EMA21/EMA55 附近，{trigger_key} 给出确认，因此可以做条件多。"
        if action == Action.SHORT:
            return f"高周期偏空，{setup_key} 回到 EMA21/EMA55 附近，{trigger_key} 给出确认，因此可以做条件空。"
        if (
            higher_bias == Bias.BULLISH
            and setup_assessment.get("execution_ready", setup_assessment["pullback_ready"])
            and setup_assessment.get("require_free_space_gate")
            and not setup_assessment.get("free_space_ready", True)
        ):
            return f"高周期偏多，{setup_key} 已进执行区，但到高周期前高的 free space 还不够。"
        if (
            higher_bias == Bias.BULLISH
            and setup_assessment["pullback_ready"]
            and setup_assessment.get("require_reversal_candle")
            and not setup_assessment.get("reversal_ready", True)
        ):
            return f"高周期偏多，{setup_key} 已进入执行区，但还没有出现明确止跌 K 线。"
        if (
            higher_bias == Bias.BULLISH
            and setup_assessment["pullback_ready"]
            and setup_assessment.get("require_divergence_gate")
            and not setup_assessment.get("divergence_ready", True)
        ):
            return f"高周期偏多，{setup_key} 已进入执行区，但还没有出现满足等级的 Bull divergence。"
        if (
            higher_bias == Bias.BEARISH
            and setup_assessment.get("execution_ready", setup_assessment["pullback_ready"])
            and setup_assessment.get("require_free_space_gate")
            and not setup_assessment.get("free_space_ready", True)
        ):
            return f"高周期偏空，{setup_key} 已进执行区，但到高周期前低的 free space 还不够。"
        if (
            higher_bias == Bias.BEARISH
            and setup_assessment["pullback_ready"]
            and setup_assessment.get("require_reversal_candle")
            and not setup_assessment.get("reversal_ready", True)
        ):
            return f"高周期偏空，{setup_key} 已进入执行区，但还没有出现明确见顶 K 线。"
        if (
            higher_bias == Bias.BEARISH
            and setup_assessment["pullback_ready"]
            and setup_assessment.get("require_divergence_gate")
            and not setup_assessment.get("divergence_ready", True)
        ):
            return f"高周期偏空，{setup_key} 已进入执行区，但还没有出现满足等级的 Bear divergence。"
        if higher_bias == Bias.BULLISH and recommended_timing == RecommendedTiming.WAIT_PULLBACK:
            return f"高周期仍偏多，但 {setup_key} 离执行区太远，先等回踩而不是追价。"
        if higher_bias == Bias.BULLISH and recommended_timing == RecommendedTiming.WAIT_CONFIRMATION:
            return f"高周期偏多，{setup_key} 已接近可用区间，但 {trigger_key} 确认还不完整。"
        if higher_bias == Bias.BEARISH and recommended_timing == RecommendedTiming.WAIT_PULLBACK:
            return f"高周期仍偏空，但 {setup_key} 离执行区太远，先等反弹而不是追空。"
        if higher_bias == Bias.BEARISH and recommended_timing == RecommendedTiming.WAIT_CONFIRMATION:
            return f"高周期偏空，{setup_key} 已接近可用区间，但 {trigger_key} 确认还不完整。"
        if not setup_assessment["aligned"]:
            return f"高周期和 {setup_key} 还没有对齐，当前更像噪音，不适合直接交易。"
        if trigger_assessment["state"] == TriggerState.NONE:
            return f"方向有了，但 {trigger_key} 还缺确认，先观望。"
        return "趋势质量和执行时机都还不够理想，先观望更稳妥。"

    def _build_diagnostics_notes(
        self,
        *,
        higher_bias: Bias,
        trend_friendly: bool,
        setup_assessment: dict[str, Any],
        trigger_assessment: dict[str, Any],
        volatility_state: VolatilityState,
        confidence: int,
        action: Action,
        setup_key: str,
        trigger_key: str,
    ) -> tuple[list[str], list[str], list[str]]:
        conflict_signals: list[str] = []
        uncertainty_notes: list[str] = []
        vetoes: list[str] = []

        if higher_bias == Bias.NEUTRAL:
            conflict_signals.append(f"{self._format_timeframe_group(tuple(self.window_config['higher_timeframes']))} 没有对齐到同一方向")
            vetoes.append("高周期环境不够清晰")
        if not setup_assessment["aligned"]:
            conflict_signals.append(f"{setup_key} 结构和高周期偏向没有对齐")
            vetoes.append(f"{setup_key} 对齐度不足")
        if not setup_assessment["structure_ready"] and higher_bias != Bias.NEUTRAL:
            uncertainty_notes.append(f"{setup_key} 虽然接近 EMA21/EMA55，但最近 swing 结构还不够理想")
        if (
            setup_assessment.get("require_free_space_gate")
            and not setup_assessment.get("free_space_ready", True)
            and setup_assessment.get("execution_ready", setup_assessment["pullback_ready"])
        ):
            uncertainty_notes.append(f"{setup_key} 已进执行区，但到高周期 swing 目标的 free space 还不够")
            vetoes.append(f"{setup_key} free space 不足")
        if (
            setup_assessment.get("require_reversal_candle")
            and not setup_assessment.get("reversal_ready", True)
            and setup_assessment["pullback_ready"]
        ):
            uncertainty_notes.append(f"{setup_key} 已进执行区，但 reversal K 线还没有明确出现")
            vetoes.append(f"{setup_key} reversal candle 未出现")
        if (
            setup_assessment.get("require_divergence_gate")
            and not setup_assessment.get("divergence_ready", True)
            and setup_assessment["pullback_ready"]
        ):
            uncertainty_notes.append(f"{setup_key} 已进执行区，但满足等级的 divergence 还没有出现")
            vetoes.append(f"{setup_key} divergence gate 未满足")
        if setup_assessment.get("state_note", {}).get("active"):
            uncertainty_notes.append(str(setup_assessment["state_note"]["message"]))
        if setup_assessment.get("divergence_enabled") and setup_assessment.get("opposing_divergence_level", 0) > 0:
            uncertainty_notes.append(
                f"{setup_key} 出现反向 divergence L{setup_assessment['opposing_divergence_level']}，当前 setup 质量不够干净"
            )
        if setup_assessment["is_extended"]:
            vetoes.append(f"价格离 {setup_key} 执行区过远")
        if trigger_assessment["state"] in {TriggerState.MIXED, TriggerState.NONE}:
            uncertainty_notes.append(f"{trigger_key} 触发还不完整或仍然混乱")
            vetoes.append(f"{trigger_key} 确认度不够")
        if volatility_state == VolatilityState.HIGH:
            uncertainty_notes.append(f"{setup_key} 波动偏高，止损距离可能被放大")
        if not trend_friendly:
            uncertainty_notes.append("当前环境不够适合激进趋势单")
        if action == Action.WAIT and confidence < int(self.config["confidence"]["action_threshold"]):
            vetoes.append(f"置信度 {confidence} 低于动作阈值 {self.config['confidence']['action_threshold']}")

        return conflict_signals, uncertainty_notes, vetoes

    def _build_timeframes_payload(self, prepared: dict[str, PreparedTimeframe]) -> TimeframesAnalysis:
        payload = {
            TIMEFRAME_TO_FIELD[timeframe]: ctx.model
            for timeframe, ctx in prepared.items()
            if timeframe in TIMEFRAME_TO_FIELD and timeframe in set(self.window_config.get("display_timeframes", ()))
        }
        return TimeframesAnalysis(**payload)

    def _build_chart_snapshots(self, prepared: dict[str, PreparedTimeframe]) -> AnalysisCharts:
        payload = {
            TIMEFRAME_TO_FIELD[timeframe]: self._build_timeframe_chart(timeframe, ctx.df)
            for timeframe, ctx in prepared.items()
            if timeframe in TIMEFRAME_TO_FIELD and timeframe in set(self.window_config.get("chart_timeframes", ()))
        }
        return AnalysisCharts(**payload)

    def _build_timeframe_chart(self, timeframe: str, df: pd.DataFrame) -> TimeframeChart:
        candle_count = int(self.config["chart"]["candles"])
        window = df.tail(candle_count).copy()
        candles = [
            ChartCandle(
                timestamp=row["timestamp"].to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for _, row in window.iterrows()
        ]
        return TimeframeChart(
            timeframe=timeframe,
            candles=candles,
            ema21=[ChartLinePoint(timestamp=row["timestamp"].to_pydatetime(), value=float(row["ema_21"])) for _, row in window.iterrows()],
            ema55=[ChartLinePoint(timestamp=row["timestamp"].to_pydatetime(), value=float(row["ema_55"])) for _, row in window.iterrows()],
            ema100=[ChartLinePoint(timestamp=row["timestamp"].to_pydatetime(), value=float(row["ema_100"])) for _, row in window.iterrows()],
            ema200=[ChartLinePoint(timestamp=row["timestamp"].to_pydatetime(), value=float(row["ema_200"])) for _, row in window.iterrows()],
        )

    @staticmethod
    def _distance_to_zone_atr(close: float, low: float, high: float, atr: float) -> float:
        if low <= close <= high:
            return 0.0
        nearest_edge = min(abs(close - low), abs(close - high))
        return nearest_edge / atr if atr else 0.0

    @staticmethod
    def _format_timeframe_group(timeframes: tuple[str, ...]) -> str:
        return " / ".join(timeframes)
