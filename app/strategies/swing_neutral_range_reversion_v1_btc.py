from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pandas as pd

from app.schemas.analysis import (
    AnalysisDiagnostics,
    AnalysisResult,
    Decision,
    EntryZone,
    MarketRegime,
    Reasoning,
    ScoreBreakdown,
    ScoreContribution,
    SetupQuality,
    StopLoss,
    TakeProfitHint,
    TriggerMaturity,
)
from app.schemas.common import Action, Bias, RecommendedTiming, TriggerState, VolatilityState
from app.schemas.request import AnalyzeRequest
from app.strategies.scoring import ScoreCard
from app.strategies.windowed_mtf import COMMON_DEFAULT_CONFIG, PreparedTimeframe, WindowedMTFStrategy
from app.utils.timeframes import validate_required_timeframes


DEFAULT_CONFIG = {
    **COMMON_DEFAULT_CONFIG,
    "setup": {
        **COMMON_DEFAULT_CONFIG["setup"],
        "require_structure_ready": True,
        "require_reversal_candle": True,
    },
    "neutral_range": {
        "max_reference_trend_score": 60,
        "max_day_trend_score": 76,
        "lookback_bars": 18,
        "min_width_atr": 2.0,
        "max_width_atr": 5.8,
        "edge_proximity_atr": 0.4,
        "sweep_buffer_atr": 0.12,
        "stop_buffer_atr": 0.15,
        "entry_pad_atr": 0.06,
        "minimum_opposite_edge_r": 1.15,
        "minimum_r_multiple": 0.75,
    },
    "confidence": {"action_threshold": 60},
    "window": {
        "higher_timeframes": ["1d", "4h"],
        "setup_timeframe": "1h",
        "trigger_timeframe": "1h",
        "reference_timeframe": "4h",
        "display_timeframes": ["1d", "4h", "1h"],
        "chart_timeframes": ["1d", "4h", "1h"],
        "trend_strength_threshold": 0,
    },
}


class SwingNeutralRangeReversionV1BTCStrategy(WindowedMTFStrategy):
    name = "swing_neutral_range_reversion_v1_btc"
    required_timeframes = ("1d", "4h", "1h")

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

        day_ctx = prepared["1d"]
        reference_ctx = prepared["4h"]
        setup_ctx = prepared["1h"]
        trigger_ctx = prepared["1h"]
        setup_key = "1h"
        trigger_key = "1h"

        volatility_state = self._derive_volatility_state(setup_ctx)
        regime_assessment = self._assess_regime(day_ctx=day_ctx, reference_ctx=reference_ctx)
        regime_ready = regime_assessment["ready"] and volatility_state != VolatilityState.HIGH
        trend_strength = regime_assessment["composite_trend_score"]

        scorecard = ScoreCard(base=50)
        reasons_for: list[str] = []
        reasons_against: list[str] = []
        risk_notes: list[str] = []

        scorecard.add(regime_assessment["score"], "neutral_regime", regime_assessment["score_note"])
        reasons_for.extend(regime_assessment["reasons_for"])
        reasons_against.extend(regime_assessment["reasons_against"])
        risk_notes.extend(regime_assessment["risk_notes"])

        candidate_bias = self._derive_candidate_bias(setup_ctx, trigger_ctx.model.close)
        setup_assessment = self._assess_setup(
            setup_ctx=setup_ctx,
            setup_key=setup_key,
            regime_ready=regime_ready,
            current_price=trigger_ctx.model.close,
            candidate_bias_override=candidate_bias,
        )
        scorecard.add(setup_assessment["score"], f"{setup_key}_setup", setup_assessment["score_note"])
        reasons_for.extend(setup_assessment["reasons_for"])
        reasons_against.extend(setup_assessment["reasons_against"])
        risk_notes.extend(setup_assessment["risk_notes"])

        trigger_assessment = self._assess_trigger(
            candidate_bias=setup_assessment["candidate_bias"],
            trigger_ctx=trigger_ctx,
            trigger_key=trigger_key,
            setup_assessment=setup_assessment,
        )
        trigger_ctx.model.trigger_state = trigger_assessment["state"]
        scorecard.add(trigger_assessment["score"], f"{trigger_key}_trigger", trigger_assessment["score_note"])
        reasons_for.extend(trigger_assessment["reasons_for"])
        reasons_against.extend(trigger_assessment["reasons_against"])
        risk_notes.extend(trigger_assessment["risk_notes"])

        if volatility_state == VolatilityState.HIGH:
            scorecard.add(-12, "volatility", f"{setup_key} ATR 百分比偏高")
            risk_notes.append(f"{setup_key} 波动偏高，均值回归单容易被假突破继续放大。")
        elif volatility_state == VolatilityState.LOW:
            scorecard.add(4, "volatility", f"{setup_key} 波动收缩，区间回归环境更可控")

        confidence = scorecard.total
        action, bias, recommended_timing = self._decide(
            regime_ready=regime_ready,
            candidate_bias=setup_assessment["candidate_bias"],
            setup_assessment=setup_assessment,
            trigger_assessment=trigger_assessment,
            confidence=confidence,
        )
        conflict_signals, uncertainty_notes, vetoes = self._build_custom_diagnostics(
            regime_ready=regime_ready,
            setup_assessment=setup_assessment,
            trigger_assessment=trigger_assessment,
            volatility_state=volatility_state,
            confidence=confidence,
            action=action,
        )

        trade_plan = self._build_trade_plan(
            action=action,
            bias=bias,
            setup_ctx=setup_ctx,
            current_price=trigger_ctx.model.close,
            setup_key=setup_key,
            setup_assessment=setup_assessment,
        )

        return AnalysisResult(
            analysis_id=uuid4().hex,
            timestamp=datetime.now(timezone.utc),
            symbol=request.symbol,
            exchange=request.exchange,
            market_type=request.market_type,
            strategy_profile=request.strategy_profile,
            timeframes=self._build_timeframes_payload(prepared),
            charts=self._build_chart_snapshots(prepared),
            market_regime=MarketRegime(
                higher_timeframe_bias=Bias.NEUTRAL,
                trend_strength=trend_strength,
                volatility_state=volatility_state,
                is_trend_friendly=regime_ready,
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
                summary=self._build_summary(
                    action=action,
                    recommended_timing=recommended_timing,
                    setup_assessment=setup_assessment,
                    trigger_assessment=trigger_assessment,
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
                setup_quality=SetupQuality(
                    setup_timeframe=setup_key,
                    higher_timeframe_bias=Bias.NEUTRAL,
                    trend_friendly=regime_ready,
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
                "regime_assessment": {
                    key: value
                    for key, value in regime_assessment.items()
                    if key not in {"reasons_for", "reasons_against", "risk_notes"}
                },
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

    def _derive_higher_timeframe_bias(self, prepared: dict[str, PreparedTimeframe]) -> tuple[Bias, int]:
        regime_assessment = self._assess_regime(day_ctx=prepared["1d"], reference_ctx=prepared["4h"])
        if not regime_assessment["ready"]:
            return Bias.NEUTRAL, regime_assessment["composite_trend_score"]
        candidate_bias = self._derive_candidate_bias(prepared["1h"], prepared["1h"].model.close)
        return candidate_bias, regime_assessment["composite_trend_score"]

    def _is_trend_friendly(
        self,
        *,
        higher_bias: Bias,
        trend_strength: int,
        volatility_state: VolatilityState,
    ) -> bool:
        cfg = dict(self.config.get("neutral_range", {}))
        composite_max = int(round((int(cfg.get("max_reference_trend_score", 60)) + int(cfg.get("max_day_trend_score", 76))) / 2))
        return higher_bias != Bias.NEUTRAL and volatility_state != VolatilityState.HIGH and trend_strength <= composite_max

    def _completed_window(self, ctx: PreparedTimeframe, lookback: int) -> pd.DataFrame:
        if len(ctx.df) <= 1:
            return ctx.df.iloc[0:0]
        start = max(0, len(ctx.df) - lookback - 1)
        return ctx.df.iloc[start:-1]

    def _range_profile(self, ctx: PreparedTimeframe) -> dict[str, float | None]:
        cfg = dict(self.config.get("neutral_range", {}))
        window = self._completed_window(ctx, int(cfg.get("lookback_bars", 18)))
        if window.empty:
            return {"high": None, "low": None, "mid": None, "width_atr": None}
        range_high = float(window["high"].max())
        range_low = float(window["low"].min())
        atr = float(ctx.model.atr14)
        width_atr = ((range_high - range_low) / atr) if atr else None
        return {
            "high": range_high,
            "low": range_low,
            "mid": (range_high + range_low) / 2,
            "width_atr": width_atr,
        }

    def _derive_candidate_bias(self, setup_ctx: PreparedTimeframe, current_price: float) -> Bias:
        cfg = dict(self.config.get("neutral_range", {}))
        edge_proximity_atr = float(cfg.get("edge_proximity_atr", 0.4))
        range_profile = self._range_profile(setup_ctx)
        if range_profile["low"] is None or range_profile["high"] is None:
            return Bias.NEUTRAL
        atr = float(setup_ctx.model.atr14)
        if atr <= 0:
            return Bias.NEUTRAL
        lower_edge_distance_atr = (current_price - float(range_profile["low"])) / atr
        upper_edge_distance_atr = (float(range_profile["high"]) - current_price) / atr
        near_lower_edge = 0.0 <= lower_edge_distance_atr <= edge_proximity_atr
        near_upper_edge = 0.0 <= upper_edge_distance_atr <= edge_proximity_atr
        if near_lower_edge and not near_upper_edge:
            return Bias.BULLISH
        if near_upper_edge and not near_lower_edge:
            return Bias.BEARISH
        return Bias.NEUTRAL

    def _assess_regime(
        self,
        *,
        day_ctx: PreparedTimeframe,
        reference_ctx: PreparedTimeframe,
    ) -> dict[str, Any]:
        cfg = dict(self.config.get("neutral_range", {}))
        max_reference_trend_score = int(cfg.get("max_reference_trend_score", 60))
        max_day_trend_score = int(cfg.get("max_day_trend_score", 76))
        composite_trend_score = int(round((day_ctx.model.trend_score + reference_ctx.model.trend_score) / 2))

        reasons_for: list[str] = []
        reasons_against: list[str] = []
        risk_notes: list[str] = []

        reference_neutral = (
            reference_ctx.model.trend_bias == Bias.NEUTRAL
            and reference_ctx.model.structure_state == "mixed"
            and int(reference_ctx.model.trend_score) <= max_reference_trend_score
        )
        day_not_extreme = int(day_ctx.model.trend_score) <= max_day_trend_score

        if reference_neutral:
            reasons_for.append(
                f"4H 维持 neutral / mixed 结构，趋势分 {reference_ctx.model.trend_score}，更像区间而不是单边。"
            )
        else:
            reasons_against.append(
                f"4H 还没有回到足够中性的结构，当前趋势分 {reference_ctx.model.trend_score}。"
            )
        if day_not_extreme:
            reasons_for.append(f"1D 趋势分 {day_ctx.model.trend_score} 没有强到直接碾压 1H 均值回归。")
        else:
            reasons_against.append(f"1D 趋势分 {day_ctx.model.trend_score} 偏强，区间单容易被更大级别单边打穿。")
            risk_notes.append("上层趋势仍然偏强，均值回归只适合做轻量研究，不适合激进扩张。")

        ready = reference_neutral and day_not_extreme
        score = 14 if ready else -16

        return {
            "ready": ready,
            "composite_trend_score": composite_trend_score,
            "score": score,
            "score_note": "这条策略只在 4H neutral regime 下工作，不在强趋势里硬做均值回归。",
            "reasons_for": reasons_for,
            "reasons_against": reasons_against,
            "risk_notes": risk_notes,
        }

    def _assess_setup(
        self,
        higher_bias: Bias | None = None,
        setup_ctx: PreparedTimeframe | None = None,
        setup_key: str | None = None,
        *,
        reference_ctx: PreparedTimeframe | None = None,
        current_price: float | None = None,
        regime_ready: bool | None = None,
        candidate_bias_override: Bias | None = None,
    ) -> dict[str, Any]:
        if setup_ctx is None or setup_key is None:
            raise ValueError("setup_ctx and setup_key are required")
        if current_price is None:
            current_price = setup_ctx.model.close
        if regime_ready is None:
            regime_ready = higher_bias != Bias.NEUTRAL if higher_bias is not None else False
        cfg = dict(self.config.get("neutral_range", {}))
        range_profile = self._range_profile(setup_ctx)
        range_low = range_profile["low"]
        range_high = range_profile["high"]
        range_mid = range_profile["mid"]
        width_atr = range_profile["width_atr"]
        atr = float(setup_ctx.model.atr14)
        min_width_atr = float(cfg.get("min_width_atr", 2.0))
        max_width_atr = float(cfg.get("max_width_atr", 5.8))
        edge_proximity_atr = float(cfg.get("edge_proximity_atr", 0.4))
        stop_buffer = float(cfg.get("stop_buffer_atr", 0.15)) * atr
        minimum_r_multiple = float(cfg.get("minimum_r_multiple", 0.75)) * atr
        minimum_opposite_edge_r = float(cfg.get("minimum_opposite_edge_r", 1.15))

        reasons_for: list[str] = []
        reasons_against: list[str] = []
        risk_notes: list[str] = []

        if not regime_ready or range_low is None or range_high is None or range_mid is None or width_atr is None:
            return {
                "aligned": False,
                "pullback_ready": False,
                "is_extended": False,
                "distance_to_value_atr": 0.0,
                "candidate_bias": Bias.NEUTRAL,
                "range_low": range_low,
                "range_high": range_high,
                "range_mid": range_mid,
                "structure_ready": False,
                "space_ready": False,
                "opposite_edge_r": None,
                "score": -12,
                "score_note": f"{setup_key} 必须先有 neutral regime 和可交易区间，才谈边缘均值回归。",
                "reasons_for": reasons_for,
                "reasons_against": ["当前 neutral regime 或 1H 区间结构还不够清晰。"],
                "risk_notes": risk_notes,
            }

        structure_ready = min_width_atr <= float(width_atr) <= max_width_atr
        in_range = float(range_low) <= current_price <= float(range_high)
        lower_edge_distance_atr = ((current_price - float(range_low)) / atr) if atr else None
        upper_edge_distance_atr = ((float(range_high) - current_price) / atr) if atr else None

        near_lower_edge = (
            lower_edge_distance_atr is not None
            and 0.0 <= lower_edge_distance_atr <= edge_proximity_atr
        )
        near_upper_edge = (
            upper_edge_distance_atr is not None
            and 0.0 <= upper_edge_distance_atr <= edge_proximity_atr
        )

        candidate_bias = Bias.NEUTRAL
        estimated_stop = None
        opposite_edge_r = None
        edge_distance = min(
            abs(float(lower_edge_distance_atr or 999.0)),
            abs(float(upper_edge_distance_atr or 999.0)),
        )

        if candidate_bias_override is not None:
            candidate_bias = candidate_bias_override
        elif higher_bias in {Bias.BULLISH, Bias.BEARISH}:
            candidate_bias = higher_bias
        elif near_lower_edge and not near_upper_edge:
            candidate_bias = Bias.BULLISH
        elif near_upper_edge and not near_lower_edge:
            candidate_bias = Bias.BEARISH
        else:
            candidate_bias = Bias.NEUTRAL

        if candidate_bias == Bias.BULLISH:
            estimated_stop = float(range_low) - stop_buffer
            risk = max(current_price - estimated_stop, minimum_r_multiple)
            opposite_edge_r = ((float(range_high) - current_price) / risk) if risk else None
            reasons_for.append(f"{setup_key} 已贴近区间下沿，接下来只看 failed breakdown 后的回归。")
        elif candidate_bias == Bias.BEARISH:
            estimated_stop = float(range_high) + stop_buffer
            risk = max(estimated_stop - current_price, minimum_r_multiple)
            opposite_edge_r = ((current_price - float(range_low)) / risk) if risk else None
            reasons_for.append(f"{setup_key} 已贴近区间上沿，接下来只看 failed breakout 后的回归。")
        else:
            reasons_against.append(f"{setup_key} 当前位置不在 1H 区间边缘，做均值回归没有赔率优势。")

        if structure_ready:
            reasons_for.append(f"{setup_key} 最近区间宽度约 {round(float(width_atr), 2)}ATR，仍属于可回归的 compact range。")
        else:
            reasons_against.append(f"{setup_key} 最近区间宽度约 {round(float(width_atr), 2)}ATR，太窄或太宽都不适合做回归。")
        if not in_range:
            reasons_against.append(f"{setup_key} 当前收盘已经离开区间内部，先等回到边缘内侧再说。")

        space_ready = opposite_edge_r is not None and opposite_edge_r >= minimum_opposite_edge_r
        if candidate_bias != Bias.NEUTRAL and opposite_edge_r is not None:
            if space_ready:
                reasons_for.append(
                    f"从当前价到区间对侧大约还有 {round(float(opposite_edge_r), 2)}R，赔率不算太薄。"
                )
            else:
                reasons_against.append(
                    f"到区间对侧只剩 {round(float(opposite_edge_r), 2)}R，均值回归赔率不足。"
                )

        execution_ready = structure_ready and in_range and candidate_bias != Bias.NEUTRAL and space_ready
        score = 15 if execution_ready else 6 if structure_ready and candidate_bias != Bias.NEUTRAL else -10
        if not structure_ready:
            score -= 4
        if not in_range:
            score -= 4
        if candidate_bias == Bias.NEUTRAL:
            score -= 6
        if candidate_bias != Bias.NEUTRAL and not space_ready:
            score -= 5

        return {
            "aligned": regime_ready and candidate_bias != Bias.NEUTRAL,
            "pullback_ready": execution_ready,
            "is_extended": not in_range,
            "distance_to_value_atr": round(edge_distance if edge_distance < 999.0 else 0.0, 4),
            "candidate_bias": candidate_bias,
            "range_low": round(float(range_low), 4),
            "range_high": round(float(range_high), 4),
            "range_mid": round(float(range_mid), 4),
            "estimated_stop": round(float(estimated_stop), 4) if estimated_stop is not None else None,
            "structure_ready": structure_ready,
            "space_ready": space_ready,
            "opposite_edge_r": round(float(opposite_edge_r), 4) if opposite_edge_r is not None else None,
            "score": score,
            "score_note": f"{setup_key} 只在 compact range 边缘寻找均值回归，不在区间中部做方向猜测。",
            "reasons_for": reasons_for,
            "reasons_against": reasons_against,
            "risk_notes": risk_notes,
        }

    def _assess_trigger(
        self,
        higher_bias: Bias | None = None,
        trigger_ctx: PreparedTimeframe | None = None,
        trigger_key: str | None = None,
        *,
        trend_strength: int | None = None,
        candidate_bias: Bias | None = None,
        setup_assessment: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if trigger_ctx is None or trigger_key is None:
            raise ValueError("trigger_ctx and trigger_key are required")
        if candidate_bias is None:
            candidate_bias = higher_bias if higher_bias in {Bias.BULLISH, Bias.BEARISH} else Bias.NEUTRAL
        if setup_assessment is None:
            range_profile = self._range_profile(trigger_ctx)
            setup_assessment = {
                "range_low": round(float(range_profile["low"]), 4) if range_profile["low"] is not None else None,
                "range_high": round(float(range_profile["high"]), 4) if range_profile["high"] is not None else None,
            }
        range_low = setup_assessment.get("range_low")
        range_high = setup_assessment.get("range_high")
        latest = trigger_ctx.df.iloc[-1]
        atr = float(trigger_ctx.model.atr14)
        sweep_buffer = float(self.config.get("neutral_range", {}).get("sweep_buffer_atr", 0.12)) * atr
        reasons_for: list[str] = []
        reasons_against: list[str] = []
        risk_notes: list[str] = []

        if candidate_bias == Bias.NEUTRAL or range_low is None or range_high is None:
            return {
                "state": TriggerState.NONE,
                "score": -10,
                "score_note": f"{trigger_key} 还没有靠到区间边缘，因此不存在可执行的回归触发。",
                "reasons_for": reasons_for,
                "reasons_against": [f"{trigger_key} 当前没有形成边缘扫单回收。"],
                "risk_notes": risk_notes,
            }

        close = float(latest["close"])
        low = float(latest["low"])
        high = float(latest["high"])
        ema21 = float(latest["ema_21"])

        if candidate_bias == Bias.BULLISH:
            swept = low <= float(range_low) - sweep_buffer
            reclaimed = close >= float(range_low)
            rejection = bool(trigger_ctx.candle_profile.get("has_bullish_rejection", False))
            regained_fast = close >= ema21

            if swept:
                reasons_for.append(f"{trigger_key} 先扫过区间下沿下方流动性。")
            else:
                reasons_against.append(f"{trigger_key} 还没有真正扫到区间下沿外侧。")
            if reclaimed:
                reasons_for.append(f"{trigger_key} 收盘重新站回区间内。")
            else:
                reasons_against.append(f"{trigger_key} 还没有收回区间下沿。")
            if rejection:
                reasons_for.append(f"{trigger_key} 出现下影 rejection。")
            else:
                reasons_against.append(f"{trigger_key} 缺少干净的下影 rejection。")
            if swept and reclaimed and rejection:
                state = TriggerState.BULLISH_CONFIRMED
                score = 16
            elif swept and reclaimed:
                state = TriggerState.MIXED
                score = 2
                risk_notes.append(f"{trigger_key} 已回到区间，但 rejection 还不够干净。")
            else:
                state = TriggerState.NONE
                score = -12
        else:
            swept = high >= float(range_high) + sweep_buffer
            reclaimed = close <= float(range_high)
            rejection = bool(trigger_ctx.candle_profile.get("has_bearish_rejection", False))
            regained_fast = close <= ema21

            if swept:
                reasons_for.append(f"{trigger_key} 先扫过区间上沿上方流动性。")
            else:
                reasons_against.append(f"{trigger_key} 还没有真正扫到区间上沿外侧。")
            if reclaimed:
                reasons_for.append(f"{trigger_key} 收盘重新压回区间内。")
            else:
                reasons_against.append(f"{trigger_key} 还没有收回区间上沿。")
            if rejection:
                reasons_for.append(f"{trigger_key} 出现上影 rejection。")
            else:
                reasons_against.append(f"{trigger_key} 缺少干净的上影 rejection。")
            if swept and reclaimed and rejection:
                state = TriggerState.BEARISH_CONFIRMED
                score = 16
            elif swept and reclaimed:
                state = TriggerState.MIXED
                score = 2
                risk_notes.append(f"{trigger_key} 已回到区间，但 rejection 还不够干净。")
            else:
                state = TriggerState.NONE
                score = -12

        return {
            "state": state,
            "score": score,
            "score_note": f"{trigger_key} 需要先扫出区间边缘，再收回并给出 rejection K 线。",
            "reasons_for": reasons_for,
            "reasons_against": reasons_against,
            "risk_notes": risk_notes,
            "regained_fast": regained_fast,
        }

    def _decide(
        self,
        *,
        higher_bias: Bias | None = None,
        trend_friendly: bool | None = None,
        setup_assessment: dict[str, Any],
        trigger_assessment: dict[str, Any],
        confidence: int,
        regime_ready: bool | None = None,
        candidate_bias: Bias | None = None,
    ) -> tuple[Action, Bias, RecommendedTiming]:
        if candidate_bias is None:
            candidate_bias = higher_bias or Bias.NEUTRAL
        if regime_ready is None:
            regime_ready = bool(trend_friendly)
        threshold = int(self.config["confidence"]["action_threshold"])
        if not regime_ready:
            return Action.WAIT, Bias.NEUTRAL, RecommendedTiming.SKIP
        if candidate_bias == Bias.NEUTRAL:
            return Action.WAIT, Bias.NEUTRAL, RecommendedTiming.WAIT_PULLBACK
        if not setup_assessment["pullback_ready"]:
            return Action.WAIT, candidate_bias, RecommendedTiming.WAIT_PULLBACK

        if (
            candidate_bias == Bias.BULLISH
            and trigger_assessment["state"] == TriggerState.BULLISH_CONFIRMED
            and confidence >= threshold
        ):
            return Action.LONG, Bias.BULLISH, RecommendedTiming.NOW
        if (
            candidate_bias == Bias.BEARISH
            and trigger_assessment["state"] == TriggerState.BEARISH_CONFIRMED
            and confidence >= threshold
        ):
            return Action.SHORT, Bias.BEARISH, RecommendedTiming.NOW
        if trigger_assessment["state"] == TriggerState.MIXED:
            return Action.WAIT, candidate_bias, RecommendedTiming.WAIT_CONFIRMATION
        return Action.WAIT, candidate_bias, RecommendedTiming.SKIP

    def _build_trade_plan(
        self,
        *,
        action: Action,
        bias: Bias,
        setup_ctx: PreparedTimeframe,
        reference_ctx: PreparedTimeframe | None = None,
        current_price: float,
        setup_key: str,
        reference_key: str | None = None,
        setup_assessment: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if setup_assessment is None:
            setup_assessment = self._assess_setup(
                higher_bias=bias,
                setup_ctx=setup_ctx,
                setup_key=setup_key,
                current_price=current_price,
                regime_ready=bias != Bias.NEUTRAL,
                candidate_bias_override=bias,
            )
        if action == Action.WAIT:
            if bias == Bias.BULLISH:
                return {
                    "entry_zone": None,
                    "stop_loss": None,
                    "take_profit_hint": None,
                    "invalidation": f"当前仍在等 {setup_key} 下沿扫单后重新收回。",
                    "invalidation_price": None,
                }
            if bias == Bias.BEARISH:
                return {
                    "entry_zone": None,
                    "stop_loss": None,
                    "take_profit_hint": None,
                    "invalidation": f"当前仍在等 {setup_key} 上沿扫单后重新压回。",
                    "invalidation_price": None,
                }
            return {
                "entry_zone": None,
                "stop_loss": None,
                "take_profit_hint": None,
                "invalidation": "当前 4H 还不是适合做区间均值回归的 neutral regime。",
                "invalidation_price": None,
            }

        cfg = dict(self.config.get("neutral_range", {}))
        latest = setup_ctx.df.iloc[-1]
        atr = float(setup_ctx.model.atr14)
        stop_buffer = float(cfg.get("stop_buffer_atr", 0.15)) * atr
        entry_pad = float(cfg.get("entry_pad_atr", 0.06)) * atr
        minimum_risk = float(cfg.get("minimum_r_multiple", 0.75)) * atr
        range_low = float(setup_assessment["range_low"])
        range_high = float(setup_assessment["range_high"])
        range_mid = float(setup_assessment["range_mid"])

        if bias == Bias.BULLISH:
            stop_price = min(float(latest["low"]), range_low) - stop_buffer
            risk = max(current_price - stop_price, minimum_risk)
            tp1 = max(range_mid, current_price + (0.8 * risk))
            tp2 = max(range_high, current_price + (1.4 * risk))
            return {
                "entry_zone": EntryZone(
                    low=round(current_price - entry_pad, 4),
                    high=round(current_price + entry_pad, 4),
                    basis=f"{setup_key} 下沿回收带",
                ),
                "stop_loss": StopLoss(
                    price=round(stop_price, 4),
                    basis=f"{setup_key} 假跌破低点下方加 ATR 缓冲",
                ),
                "take_profit_hint": TakeProfitHint(
                    tp1=round(tp1, 4),
                    tp2=round(tp2, 4),
                    basis="先看区间中轴，再看区间上沿",
                ),
                "invalidation": f"若 {setup_key} 再次失守区间下沿并延伸，均值回归逻辑失效",
                "invalidation_price": round(range_low, 4),
            }

        stop_price = max(float(latest["high"]), range_high) + stop_buffer
        risk = max(stop_price - current_price, minimum_risk)
        tp1 = min(range_mid, current_price - (0.8 * risk))
        tp2 = min(range_low, current_price - (1.4 * risk))
        return {
            "entry_zone": EntryZone(
                low=round(current_price - entry_pad, 4),
                high=round(current_price + entry_pad, 4),
                basis=f"{setup_key} 上沿回压带",
            ),
            "stop_loss": StopLoss(
                price=round(stop_price, 4),
                basis=f"{setup_key} 假突破高点上方加 ATR 缓冲",
            ),
            "take_profit_hint": TakeProfitHint(
                tp1=round(tp1, 4),
                tp2=round(tp2, 4),
                basis="先看区间中轴，再看区间下沿",
            ),
            "invalidation": f"若 {setup_key} 再次站稳区间上沿并延伸，均值回归逻辑失效",
            "invalidation_price": round(range_high, 4),
        }

    def _build_custom_diagnostics(
        self,
        *,
        regime_ready: bool,
        setup_assessment: dict[str, Any],
        trigger_assessment: dict[str, Any],
        volatility_state: VolatilityState,
        confidence: int,
        action: Action,
    ) -> tuple[list[str], list[str], list[str]]:
        conflict_signals: list[str] = []
        uncertainty_notes: list[str] = []
        vetoes: list[str] = []

        if not regime_ready:
            conflict_signals.append("4H 还不是足够中性的区间环境。")
            vetoes.append("neutral regime 未成立")
        if setup_assessment["candidate_bias"] == Bias.NEUTRAL:
            uncertainty_notes.append("当前价格不在 1H 区间边缘，均值回归赔率不足。")
            vetoes.append("不在区间边缘")
        if not setup_assessment["structure_ready"]:
            uncertainty_notes.append("1H 区间过窄或过宽，均值回归结构不够稳定。")
            vetoes.append("区间结构不理想")
        if setup_assessment["candidate_bias"] != Bias.NEUTRAL and not setup_assessment["space_ready"]:
            uncertainty_notes.append("虽然靠边，但到对侧的空间太薄。")
            vetoes.append("对侧空间不足")
        if trigger_assessment["state"] in {TriggerState.MIXED, TriggerState.NONE}:
            uncertainty_notes.append("边缘扫单有雏形，但回收和 rejection 还不完整。")
            vetoes.append("触发未确认")
        if volatility_state == VolatilityState.HIGH:
            uncertainty_notes.append("当前 1H 波动过高，均值回归更容易演化成单边扩展。")
            vetoes.append("波动过高")
        if action == Action.WAIT and confidence < int(self.config["confidence"]["action_threshold"]):
            vetoes.append(f"置信度 {confidence} 低于阈值 {self.config['confidence']['action_threshold']}")

        return conflict_signals, uncertainty_notes, vetoes

    def _build_summary(
        self,
        *,
        action: Action,
        recommended_timing: RecommendedTiming,
        setup_assessment: dict[str, Any],
        trigger_assessment: dict[str, Any],
        trigger_key: str,
    ) -> str:
        if action == Action.LONG:
            return f"4H 处于 neutral regime，{trigger_key} 下沿扫单后收回，因此按区间均值回归做多。"
        if action == Action.SHORT:
            return f"4H 处于 neutral regime，{trigger_key} 上沿扫单后压回，因此按区间均值回归做空。"
        if setup_assessment["candidate_bias"] == Bias.NEUTRAL:
            return f"{trigger_key} 当前位置不在区间边缘，先等靠边而不是在中部猜方向。"
        if recommended_timing == RecommendedTiming.WAIT_PULLBACK:
            return f"{trigger_key} 已接近边缘，但区间结构或赔率还不够理想。"
        if trigger_assessment["state"] == TriggerState.MIXED:
            return f"{trigger_key} 已有扫单回收雏形，但 rejection 还不够干净。"
        return "neutral regime 有轮廓，但执行条件还不完整，先观望。"
