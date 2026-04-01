from __future__ import annotations

from typing import Any

from app.schemas.analysis import EntryZone, StopLoss, TakeProfitHint
from app.schemas.common import Action, Bias, RecommendedTiming, StructureState, TriggerState, VolatilityState
from app.strategies.windowed_mtf import COMMON_DEFAULT_CONFIG, PreparedTimeframe, WindowedMTFStrategy


DEFAULT_CONFIG = {
    **COMMON_DEFAULT_CONFIG,
    "setup": {
        **COMMON_DEFAULT_CONFIG["setup"],
        "require_structure_ready": True,
        "require_reversal_candle": False,
    },
    "breakout": {
        "setup_lookback_bars": 20,
        "setup_proximity_atr": 0.55,
        "setup_extension_atr": 1.0,
        "trigger_lookback_bars": 12,
        "trigger_breakout_buffer_atr": 0.08,
        "trigger_extension_atr": 1.2,
        "base_lookback_bars": 8,
        "base_max_width_atr": 6.0,
        "range_expansion_ratio_min": 1.05,
        "quote_volume_ratio_min": 1.0,
    },
    "risk": {
        **COMMON_DEFAULT_CONFIG["risk"],
        "atr_buffer": 0.2,
        "minimum_r_multiple": 1.0,
    },
    "confidence": {"action_threshold": 64},
    "window": {
        "higher_timeframes": ["1d", "4h"],
        "setup_timeframe": "4h",
        "trigger_timeframe": "1h",
        "reference_timeframe": "1h",
        "display_timeframes": ["1d", "4h", "1h"],
        "chart_timeframes": ["1d", "4h", "1h"],
        "trend_strength_threshold": 65,
        "bullish_trend_strength_threshold": 72,
        "bearish_trend_strength_threshold": 68,
    },
}


class SwingBreakoutV1BTCStrategy(WindowedMTFStrategy):
    name = "swing_breakout_v1_btc"
    required_timeframes = ("1d", "4h", "1h")

    def _completed_window(self, ctx: PreparedTimeframe, lookback: int):
        df = ctx.df
        if len(df) <= 1:
            return df.iloc[0:0]
        start = max(0, len(df) - lookback - 1)
        return df.iloc[start:-1]

    def _window_extremes(self, ctx: PreparedTimeframe, lookback: int) -> dict[str, float | None]:
        window = self._completed_window(ctx, lookback)
        if window.empty:
            return {"high": None, "low": None}
        return {
            "high": float(window["high"].max()),
            "low": float(window["low"].min()),
        }

    def _base_profile(self, ctx: PreparedTimeframe, lookback: int) -> dict[str, float | None]:
        window = self._completed_window(ctx, lookback)
        if window.empty:
            return {"high": None, "low": None, "width_atr": None}
        base_high = float(window["high"].max())
        base_low = float(window["low"].min())
        atr = float(ctx.model.atr14)
        width_atr = ((base_high - base_low) / atr) if atr else None
        return {"high": base_high, "low": base_low, "width_atr": width_atr}

    def _setup_breakout_level(self, higher_bias: Bias, setup_ctx: PreparedTimeframe) -> float | None:
        lookback = int(self.config["breakout"]["setup_lookback_bars"])
        extremes = self._window_extremes(setup_ctx, lookback)
        return extremes["high"] if higher_bias == Bias.BULLISH else extremes["low"]

    def _trigger_breakout_level(self, higher_bias: Bias, trigger_ctx: PreparedTimeframe) -> float | None:
        lookback = int(self.config["breakout"]["trigger_lookback_bars"])
        extremes = self._window_extremes(trigger_ctx, lookback)
        return extremes["high"] if higher_bias == Bias.BULLISH else extremes["low"]

    def _assess_setup(
        self,
        higher_bias: Bias,
        setup_ctx: PreparedTimeframe,
        setup_key: str,
        *,
        reference_ctx: PreparedTimeframe | None = None,
        current_price: float | None = None,
    ) -> dict[str, Any]:
        reasons_for: list[str] = []
        reasons_against: list[str] = []
        risk_notes: list[str] = []
        current_price = float(current_price if current_price is not None else setup_ctx.model.close)

        if higher_bias == Bias.NEUTRAL:
            return {
                "aligned": False,
                "execution_ready": False,
                "pullback_ready": False,
                "pressure_ready": False,
                "reversal_ready": True,
                "require_reversal_candle": False,
                "divergence_enabled": False,
                "require_divergence_gate": False,
                "require_free_space_gate": False,
                "divergence_ready": False,
                "divergence_level": 0,
                "opposing_divergence_level": 0,
                "free_space_ready": True,
                "free_space_r": None,
                "free_space_min_r": 0.0,
                "is_extended": False,
                "distance_to_value_atr": 0.0,
                "structure_ready": False,
                "score": -12,
                "score_note": f"{setup_key} 突破策略要求高周期方向清晰",
                "reasons_for": reasons_for,
                "reasons_against": [f"高周期方向没有统一前，{setup_key} 的突破压力没有意义"],
                "risk_notes": risk_notes,
            }

        breakout_cfg = dict(self.config.get("breakout", {}))
        setup_level = self._setup_breakout_level(higher_bias, setup_ctx)
        base = self._base_profile(setup_ctx, int(breakout_cfg.get("base_lookback_bars", 8)))
        proximity_atr = float(breakout_cfg.get("setup_proximity_atr", 0.55))
        extension_atr = float(breakout_cfg.get("setup_extension_atr", 1.0))
        base_max_width_atr = float(breakout_cfg.get("base_max_width_atr", 6.0))
        atr = float(setup_ctx.model.atr14)

        if higher_bias == Bias.BULLISH:
            aligned = (
                setup_ctx.model.close > setup_ctx.model.ema200
                and setup_ctx.model.ema_alignment != "bearish"
                and setup_ctx.model.structure_state != StructureState.BEARISH
            )
            pressure_distance_atr = (
                (float(setup_level) - current_price) / atr if setup_level is not None and atr else None
            )
            overrun_atr = (
                (current_price - float(setup_level)) / atr if setup_level is not None and atr else None
            )
            pressure_ready = bool(
                pressure_distance_atr is not None and pressure_distance_atr <= proximity_atr
            )
            is_extended = bool(overrun_atr is not None and overrun_atr > extension_atr)
            structure_ready = bool(base["width_atr"] is not None and base["width_atr"] <= base_max_width_atr)

            if aligned:
                reasons_for.append(f"{setup_key} 仍站在 EMA200 上方，突破环境保持偏多")
            else:
                reasons_against.append(f"{setup_key} 已经不再支持多头突破环境")
            if pressure_ready and setup_level is not None:
                reasons_for.append(f"{setup_key} 已压近过去区间前高 {round(float(setup_level), 2)}")
            else:
                reasons_against.append(f"{setup_key} 离过去区间前高还偏远，突破压力不够")
            if structure_ready and base["width_atr"] is not None:
                reasons_for.append(f"{setup_key} 最近整理宽度约 {round(float(base['width_atr']), 2)}ATR，基地不算发散")
            else:
                reasons_against.append(f"{setup_key} 最近整理过宽，突破后容易回撤")
            if is_extended:
                risk_notes.append(f"{setup_key} 已经冲过前高太多，继续追多容易拿到差位置")
        else:
            aligned = (
                setup_ctx.model.close < setup_ctx.model.ema200
                and setup_ctx.model.ema_alignment != "bullish"
                and setup_ctx.model.structure_state != StructureState.BULLISH
            )
            pressure_distance_atr = (
                (current_price - float(setup_level)) / atr if setup_level is not None and atr else None
            )
            overrun_atr = (
                (float(setup_level) - current_price) / atr if setup_level is not None and atr else None
            )
            pressure_ready = bool(
                pressure_distance_atr is not None and pressure_distance_atr <= proximity_atr
            )
            is_extended = bool(overrun_atr is not None and overrun_atr > extension_atr)
            structure_ready = bool(base["width_atr"] is not None and base["width_atr"] <= base_max_width_atr)

            if aligned:
                reasons_for.append(f"{setup_key} 仍压在 EMA200 下方，突破环境保持偏空")
            else:
                reasons_against.append(f"{setup_key} 已经不再支持空头突破环境")
            if pressure_ready and setup_level is not None:
                reasons_for.append(f"{setup_key} 已贴近过去区间前低 {round(float(setup_level), 2)}")
            else:
                reasons_against.append(f"{setup_key} 离过去区间前低还偏远，突破压力不够")
            if structure_ready and base["width_atr"] is not None:
                reasons_for.append(f"{setup_key} 最近整理宽度约 {round(float(base['width_atr']), 2)}ATR，基地不算发散")
            else:
                reasons_against.append(f"{setup_key} 最近整理过宽，突破后容易反抽")
            if is_extended:
                risk_notes.append(f"{setup_key} 已经跌破前低太多，继续追空容易拿到差位置")

        execution_ready = aligned and pressure_ready and structure_ready and not is_extended
        score = 18 if execution_ready else 8 if aligned and pressure_ready and not is_extended else 3 if aligned else -12
        if not structure_ready:
            score -= 4
        if is_extended:
            score -= 8

        return {
            "aligned": aligned,
            "execution_ready": execution_ready,
            "pullback_ready": execution_ready,
            "pressure_ready": pressure_ready,
            "reversal_ready": True,
            "require_reversal_candle": False,
            "divergence_enabled": False,
            "require_divergence_gate": False,
            "require_free_space_gate": False,
            "divergence_ready": False,
            "divergence_level": 0,
            "opposing_divergence_level": 0,
            "free_space_ready": True,
            "free_space_r": None,
            "free_space_min_r": 0.0,
            "is_extended": is_extended,
            "distance_to_value_atr": round(float(pressure_distance_atr or 0.0), 4),
            "structure_ready": structure_ready,
            "score": score,
            "score_note": f"{setup_key} 重点看是否压近过去区间前高/前低，并保持较窄整理基地",
            "reasons_for": reasons_for,
            "reasons_against": reasons_against,
            "risk_notes": risk_notes,
        }

    def _assess_trigger(
        self,
        higher_bias: Bias,
        trigger_ctx: PreparedTimeframe,
        trigger_key: str,
        *,
        trend_strength: int | None = None,
    ) -> dict[str, Any]:
        if higher_bias == Bias.NEUTRAL:
            return {
                "state": TriggerState.NONE,
                "score": -8,
                "score_note": f"{trigger_key} 的局部突破不能单独替代高周期方向",
                "reasons_for": [],
                "reasons_against": [f"高周期中性时，{trigger_key} 的突破信号不做交易依据"],
                "risk_notes": [],
                "bullish_rejection": False,
                "bearish_rejection": False,
                "volume_contracting": False,
                "no_new_extreme": False,
                "regained_fast": False,
                "held_slow": False,
                "auxiliary_count": 0,
                "min_auxiliary_confirmations": 1,
            }

        latest = trigger_ctx.df.iloc[-1]
        breakout_cfg = dict(self.config.get("breakout", {}))
        breakout_level = self._trigger_breakout_level(higher_bias, trigger_ctx)
        atr = float(trigger_ctx.model.atr14)
        close = float(latest["close"])
        open_ = float(latest["open"])
        ema21 = float(latest["ema_21"])
        ema55 = float(latest["ema_55"])
        breakout_buffer = float(breakout_cfg.get("trigger_breakout_buffer_atr", 0.08)) * atr
        extension_limit = float(breakout_cfg.get("trigger_extension_atr", 1.2))
        range_expansion_ratio_min = float(breakout_cfg.get("range_expansion_ratio_min", 1.05))
        quote_volume_ratio_min = float(breakout_cfg.get("quote_volume_ratio_min", 1.0))
        candle_profile = trigger_ctx.candle_profile
        range_ratio = float(candle_profile.get("latest_range_ratio", 0.0))
        quote_volume_ratio = float(candle_profile.get("quote_volume_ratio", 0.0))
        expansion_range = range_ratio >= range_expansion_ratio_min
        expansion_volume = quote_volume_ratio >= quote_volume_ratio_min
        expansion = expansion_range or expansion_volume
        reasons_for: list[str] = []
        reasons_against: list[str] = []
        risk_notes: list[str] = []

        if higher_bias == Bias.BULLISH:
            breakout_confirmed = bool(
                breakout_level is not None and close >= float(breakout_level) + breakout_buffer
            )
            breakout_extension_atr = (
                ((close - float(breakout_level)) / atr) if breakout_level is not None and atr else 0.0
            )
            not_overextended = breakout_extension_atr <= extension_limit
            regained_fast = close > ema21
            held_slow = close > ema55
            directional_close = close > open_
            auxiliary_count = int(expansion_range) + int(expansion_volume) + int(directional_close)

            if breakout_confirmed and breakout_level is not None:
                reasons_for.append(f"{trigger_key} 收盘突破过去区间前高 {round(float(breakout_level), 2)}")
            else:
                reasons_against.append(f"{trigger_key} 还没有形成有效突破收盘")
            if regained_fast:
                reasons_for.append(f"{trigger_key} 仍站在 EMA21 上方，突破后没有立刻失速")
            else:
                reasons_against.append(f"{trigger_key} 突破后仍站不稳 EMA21")
            if held_slow:
                reasons_for.append(f"{trigger_key} 同时站在 EMA55 上方，突破不是纯刺穿")
            else:
                reasons_against.append(f"{trigger_key} 还没稳稳站到 EMA55 上方")
            if expansion:
                reasons_for.append(f"{trigger_key} 有范围或成交额扩张，突破不像闷穿")
            else:
                reasons_against.append(f"{trigger_key} 缺少范围/成交额扩张，突破质量偏弱")
            if not not_overextended:
                risk_notes.append(f"{trigger_key} 已经脱离突破位太远，下一根再追会变差")

            if breakout_confirmed and regained_fast and held_slow and expansion and not_overextended:
                state = TriggerState.BULLISH_CONFIRMED
                score = 15
            elif breakout_confirmed and (regained_fast or held_slow):
                state = TriggerState.MIXED
                score = 2
                risk_notes.append(f"{trigger_key} 有突破轮廓，但还不像高质量顺势突破")
            else:
                state = TriggerState.NONE
                score = -12
                risk_notes.append(f"{trigger_key} 还没有走出值得追随的突破结构")
        else:
            breakout_confirmed = bool(
                breakout_level is not None and close <= float(breakout_level) - breakout_buffer
            )
            breakout_extension_atr = (
                ((float(breakout_level) - close) / atr) if breakout_level is not None and atr else 0.0
            )
            not_overextended = breakout_extension_atr <= extension_limit
            regained_fast = close < ema21
            held_slow = close < ema55
            directional_close = close < open_
            auxiliary_count = int(expansion_range) + int(expansion_volume) + int(directional_close)

            if breakout_confirmed and breakout_level is not None:
                reasons_for.append(f"{trigger_key} 收盘跌破过去区间前低 {round(float(breakout_level), 2)}")
            else:
                reasons_against.append(f"{trigger_key} 还没有形成有效破位收盘")
            if regained_fast:
                reasons_for.append(f"{trigger_key} 仍压在 EMA21 下方，破位后没有立刻被拉回")
            else:
                reasons_against.append(f"{trigger_key} 破位后仍站不稳 EMA21 下方")
            if held_slow:
                reasons_for.append(f"{trigger_key} 同时压在 EMA55 下方，破位不是纯刺穿")
            else:
                reasons_against.append(f"{trigger_key} 还没稳稳压到 EMA55 下方")
            if expansion:
                reasons_for.append(f"{trigger_key} 有范围或成交额扩张，破位不像闷穿")
            else:
                reasons_against.append(f"{trigger_key} 缺少范围/成交额扩张，破位质量偏弱")
            if not not_overextended:
                risk_notes.append(f"{trigger_key} 已经脱离破位点太远，下一根再追会变差")

            if breakout_confirmed and regained_fast and held_slow and expansion and not_overextended:
                state = TriggerState.BEARISH_CONFIRMED
                score = 15
            elif breakout_confirmed and (regained_fast or held_slow):
                state = TriggerState.MIXED
                score = 2
                risk_notes.append(f"{trigger_key} 有破位轮廓，但还不像高质量顺势下破")
            else:
                state = TriggerState.NONE
                score = -12
                risk_notes.append(f"{trigger_key} 还没有走出值得追随的破位结构")

        return {
            "state": state,
            "score": score,
            "score_note": f"{trigger_key} 需要有效突破/破位前高前低，并伴随范围或成交额扩张",
            "reasons_for": reasons_for,
            "reasons_against": reasons_against,
            "risk_notes": risk_notes,
            "recent_low": None,
            "prior_low": None,
            "recent_high": None,
            "prior_high": None,
            "bullish_rejection": False,
            "bearish_rejection": False,
            "volume_contracting": not expansion,
            "no_new_extreme": breakout_confirmed,
            "regained_fast": regained_fast,
            "held_slow": held_slow,
            "auxiliary_count": auxiliary_count,
            "min_auxiliary_confirmations": 1,
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
        threshold = int(self.config["confidence"]["action_threshold"])
        if higher_bias == Bias.BULLISH:
            if (
                trend_friendly
                and setup_assessment["aligned"]
                and setup_assessment["pullback_ready"]
                and trigger_assessment["state"] == TriggerState.BULLISH_CONFIRMED
                and confidence >= threshold
            ):
                return Action.LONG, Bias.BULLISH, RecommendedTiming.NOW
            if setup_assessment["is_extended"]:
                return Action.WAIT, Bias.BULLISH, RecommendedTiming.WAIT_PULLBACK
            return Action.WAIT, Bias.BULLISH, RecommendedTiming.WAIT_CONFIRMATION
        if higher_bias == Bias.BEARISH:
            if (
                trend_friendly
                and setup_assessment["aligned"]
                and setup_assessment["pullback_ready"]
                and trigger_assessment["state"] == TriggerState.BEARISH_CONFIRMED
                and confidence >= threshold
            ):
                return Action.SHORT, Bias.BEARISH, RecommendedTiming.NOW
            if setup_assessment["is_extended"]:
                return Action.WAIT, Bias.BEARISH, RecommendedTiming.WAIT_PULLBACK
            return Action.WAIT, Bias.BEARISH, RecommendedTiming.WAIT_CONFIRMATION
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
                    "invalidation": f"当前仍在等待 {reference_key} 给出有效突破，不直接追多。",
                    "invalidation_price": None,
                }
            if bias == Bias.BEARISH:
                return {
                    "entry_zone": None,
                    "stop_loss": None,
                    "take_profit_hint": None,
                    "invalidation": f"当前仍在等待 {reference_key} 给出有效破位，不直接追空。",
                    "invalidation_price": None,
                }
            return {
                "entry_zone": None,
                "stop_loss": None,
                "take_profit_hint": None,
                "invalidation": "高周期方向不清晰，突破单不成立。",
                "invalidation_price": None,
            }

        breakout_cfg = dict(self.config.get("breakout", {}))
        atr_buffer = float(reference_ctx.model.atr14) * float(self.config["risk"]["atr_buffer"])
        minimum_risk = float(reference_ctx.model.atr14) * float(self.config["risk"]["minimum_r_multiple"])
        trigger_level = self._trigger_breakout_level(bias, reference_ctx)
        setup_level = self._setup_breakout_level(bias, setup_ctx)
        breakout_level = None
        if trigger_level is not None and setup_level is not None:
            breakout_level = max(trigger_level, setup_level) if bias == Bias.BULLISH else min(trigger_level, setup_level)
        else:
            breakout_level = trigger_level if trigger_level is not None else setup_level
        base = self._base_profile(reference_ctx, int(breakout_cfg.get("base_lookback_bars", 8)))

        if bias == Bias.BULLISH:
            base_low = float(base["low"] or reference_ctx.model.swing_low or reference_ctx.model.ema55)
            stop_price = base_low - atr_buffer
            risk = max(current_price - stop_price, minimum_risk)
            tp1 = current_price + risk
            tp2 = current_price + (3 * risk)
            entry_low = float(breakout_level or current_price)
            entry_high = entry_low + atr_buffer
            invalidation_price = base_low
            return {
                "entry_zone": EntryZone(
                    low=round(entry_low, 4),
                    high=round(entry_high, 4),
                    basis=f"{reference_key} 突破前高带",
                ),
                "stop_loss": StopLoss(
                    price=round(stop_price, 4),
                    basis=f"{reference_key} 最近整理低点下方加 ATR 缓冲",
                ),
                "take_profit_hint": TakeProfitHint(
                    tp1=round(tp1, 4),
                    tp2=round(tp2, 4),
                    basis="突破策略先看 1R，延伸目标按 3R 估算",
                ),
                "invalidation": f"若 {reference_key} 突破后重新跌回整理低点下方，则多头突破失效",
                "invalidation_price": round(invalidation_price, 4),
            }

        base_high = float(base["high"] or reference_ctx.model.swing_high or reference_ctx.model.ema55)
        stop_price = base_high + atr_buffer
        risk = max(stop_price - current_price, minimum_risk)
        tp1 = current_price - risk
        tp2 = current_price - (3 * risk)
        entry_high = float(breakout_level or current_price)
        entry_low = entry_high - atr_buffer
        invalidation_price = base_high
        return {
            "entry_zone": EntryZone(
                low=round(entry_low, 4),
                high=round(entry_high, 4),
                basis=f"{reference_key} 跌破前低带",
            ),
            "stop_loss": StopLoss(
                price=round(stop_price, 4),
                basis=f"{reference_key} 最近整理高点上方加 ATR 缓冲",
            ),
            "take_profit_hint": TakeProfitHint(
                tp1=round(tp1, 4),
                tp2=round(tp2, 4),
                basis="破位策略先看 1R，延伸目标按 3R 估算",
            ),
            "invalidation": f"若 {reference_key} 破位后重新站回整理高点上方，则空头突破失效",
            "invalidation_price": round(invalidation_price, 4),
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
            return f"高周期偏多，{setup_key} 已压近前高，{trigger_key} 给出有效突破，因此可以顺势做多。"
        if action == Action.SHORT:
            return f"高周期偏空，{setup_key} 已贴近前低，{trigger_key} 给出有效破位，因此可以顺势做空。"
        if higher_bias == Bias.BULLISH and setup_assessment.get("is_extended"):
            return f"高周期偏多，但 {trigger_key} 已脱离突破位太远，先别追。"
        if higher_bias == Bias.BULLISH and setup_assessment.get("pressure_ready"):
            return f"高周期偏多，{setup_key} 已压近前高，但 {trigger_key} 还没有完成有效突破。"
        if higher_bias == Bias.BULLISH:
            return f"高周期偏多，但 {setup_key} 还没压近关键前高，先等突破压力成形。"
        if higher_bias == Bias.BEARISH and setup_assessment.get("is_extended"):
            return f"高周期偏空，但 {trigger_key} 已脱离破位点太远，先别追。"
        if higher_bias == Bias.BEARISH and setup_assessment.get("pressure_ready"):
            return f"高周期偏空，{setup_key} 已贴近前低，但 {trigger_key} 还没有完成有效破位。"
        if higher_bias == Bias.BEARISH:
            return f"高周期偏空，但 {setup_key} 还没贴近关键前低，先等破位压力成形。"
        return "高周期没有形成统一方向，突破策略先不出手。"

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
            conflict_signals.append(f"{self._format_timeframe_group(tuple(self.window_config['higher_timeframes']))} 没有统一方向")
            vetoes.append("高周期方向不清晰")
        if not setup_assessment["aligned"]:
            conflict_signals.append(f"{setup_key} 自身结构没有和高周期方向对齐")
            vetoes.append(f"{setup_key} 环境不支持顺势突破")
        if not setup_assessment.get("pressure_ready", False) and higher_bias != Bias.NEUTRAL:
            uncertainty_notes.append(f"{setup_key} 还没有真正压近关键前高/前低")
        if not setup_assessment["structure_ready"] and higher_bias != Bias.NEUTRAL:
            uncertainty_notes.append(f"{setup_key} 整理基地偏宽，突破后容易回撤或反抽")
            vetoes.append(f"{setup_key} 整理结构不够紧")
        if setup_assessment["is_extended"]:
            vetoes.append(f"{trigger_key} 已经脱离突破位太远")
        if trigger_assessment["state"] in {TriggerState.MIXED, TriggerState.NONE}:
            uncertainty_notes.append(f"{trigger_key} 还没有完成高质量突破/破位")
            vetoes.append(f"{trigger_key} 确认度不够")
        if volatility_state == VolatilityState.HIGH:
            uncertainty_notes.append(f"{trigger_key} 波动偏高，下一根继续追会更吃位置")
        if not trend_friendly:
            uncertainty_notes.append("当前趋势强度还不够理想，突破延续性未必够")
        if action == Action.WAIT and confidence < int(self.config["confidence"]["action_threshold"]):
            vetoes.append(f"置信度 {confidence} 低于动作阈值 {self.config['confidence']['action_threshold']}")

        return conflict_signals, uncertainty_notes, vetoes
