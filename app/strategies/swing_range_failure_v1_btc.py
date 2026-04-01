from __future__ import annotations

from typing import Any

from app.schemas.analysis import EntryZone, StopLoss, TakeProfitHint
from app.schemas.common import Action, Bias, RecommendedTiming, TriggerState, VolatilityState
from app.strategies.windowed_mtf import COMMON_DEFAULT_CONFIG, PreparedTimeframe, WindowedMTFStrategy


DEFAULT_CONFIG = {
    **COMMON_DEFAULT_CONFIG,
    "setup": {
        **COMMON_DEFAULT_CONFIG["setup"],
        "require_structure_ready": True,
        "require_reversal_candle": False,
    },
    "range_failure": {
        "lookback_bars": 12,
        "min_width_atr": 1.6,
        "max_width_atr": 6.0,
        "edge_proximity_atr": 0.45,
        "sweep_buffer_atr": 0.12,
        "stop_buffer_atr": 0.12,
        "entry_pad_atr": 0.08,
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


class SwingRangeFailureV1BTCStrategy(WindowedMTFStrategy):
    name = "swing_range_failure_v1_btc"
    required_timeframes = ("1d", "4h", "1h")

    def _completed_window(self, ctx: PreparedTimeframe, lookback: int):
        df = ctx.df
        if len(df) <= 1:
            return df.iloc[0:0]
        start = max(0, len(df) - lookback - 1)
        return df.iloc[start:-1]

    def _range_profile(self, ctx: PreparedTimeframe) -> dict[str, float | None]:
        cfg = dict(self.config.get("range_failure", {}))
        window = self._completed_window(ctx, int(cfg.get("lookback_bars", 12)))
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

    def _derive_higher_timeframe_bias(self, prepared: dict[str, PreparedTimeframe]) -> tuple[Bias, int]:
        day_ctx = prepared["1d"]
        setup_ctx = prepared["4h"]
        trend_strength = int(round((day_ctx.model.trend_score + setup_ctx.model.trend_score) / 2))

        if day_ctx.model.trend_bias == Bias.BULLISH and setup_ctx.model.trend_bias != Bias.BEARISH:
            return Bias.BULLISH, trend_strength
        if day_ctx.model.trend_bias == Bias.BEARISH and setup_ctx.model.trend_bias != Bias.BULLISH:
            return Bias.BEARISH, trend_strength
        return Bias.NEUTRAL, trend_strength

    def _is_trend_friendly(
        self,
        *,
        higher_bias: Bias,
        trend_strength: int,
        volatility_state: VolatilityState,
    ) -> bool:
        return higher_bias != Bias.NEUTRAL and volatility_state != VolatilityState.HIGH

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
        range_cfg = dict(self.config.get("range_failure", {}))
        range_profile = self._range_profile(setup_ctx)
        range_low = range_profile["low"]
        range_high = range_profile["high"]
        width_atr = range_profile["width_atr"]
        min_width_atr = float(range_cfg.get("min_width_atr", 1.6))
        max_width_atr = float(range_cfg.get("max_width_atr", 6.0))
        edge_proximity_atr = float(range_cfg.get("edge_proximity_atr", 0.45))
        atr = float(setup_ctx.model.atr14)

        if higher_bias == Bias.NEUTRAL or range_low is None or range_high is None or width_atr is None:
            return {
                "aligned": False,
                "execution_ready": False,
                "pullback_ready": False,
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
                "score_note": f"{setup_key} 需要先形成可交易的 1H 区间，再等假突破失败",
                "reasons_for": reasons_for,
                "reasons_against": [f"{setup_key} 还没有形成可识别的 range-failure 环境"],
                "risk_notes": risk_notes,
            }

        aligned = True
        structure_ready = min_width_atr <= float(width_atr) <= max_width_atr
        in_range = float(range_low) <= current_price <= float(range_high)

        if higher_bias == Bias.BULLISH:
            edge_distance_atr = ((current_price - float(range_low)) / atr) if atr else None
            edge_ready = edge_distance_atr is not None and 0 <= edge_distance_atr <= edge_proximity_atr
            if edge_ready:
                reasons_for.append(f"{setup_key} 靠近最近 1H 区间下沿，适合观察 failed breakdown")
            else:
                reasons_against.append(f"{setup_key} 离最近 1H 区间下沿还不够近")
        else:
            edge_distance_atr = ((float(range_high) - current_price) / atr) if atr else None
            edge_ready = edge_distance_atr is not None and 0 <= edge_distance_atr <= edge_proximity_atr
            if edge_ready:
                reasons_for.append(f"{setup_key} 靠近最近 1H 区间上沿，适合观察 failed breakout")
            else:
                reasons_against.append(f"{setup_key} 离最近 1H 区间上沿还不够近")

        if structure_ready:
            reasons_for.append(f"{setup_key} 最近 1H 区间宽度约 {round(float(width_atr), 2)}ATR，没有发散到不可交易")
        else:
            reasons_against.append(f"{setup_key} 最近 1H 区间宽度约 {round(float(width_atr), 2)}ATR，太窄或太宽")
        if not in_range:
            reasons_against.append(f"{setup_key} 当前收盘还没回到区间内部")

        execution_ready = aligned and structure_ready and edge_ready and in_range
        score = 16 if execution_ready else 8 if structure_ready and in_range else -10
        if not structure_ready:
            score -= 4
        if not in_range:
            score -= 4

        return {
            "aligned": aligned,
            "execution_ready": execution_ready,
            "pullback_ready": execution_ready,
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
            "is_extended": not in_range,
            "distance_to_value_atr": round(float(edge_distance_atr or 0.0), 4),
            "structure_ready": structure_ready,
            "score": score,
            "score_note": f"{setup_key} 重点看最近 1H 区间是否紧凑，并且当前是否贴近区间边缘",
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
        range_profile = self._range_profile(trigger_ctx)
        range_low = range_profile["low"]
        range_high = range_profile["high"]
        latest = trigger_ctx.df.iloc[-1]
        atr = float(trigger_ctx.model.atr14)
        sweep_buffer = float(self.config.get("range_failure", {}).get("sweep_buffer_atr", 0.12)) * atr
        reasons_for: list[str] = []
        reasons_against: list[str] = []
        risk_notes: list[str] = []

        if higher_bias == Bias.NEUTRAL or range_low is None or range_high is None:
            return {
                "state": TriggerState.NONE,
                "score": -10,
                "score_note": f"{trigger_key} 没有可以执行的 range-failure 方向",
                "reasons_for": reasons_for,
                "reasons_against": [f"{trigger_key} 当前没有清晰的 range-failure 触发"],
                "risk_notes": risk_notes,
                "recent_low": None,
                "prior_low": None,
                "recent_high": None,
                "prior_high": None,
                "bullish_rejection": False,
                "bearish_rejection": False,
                "volume_contracting": False,
                "no_new_extreme": False,
                "regained_fast": False,
                "held_slow": False,
                "auxiliary_count": 0,
                "min_auxiliary_confirmations": 1,
            }

        close = float(latest["close"])
        low = float(latest["low"])
        high = float(latest["high"])
        ema21 = float(latest["ema_21"])

        if higher_bias == Bias.BULLISH:
            swept = low <= float(range_low) - sweep_buffer
            reclaimed = close >= float(range_low)
            rejection = bool(trigger_ctx.candle_profile.get("has_bullish_rejection", False))
            regained_fast = close >= ema21
            held_slow = reclaimed

            if swept:
                reasons_for.append(f"{trigger_key} 先扫破最近 1H 区间下沿")
            else:
                reasons_against.append(f"{trigger_key} 还没有形成区间下沿下方的流动性扫单")
            if reclaimed:
                reasons_for.append(f"{trigger_key} 收盘重新回到区间内部")
            else:
                reasons_against.append(f"{trigger_key} 还没有收回区间下沿")
            if rejection:
                reasons_for.append(f"{trigger_key} 出现明显下影拒绝回落")
            else:
                reasons_against.append(f"{trigger_key} 缺少明显下影拒绝")

            if swept and reclaimed and rejection:
                state = TriggerState.BULLISH_CONFIRMED
                score = 15
            elif swept and reclaimed:
                state = TriggerState.MIXED
                score = 1
                risk_notes.append(f"{trigger_key} 已经回到区间内，但没有足够干净的拒绝影线")
            else:
                state = TriggerState.NONE
                score = -12
        else:
            swept = high >= float(range_high) + sweep_buffer
            reclaimed = close <= float(range_high)
            rejection = bool(trigger_ctx.candle_profile.get("has_bearish_rejection", False))
            regained_fast = close <= ema21
            held_slow = reclaimed

            if swept:
                reasons_for.append(f"{trigger_key} 先扫过最近 1H 区间上沿")
            else:
                reasons_against.append(f"{trigger_key} 还没有形成区间上沿上方的流动性扫单")
            if reclaimed:
                reasons_for.append(f"{trigger_key} 收盘重新压回区间内部")
            else:
                reasons_against.append(f"{trigger_key} 还没有重新压回区间上沿下方")
            if rejection:
                reasons_for.append(f"{trigger_key} 出现明显上影拒绝拉升")
            else:
                reasons_against.append(f"{trigger_key} 缺少明显上影拒绝")

            if swept and reclaimed and rejection:
                state = TriggerState.BEARISH_CONFIRMED
                score = 15
            elif swept and reclaimed:
                state = TriggerState.MIXED
                score = 1
                risk_notes.append(f"{trigger_key} 已经回到区间内，但没有足够干净的拒绝影线")
            else:
                state = TriggerState.NONE
                score = -12

        return {
            "state": state,
            "score": score,
            "score_note": f"{trigger_key} 需要扫破区间边缘后重新收回，并出现 rejection K 线",
            "reasons_for": reasons_for,
            "reasons_against": reasons_against,
            "risk_notes": risk_notes,
            "recent_low": round(float(range_low), 4),
            "prior_low": round(float(range_low), 4),
            "recent_high": round(float(range_high), 4),
            "prior_high": round(float(range_high), 4),
            "bullish_rejection": bool(trigger_ctx.candle_profile.get("has_bullish_rejection", False)),
            "bearish_rejection": bool(trigger_ctx.candle_profile.get("has_bearish_rejection", False)),
            "volume_contracting": bool(trigger_ctx.candle_profile.get("is_volume_contracting", False)),
            "no_new_extreme": reclaimed,
            "regained_fast": regained_fast,
            "held_slow": held_slow,
            "auxiliary_count": int(rejection) + int(reclaimed),
            "min_auxiliary_confirmations": 2,
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
                and setup_assessment["pullback_ready"]
                and trigger_assessment["state"] == TriggerState.BULLISH_CONFIRMED
                and confidence >= threshold
            ):
                return Action.LONG, Bias.BULLISH, RecommendedTiming.NOW
            if trigger_assessment["state"] == TriggerState.MIXED or setup_assessment["is_extended"]:
                return Action.WAIT, Bias.BULLISH, RecommendedTiming.WAIT_CONFIRMATION
            return Action.WAIT, Bias.BULLISH, RecommendedTiming.SKIP

        if higher_bias == Bias.BEARISH:
            if (
                trend_friendly
                and setup_assessment["pullback_ready"]
                and trigger_assessment["state"] == TriggerState.BEARISH_CONFIRMED
                and confidence >= threshold
            ):
                return Action.SHORT, Bias.BEARISH, RecommendedTiming.NOW
            if trigger_assessment["state"] == TriggerState.MIXED or setup_assessment["is_extended"]:
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
                    "invalidation": f"当前仍在等待 {setup_key} 完整走出扫低失败并回到区间内。",
                    "invalidation_price": None,
                }
            if bias == Bias.BEARISH:
                return {
                    "entry_zone": None,
                    "stop_loss": None,
                    "take_profit_hint": None,
                    "invalidation": f"当前仍在等待 {setup_key} 完整走出扫高失败并回到区间内。",
                    "invalidation_price": None,
                }
            return {
                "entry_zone": None,
                "stop_loss": None,
                "take_profit_hint": None,
                "invalidation": "当前没有可执行的宏观方向或区间失败结构。",
                "invalidation_price": None,
            }

        range_profile = self._range_profile(setup_ctx)
        latest = setup_ctx.df.iloc[-1]
        atr = float(setup_ctx.model.atr14)
        stop_buffer = float(self.config.get("range_failure", {}).get("stop_buffer_atr", 0.12)) * atr
        entry_pad = float(self.config.get("range_failure", {}).get("entry_pad_atr", 0.08)) * atr
        range_low = float(range_profile["low"] or current_price)
        range_high = float(range_profile["high"] or current_price)
        range_mid = float(range_profile["mid"] or current_price)

        if bias == Bias.BULLISH:
            stop_price = min(float(latest["low"]), range_low) - stop_buffer
            risk = max(current_price - stop_price, atr * 0.8)
            tp1 = max(range_mid, current_price + risk)
            tp2 = max(range_high, current_price + (2 * risk))
            return {
                "entry_zone": EntryZone(
                    low=round(current_price - entry_pad, 4),
                    high=round(current_price + entry_pad, 4),
                    basis=f"{setup_key} 区间下沿失败后的回收带",
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
                "invalidation": f"若 {setup_key} 再次失守假跌破低点，则 failed breakdown 失效",
                "invalidation_price": round(min(float(latest['low']), range_low), 4),
            }

        stop_price = max(float(latest["high"]), range_high) + stop_buffer
        risk = max(stop_price - current_price, atr * 0.8)
        tp1 = min(range_mid, current_price - risk)
        tp2 = min(range_low, current_price - (2 * risk))
        return {
            "entry_zone": EntryZone(
                low=round(current_price - entry_pad, 4),
                high=round(current_price + entry_pad, 4),
                basis=f"{setup_key} 区间上沿失败后的回压带",
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
            "invalidation": f"若 {setup_key} 再次站稳假突破高点，则 failed breakout 失效",
            "invalidation_price": round(max(float(latest['high']), range_high), 4),
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
            return f"日线偏多，{trigger_key} 先扫破最近 1H 区间下沿又收回，因此按 failed breakdown 做多。"
        if action == Action.SHORT:
            return f"日线偏空，{trigger_key} 先扫过最近 1H 区间上沿又压回，因此按 failed breakout 做空。"
        if higher_bias == Bias.BULLISH:
            return f"日线偏多，但 {trigger_key} 还没有把区间下沿假跌破走完整，先等。"
        if higher_bias == Bias.BEARISH:
            return f"日线偏空，但 {trigger_key} 还没有把区间上沿假突破走完整，先等。"
        return "当前宏观方向和 1H 区间失败结构没有同时成立，先观望。"
