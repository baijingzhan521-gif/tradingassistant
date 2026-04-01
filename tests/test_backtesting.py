from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from app.backtesting.service import (
    BACKTEST_SNAPSHOT_TAIL_BARS,
    POSITION_MAP_COLUMNS,
    BacktestAssumptions,
    BacktestService,
    BacktestTrade,
    _OpenPosition,
    _PendingEntry,
)
from app.data.exchange_client import ExchangeClientFactory
from app.data.ohlcv_service import OhlcvService
from app.indicators.swings import identify_swings
from app.schemas.common import Action, Bias, RecommendedTiming, TriggerState, VolatilityState
from app.services.strategy_service import StrategyService
from app.strategies.swing_trend_long_regime_gate_v1 import (
    DEFAULT_CONFIG as SWING_LONG_REGIME_GATE_DEFAULT_CONFIG,
    SwingTrendLongRegimeGateV1Strategy,
)
from tests.conftest import make_ohlcv_frame


def make_service() -> BacktestService:
    assumptions = BacktestAssumptions(
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        intraday_max_hold_bars=10,
        swing_max_hold_bars=10,
        cache_dir="artifacts/backtests/test-cache",
    )
    return BacktestService(
        ohlcv_service=OhlcvService(ExchangeClientFactory()),
        strategy_service=StrategyService(),
        assumptions=assumptions,
    )


def test_long_trade_scales_out_to_tp2() -> None:
    service = make_service()
    position = _OpenPosition(
        symbol="BTC/USDT:USDT",
        strategy_profile="intraday_mtf_v1",
        side=Action.LONG,
        higher_bias=Bias.BULLISH,
        trend_strength=90,
        signal_time=datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc),
        entry_time=datetime(2026, 3, 1, 0, 3, tzinfo=timezone.utc),
        entry_price=100.0,
        initial_stop_price=99.0,
        current_stop_price=99.0,
        tp1_price=101.0,
        tp2_price=102.0,
        take_profit_mode="scaled",
        fixed_take_profit_r=None,
        confidence=90,
    )

    first_candle = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-03-01T00:06:00Z"),
            "high": 101.2,
            "low": 100.1,
            "close": 101.0,
        }
    )
    second_candle = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-03-01T00:09:00Z"),
            "high": 102.2,
            "low": 100.8,
            "close": 101.8,
        }
    )

    assert service._update_open_position(position=position, candle=first_candle, max_hold_bars=10) is None
    trade = service._update_open_position(position=position, candle=second_candle, max_hold_bars=10)

    assert trade is not None
    assert trade.exit_reason == "tp2"
    assert trade.tp1_hit is True
    assert trade.tp2_hit is True
    assert trade.pnl_r == 1.5


def test_long_trade_after_tp1_marks_breakeven_reason_under_conservative_exit() -> None:
    service = make_service()
    position = _OpenPosition(
        symbol="BTC/USDT:USDT",
        strategy_profile="swing_trend_long_regime_gate_v1",
        side=Action.LONG,
        higher_bias=Bias.BULLISH,
        trend_strength=92,
        signal_time=datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc),
        entry_time=datetime(2026, 3, 1, 1, 0, tzinfo=timezone.utc),
        entry_price=100.0,
        initial_stop_price=99.0,
        current_stop_price=100.0,
        tp1_price=101.0,
        tp2_price=103.0,
        take_profit_mode="scaled",
        fixed_take_profit_r=None,
        confidence=90,
        remaining_qty=0.5,
        realized_pnl_quote=0.5,
        tp1_hit=True,
    )

    candle = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-03-01T02:00:00Z"),
            "high": 100.4,
            "low": 99.8,
            "close": 100.1,
        }
    )

    trade = service._update_open_position(position=position, candle=candle, max_hold_bars=10)

    assert trade is not None
    assert trade.exit_reason == "breakeven_after_tp1"
    assert trade.tp1_hit is True
    assert trade.pnl_r == 0.5


def test_summary_computes_payoff_and_win_rate() -> None:
    service = make_service()
    trades = [
        BacktestTrade(
            symbol="ETH/USDT:USDT",
            strategy_profile="swing_trend_v1",
            side="LONG",
            higher_bias="bullish",
            trend_strength=92,
            signal_time="2026-01-01T00:00:00+00:00",
            entry_time="2026-01-01T01:00:00+00:00",
            exit_time="2026-01-03T01:00:00+00:00",
            entry_price=100.0,
            exit_price=102.0,
            stop_price=99.0,
            tp1_price=101.0,
            tp2_price=103.0,
            bars_held=20,
            exit_reason="tp2",
            confidence=85,
            tp1_hit=True,
            tp2_hit=True,
            pnl_pct=2.0,
            pnl_r=1.5,
            gross_pnl_quote=1.5,
            fees_quote=0.0,
        ),
        BacktestTrade(
            symbol="ETH/USDT:USDT",
            strategy_profile="swing_trend_v1",
            side="SHORT",
            higher_bias="bearish",
            trend_strength=88,
            signal_time="2026-01-05T00:00:00+00:00",
            entry_time="2026-01-05T01:00:00+00:00",
            exit_time="2026-01-06T01:00:00+00:00",
            entry_price=100.0,
            exit_price=101.0,
            stop_price=101.0,
            tp1_price=99.0,
            tp2_price=98.0,
            bars_held=8,
            exit_reason="stop_loss",
            confidence=78,
            tp1_hit=False,
            tp2_hit=False,
            pnl_pct=-1.0,
            pnl_r=-1.0,
            gross_pnl_quote=-1.0,
            fees_quote=0.0,
        ),
    ]

    summary = service._summarize_trades(
        trades=trades,
        strategy_profile="swing_trend_v1",
        symbol="ETH/USDT:USDT",
        signals_now=2,
        skipped_entries=0,
    )

    assert summary.total_trades == 2
    assert summary.win_rate == 50.0
    assert summary.payoff_ratio == 1.5
    assert summary.profit_factor == 1.5
    assert summary.expectancy_r == 0.25


def test_fixed_r_full_exit_closes_entire_position_at_target() -> None:
    assumptions = BacktestAssumptions(
        exit_profile="fixed_1_5r_full",
        take_profit_mode="fixed_r",
        fixed_take_profit_r=1.5,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        intraday_max_hold_bars=10,
        swing_max_hold_bars=10,
        cache_dir="artifacts/backtests/test-cache",
    )
    service = BacktestService(
        ohlcv_service=OhlcvService(ExchangeClientFactory()),
        strategy_service=StrategyService(),
        assumptions=assumptions,
    )
    position = _OpenPosition(
        symbol="BTC/USDT:USDT",
        strategy_profile="intraday_mtf_v1",
        side=Action.LONG,
        higher_bias=Bias.BULLISH,
        trend_strength=90,
        signal_time=datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc),
        entry_time=datetime(2026, 3, 1, 0, 3, tzinfo=timezone.utc),
        entry_price=100.0,
        initial_stop_price=99.0,
        current_stop_price=99.0,
        tp1_price=101.5,
        tp2_price=101.5,
        take_profit_mode="fixed_r",
        fixed_take_profit_r=1.5,
        confidence=90,
    )

    candle = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-03-01T00:06:00Z"),
            "high": 101.6,
            "low": 100.2,
            "close": 101.3,
        }
    )

    trade = service._update_open_position(position=position, candle=candle, max_hold_bars=10)

    assert trade is not None
    assert trade.exit_reason == "take_profit_1.5R"
    assert trade.tp1_hit is True
    assert trade.tp2_hit is True
    assert trade.pnl_r == 1.5


def test_backtest_signal_respects_bias_specific_trend_thresholds() -> None:
    service = make_service()
    strategy = SwingTrendLongRegimeGateV1Strategy(SWING_LONG_REGIME_GATE_DEFAULT_CONFIG)
    timestamp = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
    frame = pd.DataFrame([{"timestamp": pd.Timestamp(timestamp)}])
    enriched = {"1d": frame, "4h": frame, "1h": frame}

    class DummyPrepared:
        class model:
            close = 100.0

    dummy = DummyPrepared()
    service._build_snapshot = lambda *args, **kwargs: dummy  # type: ignore[method-assign]
    strategy._derive_higher_timeframe_bias = lambda prepared: (Bias.BULLISH, 80)  # type: ignore[method-assign]
    strategy._derive_volatility_state = lambda prepared: VolatilityState.NORMAL  # type: ignore[method-assign]
    strategy._assess_setup = lambda higher_bias, ctx, key, **kwargs: {  # type: ignore[method-assign]
        "aligned": True,
        "execution_ready": True,
        "pullback_ready": True,
        "reversal_ready": True,
        "require_reversal_candle": True,
        "require_free_space_gate": False,
        "free_space_ready": True,
        "is_extended": False,
        "score": 20,
        "score_note": "setup ok",
    }
    strategy._assess_trigger = lambda higher_bias, ctx, key, **kwargs: {  # type: ignore[method-assign]
        "state": TriggerState.BULLISH_CONFIRMED,
        "score": 12,
        "score_note": "trigger ok",
        "no_new_extreme": True,
        "regained_fast": True,
        "held_slow": True,
        "auxiliary_count": 2,
        "bullish_rejection": True,
        "bearish_rejection": False,
        "volume_contracting": True,
    }
    strategy._build_trade_plan = lambda **kwargs: {  # type: ignore[method-assign]
        "entry_zone": None,
        "stop_loss": None,
        "take_profit_hint": None,
        "invalidation": "wait",
        "invalidation_price": None,
    }

    signal = service._evaluate_signal(
        strategy=strategy,
        strategy_profile=strategy.name,
        enriched=enriched,
        indices={"1d": 0, "4h": 0, "1h": 0},
        timestamp=timestamp,
    )

    assert signal.action == Action.WAIT
    assert signal.recommended_timing == RecommendedTiming.SKIP


def test_evaluate_signal_reuses_cached_snapshots_when_higher_timeframes_do_not_advance() -> None:
    service = make_service()
    strategy = SwingTrendLongRegimeGateV1Strategy(SWING_LONG_REGIME_GATE_DEFAULT_CONFIG)
    first_ts = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
    second_ts = datetime(2026, 3, 1, 1, 0, tzinfo=timezone.utc)
    higher_frame = pd.DataFrame([{"timestamp": pd.Timestamp(first_ts)}])
    trigger_frame = pd.DataFrame(
        [{"timestamp": pd.Timestamp(first_ts)}, {"timestamp": pd.Timestamp(second_ts)}]
    )
    enriched = {"1d": higher_frame, "4h": higher_frame, "1h": trigger_frame}
    snapshot_cache: dict[str, tuple[int, object]] = {}
    build_calls: list[tuple[str, int]] = []

    def build_snapshot(*args, **kwargs):
        timeframe = args[1]
        idx = args[3]
        build_calls.append((timeframe, idx))
        return SimpleNamespace(model=SimpleNamespace(close=100.0))

    service._build_snapshot = build_snapshot  # type: ignore[method-assign]
    strategy._derive_higher_timeframe_bias = lambda prepared: (Bias.BULLISH, 80)  # type: ignore[method-assign]
    strategy._derive_volatility_state = lambda prepared: VolatilityState.NORMAL  # type: ignore[method-assign]
    strategy._assess_setup = lambda higher_bias, ctx, key, **kwargs: {  # type: ignore[method-assign]
        "aligned": True,
        "execution_ready": True,
        "pullback_ready": True,
        "reversal_ready": True,
        "require_reversal_candle": False,
        "require_free_space_gate": False,
        "free_space_ready": True,
        "is_extended": False,
        "score": 20,
        "score_note": "setup ok",
    }
    strategy._assess_trigger = lambda higher_bias, ctx, key, **kwargs: {  # type: ignore[method-assign]
        "state": TriggerState.BULLISH_CONFIRMED,
        "score": 12,
        "score_note": "trigger ok",
        "no_new_extreme": True,
        "regained_fast": True,
        "held_slow": True,
        "auxiliary_count": 2,
        "bullish_rejection": True,
        "bearish_rejection": False,
        "volume_contracting": True,
    }
    strategy._build_trade_plan = lambda **kwargs: {  # type: ignore[method-assign]
        "entry_zone": None,
        "stop_loss": None,
        "take_profit_hint": None,
        "invalidation": "wait",
        "invalidation_price": None,
    }

    service._evaluate_signal(
        strategy=strategy,
        strategy_profile=strategy.name,
        enriched=enriched,
        indices={"1d": 0, "4h": 0, "1h": 0},
        timestamp=first_ts,
        snapshot_cache=snapshot_cache,  # type: ignore[arg-type]
    )
    service._evaluate_signal(
        strategy=strategy,
        strategy_profile=strategy.name,
        enriched=enriched,
        indices={"1d": 0, "4h": 0, "1h": 1},
        timestamp=second_ts,
        snapshot_cache=snapshot_cache,  # type: ignore[arg-type]
    )

    assert build_calls == [("1d", 0), ("4h", 0), ("1h", 0), ("1h", 1)]


def test_build_snapshot_uses_precomputed_position_map_columns_and_bounded_tail() -> None:
    service = make_service()
    strategy = SwingTrendLongRegimeGateV1Strategy(SWING_LONG_REGIME_GATE_DEFAULT_CONFIG)
    frame = make_ohlcv_frame(periods=90, start_price=2600, up_step=1.1, freq="h")
    enriched = service._enrich_frame(strategy, "1h", frame)

    snapshot = service._build_snapshot(strategy, "1h", enriched, 60)
    row = enriched.iloc[60]

    assert snapshot.band_upper == pytest.approx(float(row["band_upper"]))
    assert snapshot.band_lower == pytest.approx(float(row["band_lower"]))
    assert snapshot.band_volatility_unit == pytest.approx(float(row["band_volatility_unit"]))
    assert snapshot.axis_distance_vol == pytest.approx(float(row["axis_distance_vol"]))
    assert snapshot.ema55_distance_vol == pytest.approx(float(row["ema55_distance_vol"]))
    assert snapshot.band_position == pytest.approx(float(row["band_position"]))
    assert len(snapshot.df) == BACKTEST_SNAPSHOT_TAIL_BARS
    assert snapshot.df.iloc[-1]["timestamp"] == row["timestamp"]


def test_build_snapshot_fallback_matches_precomputed_std_position_map() -> None:
    service = make_service()
    config = deepcopy(SWING_LONG_REGIME_GATE_DEFAULT_CONFIG)
    config["position_map"] = {
        **config["position_map"],
        "band_volatility_mode": "std",
        "band_std_period": 20,
        "band_std_mult": 2.0,
    }
    strategy = SwingTrendLongRegimeGateV1Strategy(config)
    frame = make_ohlcv_frame(periods=90, start_price=2600, up_step=1.1, freq="h")
    enriched = service._enrich_frame(strategy, "1h", frame)
    fallback_enriched = enriched.drop(columns=list(POSITION_MAP_COLUMNS))

    snapshot_precomputed = service._build_snapshot(strategy, "1h", enriched, 60)
    snapshot_fallback = service._build_snapshot(strategy, "1h", fallback_enriched, 60)

    assert snapshot_fallback.band_upper == pytest.approx(snapshot_precomputed.band_upper)
    assert snapshot_fallback.band_lower == pytest.approx(snapshot_precomputed.band_lower)
    assert snapshot_fallback.band_volatility_unit == pytest.approx(snapshot_precomputed.band_volatility_unit)
    assert snapshot_fallback.axis_distance_vol == pytest.approx(snapshot_precomputed.axis_distance_vol)
    assert snapshot_fallback.ema55_distance_vol == pytest.approx(snapshot_precomputed.ema55_distance_vol)
    assert snapshot_fallback.band_position == pytest.approx(snapshot_precomputed.band_position)


def test_open_pending_entry_supports_scaled_target_overrides() -> None:
    assumptions = BacktestAssumptions(
        exit_profile="scaled_1r_to_3r_be",
        take_profit_mode="scaled",
        scaled_tp1_r=1.0,
        scaled_tp2_r=3.0,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
    )
    service = BacktestService(
        ohlcv_service=OhlcvService(ExchangeClientFactory()),
        strategy_service=StrategyService(),
        assumptions=assumptions,
    )
    pending_entry = _PendingEntry(
        signal=type(
            "Signal",
            (),
            {
                "action": Action.LONG,
                "stop_price": 99.0,
                "tp1_price": 101.0,
                "tp2_price": 102.0,
                "confidence": 90,
                "timestamp": datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc),
            },
        )()
    )
    candle = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-03-01T01:00:00Z"),
            "open": 100.0,
        }
    )

    position = service._open_pending_entry(
        symbol="BTC/USDT:USDT",
        strategy_profile="swing_trend_long_regime_gate_v1",
        pending_entry=pending_entry,
        candle=candle,
    )

    assert position is not None
    assert position.tp1_price == 101.0
    assert position.tp2_price == 103.0
    assert position.take_profit_mode == "scaled"
    assert position.move_stop_to_entry_after_tp1 is True


def test_open_pending_entry_supports_side_specific_exit_configs() -> None:
    assumptions = BacktestAssumptions(
        exit_profile="asymmetric_exit",
        take_profit_mode="scaled",
        scaled_tp1_r=1.0,
        scaled_tp2_r=2.0,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        long_exit={"take_profit_mode": "scaled", "scaled_tp1_r": 1.0, "scaled_tp2_r": 3.0},
        short_exit={"take_profit_mode": "fixed_r", "fixed_take_profit_r": 3.0},
    )
    service = BacktestService(
        ohlcv_service=OhlcvService(ExchangeClientFactory()),
        strategy_service=StrategyService(),
        assumptions=assumptions,
    )
    long_pending = _PendingEntry(
        signal=type(
            "Signal",
            (),
            {
                "action": Action.LONG,
                "stop_price": 99.0,
                "tp1_price": 101.0,
                "tp2_price": 102.0,
                "confidence": 90,
                "timestamp": datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc),
            },
        )()
    )
    short_pending = _PendingEntry(
        signal=type(
            "Signal",
            (),
            {
                "action": Action.SHORT,
                "stop_price": 101.0,
                "tp1_price": 99.0,
                "tp2_price": 98.0,
                "confidence": 90,
                "timestamp": datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc),
            },
        )()
    )
    candle = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-03-01T01:00:00Z"),
            "open": 100.0,
        }
    )

    long_position = service._open_pending_entry(
        symbol="BTC/USDT:USDT",
        strategy_profile="swing_trend_long_regime_gate_v1",
        pending_entry=long_pending,
        candle=candle,
    )
    short_position = service._open_pending_entry(
        symbol="BTC/USDT:USDT",
        strategy_profile="swing_trend_long_regime_gate_v1",
        pending_entry=short_pending,
        candle=candle,
    )

    assert long_position is not None
    assert short_position is not None
    assert long_position.take_profit_mode == "scaled"
    assert long_position.tp2_price == 103.0
    assert short_position.take_profit_mode == "fixed_r"
    assert short_position.fixed_take_profit_r == 3.0
    assert short_position.tp1_price == 97.0
    assert short_position.tp2_price == 97.0


def test_identify_swings_confirmed_mode_delays_marker_until_confirmation_bar() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=7, freq="h", tz="UTC"),
            "high": [1.0, 2.0, 5.0, 2.0, 1.0, 3.0, 2.0],
            "low": [3.0, 2.0, 1.0, 2.0, 4.0, 1.0, 2.0],
        }
    )

    centered = identify_swings(frame, window=1, mode="centered")
    confirmed = identify_swings(frame, window=1, mode="confirmed")

    assert centered.loc[2, "swing_high_marker"] == 5.0
    assert pd.isna(confirmed.loc[2, "swing_high_marker"])
    assert confirmed.loc[3, "swing_high_marker"] == 5.0
    assert centered.loc[2, "swing_low_marker"] == 1.0
    assert pd.isna(confirmed.loc[2, "swing_low_marker"])
    assert confirmed.loc[3, "swing_low_marker"] == 1.0


def test_run_symbol_strategy_with_frames_matches_history_loading_path() -> None:
    service = make_service()
    start = datetime(2025, 2, 1, tzinfo=timezone.utc)
    end = datetime(2025, 2, 15, tzinfo=timezone.utc)
    full_frames = {
        "1d": make_ohlcv_frame(periods=360, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=360, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=360, start_price=2600, up_step=1.1, freq="h"),
    }
    limited_frames = {
        timeframe: frame.loc[frame["timestamp"] < pd.Timestamp(end)].reset_index(drop=True)
        for timeframe, frame in full_frames.items()
    }

    service._load_history = lambda **kwargs: limited_frames  # type: ignore[method-assign]

    summary_loaded, trades_loaded = service._run_symbol_strategy(
        exchange="binance",
        market_type="perpetual",
        symbol="BTC/USDT:USDT",
        strategy_profile="swing_trend_long_regime_gate_v1",
        start=start,
        end=end,
    )
    summary_preloaded, trades_preloaded = service.run_symbol_strategy_with_frames(
        symbol="BTC/USDT:USDT",
        strategy_profile="swing_trend_long_regime_gate_v1",
        start=start,
        end=end,
        frames=full_frames,
    )
    service._load_history = lambda **kwargs: full_frames  # type: ignore[method-assign]
    enriched = service.prepare_enriched_history(
        exchange="binance",
        market_type="perpetual",
        symbol="BTC/USDT:USDT",
        strategy_profile="swing_trend_long_regime_gate_v1",
        start=start,
        end=end,
    )
    summary_enriched, trades_enriched = service.run_symbol_strategy_with_enriched_frames(
        symbol="BTC/USDT:USDT",
        strategy_profile="swing_trend_long_regime_gate_v1",
        start=start,
        end=end,
        enriched_frames=enriched,
    )

    assert asdict(summary_loaded) == asdict(summary_preloaded)
    assert [asdict(item) for item in trades_loaded] == [asdict(item) for item in trades_preloaded]
    assert asdict(summary_loaded) == asdict(summary_enriched)
    assert [asdict(item) for item in trades_loaded] == [asdict(item) for item in trades_enriched]


def test_confluence_setup_profile_runs_through_enriched_backtest_path() -> None:
    service = make_service()
    start = datetime(2025, 2, 1, tzinfo=timezone.utc)
    end = datetime(2025, 2, 15, tzinfo=timezone.utc)
    full_frames = {
        "1d": make_ohlcv_frame(periods=360, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=360, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=360, start_price=2600, up_step=1.1, freq="h"),
    }

    summary_frames, trades_frames = service.run_symbol_strategy_with_frames(
        symbol="BTC/USDT:USDT",
        strategy_profile="swing_trend_confluence_setup_v1",
        start=start,
        end=end,
        frames=full_frames,
    )
    enriched = service._prepare_enriched_frames("swing_trend_confluence_setup_v1", full_frames)
    summary_enriched, trades_enriched = service.run_symbol_strategy_with_enriched_frames(
        symbol="BTC/USDT:USDT",
        strategy_profile="swing_trend_confluence_setup_v1",
        start=start,
        end=end,
        enriched_frames=enriched,
    )

    assert asdict(summary_frames) == asdict(summary_enriched)
    assert [asdict(item) for item in trades_frames] == [asdict(item) for item in trades_enriched]


def test_level_aware_confirmation_profile_runs_through_enriched_backtest_path() -> None:
    service = make_service()
    start = datetime(2025, 2, 1, tzinfo=timezone.utc)
    end = datetime(2025, 2, 15, tzinfo=timezone.utc)
    full_frames = {
        "1d": make_ohlcv_frame(periods=360, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=360, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=360, start_price=2600, up_step=1.1, freq="h"),
    }

    summary_frames, trades_frames = service.run_symbol_strategy_with_frames(
        symbol="BTC/USDT:USDT",
        strategy_profile="swing_trend_level_aware_confirmation_v1",
        start=start,
        end=end,
        frames=full_frames,
    )
    enriched = service._prepare_enriched_frames("swing_trend_level_aware_confirmation_v1", full_frames)
    summary_enriched, trades_enriched = service.run_symbol_strategy_with_enriched_frames(
        symbol="BTC/USDT:USDT",
        strategy_profile="swing_trend_level_aware_confirmation_v1",
        start=start,
        end=end,
        enriched_frames=enriched,
    )

    assert asdict(summary_frames) == asdict(summary_enriched)
    assert [asdict(item) for item in trades_frames] == [asdict(item) for item in trades_enriched]


def test_axis_band_state_note_profile_runs_through_enriched_backtest_path() -> None:
    service = make_service()
    start = datetime(2025, 2, 1, tzinfo=timezone.utc)
    end = datetime(2025, 2, 15, tzinfo=timezone.utc)
    full_frames = {
        "1d": make_ohlcv_frame(periods=360, start_price=1800, up_step=8.0, freq="D"),
        "4h": make_ohlcv_frame(periods=360, start_price=2200, up_step=1.8, freq="4h"),
        "1h": make_ohlcv_frame(periods=360, start_price=2600, up_step=1.1, freq="h"),
    }

    summary_frames, trades_frames = service.run_symbol_strategy_with_frames(
        symbol="BTC/USDT:USDT",
        strategy_profile="swing_trend_axis_band_state_note_v1",
        start=start,
        end=end,
        frames=full_frames,
    )
    enriched = service._prepare_enriched_frames("swing_trend_axis_band_state_note_v1", full_frames)
    summary_enriched, trades_enriched = service.run_symbol_strategy_with_enriched_frames(
        symbol="BTC/USDT:USDT",
        strategy_profile="swing_trend_axis_band_state_note_v1",
        start=start,
        end=end,
        enriched_frames=enriched,
    )

    assert asdict(summary_frames) == asdict(summary_enriched)
    assert [asdict(item) for item in trades_frames] == [asdict(item) for item in trades_enriched]
