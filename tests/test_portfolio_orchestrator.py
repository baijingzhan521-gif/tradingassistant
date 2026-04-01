from __future__ import annotations

import pytest

from app.backtesting.portfolio_orchestrator import (
    PortfolioConfig,
    PortfolioState,
    apply_drawdown_circuit_breaker,
    compute_dynamic_leverage,
    REGIME_WEIGHTS,
    REGIME_MAX_LEVERAGE,
    PortfolioOrchestrator,
)
from app.backtesting.service import BacktestAssumptions, BacktestTrade
from app.indicators.regime_classifier import MarketRegimeType


class TestDynamicLeverage:
    def test_bull_trend_max_leverage(self) -> None:
        config = PortfolioConfig()
        lev = compute_dynamic_leverage(
            regime=MarketRegimeType.BULL_TREND.value,
            confidence=100,
            current_drawdown_pct=0.0,
            realized_vol=0.5,
            config=config,
        )
        assert lev == 3.0  # Max for bull trend

    def test_high_vol_chop_reduces_leverage(self) -> None:
        config = PortfolioConfig()
        lev = compute_dynamic_leverage(
            regime=MarketRegimeType.HIGH_VOL_CHOP.value,
            confidence=80,
            current_drawdown_pct=0.0,
            realized_vol=0.5,
            config=config,
        )
        assert lev <= 0.5

    def test_drawdown_reduces_leverage(self) -> None:
        config = PortfolioConfig()
        lev_no_dd = compute_dynamic_leverage(
            regime=MarketRegimeType.BULL_TREND.value,
            confidence=80,
            current_drawdown_pct=0.0,
            realized_vol=0.5,
            config=config,
        )
        lev_with_dd = compute_dynamic_leverage(
            regime=MarketRegimeType.BULL_TREND.value,
            confidence=80,
            current_drawdown_pct=15.0,
            realized_vol=0.5,
            config=config,
        )
        assert lev_with_dd < lev_no_dd

    def test_high_vol_reduces_leverage(self) -> None:
        config = PortfolioConfig()
        lev_normal = compute_dynamic_leverage(
            regime=MarketRegimeType.BULL_TREND.value,
            confidence=80,
            current_drawdown_pct=0.0,
            realized_vol=0.5,
            config=config,
        )
        lev_high_vol = compute_dynamic_leverage(
            regime=MarketRegimeType.BULL_TREND.value,
            confidence=80,
            current_drawdown_pct=0.0,
            realized_vol=1.5,
            config=config,
        )
        assert lev_high_vol < lev_normal

    def test_leverage_never_exceeds_max(self) -> None:
        config = PortfolioConfig(max_leverage=2.0)
        lev = compute_dynamic_leverage(
            regime=MarketRegimeType.BULL_TREND.value,
            confidence=100,
            current_drawdown_pct=0.0,
            realized_vol=0.1,
            config=config,
        )
        assert lev <= 2.0

    def test_leverage_never_negative(self) -> None:
        config = PortfolioConfig()
        lev = compute_dynamic_leverage(
            regime=MarketRegimeType.HIGH_VOL_CHOP.value,
            confidence=10,
            current_drawdown_pct=24.0,
            realized_vol=3.0,
            config=config,
        )
        assert lev >= 0.0


class TestDrawdownCircuitBreaker:
    def test_no_drawdown_stays_level_0(self) -> None:
        state = PortfolioState(equity=10000.0, peak_equity=10000.0)
        config = PortfolioConfig()
        result = apply_drawdown_circuit_breaker(state, config)
        assert result.drawdown_level == 0
        assert result.leverage_multiplier == 1.0

    def test_level1_at_10pct_drawdown(self) -> None:
        state = PortfolioState(equity=9000.0, peak_equity=10000.0)
        config = PortfolioConfig()
        result = apply_drawdown_circuit_breaker(state, config)
        assert result.drawdown_level == 1
        assert result.leverage_multiplier == 0.7

    def test_level2_at_20pct_drawdown(self) -> None:
        state = PortfolioState(equity=8000.0, peak_equity=10000.0)
        config = PortfolioConfig()
        result = apply_drawdown_circuit_breaker(state, config)
        assert result.drawdown_level == 2
        assert result.leverage_multiplier == 0.4

    def test_level3_at_25pct_drawdown_triggers_cooldown(self) -> None:
        state = PortfolioState(equity=7500.0, peak_equity=10000.0)
        config = PortfolioConfig()
        result = apply_drawdown_circuit_breaker(state, config)
        assert result.drawdown_level == 3
        assert result.leverage_multiplier == 0.0
        assert result.cooldown_remaining == config.drawdown_cooldown_bars - 1

    def test_recovery_after_new_peak(self) -> None:
        state = PortfolioState(
            equity=10500.0,
            peak_equity=10000.0,
            drawdown_level=1,
            cooldown_remaining=0,
        )
        config = PortfolioConfig()
        result = apply_drawdown_circuit_breaker(state, config)
        assert result.drawdown_level == 0
        assert result.peak_equity == 10500.0

    def test_cooldown_ticks_down(self) -> None:
        state = PortfolioState(
            equity=7500.0,
            peak_equity=10000.0,
            drawdown_level=3,
            cooldown_remaining=10,
        )
        config = PortfolioConfig()
        result = apply_drawdown_circuit_breaker(state, config)
        assert result.cooldown_remaining == 9


class TestRegimeWeights:
    def test_all_regimes_have_weights(self) -> None:
        for regime in MarketRegimeType:
            assert regime.value in REGIME_WEIGHTS

    def test_weights_sum_to_at_most_1(self) -> None:
        for regime, weights in REGIME_WEIGHTS.items():
            total = sum(weights.values())
            assert total <= 1.001, f"{regime} weights sum to {total}"

    def test_all_regimes_have_leverage_limits(self) -> None:
        for regime in MarketRegimeType:
            assert regime.value in REGIME_MAX_LEVERAGE


class TestPortfolioConfig:
    def test_default_config(self) -> None:
        config = PortfolioConfig()
        assert len(config.strategy_profiles) == 3
        assert config.max_leverage == 3.0
        assert config.drawdown_level3_pct == 25.0


class TestCombinePortfolio:
    def test_empty_trades_returns_zeros(self) -> None:
        from app.data.exchange_client import ExchangeClientFactory
        from app.data.ohlcv_service import OhlcvService
        from app.services.strategy_service import StrategyService

        orch = PortfolioOrchestrator(
            ohlcv_service=OhlcvService(ExchangeClientFactory()),
            strategy_service=StrategyService(),
        )
        result = orch._combine_portfolio([], {})
        assert result["total_trades"] == 0
        assert result["combined_cumulative_r"] == 0.0

    def test_combine_with_trades(self) -> None:
        from app.data.exchange_client import ExchangeClientFactory
        from app.data.ohlcv_service import OhlcvService
        from app.services.strategy_service import StrategyService

        orch = PortfolioOrchestrator(
            ohlcv_service=OhlcvService(ExchangeClientFactory()),
            strategy_service=StrategyService(),
        )
        trades = [
            BacktestTrade(
                symbol="BTC/USDT:USDT",
                strategy_profile="trend_following_v1",
                side="LONG",
                higher_bias="bullish",
                trend_strength=80,
                signal_time="2026-01-01T00:00:00+00:00",
                entry_time="2026-01-01T01:00:00+00:00",
                exit_time="2026-01-10T01:00:00+00:00",
                entry_price=50000.0,
                exit_price=55000.0,
                stop_price=48000.0,
                tp1_price=52000.0,
                tp2_price=60000.0,
                bars_held=100,
                exit_reason="trailing_stop",
                confidence=75,
                tp1_hit=True,
                tp2_hit=False,
                pnl_pct=10.0,
                pnl_r=2.5,
                gross_pnl_quote=5000.0,
                fees_quote=50.0,
                leverage=2.0,
            ),
            BacktestTrade(
                symbol="BTC/USDT:USDT",
                strategy_profile="mean_reversion_v1",
                side="SHORT",
                higher_bias="neutral",
                trend_strength=30,
                signal_time="2026-01-15T00:00:00+00:00",
                entry_time="2026-01-15T01:00:00+00:00",
                exit_time="2026-01-16T01:00:00+00:00",
                entry_price=55000.0,
                exit_price=56000.0,
                stop_price=56000.0,
                tp1_price=54000.0,
                tp2_price=53000.0,
                bars_held=10,
                exit_reason="stop_loss",
                confidence=60,
                tp1_hit=False,
                tp2_hit=False,
                pnl_pct=-1.8,
                pnl_r=-1.0,
                gross_pnl_quote=-1000.0,
                fees_quote=55.0,
                leverage=1.0,
            ),
        ]
        strat_trades = {
            "trend_following_v1": [trades[0]],
            "mean_reversion_v1": [trades[1]],
        }
        result = orch._combine_portfolio(trades, strat_trades)
        assert result["total_trades"] == 2
        assert result["combined_cumulative_r"] == 1.5
        assert result["combined_win_rate"] == 50.0
        assert result["combined_profit_factor"] == 2.5
        assert len(result["equity_curve"]) == 2
