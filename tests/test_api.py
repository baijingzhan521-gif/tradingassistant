from __future__ import annotations

from datetime import datetime, timezone

from app.api.dependencies import get_analysis_service
from app.api.dependencies import get_backtest_artifact_service
from app.schemas.response import (
    BacktestTradeSnapshot,
    WorkspaceBacktestTradeBookResponse,
    WorkspaceBatchAnalysisItem,
    WorkspaceBatchAnalysisResponse,
)


def _batch_payload_from_sample(sample_analysis_result):
    swing = sample_analysis_result.model_copy(update={"strategy_profile": "swing_trend_long_regime_gate_v1"})
    intraday = sample_analysis_result.model_copy(
        update={
            "analysis_id": "intraday-analysis-id",
            "strategy_profile": "intraday_mtf_v1",
            "timeframes": sample_analysis_result.timeframes.model_copy(
                update={
                    "day_1": None,
                    "hour_4": None,
                    "min_3": sample_analysis_result.timeframes.hour_1.model_copy(
                        update={"timeframe": "3m"}
                    ),
                }
            ),
            "charts": None,
        }
    )
    return WorkspaceBatchAnalysisResponse(
        batch_id="workspace-batch-id",
        timestamp=swing.timestamp,
        symbol=swing.symbol,
        exchange=swing.exchange,
        market_type=swing.market_type,
        strategy_profiles=["swing_trend_long_regime_gate_v1", "intraday_mtf_v1"],
        analyses=[
            WorkspaceBatchAnalysisItem(strategy_profile="swing_trend_long_regime_gate_v1", analysis=swing),
            WorkspaceBatchAnalysisItem(strategy_profile="intraday_mtf_v1", analysis=intraday),
        ],
    )


class FakeAnalysisService:
    def __init__(self, sample_analysis_result):
        self._sample = sample_analysis_result
        self._batch = _batch_payload_from_sample(sample_analysis_result)

    def analyze(self, request, db):
        return self._sample

    def analyze_batch(self, request, db):
        return self._batch

    def get_analysis(self, db, analysis_id):
        return self._sample

    def list_analyses(self, db, limit=50, offset=0):
        return {"items": [], "total": 0}


class FakeBacktestArtifactService:
    def get_best_btc_trade_book(self) -> WorkspaceBacktestTradeBookResponse:
        timestamp = datetime(2026, 3, 19, 12, 8, tzinfo=timezone.utc)
        return WorkspaceBacktestTradeBookResponse(
            dataset="confirmed_swing_ablation_latest_best_profile",
            dataset_generated_at="20260319T120857Z",
            symbol="BTC/USDT:USDT",
            strategy_profile="swing_trend_long_regime_gate_v1",
            swing_detection_mode="confirmed",
            exit_profile_label="LONG scaled 1R -> 3R, SHORT fixed 1.5R",
            total_trades=2,
            profit_factor=1.68,
            expectancy_r=0.28,
            cumulative_r=0.77,
            max_drawdown_r=1.12,
            long_trades=1,
            long_r=1.88,
            short_trades=1,
            short_r=-1.11,
            ranking_source_csv="artifacts/backtests/btc_confirmed_swing_ablation/confirmed_swing_ablation_20260319T120857Z.csv",
            trades_source_csv="artifacts/backtests/btc_confirmed_swing_ablation/confirmed_swing_ablation_20260319T120857Z_trades.csv",
            trades=[
                BacktestTradeSnapshot(
                    sequence=1,
                    symbol="BTC/USDT:USDT",
                    strategy_profile="swing_trend_long_regime_gate_v1",
                    side="LONG",
                    signal_time=timestamp,
                    entry_time=timestamp,
                    exit_time=timestamp,
                    entry_price=100000.0,
                    exit_price=101500.0,
                    stop_price=99000.0,
                    tp1_price=101000.0,
                    tp2_price=103000.0,
                    bars_held=12,
                    exit_reason="tp2",
                    confidence=99,
                    tp1_hit=True,
                    tp2_hit=True,
                    pnl_pct=1.5,
                    pnl_r=1.88,
                    cumulative_r_after_trade=1.88,
                ),
                BacktestTradeSnapshot(
                    sequence=2,
                    symbol="BTC/USDT:USDT",
                    strategy_profile="swing_trend_long_regime_gate_v1",
                    side="SHORT",
                    signal_time=timestamp,
                    entry_time=timestamp,
                    exit_time=timestamp,
                    entry_price=98000.0,
                    exit_price=99000.0,
                    stop_price=99000.0,
                    tp1_price=97000.0,
                    tp2_price=96500.0,
                    bars_held=8,
                    exit_reason="stop_loss",
                    confidence=96,
                    tp1_hit=False,
                    tp2_hit=False,
                    pnl_pct=-1.02,
                    pnl_r=-1.11,
                    cumulative_r_after_trade=0.77,
                ),
            ],
        )


def test_analyze_endpoint_returns_structured_payload(client, app_instance, sample_analysis_result) -> None:
    app_instance.dependency_overrides[get_analysis_service] = lambda: FakeAnalysisService(sample_analysis_result)

    payload = {
        "symbol": "ETH/USDT:USDT",
        "market_type": "perpetual",
        "exchange": "binance",
        "timeframes": ["1d", "4h", "1h", "15m"],
        "strategy_profile": "trend_pullback_v1",
        "lookback": 300,
    }
    response = client.post("/analyze", json=payload)
    data = response.json()

    assert response.status_code == 201
    assert data["analysis_id"] == "test-analysis-id"
    assert data["decision"]["action"] == "LONG"
    assert data["decision"]["bias"] == "bullish"
    assert "1d" in data["timeframes"]
    assert "diagnostics" in data
    assert "reasoning" in data


def test_analyze_endpoint_rejects_incomplete_timeframes(client) -> None:
    payload = {
        "symbol": "ETH/USDT:USDT",
        "market_type": "perpetual",
        "exchange": "binance",
        "timeframes": ["4h", "1h", "15m"],
        "strategy_profile": "trend_pullback_v1",
        "lookback": 300,
    }

    response = client.post("/analyze", json=payload)
    assert response.status_code == 422


def test_workspace_batch_endpoint_returns_intraday_and_swing(client, app_instance, sample_analysis_result) -> None:
    app_instance.dependency_overrides[get_analysis_service] = lambda: FakeAnalysisService(sample_analysis_result)

    payload = {
        "symbol": "ETH/USDT:USDT",
        "market_type": "perpetual",
        "exchange": "binance",
        "strategy_profiles": ["swing_trend_long_regime_gate_v1", "intraday_mtf_v1"],
        "lookback": 300,
    }

    response = client.post("/workspace/analyze/batch", json=payload)
    data = response.json()

    assert response.status_code == 201
    assert data["batch_id"] == "workspace-batch-id"
    assert data["strategy_profiles"] == ["swing_trend_long_regime_gate_v1", "intraday_mtf_v1"]
    assert len(data["analyses"]) == 2
    assert {item["strategy_profile"] for item in data["analyses"]} == {"swing_trend_long_regime_gate_v1", "intraday_mtf_v1"}


def test_workspace_backtest_trade_book_endpoint_returns_latest_btc_trade_book(client, app_instance) -> None:
    app_instance.dependency_overrides[get_backtest_artifact_service] = lambda: FakeBacktestArtifactService()

    response = client.get("/workspace/backtest-trades/btc-best")
    data = response.json()

    assert response.status_code == 200
    assert data["strategy_profile"] == "swing_trend_long_regime_gate_v1"
    assert data["symbol"] == "BTC/USDT:USDT"
    assert data["total_trades"] == 2
    assert len(data["trades"]) == 2
    assert data["trades"][0]["sequence"] == 1
    assert data["trades"][1]["cumulative_r_after_trade"] == 0.77
