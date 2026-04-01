from __future__ import annotations

from datetime import UTC, datetime, timedelta

import ccxt

from app.data.ohlcv_service import OhlcvService, settings


class _FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def fetch_ohlcv(self, symbol: str, *, timeframe: str, since: int, limit: int) -> list[list[float]]:
        self.calls += 1
        if self.calls == 1:
            raise ccxt.DDoSProtection("429 Too Many Requests")
        return [
            [float(since), 1.0, 1.0, 1.0, 1.0, 1.0],
            [float(since + 3 * 60 * 1000), 1.0, 1.0, 1.0, 1.0, 1.0],
        ]


class _FakeExchangeFactory:
    def __init__(self, client: _FakeClient) -> None:
        self._client = client
        self.ensure_symbol_calls = 0

    def get_client(self, exchange: str, market_type: str) -> _FakeClient:
        return self._client

    def ensure_symbol(self, exchange: str, market_type: str, symbol: str) -> None:
        self.ensure_symbol_calls += 1


def test_fetch_ohlcv_range_retries_on_rate_limit(monkeypatch) -> None:
    client = _FakeClient()
    service = OhlcvService(_FakeExchangeFactory(client))

    sleep_calls: list[float] = []
    monkeypatch.setattr(settings, "ccxt_max_retries", 2)
    monkeypatch.setattr(settings, "ccxt_retry_delay_ms", 100)
    monkeypatch.setattr("app.data.ohlcv_service.time.sleep", lambda seconds: sleep_calls.append(seconds))

    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=9)
    frame = service.fetch_ohlcv_range(
        exchange="binance",
        market_type="perpetual",
        symbol="BTC/USDT:USDT",
        timeframe="3m",
        start=start,
        end=end,
        limit_per_call=1000,
    )

    assert len(frame) >= 2
    assert client.calls >= 2
    assert len(sleep_calls) == 1
    assert sleep_calls[0] >= 2.0
