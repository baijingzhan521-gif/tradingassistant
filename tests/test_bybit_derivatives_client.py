from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import parse_qs

import httpx

from app.core.config import Settings
from app.data.bybit_derivatives_client import BybitDerivativesClient


def test_bybit_client_fetches_funding_history_across_pages() -> None:
    start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

    def handler(request: httpx.Request) -> httpx.Response:
        query = {key: values[0] for key, values in parse_qs(request.url.query.decode()).items()}
        end_time = int(query["endTime"])
        if end_time >= 1704081600000:
            payload = {
                "retCode": 0,
                "retMsg": "OK",
                "result": {
                    "list": [
                        {"symbol": "BTCUSDT", "fundingRate": "0.0003", "fundingRateTimestamp": "1704096000000"},
                        {"symbol": "BTCUSDT", "fundingRate": "0.0002", "fundingRateTimestamp": "1704081600000"},
                    ]
                },
            }
        else:
            payload = {
                "retCode": 0,
                "retMsg": "OK",
                "result": {
                    "list": [
                        {"symbol": "BTCUSDT", "fundingRate": "0.0001", "fundingRateTimestamp": "1704067200000"},
                    ]
                },
            }
        return httpx.Response(200, json=payload)

    client = BybitDerivativesClient(
        Settings(ccxt_timeout_ms=1_000, ccxt_max_retries=0),
        transport=httpx.MockTransport(handler),
    )
    try:
        frame = client.fetch_funding_history(symbol="BTCUSDT", start=start, end=end)
    finally:
        client.close()

    assert frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z").tolist() == [
        "2024-01-01 00:00:00+0000",
        "2024-01-01 04:00:00+0000",
        "2024-01-01 08:00:00+0000",
    ]
    assert frame["funding_rate_event"].tolist() == [0.0001, 0.0002, 0.0003]


def test_bybit_client_fetches_mark_price_kline_series() -> None:
    start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 3, 0, tzinfo=timezone.utc)

    def handler(_: httpx.Request) -> httpx.Response:
        payload = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    ["1704074400000", "101", "102", "100", "101.5"],
                    ["1704070800000", "100", "101", "99", "100.5"],
                    ["1704067200000", "99", "100", "98", "99.5"],
                ]
            },
        }
        return httpx.Response(200, json=payload)

    client = BybitDerivativesClient(
        Settings(ccxt_timeout_ms=1_000, ccxt_max_retries=0),
        transport=httpx.MockTransport(handler),
    )
    try:
        frame = client.fetch_mark_price_klines(symbol="BTCUSDT", start=start, end=end)
    finally:
        client.close()

    assert frame["mark_close"].tolist() == [99.5, 100.5, 101.5]
    assert frame["timestamp"].dt.hour.tolist() == [0, 1, 2]
