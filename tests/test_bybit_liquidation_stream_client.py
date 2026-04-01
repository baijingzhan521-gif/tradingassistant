from __future__ import annotations

import json

from app.core.config import Settings
from app.data.bybit_liquidation_stream_client import BybitLiquidationStreamClient


class FakeWebSocket:
    def __init__(self, messages: list[str]) -> None:
        self.messages = iter(messages)
        self.sent: list[str] = []

    def __enter__(self) -> "FakeWebSocket":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def send(self, payload: str) -> None:
        self.sent.append(payload)

    def recv(self, timeout: float | None = None) -> str:
        return next(self.messages)


def test_stream_client_subscribes_and_forwards_liquidation_payloads() -> None:
    captured: list[dict] = []
    fake_socket = FakeWebSocket(
        [
            json.dumps({"success": True, "op": "subscribe"}),
            json.dumps({"topic": "tickers.BTCUSDT", "data": [{"symbol": "BTCUSDT"}]}),
            json.dumps(
                {
                    "topic": "allLiquidation.BTCUSDT",
                    "type": "snapshot",
                    "ts": 1710000000000,
                    "data": [{"T": 1710000000000, "s": "BTCUSDT", "S": "Buy", "v": "0.5", "p": "50000"}],
                }
            ),
        ]
    )

    client = BybitLiquidationStreamClient(
        Settings(ccxt_timeout_ms=1_000, ccxt_trust_env=False),
        connector=lambda *args, **kwargs: fake_socket,
    )
    summary = client.stream(symbols=["BTCUSDT"], on_payload=captured.append, max_messages=1, max_runtime_seconds=5)

    assert summary["ack_received"] is True
    assert summary["liquidation_messages"] == 1
    assert summary["liquidation_events"] == 1
    assert len(captured) == 1
    subscribe_payload = json.loads(fake_socket.sent[0])
    assert subscribe_payload == {"op": "subscribe", "args": ["allLiquidation.BTCUSDT"]}
