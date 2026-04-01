from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as websocket_connect

from app.core.config import Settings, get_settings
from app.core.exceptions import ExternalServiceError


logger = logging.getLogger(__name__)


class BybitLiquidationStreamClient:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        connector: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.connector = connector or websocket_connect

    def stream(
        self,
        *,
        symbols: Sequence[str],
        on_payload: Callable[[dict[str, Any]], None],
        max_messages: int | None = None,
        max_runtime_seconds: float | None = None,
    ) -> dict[str, int | float | bool]:
        if not symbols:
            raise ValueError("symbols must not be empty")

        topics = [f"allLiquidation.{symbol}" for symbol in symbols]
        proxy = self.settings.ccxt_https_proxy or self.settings.ccxt_http_proxy
        if proxy is None and not self.settings.ccxt_trust_env:
            proxy = None

        started_at = time.monotonic()
        raw_messages = 0
        liquidation_messages = 0
        liquidation_events = 0
        ack_received = False

        try:
            with self.connector(
                self.settings.bybit_public_linear_ws_url,
                open_timeout=self.settings.ccxt_timeout_ms / 1000,
                ping_interval=20,
                ping_timeout=20,
                proxy=proxy if proxy is not None else self.settings.ccxt_trust_env,
            ) as websocket:
                websocket.send(json.dumps({"op": "subscribe", "args": topics}))

                while True:
                    elapsed = time.monotonic() - started_at
                    if max_runtime_seconds is not None and elapsed >= max_runtime_seconds:
                        break

                    recv_timeout = 1.0
                    if max_runtime_seconds is not None:
                        recv_timeout = min(recv_timeout, max(0.1, max_runtime_seconds - elapsed))

                    try:
                        message = websocket.recv(timeout=recv_timeout)
                    except TimeoutError:
                        if max_runtime_seconds is not None and (time.monotonic() - started_at) >= max_runtime_seconds:
                            break
                        continue

                    payload = json.loads(message)
                    raw_messages += 1

                    if payload.get("op") == "subscribe":
                        if not payload.get("success", False):
                            raise ExternalServiceError(f"Bybit liquidation subscribe failed: {payload}")
                        ack_received = True
                        continue

                    topic = str(payload.get("topic") or "")
                    if not topic.startswith("allLiquidation."):
                        logger.debug("Ignoring non-liquidation websocket payload keys=%s", sorted(payload.keys()))
                        continue

                    liquidation_messages += 1
                    liquidation_events += len(payload.get("data") or [])
                    on_payload(payload)

                    if max_messages is not None and liquidation_messages >= max_messages:
                        break
        except (ConnectionClosed, OSError, ValueError) as exc:
            raise ExternalServiceError(f"Bybit liquidation websocket stream failed: {exc}") from exc

        return {
            "symbols": len(symbols),
            "raw_messages": raw_messages,
            "liquidation_messages": liquidation_messages,
            "liquidation_events": liquidation_events,
            "ack_received": ack_received,
            "runtime_seconds": round(time.monotonic() - started_at, 4),
        }
