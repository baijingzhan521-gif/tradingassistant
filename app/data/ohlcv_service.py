from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import ccxt
import pandas as pd

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError, SymbolNotFoundError, UnsupportedTimeframeError
from app.data.exchange_client import ExchangeClientFactory
from app.utils.timeframes import SUPPORTED_TIMEFRAMES, TIMEFRAME_TO_MINUTES


logger = logging.getLogger(__name__)
settings = get_settings()


class OhlcvService:
    def __init__(self, exchange_factory: ExchangeClientFactory) -> None:
        self.exchange_factory = exchange_factory

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        message = str(error).lower()
        return isinstance(error, ccxt.DDoSProtection) or "too many requests" in message or "429" in message

    @staticmethod
    def _retry_delay_seconds(*, attempt: int, rate_limited: bool) -> float:
        base_delay = max(settings.ccxt_retry_delay_ms / 1000.0, 0.1)
        delay = base_delay * (2 ** max(attempt - 1, 0))
        if rate_limited:
            delay = max(delay, float(attempt * 2))
        return delay

    def fetch_ohlcv(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> pd.DataFrame:
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise UnsupportedTimeframeError(f"Unsupported timeframe: {timeframe}")

        client = self.exchange_factory.get_client(exchange, market_type)

        rows: list[list[Any]] | None = None
        last_error: Exception | None = None
        for attempt in range(1, settings.ccxt_max_retries + 2):
            try:
                self.exchange_factory.ensure_symbol(exchange, market_type, symbol)
                rows = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                break
            except ccxt.BadSymbol as exc:
                raise SymbolNotFoundError(f"Symbol not found on {exchange}: {symbol}") from exc
            except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
                last_error = exc
                if attempt > settings.ccxt_max_retries:
                    raise ExternalServiceError(
                        f"Failed to fetch OHLCV from {exchange}: {exc}. "
                        "Check direct network reachability or configure CCXT_HTTP_PROXY/CCXT_HTTPS_PROXY "
                        "or HTTP_PROXY/HTTPS_PROXY."
                    ) from exc
                logger.warning(
                    "Retrying OHLCV fetch exchange=%s symbol=%s timeframe=%s attempt=%s/%s error=%s",
                    exchange,
                    symbol,
                    timeframe,
                    attempt,
                    settings.ccxt_max_retries + 1,
                    exc,
                )
                time.sleep(settings.ccxt_retry_delay_ms / 1000)

        if rows is None:
            raise ExternalServiceError(f"Failed to fetch OHLCV from {exchange}: {last_error}")

        if not rows:
            raise ExternalServiceError(f"No OHLCV data returned for {symbol} on timeframe {timeframe}")

        frame = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        frame["timestamp"] = pd.to_datetime(frame["ts"], unit="ms", utc=True)
        frame = frame.drop(columns=["ts"]).sort_values("timestamp").reset_index(drop=True)
        if len(frame) < 200:
            raise ExternalServiceError(
                f"Not enough OHLCV data for {symbol} {timeframe}. Need at least 200 candles, got {len(frame)}."
            )
        logger.info(
            "Fetched OHLCV: exchange=%s symbol=%s market_type=%s timeframe=%s candles=%s",
            exchange,
            symbol,
            market_type,
            timeframe,
            len(frame),
        )
        return frame

    def fetch_multi_timeframe(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframes: list[str],
        limit: int,
    ) -> dict[str, pd.DataFrame]:
        return {
            timeframe: self.fetch_ohlcv(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
            )
            for timeframe in timeframes
        }

    def fetch_ohlcv_range(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit_per_call: int = 1500,
    ) -> pd.DataFrame:
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise UnsupportedTimeframeError(f"Unsupported timeframe: {timeframe}")
        if start >= end:
            raise ExternalServiceError("Invalid OHLCV range: start must be earlier than end")

        client = self.exchange_factory.get_client(exchange, market_type)
        timeframe_ms = int(TIMEFRAME_TO_MINUTES[timeframe] * 60 * 1000)
        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        rows: list[list[Any]] = []
        seen_timestamps: set[int] = set()

        while since_ms < end_ms:
            batch: list[list[Any]] | None = None
            for attempt in range(1, settings.ccxt_max_retries + 2):
                try:
                    self.exchange_factory.ensure_symbol(exchange, market_type, symbol)
                    batch = client.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit_per_call)
                    break
                except ccxt.BadSymbol as exc:
                    raise SymbolNotFoundError(f"Symbol not found on {exchange}: {symbol}") from exc
                except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
                    if attempt > settings.ccxt_max_retries:
                        raise ExternalServiceError(
                            f"Failed to fetch OHLCV range from {exchange}: {exc}. "
                            "Check direct network reachability or configure CCXT_HTTP_PROXY/CCXT_HTTPS_PROXY "
                            "or HTTP_PROXY/HTTPS_PROXY."
                        ) from exc
                    rate_limited = self._is_rate_limit_error(exc)
                    delay_seconds = self._retry_delay_seconds(attempt=attempt, rate_limited=rate_limited)
                    logger.warning(
                        "Retrying OHLCV range fetch exchange=%s symbol=%s timeframe=%s since_ms=%s attempt=%s/%s "
                        "rate_limited=%s sleep_s=%.2f error=%s",
                        exchange,
                        symbol,
                        timeframe,
                        since_ms,
                        attempt,
                        settings.ccxt_max_retries + 1,
                        rate_limited,
                        delay_seconds,
                        exc,
                    )
                    time.sleep(delay_seconds)

            if batch is None:
                raise ExternalServiceError(
                    f"Failed to fetch OHLCV range from {exchange}: empty retry result for {symbol} {timeframe}."
                )

            if not batch:
                break

            advanced = False
            for item in batch:
                ts = int(item[0])
                if ts >= end_ms:
                    continue
                if ts not in seen_timestamps:
                    rows.append(item)
                    seen_timestamps.add(ts)
                if ts + timeframe_ms > since_ms:
                    since_ms = ts + timeframe_ms
                    advanced = True

            if not advanced:
                since_ms += timeframe_ms

        if not rows:
            raise ExternalServiceError(
                f"No OHLCV data returned for {symbol} on timeframe {timeframe} between {start.isoformat()} and {end.isoformat()}"
            )

        frame = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        frame["timestamp"] = pd.to_datetime(frame["ts"], unit="ms", utc=True)
        frame = frame.drop(columns=["ts"]).sort_values("timestamp").reset_index(drop=True)
        logger.info(
            "Fetched OHLCV range: exchange=%s symbol=%s market_type=%s timeframe=%s candles=%s start=%s end=%s",
            exchange,
            symbol,
            market_type,
            timeframe,
            len(frame),
            frame["timestamp"].iloc[0].isoformat(),
            frame["timestamp"].iloc[-1].isoformat(),
        )
        return frame

    def fetch_multi_timeframe_range(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframes: list[str],
        start: datetime,
        end: datetime,
        limit_per_call: int = 1500,
    ) -> dict[str, pd.DataFrame]:
        return {
            timeframe: self.fetch_ohlcv_range(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                limit_per_call=limit_per_call,
            )
            for timeframe in timeframes
        }
