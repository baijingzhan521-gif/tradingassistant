from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable

import httpx
import pandas as pd

from app.core.config import Settings, get_settings
from app.core.exceptions import ExternalServiceError


logger = logging.getLogger(__name__)

BYBIT_BASE_URL = "https://api.bybit.com"
MS_IN_HOUR = 60 * 60 * 1000


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class BybitDerivativesClient:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        proxy = self.settings.ccxt_https_proxy or self.settings.ccxt_http_proxy
        self.client = httpx.Client(
            base_url=BYBIT_BASE_URL,
            timeout=self.settings.ccxt_timeout_ms / 1000,
            trust_env=self.settings.ccxt_trust_env,
            proxy=proxy,
            transport=transport,
        )

    def fetch_funding_history(self, *, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        rows = self._fetch_descending_rows(
            path="/v5/market/funding/history",
            base_params={"category": "linear", "symbol": symbol},
            start=start,
            end=end,
            limit=200,
            start_key="startTime",
            end_key="endTime",
            result_key="list",
            timestamp_parser=lambda item: int(item["fundingRateTimestamp"]),
        )
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([int(item["fundingRateTimestamp"]) for item in rows], unit="ms", utc=True),
                "funding_rate_event": [float(item["fundingRate"]) for item in rows],
            }
        )
        return frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    def fetch_open_interest_history(self, *, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        rows = self._fetch_descending_rows(
            path="/v5/market/open-interest",
            base_params={"category": "linear", "symbol": symbol, "intervalTime": "1h"},
            start=start,
            end=end,
            limit=200,
            start_key="startTime",
            end_key="endTime",
            result_key="list",
            timestamp_parser=lambda item: int(item["timestamp"]),
        )
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([int(item["timestamp"]) for item in rows], unit="ms", utc=True),
                "open_interest": [float(item["openInterest"]) for item in rows],
            }
        )
        return frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    def fetch_mark_price_klines(self, *, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return self._fetch_kline_series(
            path="/v5/market/mark-price-kline",
            symbol=symbol,
            start=start,
            end=end,
            value_name="mark_close",
        )

    def fetch_index_price_klines(self, *, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return self._fetch_kline_series(
            path="/v5/market/index-price-kline",
            symbol=symbol,
            start=start,
            end=end,
            value_name="index_close",
        )

    def fetch_premium_index_klines(self, *, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return self._fetch_kline_series(
            path="/v5/market/premium-index-price-kline",
            symbol=symbol,
            start=start,
            end=end,
            value_name="premium_close",
        )

    def close(self) -> None:
        self.client.close()

    def _fetch_kline_series(
        self,
        *,
        path: str,
        symbol: str,
        start: datetime,
        end: datetime,
        value_name: str,
    ) -> pd.DataFrame:
        rows = self._fetch_descending_rows(
            path=path,
            base_params={"category": "linear", "symbol": symbol, "interval": "60"},
            start=start,
            end=end,
            limit=1000,
            start_key="start",
            end_key="end",
            result_key="list",
            timestamp_parser=lambda item: int(item[0]),
        )
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([int(item[0]) for item in rows], unit="ms", utc=True),
                value_name: [float(item[4]) for item in rows],
            }
        )
        return frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    def _fetch_descending_rows(
        self,
        *,
        path: str,
        base_params: dict[str, Any],
        start: datetime,
        end: datetime,
        limit: int,
        start_key: str,
        end_key: str,
        result_key: str,
        timestamp_parser: Callable[[Any], int],
    ) -> list[Any]:
        start_ms = int(_to_utc(start).timestamp() * 1000)
        end_ms = int(_to_utc(end).timestamp() * 1000) - 1
        if start_ms > end_ms:
            raise ExternalServiceError("Invalid Bybit range: start must be earlier than end")

        rows: list[Any] = []
        seen_timestamps: set[int] = set()
        current_end_ms = end_ms

        while current_end_ms >= start_ms:
            params = dict(base_params)
            params[start_key] = start_ms
            params[end_key] = current_end_ms
            params["limit"] = limit
            payload = self._get(path, params=params)
            batch = payload.get("result", {}).get(result_key, [])
            if not batch:
                break

            oldest_ts: int | None = None
            for item in batch:
                item_ts = timestamp_parser(item)
                oldest_ts = item_ts if oldest_ts is None else min(oldest_ts, item_ts)
                if start_ms <= item_ts <= end_ms and item_ts not in seen_timestamps:
                    rows.append(item)
                    seen_timestamps.add(item_ts)

            if oldest_ts is None or oldest_ts >= current_end_ms:
                break
            current_end_ms = oldest_ts - 1
            if oldest_ts < start_ms:
                break

        return rows

    def _get(self, path: str, *, params: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.settings.ccxt_max_retries + 2):
            try:
                response = self.client.get(path, params=params)
                response.raise_for_status()
                payload = response.json()
                if payload.get("retCode") != 0:
                    raise ExternalServiceError(
                        f"Bybit API returned retCode={payload.get('retCode')} retMsg={payload.get('retMsg')}"
                    )
                return payload
            except (httpx.HTTPError, ValueError, ExternalServiceError) as exc:
                last_error = exc
                if attempt > self.settings.ccxt_max_retries:
                    raise ExternalServiceError(f"Bybit request failed path={path} params={params}: {exc}") from exc
                logger.warning(
                    "Retrying Bybit request path=%s attempt=%s/%s error=%s",
                    path,
                    attempt,
                    self.settings.ccxt_max_retries + 1,
                    exc,
                )
                time.sleep(self.settings.ccxt_retry_delay_ms / 1000)

        raise ExternalServiceError(f"Bybit request failed path={path} params={params}: {last_error}")
