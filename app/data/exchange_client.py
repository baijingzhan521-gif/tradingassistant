from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import ccxt

from app.core.config import Settings, get_settings
from app.core.exceptions import SymbolNotFoundError, UnsupportedExchangeError, UnsupportedMarketTypeError
from app.schemas.common import MarketType


logger = logging.getLogger(__name__)


class ExchangeClientFactory:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._clients: dict[tuple[str, str], Any] = {}

    def get_client(self, exchange: str, market_type: str) -> Any:
        key = (exchange, market_type)
        if key in self._clients:
            return self._clients[key]

        if exchange != "binance":
            raise UnsupportedExchangeError(f"Unsupported exchange: {exchange}")

        if market_type != MarketType.PERPETUAL.value:
            raise UnsupportedMarketTypeError(
                f"Unsupported market type for MVP: {market_type}. Only 'perpetual' is supported."
            )

        client = ccxt.binanceusdm(
            {
                "enableRateLimit": True,
                "timeout": self.settings.ccxt_timeout_ms,
            }
        )
        client.session.trust_env = self.settings.ccxt_trust_env
        explicit_http_proxy = self.settings.ccxt_http_proxy
        explicit_https_proxy = self.settings.ccxt_https_proxy or explicit_http_proxy
        if explicit_http_proxy or explicit_https_proxy:
            client.session.proxies.update(
                {
                    "http": explicit_http_proxy or explicit_https_proxy,
                    "https": explicit_https_proxy or explicit_http_proxy,
                }
            )
        if self.settings.binance_use_sandbox:
            client.set_sandbox_mode(True)
        if self.settings.binance_hostname:
            client.hostname = self.settings.binance_hostname
        if self.settings.ccxt_socks_proxy:
            client.socksProxy = self.settings.ccxt_socks_proxy

        self._clients[key] = client
        logger.info(
            "Created exchange client exchange=%s market_type=%s ccxt_id=%s sandbox=%s hostname=%s trust_env=%s session_proxy=%s",
            exchange,
            market_type,
            client.id,
            self.settings.binance_use_sandbox,
            self.settings.binance_hostname or "default",
            self.settings.ccxt_trust_env,
            explicit_https_proxy or explicit_http_proxy or "none",
        )
        return client

    def list_symbols(self, exchange: str, market_type: str, limit: int = 200) -> list[str]:
        client = self.get_client(exchange, market_type)
        client.load_markets()
        symbols = [
            symbol
            for symbol in client.symbols
            if symbol.endswith(":USDT") and "/USDT" in symbol and client.markets[symbol].get("active", True)
        ]
        return sorted(symbols)[:limit]

    def ensure_symbol(self, exchange: str, market_type: str, symbol: str) -> None:
        client = self.get_client(exchange, market_type)
        client.load_markets()
        if symbol not in client.symbols:
            raise SymbolNotFoundError(f"Symbol not found on {exchange} {market_type}: {symbol}")


@lru_cache(maxsize=1)
def get_exchange_client_factory() -> ExchangeClientFactory:
    return ExchangeClientFactory(get_settings())
