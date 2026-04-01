from __future__ import annotations

from app.core.config import Settings
from app.data.exchange_client import ExchangeClientFactory


def test_exchange_client_applies_trust_env_and_session_proxies() -> None:
    settings = Settings(
        ccxt_trust_env=True,
        ccxt_http_proxy="http://127.0.0.1:8118",
        ccxt_https_proxy="http://127.0.0.1:8118",
    )
    factory = ExchangeClientFactory(settings)

    client = factory.get_client("binance", "perpetual")

    assert client.id == "binanceusdm"
    assert client.session.trust_env is True
    assert client.session.proxies["http"] == "http://127.0.0.1:8118"
    assert client.session.proxies["https"] == "http://127.0.0.1:8118"
