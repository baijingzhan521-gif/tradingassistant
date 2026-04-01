from __future__ import annotations

import os
import socket
import sys
from typing import Optional

import ccxt
import requests


BINANCE_EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"
LOCAL_PROXY_CANDIDATES = (
    "http://127.0.0.1:8118",
    "http://127.0.0.1:8119",
    "http://127.0.0.1:6270",
)


def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def discover_local_proxy() -> Optional[str]:
    for candidate in LOCAL_PROXY_CANDIDATES:
        host, port = candidate.replace("http://", "").split(":")
        if is_port_open(host, int(port)):
            return candidate
    return None


def test_http(proxy_url: Optional[str] = None) -> tuple[bool, str]:
    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
    try:
        response = requests.get(BINANCE_EXCHANGE_INFO_URL, timeout=15, proxies=proxies)
        return True, f"HTTP {response.status_code}"
    except requests.RequestException as exc:
        return False, str(exc)


def test_ccxt(proxy_url: Optional[str] = None) -> tuple[bool, str]:
    try:
        exchange = ccxt.binanceusdm({"enableRateLimit": True, "timeout": 15000})
        exchange.session.trust_env = True
        if proxy_url:
            exchange.session.proxies.update({"http": proxy_url, "https": proxy_url})
        exchange.load_markets()
        return True, f"loaded {len(exchange.symbols)} symbols"
    except Exception as exc:  # pragma: no cover - operational script
        return False, str(exc)


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"HTTP_PROXY: {os.getenv('HTTP_PROXY') or ''}")
    print(f"HTTPS_PROXY: {os.getenv('HTTPS_PROXY') or ''}")

    direct_ok, direct_message = test_http()
    print(f"Direct HTTP check: {'OK' if direct_ok else 'FAIL'} ({direct_message})")

    proxy_url = os.getenv("CCXT_HTTPS_PROXY") or os.getenv("HTTPS_PROXY") or discover_local_proxy()
    if proxy_url:
        proxy_ok, proxy_message = test_http(proxy_url=proxy_url)
        print(f"Proxy HTTP check [{proxy_url}]: {'OK' if proxy_ok else 'FAIL'} ({proxy_message})")
        ccxt_ok, ccxt_message = test_ccxt(proxy_url=proxy_url)
        print(f"ccxt market load [{proxy_url}]: {'OK' if ccxt_ok else 'FAIL'} ({ccxt_message})")
    else:
        print("No proxy candidate detected from env or common local ports.")
        ccxt_ok, ccxt_message = test_ccxt()
        print(f"ccxt market load [direct]: {'OK' if ccxt_ok else 'FAIL'} ({ccxt_message})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
