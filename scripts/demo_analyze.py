from __future__ import annotations

import argparse
import json
import sys

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call the local /analyze endpoint with a simple ETH-style payload.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--symbol", default="ETH/USDT:USDT", help="Unified ccxt symbol")
    parser.add_argument("--exchange", default="binance", help="Exchange id")
    parser.add_argument("--market-type", default="perpetual", help="Market type")
    parser.add_argument("--strategy-profile", default="trend_pullback_v1", help="Strategy profile")
    parser.add_argument("--lookback", type=int, default=300, help="Number of candles per timeframe")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = {
        "symbol": args.symbol,
        "market_type": args.market_type,
        "exchange": args.exchange,
        "timeframes": ["1d", "4h", "1h", "15m"],
        "strategy_profile": args.strategy_profile,
        "lookback": args.lookback,
    }

    try:
        response = httpx.post(f"{args.base_url.rstrip('/')}/analyze", json=payload, timeout=args.timeout)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        print(f"Request failed with status {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
        return 1
    except httpx.HTTPError as exc:
        print(
            f"Failed to reach {args.base_url.rstrip('/')}/analyze: {exc}. "
            "Make sure the FastAPI server is running and reachable.",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
