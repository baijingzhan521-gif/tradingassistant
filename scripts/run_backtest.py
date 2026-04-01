from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.service import BacktestService
from app.backtesting.service import BacktestAssumptions
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.services.strategy_service import StrategyService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline backtests for trading-assistant strategies.")
    parser.add_argument(
        "--symbols",
        default="BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT",
        help="Comma-separated ccxt unified symbols.",
    )
    parser.add_argument(
        "--strategy-profiles",
        default="swing_trend_v1,intraday_mtf_v1",
        help="Comma-separated strategy profiles.",
    )
    parser.add_argument("--exchange", default="binance", help="Exchange id.")
    parser.add_argument("--market-type", default="perpetual", help="Market type.")
    parser.add_argument("--start", default=None, help="UTC start date, e.g. 2024-03-18")
    parser.add_argument("--end", default=None, help="UTC end date, e.g. 2026-03-18")
    parser.add_argument("--years", type=int, default=2, help="Fallback years lookback when --start is omitted.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/backtests",
        help="Directory for report json/csv output.",
    )
    parser.add_argument(
        "--take-profit-mode",
        choices=["scaled", "fixed_r"],
        default="scaled",
        help="Profit taking mode. 'scaled' keeps the current 1R/2R partial model; 'fixed_r' exits full size at a fixed R target.",
    )
    parser.add_argument(
        "--fixed-take-profit-r",
        type=float,
        default=None,
        help="Required when --take-profit-mode=fixed_r. Example: 1.5 or 2.0",
    )
    parser.add_argument("--scaled-tp1-r", type=float, default=None, help="Optional scaled TP1 target in R.")
    parser.add_argument("--scaled-tp2-r", type=float, default=None, help="Optional scaled TP2 target in R.")
    parser.add_argument(
        "--long-exit-json",
        default=None,
        help='Optional JSON override for LONG exits, e.g. \'{"take_profit_mode":"scaled","scaled_tp1_r":1,"scaled_tp2_r":3}\'',
    )
    parser.add_argument(
        "--short-exit-json",
        default=None,
        help='Optional JSON override for SHORT exits, e.g. \'{"take_profit_mode":"fixed_r","fixed_take_profit_r":3}\'',
    )
    parser.add_argument(
        "--exit-profile",
        default=None,
        help="Optional label written into the report filename.",
    )
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def parse_exit_json(value: str | None, *, label: str) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must decode to a JSON object")
    return parsed


def infer_exit_profile(args: argparse.Namespace) -> str:
    if args.exit_profile:
        return args.exit_profile
    if args.long_exit_json or args.short_exit_json:
        return "side_asymmetric_exit"
    if args.take_profit_mode == "fixed_r" and args.fixed_take_profit_r is not None:
        return f"fixed_{args.fixed_take_profit_r:g}R_full"
    if args.scaled_tp1_r is not None or args.scaled_tp2_r is not None:
        tp1 = args.scaled_tp1_r if args.scaled_tp1_r is not None else 1.0
        tp2 = args.scaled_tp2_r if args.scaled_tp2_r is not None else 2.0
        return f"scaled_{tp1:g}r_to_{tp2:g}r_be"
    return "scaled_1r_to_2r"


def print_summary(report) -> None:
    print(f"Backtest window: {report.start} -> {report.end}")
    print(f"Exit profile: {report.assumptions.get('exit_profile')}")
    print(f"Symbols: {', '.join(report.symbols)}")
    print("Overall:")
    for item in report.overall:
        print(
            f"  {item.strategy_profile}: trades={item.total_trades} win_rate={item.win_rate:.2f}% "
            f"payoff={item.payoff_ratio:.2f} profit_factor={item.profit_factor:.2f} "
            f"expectancy_r={item.expectancy_r:.2f} cumulative_r={item.cumulative_r:.2f}"
        )
    print("By symbol:")
    for item in report.by_symbol:
        print(
            f"  {item.strategy_profile} | {item.symbol}: trades={item.total_trades} win_rate={item.win_rate:.2f}% "
            f"payoff={item.payoff_ratio:.2f} profit_factor={item.profit_factor:.2f} "
            f"expectancy_r={item.expectancy_r:.2f} cumulative_r={item.cumulative_r:.2f}"
        )


def main() -> None:
    args = parse_args()
    configure_logging()

    now = datetime.now(timezone.utc)
    end = parse_date(args.end) if args.end else now
    start = parse_date(args.start) if args.start else end - timedelta(days=args.years * 365)
    exit_profile = infer_exit_profile(args)

    service = BacktestService(
        ohlcv_service=OhlcvService(get_exchange_client_factory()),
        strategy_service=StrategyService(),
        assumptions=BacktestAssumptions(
            exit_profile=exit_profile,
            take_profit_mode=args.take_profit_mode,
            fixed_take_profit_r=args.fixed_take_profit_r,
            scaled_tp1_r=args.scaled_tp1_r,
            scaled_tp2_r=args.scaled_tp2_r,
            long_exit=parse_exit_json(args.long_exit_json, label="--long-exit-json"),
            short_exit=parse_exit_json(args.short_exit_json, label="--short-exit-json"),
        ),
    )
    report = service.run(
        exchange=args.exchange,
        market_type=args.market_type,
        symbols=[item.strip().upper() for item in args.symbols.split(",") if item.strip()],
        strategy_profiles=[item.strip().lower() for item in args.strategy_profiles.split(",") if item.strip()],
        start=start,
        end=end,
    )
    json_path, csv_path = service.save_report(report, Path(args.output_dir))
    print_summary(report)
    print(f"Saved report JSON: {json_path}")
    print(f"Saved trades CSV: {csv_path}")


if __name__ == "__main__":
    main()
