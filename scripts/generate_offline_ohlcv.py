"""Generate realistic synthetic OHLCV data for offline backtesting.

This script creates CSV files in the BacktestService cache directory so that
backtests can run without any exchange API connection.  Data is generated using
Geometric Brownian Motion (GBM) with parameters calibrated to historical BTC
volatility, producing price paths that exhibit realistic trends, pullbacks,
range-bound periods, and volatility clustering.

Usage
-----
    # Generate default 2-year window for BTC
    python scripts/generate_offline_ohlcv.py

    # Custom parameters
    python scripts/generate_offline_ohlcv.py \
        --symbol "BTC/USDT:USDT" \
        --start 2020-01-01 \
        --end 2026-03-19 \
        --start-price 30000 \
        --cache-dir artifacts/backtests/cache

The generated CSVs are placed in ``artifacts/backtests/cache/`` using the
exact naming convention expected by ``BacktestService._cache_path``, so
``run_backtest.py`` and ``run_walk_forward.py`` will automatically pick them
up without touching the network.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.timeframes import TIMEFRAME_TO_MINUTES, get_strategy_required_timeframes

# ---------------------------------------------------------------------------
# GBM parameters calibrated to ~BTC daily behaviour
# ---------------------------------------------------------------------------
DEFAULT_ANNUAL_DRIFT = 0.15          # ~15% annual drift
DEFAULT_ANNUAL_VOL = 0.65           # ~65% annualised volatility (BTC-like)
REGIME_CHANGE_PROB = 0.02           # 2% chance per day to switch regime
VOL_CLUSTER_HALF_LIFE_DAYS = 15     # GARCH-like half-life

STRATEGIES_TO_COVER = [
    "swing_trend_long_regime_gate_v1",
    "swing_trend_simple_candidate_v2",
    "swing_trend_long_regime_short_no_reversal_no_aux_v1",
    "trend_following_v1",
    "swing_improved_v1",
    "mean_reversion_v1",
]


# ---------------------------------------------------------------------------
# Core data generation
# ---------------------------------------------------------------------------

def _generate_1h_ohlcv(
    *,
    start: datetime,
    end: datetime,
    start_price: float,
    annual_drift: float,
    annual_vol: float,
    seed: int,
) -> pd.DataFrame:
    """Generate 1-hour OHLCV using GBM with regime switching and vol clustering."""
    rng = np.random.default_rng(seed)

    # Number of 1-hour bars
    total_hours = int((end - start).total_seconds() / 3600)
    timestamps = [start + timedelta(hours=i) for i in range(total_hours)]

    # Convert annual parameters to hourly
    hours_per_year = 365.25 * 24
    mu_h = annual_drift / hours_per_year
    sigma_h = annual_vol / np.sqrt(hours_per_year)

    # Generate regime sequence (bull / bear / range)
    regimes = _generate_regimes(total_hours, rng)

    # Generate vol-clustering multiplier (GARCH-like)
    vol_mult = _generate_vol_clustering(total_hours, rng)

    # GBM path
    closes = np.empty(total_hours)
    closes[0] = start_price

    for i in range(1, total_hours):
        regime = regimes[i]
        drift_adj = _regime_drift(regime, mu_h)
        vol_adj = sigma_h * vol_mult[i] * _regime_vol_scale(regime)

        z = rng.standard_normal()
        closes[i] = closes[i - 1] * np.exp(drift_adj - 0.5 * vol_adj**2 + vol_adj * z)
        closes[i] = max(closes[i], 1.0)  # price floor

    # Generate OHLV from close path
    opens = np.roll(closes, 1)
    opens[0] = start_price

    # Intra-bar high/low derived from close and a noise factor
    bar_vol = sigma_h * vol_mult * 0.7
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, bar_vol)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, bar_vol)))
    lows = np.maximum(lows, 1.0)

    # Volume: base level + volatility-correlated component
    base_volume = 50_000.0
    volume = base_volume * (1.0 + 2.0 * vol_mult) * (1.0 + np.abs(rng.normal(0, 0.3, total_hours)))

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(closes, 2),
        "volume": np.round(volume, 2),
    })


def _generate_regimes(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate regime labels: 0=bull, 1=bear, 2=range."""
    regimes = np.zeros(n, dtype=int)
    current = 0  # start bull
    for i in range(1, n):
        if rng.random() < REGIME_CHANGE_PROB / 24:  # per-hour probability
            current = rng.integers(0, 3)
        regimes[i] = current
    return regimes


def _regime_drift(regime: int, base_mu: float) -> float:
    if regime == 0:  # bull
        return base_mu * 2.5
    if regime == 1:  # bear
        return -base_mu * 2.0
    return base_mu * 0.1  # range


def _regime_vol_scale(regime: int) -> float:
    if regime == 0:  # bull
        return 0.9
    if regime == 1:  # bear
        return 1.4
    return 0.6  # range


def _generate_vol_clustering(n: int, rng: np.random.Generator) -> np.ndarray:
    """Approximate GARCH-like vol clustering using an AR(1) process on log-vol."""
    half_life_hours = VOL_CLUSTER_HALF_LIFE_DAYS * 24
    alpha = np.exp(-np.log(2) / half_life_hours)

    log_vol = np.zeros(n)
    for i in range(1, n):
        log_vol[i] = alpha * log_vol[i - 1] + np.sqrt(1 - alpha**2) * rng.standard_normal() * 0.3

    return np.exp(log_vol)  # multiplicative factor, centred around 1.0


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------

def _resample_to_timeframe(df_1h: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 1-hour data to a coarser timeframe."""
    minutes = TIMEFRAME_TO_MINUTES[timeframe]
    if minutes <= 60:
        return df_1h.copy()

    df = df_1h.set_index("timestamp")
    rule = f"{minutes}min"
    resampled = df.resample(rule, closed="left", label="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    resampled = resampled.reset_index()
    return resampled


# ---------------------------------------------------------------------------
# Cache file naming (mirrors BacktestService._cache_path)
# ---------------------------------------------------------------------------

def _cache_filename(symbol: str, timeframe: str, start: datetime, end: datetime) -> str:
    safe_symbol = symbol.lower().replace("/", "_").replace(":", "_")
    return f"{safe_symbol}_{timeframe}_{start:%Y%m%d}_{end:%Y%m%d}.csv"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic OHLCV data for offline backtesting.",
    )
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--start", default="2024-03-19", help="Backtest start (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-03-19", help="Backtest end (YYYY-MM-DD)")
    parser.add_argument("--start-price", type=float, default=64000.0, help="Opening price")
    parser.add_argument("--annual-drift", type=float, default=DEFAULT_ANNUAL_DRIFT)
    parser.add_argument("--annual-vol", type=float, default=DEFAULT_ANNUAL_VOL)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--lookback", type=int, default=300, help="Lookback bars for padding")
    parser.add_argument("--cache-dir", default="artifacts/backtests/cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    # Determine all timeframes needed across all strategies
    all_timeframes: set[str] = set()
    for strategy in STRATEGIES_TO_COVER:
        try:
            all_timeframes.update(get_strategy_required_timeframes(strategy))
        except Exception:
            pass
    if not all_timeframes:
        all_timeframes = {"1d", "4h", "1h"}

    # Compute padded start (lookback)
    max_minutes = max(TIMEFRAME_TO_MINUTES[tf] for tf in all_timeframes)
    preload_minutes = max_minutes * args.lookback
    padded_start = start - timedelta(minutes=preload_minutes)

    print(f"Symbol:       {args.symbol}")
    print(f"Window:       {start.date()} -> {end.date()}")
    print(f"Padded start: {padded_start.date()} (lookback={args.lookback})")
    print(f"Timeframes:   {sorted(all_timeframes)}")
    print(f"Start price:  {args.start_price}")
    print(f"Seed:         {args.seed}")
    print()

    # Generate 1-hour base data
    print("Generating 1-hour OHLCV data...")
    df_1h = _generate_1h_ohlcv(
        start=padded_start,
        end=end,
        start_price=args.start_price,
        annual_drift=args.annual_drift,
        annual_vol=args.annual_vol,
        seed=args.seed,
    )
    print(f"  Generated {len(df_1h)} hourly candles")
    print(f"  Price range: {df_1h['low'].min():.2f} - {df_1h['high'].max():.2f}")
    print()

    # Create cache directory
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save per-timeframe
    for tf in sorted(all_timeframes):
        if tf == "1h":
            df_tf = df_1h.copy()
        else:
            print(f"Resampling to {tf}...")
            df_tf = _resample_to_timeframe(df_1h, tf)

        filename = _cache_filename(args.symbol, tf, padded_start, end)
        filepath = cache_dir / filename
        df_tf.to_csv(filepath, index=False)
        print(f"  Saved {tf}: {len(df_tf)} candles -> {filepath}")

    print()
    print("Done! Cache files are ready for offline backtesting.")
    print()
    print("Run a backtest with:")
    print(f"  python scripts/run_backtest.py \\")
    print(f'    --symbols "{args.symbol}" \\')
    print(f"    --strategy-profiles swing_trend_long_regime_gate_v1 \\")
    print(f"    --start {args.start} --end {args.end}")


if __name__ == "__main__":
    main()
