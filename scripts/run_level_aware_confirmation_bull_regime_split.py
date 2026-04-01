from __future__ import annotations

import csv
import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.logging import configure_logging
from app.data.exchange_client import ExchangeClientFactory
from app.data.ohlcv_service import OhlcvService
from app.indicators.ema import apply_ema_indicators
from app.indicators.market_structure import determine_ema_alignment, determine_trend_bias


INPUT_TRADES = (
    ROOT
    / "artifacts/backtests/level_aware_confirmation_compare_full_2020/backtest_long_scaled1_3_short_fixed1_5_20260323T135105Z_trades.csv"
)
OUTPUT_DIR = ROOT / "artifacts/backtests/level_aware_confirmation_bull_regime_split"

BASE_PROFILE = "swing_trend_long_regime_gate_v1"
LEVEL_PROFILE = "swing_trend_level_aware_confirmation_v1"
EXCHANGE = "binance"
MARKET_TYPE = "perpetual"
SYMBOL = "BTC/USDT:USDT"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_None_"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        values: list[str] = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_trades() -> pd.DataFrame:
    trades = pd.read_csv(
        INPUT_TRADES,
        parse_dates=["signal_time", "entry_time", "exit_time"],
    )
    trades = trades[trades["strategy_profile"].isin([BASE_PROFILE, LEVEL_PROFILE])].copy()
    trades = trades.sort_values("signal_time").reset_index(drop=True)
    return trades


def enrich_timeframe(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = apply_ema_indicators(frame.sort_values("timestamp").reset_index(drop=True), periods=(21, 55, 100, 200))
    ema_alignment = []
    trend_bias = []
    for row in enriched.itertuples(index=False):
        alignment = determine_ema_alignment(
            float(row.ema_21),
            float(row.ema_55),
            float(row.ema_100),
            float(row.ema_200),
        )
        bias = determine_trend_bias(float(row.close), float(row.ema_200), alignment)
        ema_alignment.append(alignment.value)
        trend_bias.append(bias.value)
    enriched["ema_alignment"] = ema_alignment
    enriched["trend_bias"] = trend_bias
    return enriched


def build_regime_frame(*, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    padded_start = (start - timedelta(days=260)).to_pydatetime()
    end_dt = (end + timedelta(days=1)).to_pydatetime()

    exchange_factory = ExchangeClientFactory()
    ohlcv_service = OhlcvService(exchange_factory)

    day = ohlcv_service.fetch_ohlcv_range(
        exchange=EXCHANGE,
        market_type=MARKET_TYPE,
        symbol=SYMBOL,
        timeframe="1d",
        start=padded_start,
        end=end_dt,
    )
    h4 = ohlcv_service.fetch_ohlcv_range(
        exchange=EXCHANGE,
        market_type=MARKET_TYPE,
        symbol=SYMBOL,
        timeframe="4h",
        start=padded_start,
        end=end_dt,
    )

    day = enrich_timeframe(day)
    h4 = enrich_timeframe(h4)

    day["day_above_ema200"] = day["close"] > day["ema_200"]
    day["day_ema200_slope"] = day["ema_200"].diff()
    day["day_ema200_slope_non_negative"] = day["day_ema200_slope"] >= 0.0
    day_view = day[
        [
            "timestamp",
            "close",
            "ema_200",
            "day_above_ema200",
            "day_ema200_slope",
            "day_ema200_slope_non_negative",
        ]
    ].rename(
        columns={
            "timestamp": "day_timestamp",
            "close": "day_close",
            "ema_200": "day_ema200",
        }
    )

    h4_view = h4[
        [
            "timestamp",
            "close",
            "ema_21",
            "ema_55",
            "ema_100",
            "ema_200",
            "ema_alignment",
            "trend_bias",
        ]
    ].rename(
        columns={
            "timestamp": "h4_timestamp",
            "close": "h4_close",
            "ema_21": "h4_ema21",
            "ema_55": "h4_ema55",
            "ema_100": "h4_ema100",
            "ema_200": "h4_ema200",
            "ema_alignment": "h4_ema_alignment",
            "trend_bias": "h4_trend_bias",
        }
    )

    signal_frame = pd.DataFrame({"signal_time": pd.date_range(start=start.floor("h"), end=end.ceil("h"), freq="1h", tz="UTC")})
    signal_frame = pd.merge_asof(
        signal_frame.sort_values("signal_time"),
        day_view.sort_values("day_timestamp"),
        left_on="signal_time",
        right_on="day_timestamp",
        direction="backward",
    )
    signal_frame = pd.merge_asof(
        signal_frame.sort_values("signal_time"),
        h4_view.sort_values("h4_timestamp"),
        left_on="signal_time",
        right_on="h4_timestamp",
        direction="backward",
    )
    signal_frame["bull_regime"] = (
        signal_frame["day_above_ema200"].fillna(False)
        & signal_frame["day_ema200_slope_non_negative"].fillna(False)
        & (signal_frame["h4_trend_bias"] == "bullish")
    )
    signal_frame["regime_label"] = signal_frame["bull_regime"].map({True: "bull_regime", False: "non_bull_regime"})
    return signal_frame


def attach_regime(trades: pd.DataFrame, regime_frame: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge_asof(
        trades.sort_values("signal_time"),
        regime_frame.sort_values("signal_time"),
        on="signal_time",
        direction="backward",
    )
    merged["regime_label"] = merged["regime_label"].fillna("non_bull_regime")
    merged["bull_regime"] = merged["bull_regime"].fillna(False)
    return merged


def summarize_trades(trades: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for regime_label in ("bull_regime", "non_bull_regime"):
        regime_trades = trades[trades["regime_label"] == regime_label]
        for strategy_profile, profile_trades in regime_trades.groupby("strategy_profile"):
            rows.append(
                {
                    "regime_label": regime_label,
                    "strategy_profile": strategy_profile,
                    "trades": int(len(profile_trades)),
                    "win_rate_pct": round(float((profile_trades["pnl_r"] > 0).mean() * 100.0), 2),
                    "expectancy_r": round(float(profile_trades["pnl_r"].mean()), 4),
                    "cumulative_r": round(float(profile_trades["pnl_r"].sum()), 4),
                    "avg_bars_held": round(float(profile_trades["bars_held"].mean()), 2),
                    "tp1_hit_rate_pct": round(float(profile_trades["tp1_hit"].mean() * 100.0), 2),
                    "tp2_hit_rate_pct": round(float(profile_trades["tp2_hit"].mean() * 100.0), 2),
                }
            )
    return rows


def summarize_side_breakdown(trades: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = trades.groupby(["regime_label", "strategy_profile", "side"])
    for (regime_label, strategy_profile, side), frame in grouped:
        rows.append(
            {
                "regime_label": regime_label,
                "strategy_profile": strategy_profile,
                "side": side,
                "trades": int(len(frame)),
                "win_rate_pct": round(float((frame["pnl_r"] > 0).mean() * 100.0), 2),
                "expectancy_r": round(float(frame["pnl_r"].mean()), 4),
                "cumulative_r": round(float(frame["pnl_r"].sum()), 4),
                "avg_bars_held": round(float(frame["bars_held"].mean()), 2),
            }
        )
    return rows


def summarize_monthly(trades: pd.DataFrame) -> list[dict[str, Any]]:
    trades = trades.copy()
    trades["month"] = trades["signal_time"].dt.strftime("%Y-%m")
    grouped = (
        trades.groupby(["regime_label", "month", "strategy_profile"])["pnl_r"]
        .sum()
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for regime_label, regime_group in grouped.groupby("regime_label"):
        pivot = regime_group.pivot(index="month", columns="strategy_profile", values="pnl_r").fillna(0.0)
        for month, row in pivot.iterrows():
            base_r = float(row.get(BASE_PROFILE, 0.0))
            level_r = float(row.get(LEVEL_PROFILE, 0.0))
            rows.append(
                {
                    "regime_label": regime_label,
                    "month": month,
                    "base_pnl_r": round(base_r, 4),
                    "level_pnl_r": round(level_r, 4),
                    "delta_r": round(level_r - base_r, 4),
                }
            )
    return rows


def build_report(
    *,
    trade_summary: list[dict[str, Any]],
    side_summary: list[dict[str, Any]],
    monthly_rows: list[dict[str, Any]],
) -> str:
    bull_rows = [row for row in trade_summary if row["regime_label"] == "bull_regime"]
    non_bull_rows = [row for row in trade_summary if row["regime_label"] == "non_bull_regime"]
    bull_delta = next((row["cumulative_r"] for row in bull_rows if row["strategy_profile"] == LEVEL_PROFILE), 0.0) - next(
        (row["cumulative_r"] for row in bull_rows if row["strategy_profile"] == BASE_PROFILE),
        0.0,
    )
    non_bull_delta = next(
        (row["cumulative_r"] for row in non_bull_rows if row["strategy_profile"] == LEVEL_PROFILE),
        0.0,
    ) - next(
        (row["cumulative_r"] for row in non_bull_rows if row["strategy_profile"] == BASE_PROFILE),
        0.0,
    )

    recommendation = (
        "bull_variant_candidate"
        if bull_delta > 0 and non_bull_delta < 0
        else "not_supported_as_regime_variant"
    )

    lines = [
        "# Level-Aware Confirmation Bull-Regime Split",
        "",
        "- 这轮验证只回答一个问题：`level-aware confirmation` 是否更适合被理解成 `bull-regime` 变体，而不是全局主线升级。",
        "- bull regime 预先定义为：`1D close > EMA200`、`1D EMA200 slope >= 0`、`4H trend_bias = bullish`。",
        "- 这里不改策略，不重跑 entry，只把现有长样本交易按 `signal_time` 切到 bull / non-bull 两个子样本里。",
        "",
        f"- bull 子样本里，Level-Aware 相对 Base 的累计差值：`{bull_delta:.4f}R`",
        f"- non-bull 子样本里，Level-Aware 相对 Base 的累计差值：`{non_bull_delta:.4f}R`",
        f"- 当前结论：`{recommendation}`",
        "",
        "## Trade Summary",
        "",
        markdown_table(
            trade_summary,
            [
                ("regime_label", "Regime"),
                ("strategy_profile", "Strategy"),
                ("trades", "Trades"),
                ("win_rate_pct", "Win %"),
                ("expectancy_r", "Exp R"),
                ("cumulative_r", "Cum R"),
                ("avg_bars_held", "Avg Bars"),
                ("tp1_hit_rate_pct", "TP1 %"),
                ("tp2_hit_rate_pct", "TP2 %"),
            ],
        ),
        "",
        "## Side Breakdown",
        "",
        markdown_table(
            side_summary,
            [
                ("regime_label", "Regime"),
                ("strategy_profile", "Strategy"),
                ("side", "Side"),
                ("trades", "Trades"),
                ("win_rate_pct", "Win %"),
                ("expectancy_r", "Exp R"),
                ("cumulative_r", "Cum R"),
                ("avg_bars_held", "Avg Bars"),
            ],
        ),
        "",
        "## Worst Months By Regime",
        "",
        "Bull regime worst months:",
        "",
        markdown_table(
            sorted([row for row in monthly_rows if row["regime_label"] == "bull_regime"], key=lambda item: item["delta_r"])[:6],
            [
                ("month", "Month"),
                ("base_pnl_r", "Base R"),
                ("level_pnl_r", "Level R"),
                ("delta_r", "Delta R"),
            ],
        ),
        "",
        "Non-bull regime worst months:",
        "",
        markdown_table(
            sorted([row for row in monthly_rows if row["regime_label"] == "non_bull_regime"], key=lambda item: item["delta_r"])[:6],
            [
                ("month", "Month"),
                ("base_pnl_r", "Base R"),
                ("level_pnl_r", "Level R"),
                ("delta_r", "Delta R"),
            ],
        ),
        "",
        "## Engineering Judgment",
        "",
        "- 如果 bull 子样本内显著更好、而 non-bull 子样本内显著更差，这条线才有资格继续作为 bull 变体研究。",
        "- 如果 bull 子样本内也没有稳定提升，那这条线就不值得再保留成 regime variant。",
        "- 这里仍然只是分层验证，不是 sequence-aware hybrid 回测；即便成立，也还没证明“实盘切换版本”一定更优。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    configure_logging()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    ensure_output_dir()

    trades = load_trades()
    start = trades["signal_time"].min()
    end = trades["signal_time"].max()

    regime_frame = build_regime_frame(start=start, end=end)
    trades_with_regime = attach_regime(trades, regime_frame)

    trade_summary = summarize_trades(trades_with_regime)
    side_summary = summarize_side_breakdown(trades_with_regime)
    monthly_rows = summarize_monthly(trades_with_regime)

    write_csv(OUTPUT_DIR / "trade_summary.csv", trade_summary)
    write_csv(OUTPUT_DIR / "side_summary.csv", side_summary)
    write_csv(OUTPUT_DIR / "monthly_breakdown.csv", monthly_rows)
    trades_with_regime.to_csv(OUTPUT_DIR / "trades_with_regime.csv", index=False)
    regime_frame.to_csv(OUTPUT_DIR / "regime_frame.csv", index=False)
    (OUTPUT_DIR / "report.md").write_text(
        build_report(
            trade_summary=trade_summary,
            side_summary=side_summary,
            monthly_rows=monthly_rows,
        ),
        encoding="utf-8",
    )

    print(f"Saved trade summary CSV: {OUTPUT_DIR / 'trade_summary.csv'}")
    print(f"Saved side summary CSV: {OUTPUT_DIR / 'side_summary.csv'}")
    print(f"Saved monthly breakdown CSV: {OUTPUT_DIR / 'monthly_breakdown.csv'}")
    print(f"Saved trades with regime CSV: {OUTPUT_DIR / 'trades_with_regime.csv'}")
    print(f"Saved regime frame CSV: {OUTPUT_DIR / 'regime_frame.csv'}")
    print(f"Saved report: {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
