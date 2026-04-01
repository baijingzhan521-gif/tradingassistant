from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.service import BacktestAssumptions, BacktestService, BacktestSummary, BacktestTrade
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.services.strategy_service import StrategyService


SYMBOL = "BTC/USDT:USDT"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "backtests" / "trend_divergence_standalone_one_rule_gate"
BASELINE_PROFILE = "swing_trend_divergence_v1"
DEFAULT_PROFILES = [
    BASELINE_PROFILE,
    "swing_trend_divergence_no_reversal_v1",
    "swing_trend_divergence_min_level3_v1",
    "swing_trend_long_divergence_gate_v1",
]
WINDOW_STARTS = {
    "full_2020": "2020-01-01",
    "two_year": "2024-03-19",
}
COST_SCENARIOS = {
    "base": {},
    "stress_x2": {"taker_fee_bps": 10.0, "slippage_bps": 4.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone one-rule promotion gate for trend-divergence candidates.")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    parser.add_argument("--primary-start", default=WINDOW_STARTS["full_2020"])
    parser.add_argument("--secondary-start", default=WINDOW_STARTS["two_year"])
    parser.add_argument("--end", default=None, help="Defaults to latest available data at runtime.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def make_service(*, assumption_overrides: dict[str, Any] | None = None) -> BacktestService:
    assumptions = {
        "exit_profile": "long_scaled1_3_short_fixed1_5",
        "take_profit_mode": "scaled",
        "long_exit": {"take_profit_mode": "scaled", "scaled_tp1_r": 1.0, "scaled_tp2_r": 3.0},
        "short_exit": {"take_profit_mode": "fixed_r", "fixed_take_profit_r": 1.5},
        "swing_detection_mode": "confirmed",
        "cache_dir": "artifacts/backtests/cache",
    }
    if assumption_overrides:
        assumptions.update(assumption_overrides)
    return BacktestService(
        ohlcv_service=OhlcvService(get_exchange_client_factory()),
        strategy_service=StrategyService(),
        assumptions=BacktestAssumptions(**assumptions),
    )


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def resolve_end_from_history(enriched_history: dict[str, dict[str, pd.DataFrame]]) -> datetime:
    profile_ends: list[pd.Timestamp] = []
    for frames in enriched_history.values():
        profile_ends.append(min(pd.Timestamp(frame["timestamp"].max()) for frame in frames.values()))
    return min(profile_ends).to_pydatetime()


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compounded_return_factor(trades: list[BacktestTrade]) -> float:
    factor = 1.0
    for trade in trades:
        factor *= 1.0 + (float(trade.pnl_pct) / 100.0)
    return factor


def geometric_return_pct(trades: list[BacktestTrade]) -> float:
    if not trades:
        return 0.0
    return round((compounded_return_factor(trades) - 1.0) * 100.0, 4)


def additive_return_pct(trades: list[BacktestTrade]) -> float:
    return round(float(sum(float(trade.pnl_pct) for trade in trades)), 4)


def cagr_pct(*, trades: list[BacktestTrade], window_start: datetime, window_end: datetime) -> float:
    if not trades:
        return 0.0
    elapsed_days = max((window_end - window_start).total_seconds() / 86400.0, 1e-9)
    years = elapsed_days / 365.25
    if years <= 0:
        return 0.0
    factor = compounded_return_factor(trades)
    if factor <= 0.0:
        return -100.0
    return round(((factor ** (1.0 / years)) - 1.0) * 100.0, 4)


def summary_row(
    *,
    cost_scenario: str,
    window: str,
    window_start: datetime,
    window_end: datetime,
    strategy_profile: str,
    summary: BacktestSummary,
    trades: list[BacktestTrade],
) -> dict[str, Any]:
    return {
        "cost_scenario": cost_scenario,
        "window": window,
        "window_start": window_start.date().isoformat(),
        "window_end": window_end.date().isoformat(),
        "strategy_profile": strategy_profile,
        "trades": int(summary.total_trades),
        "win_rate_pct": round(float(summary.win_rate), 2),
        "profit_factor": round(float(summary.profit_factor), 4),
        "expectancy_r": round(float(summary.expectancy_r), 4),
        "cum_r": round(float(summary.cumulative_r), 4),
        "max_dd_r": round(float(summary.max_drawdown_r), 4),
        "avg_holding_bars": round(float(summary.avg_holding_bars), 2),
        "geometric_return_pct": geometric_return_pct(trades),
        "additive_return_pct": additive_return_pct(trades),
        "cagr_pct": cagr_pct(trades=trades, window_start=window_start, window_end=window_end),
    }


def trade_rows(
    *,
    cost_scenario: str,
    window: str,
    strategy_profile: str,
    trades: list[BacktestTrade],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trade in trades:
        row = asdict(trade)
        row["cost_scenario"] = cost_scenario
        row["window"] = window
        row["strategy_profile"] = strategy_profile
        rows.append(row)
    return rows


def build_trade_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "cost_scenario",
                "window",
                "strategy_profile",
                "signal_time",
                "entry_time",
                "exit_time",
                "pnl_pct",
                "pnl_r",
            ]
        )
    frame = pd.DataFrame(rows)
    for column in ("signal_time", "entry_time", "exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    frame["year"] = frame["signal_time"].dt.year.astype(int)
    frame = frame.sort_values(["cost_scenario", "window", "strategy_profile", "entry_time"]).reset_index(drop=True)
    return frame


def build_yearly_geometric_returns(trade_frame: pd.DataFrame) -> list[dict[str, Any]]:
    if trade_frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    grouped = trade_frame.groupby(["cost_scenario", "window", "strategy_profile", "year"], sort=True, observed=True)
    for (cost_scenario, window, strategy_profile, year), group in grouped:
        compounded = float((1.0 + group["pnl_pct"] / 100.0).prod() - 1.0)
        rows.append(
            {
                "cost_scenario": cost_scenario,
                "window": window,
                "strategy_profile": strategy_profile,
                "year": int(year),
                "trades": int(len(group)),
                "geometric_return_pct": round(compounded * 100.0, 4),
                "additive_return_pct": round(float(group["pnl_pct"].sum()), 4),
            }
        )
    return rows


def build_cost_sensitivity(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(summary_rows)
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(["window", "strategy_profile"], sort=True, observed=True)
    for (window, strategy_profile), group in grouped:
        base = group[group["cost_scenario"] == "base"]
        stress = group[group["cost_scenario"] == "stress_x2"]
        if base.empty or stress.empty:
            continue
        base_row = base.iloc[0]
        stress_row = stress.iloc[0]
        rows.append(
            {
                "window": window,
                "strategy_profile": strategy_profile,
                "base_trades": int(base_row["trades"]),
                "stress_trades": int(stress_row["trades"]),
                "base_geometric_return_pct": round(float(base_row["geometric_return_pct"]), 4),
                "stress_geometric_return_pct": round(float(stress_row["geometric_return_pct"]), 4),
                "delta_geometric_return_pct": round(
                    float(stress_row["geometric_return_pct"] - base_row["geometric_return_pct"]), 4
                ),
                "base_cagr_pct": round(float(base_row["cagr_pct"]), 4),
                "stress_cagr_pct": round(float(stress_row["cagr_pct"]), 4),
                "delta_cagr_pct": round(float(stress_row["cagr_pct"] - base_row["cagr_pct"]), 4),
                "base_profit_factor": round(float(base_row["profit_factor"]), 4),
                "stress_profit_factor": round(float(stress_row["profit_factor"]), 4),
                "delta_profit_factor": round(float(stress_row["profit_factor"] - base_row["profit_factor"]), 4),
                "base_max_dd_r": round(float(base_row["max_dd_r"]), 4),
                "stress_max_dd_r": round(float(stress_row["max_dd_r"]), 4),
                "delta_max_dd_r": round(float(stress_row["max_dd_r"] - base_row["max_dd_r"]), 4),
            }
        )
    return rows


def build_trade_concentration(
    trade_frame: pd.DataFrame, yearly_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if trade_frame.empty:
        return []
    yearly_frame = pd.DataFrame(yearly_rows)
    rows: list[dict[str, Any]] = []
    grouped = trade_frame.groupby(["cost_scenario", "window", "strategy_profile"], sort=True, observed=True)
    for (cost_scenario, window, strategy_profile), group in grouped:
        positive = group.loc[group["pnl_r"] > 0.0, "pnl_r"].sort_values(ascending=False)
        total_positive_pnl_r = float(positive.sum())
        top3_positive_pnl_r = float(positive.head(3).sum())
        if total_positive_pnl_r > 0:
            top3_share = (top3_positive_pnl_r / total_positive_pnl_r) * 100.0
        else:
            top3_share = 100.0

        yearly = yearly_frame[
            (yearly_frame["cost_scenario"] == cost_scenario)
            & (yearly_frame["window"] == window)
            & (yearly_frame["strategy_profile"] == strategy_profile)
        ]
        positive_year_returns = yearly.loc[yearly["geometric_return_pct"] > 0.0, "geometric_return_pct"]
        if not positive_year_returns.empty:
            best_year_geometric_pct = float(positive_year_returns.max())
            best_year_share = float((best_year_geometric_pct / positive_year_returns.sum()) * 100.0)
        else:
            best_year_geometric_pct = 0.0
            best_year_share = 100.0

        rows.append(
            {
                "cost_scenario": cost_scenario,
                "window": window,
                "strategy_profile": strategy_profile,
                "trades": int(len(group)),
                "total_positive_pnl_r": round(total_positive_pnl_r, 4),
                "top3_positive_pnl_r": round(top3_positive_pnl_r, 4),
                "top3_trades_pnl_share_pct": round(top3_share, 4),
                "best_year_geometric_pct": round(best_year_geometric_pct, 4),
                "best_year_geometric_pct_share": round(best_year_share, 4),
            }
        )
    return rows


def _summary_lookup(summary_rows: list[dict[str, Any]], *, cost: str, window: str, profile: str) -> dict[str, Any]:
    for row in summary_rows:
        if row["cost_scenario"] == cost and row["window"] == window and row["strategy_profile"] == profile:
            return row
    raise KeyError(f"Missing summary row: cost={cost}, window={window}, profile={profile}")


def _concentration_lookup(
    concentration_rows: list[dict[str, Any]], *, cost: str, window: str, profile: str
) -> dict[str, Any]:
    for row in concentration_rows:
        if row["cost_scenario"] == cost and row["window"] == window and row["strategy_profile"] == profile:
            return row
    raise KeyError(f"Missing concentration row: cost={cost}, window={window}, profile={profile}")


def build_promotion_decision(
    *,
    summary_rows: list[dict[str, Any]],
    concentration_rows: list[dict[str, Any]],
    baseline_profile: str,
    profiles: list[str],
) -> list[dict[str, Any]]:
    baseline_base_full = _summary_lookup(
        summary_rows, cost="base", window="full_2020", profile=baseline_profile
    )
    baseline_geo = float(baseline_base_full["geometric_return_pct"])

    rows: list[dict[str, Any]] = []
    for profile in profiles:
        if profile == baseline_profile:
            continue
        base_full = _summary_lookup(summary_rows, cost="base", window="full_2020", profile=profile)
        stress_full = _summary_lookup(summary_rows, cost="stress_x2", window="full_2020", profile=profile)
        concentration = _concentration_lookup(
            concentration_rows, cost="base", window="full_2020", profile=profile
        )

        pass_geo_vs_baseline = float(base_full["geometric_return_pct"]) > baseline_geo
        pass_base_pf = float(base_full["profit_factor"]) >= 1.10
        pass_base_max_dd = float(base_full["max_dd_r"]) <= 6.0
        pass_stress_geo = float(stress_full["geometric_return_pct"]) > 0.0
        pass_stress_pf = float(stress_full["profit_factor"]) >= 1.00
        pass_trades_floor = int(base_full["trades"]) >= 10
        pass_top3_share = float(concentration["top3_trades_pnl_share_pct"]) <= 65.0
        pass_best_year_share = float(concentration["best_year_geometric_pct_share"]) <= 80.0

        promoted = (
            pass_geo_vs_baseline
            and pass_base_pf
            and pass_base_max_dd
            and pass_stress_geo
            and pass_stress_pf
            and pass_trades_floor
            and pass_top3_share
            and pass_best_year_share
        )
        status = "promoted_standalone_candidate" if promoted else "rejected_fragile_or_unprofitable"

        rows.append(
            {
                "baseline_profile": baseline_profile,
                "candidate_profile": profile,
                "base_full_2020_baseline_geometric_return_pct": round(baseline_geo, 4),
                "base_full_2020_candidate_geometric_return_pct": round(
                    float(base_full["geometric_return_pct"]), 4
                ),
                "base_full_2020_candidate_cagr_pct": round(float(base_full["cagr_pct"]), 4),
                "base_full_2020_candidate_profit_factor": round(float(base_full["profit_factor"]), 4),
                "base_full_2020_candidate_max_dd_r": round(float(base_full["max_dd_r"]), 4),
                "stress_full_2020_candidate_geometric_return_pct": round(
                    float(stress_full["geometric_return_pct"]), 4
                ),
                "stress_full_2020_candidate_cagr_pct": round(float(stress_full["cagr_pct"]), 4),
                "stress_full_2020_candidate_profit_factor": round(float(stress_full["profit_factor"]), 4),
                "base_full_2020_candidate_trades": int(base_full["trades"]),
                "base_full_2020_top3_trades_pnl_share_pct": round(
                    float(concentration["top3_trades_pnl_share_pct"]), 4
                ),
                "base_full_2020_best_year_geometric_pct_share": round(
                    float(concentration["best_year_geometric_pct_share"]), 4
                ),
                "pass_base_geo_vs_baseline": pass_geo_vs_baseline,
                "pass_base_pf": pass_base_pf,
                "pass_base_max_dd": pass_base_max_dd,
                "pass_stress_geo": pass_stress_geo,
                "pass_stress_pf": pass_stress_pf,
                "pass_trades_floor": pass_trades_floor,
                "pass_top3_share": pass_top3_share,
                "pass_best_year_share": pass_best_year_share,
                "status": status,
                "next_route": "promote_and_compare_with_mainline"
                if promoted
                else "freeze_trend_divergence_and_keep_switch_baseline",
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_None_"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        values: list[str] = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(
    *,
    summary_rows: list[dict[str, Any]],
    yearly_rows: list[dict[str, Any]],
    concentration_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    profiles: list[str],
) -> str:
    base_full = [
        row
        for row in summary_rows
        if row["cost_scenario"] == "base" and row["window"] == "full_2020" and row["strategy_profile"] in profiles
    ]
    base_full = sorted(base_full, key=lambda row: float(row["geometric_return_pct"]), reverse=True)
    base_full_yearly = [
        row
        for row in yearly_rows
        if row["cost_scenario"] == "base" and row["window"] == "full_2020" and row["strategy_profile"] in profiles
    ]
    base_full_yearly = sorted(base_full_yearly, key=lambda row: (row["strategy_profile"], row["year"]))
    base_full_concentration = [
        row
        for row in concentration_rows
        if row["cost_scenario"] == "base" and row["window"] == "full_2020" and row["strategy_profile"] in profiles
    ]
    base_full_concentration = sorted(base_full_concentration, key=lambda row: row["strategy_profile"])
    promoted = [row["candidate_profile"] for row in decision_rows if row["status"] == "promoted_standalone_candidate"]
    conclusion = (
        f"promoted candidates: {', '.join(promoted)}"
        if promoted
        else "no candidate promoted; freeze trend-divergence family and move to next orthogonal one-rule gate."
    )
    return "\n".join(
        [
            "# Trend-Divergence Standalone One-Rule Gate",
            "",
            "- scope: standalone profitability promotion gate for trend-divergence candidates",
            "- baseline profile: `swing_trend_divergence_v1`",
            "- this gate uses low-freedom one-rule candidates only",
            "",
            "## Base Full_2020 Summary",
            "",
            markdown_table(
                base_full,
                [
                    ("strategy_profile", "Profile"),
                    ("trades", "Trades"),
                    ("profit_factor", "PF"),
                    ("cum_r", "Cum R"),
                    ("max_dd_r", "MaxDD R"),
                    ("geometric_return_pct", "Geom Return %"),
                    ("additive_return_pct", "Add Return %"),
                    ("cagr_pct", "CAGR %"),
                ],
            ),
            "",
            "## Base Full_2020 Yearly Geometric Returns",
            "",
            markdown_table(
                base_full_yearly,
                [
                    ("strategy_profile", "Profile"),
                    ("year", "Year"),
                    ("trades", "Trades"),
                    ("geometric_return_pct", "Geom Return %"),
                    ("additive_return_pct", "Add Return %"),
                ],
            ),
            "",
            "## Base Full_2020 Trade Concentration",
            "",
            markdown_table(
                base_full_concentration,
                [
                    ("strategy_profile", "Profile"),
                    ("trades", "Trades"),
                    ("top3_trades_pnl_share_pct", "Top3 Share %"),
                    ("best_year_geometric_pct", "Best Year Geom %"),
                    ("best_year_geometric_pct_share", "Best Year Share %"),
                ],
            ),
            "",
            "## Promotion Decision",
            "",
            markdown_table(
                decision_rows,
                [
                    ("candidate_profile", "Candidate"),
                    ("base_full_2020_candidate_geometric_return_pct", "Base Geom %"),
                    ("base_full_2020_candidate_cagr_pct", "Base CAGR %"),
                    ("base_full_2020_candidate_profit_factor", "Base PF"),
                    ("base_full_2020_candidate_max_dd_r", "Base MaxDD R"),
                    ("stress_full_2020_candidate_geometric_return_pct", "Stress Geom %"),
                    ("stress_full_2020_candidate_cagr_pct", "Stress CAGR %"),
                    ("stress_full_2020_candidate_profit_factor", "Stress PF"),
                    ("base_full_2020_top3_trades_pnl_share_pct", "Top3 Share %"),
                    ("base_full_2020_best_year_geometric_pct_share", "Best Year Share %"),
                    ("status", "Status"),
                ],
            ),
            "",
            "## Conclusion",
            "",
            f"- {conclusion}",
        ]
    )


def main() -> None:
    args = parse_args()
    configure_logging()

    profiles = list(dict.fromkeys(args.profiles))
    if BASELINE_PROFILE not in profiles:
        raise ValueError(f"profiles must include baseline profile: {BASELINE_PROFILE}")
    if len(profiles) < 2:
        raise ValueError("profiles must include at least baseline plus one candidate.")

    primary_start = parse_date(args.primary_start)
    secondary_start = parse_date(args.secondary_start)
    if secondary_start <= primary_start:
        raise ValueError("secondary-start must be later than primary-start")
    requested_end = parse_date(args.end) if args.end else datetime.now(timezone.utc)

    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    history_service = make_service()
    enriched_history = {
        profile: history_service.prepare_enriched_history(
            exchange=args.exchange,
            market_type=args.market_type,
            symbol=args.symbol,
            strategy_profile=profile,
            start=primary_start,
            end=requested_end,
        )
        for profile in profiles
    }
    resolved_end = resolve_end_from_history(enriched_history)
    windows = {
        "full_2020": (primary_start, resolved_end),
        "two_year": (secondary_start, resolved_end),
    }

    summary_rows: list[dict[str, Any]] = []
    trade_row_list: list[dict[str, Any]] = []
    for cost_scenario, overrides in COST_SCENARIOS.items():
        service = make_service(assumption_overrides=overrides)
        for window, (window_start, window_end) in windows.items():
            for profile in profiles:
                summary, trades = service.run_symbol_strategy_with_enriched_frames(
                    symbol=args.symbol,
                    strategy_profile=profile,
                    start=window_start,
                    end=window_end,
                    enriched_frames=enriched_history[profile],
                )
                summary_rows.append(
                    summary_row(
                        cost_scenario=cost_scenario,
                        window=window,
                        window_start=window_start,
                        window_end=window_end,
                        strategy_profile=profile,
                        summary=summary,
                        trades=trades,
                    )
                )
                trade_row_list.extend(
                    trade_rows(
                        cost_scenario=cost_scenario,
                        window=window,
                        strategy_profile=profile,
                        trades=trades,
                    )
                )

    trade_frame = build_trade_frame(trade_row_list)
    yearly_rows = build_yearly_geometric_returns(trade_frame)
    cost_rows = build_cost_sensitivity(summary_rows)
    concentration_rows = build_trade_concentration(trade_frame, yearly_rows)
    decision_rows = build_promotion_decision(
        summary_rows=summary_rows,
        concentration_rows=concentration_rows,
        baseline_profile=BASELINE_PROFILE,
        profiles=profiles,
    )

    report = build_report(
        summary_rows=summary_rows,
        yearly_rows=yearly_rows,
        concentration_rows=concentration_rows,
        decision_rows=decision_rows,
        profiles=profiles,
    )

    write_csv(output_dir / "summary_all.csv", summary_rows)
    write_csv(output_dir / "yearly_geometric_returns.csv", yearly_rows)
    write_csv(output_dir / "cost_sensitivity.csv", cost_rows)
    write_csv(output_dir / "trade_concentration.csv", concentration_rows)
    write_csv(output_dir / "promotion_decision.csv", decision_rows)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"saved trend-divergence standalone one-rule gate artifacts to {output_dir}")


if __name__ == "__main__":
    main()
