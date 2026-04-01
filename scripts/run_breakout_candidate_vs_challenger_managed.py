from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.service import BacktestAssumptions, BacktestService
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.services.strategy_service import StrategyService
from scripts.post_tp1_managed_replay import build_service
from scripts.run_range_failure_vs_challenger_managed import (
    COST_SCENARIOS,
    DEFAULT_BASELINE_OVERLAY_PROFILE,
    DEFAULT_BASELINE_STRATEGY_PROFILE,
    baseline_label,
    ensure_output_dir,
    parse_date,
    resolve_end_from_history,
    run_managed_baseline_with_helper,
    trades_frame,
    write_csv,
)


SYMBOL = "BTC/USDT:USDT"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "backtests" / "breakout_candidate_vs_challenger_managed"
DEFAULT_CANDIDATE_PROFILE = "swing_breakout_setup_proximity_045_v1_btc"
WINDOWS = ("two_year", "full_2020")
WINDOW_STARTS = {
    "two_year": "2024-03-19",
    "full_2020": "2020-01-01",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent pool comparison: breakout candidate vs challenger_managed.")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--baseline-strategy-profile", default=DEFAULT_BASELINE_STRATEGY_PROFILE)
    parser.add_argument("--baseline-overlay-profile", default=DEFAULT_BASELINE_OVERLAY_PROFILE)
    parser.add_argument("--candidate-profile", default=DEFAULT_CANDIDATE_PROFILE)
    parser.add_argument("--primary-start", default=WINDOW_STARTS["full_2020"])
    parser.add_argument("--secondary-start", default=WINDOW_STARTS["two_year"])
    parser.add_argument("--end", default=None, help="Defaults to latest available data at runtime.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def make_candidate_service(*, assumption_overrides: dict[str, Any] | None = None) -> BacktestService:
    assumptions = {
        "exit_profile": "breakout_strategy_defined",
        "take_profit_mode": "scaled",
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


def geometric_return_pct_from_frame(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    compounded = float((1.0 + frame["pnl_pct"] / 100.0).prod() - 1.0)
    return round(compounded * 100.0, 4)


def cagr_pct_from_frame(frame: pd.DataFrame, *, window_start: datetime, window_end: datetime) -> float:
    if frame.empty:
        return 0.0
    compounded = float((1.0 + frame["pnl_pct"] / 100.0).prod())
    elapsed_days = max((window_end - window_start).total_seconds() / 86400.0, 1e-9)
    years = elapsed_days / 365.25
    if years <= 0.0:
        return 0.0
    if compounded <= 0.0:
        return -100.0
    return round(((compounded ** (1.0 / years)) - 1.0) * 100.0, 4)


def summary_row(
    *,
    cost_scenario: str,
    window: str,
    window_start: datetime,
    window_end: datetime,
    profile_kind: str,
    strategy_profile: str,
    profile_label: str,
    summary,
    trades_frame_obj: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "cost_scenario": cost_scenario,
        "window": window,
        "window_start": window_start.date().isoformat(),
        "window_end": window_end.date().isoformat(),
        "profile_kind": profile_kind,
        "strategy_profile": strategy_profile,
        "profile_label": profile_label,
        "trades": int(summary.total_trades),
        "win_rate_pct": round(float(summary.win_rate), 2),
        "profit_factor": round(float(summary.profit_factor), 4),
        "expectancy_r": round(float(summary.expectancy_r), 4),
        "cum_r": round(float(summary.cumulative_r), 4),
        "max_dd_r": round(float(summary.max_drawdown_r), 4),
        "avg_holding_bars": round(float(summary.avg_holding_bars), 2),
        "geometric_return_pct": geometric_return_pct_from_frame(trades_frame_obj),
        "additive_return_pct": round(float(trades_frame_obj["pnl_pct"].sum()), 4) if not trades_frame_obj.empty else 0.0,
        "cagr_pct": cagr_pct_from_frame(trades_frame_obj, window_start=window_start, window_end=window_end),
    }


def max_drawdown_r_from_frame(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    ordered = frame.sort_values("entry_time")
    cumulative = ordered["pnl_r"].cumsum()
    drawdown = cumulative - cumulative.cummax()
    return round(abs(float(drawdown.min())), 4)


def build_side_summary(trades_all: pd.DataFrame) -> list[dict[str, Any]]:
    if trades_all.empty:
        return []
    rows: list[dict[str, Any]] = []
    grouped = trades_all.groupby(
        ["cost_scenario", "window", "profile_kind", "strategy_profile", "profile_label", "side"],
        observed=True,
        sort=True,
    )
    for keys, group in grouped:
        cost_scenario, window, profile_kind, strategy_profile, profile_label, side = keys
        wins = float(group.loc[group["pnl_r"] > 0.0, "pnl_r"].sum())
        losses = float(-group.loc[group["pnl_r"] < 0.0, "pnl_r"].sum())
        pf = round(wins / losses, 4) if losses > 0 else None
        rows.append(
            {
                "cost_scenario": cost_scenario,
                "window": window,
                "profile_kind": profile_kind,
                "strategy_profile": strategy_profile,
                "profile_label": profile_label,
                "side": side,
                "trades": int(len(group)),
                "profit_factor": pf,
                "expectancy_r": round(float(group["pnl_r"].mean()), 4),
                "cum_r": round(float(group["pnl_r"].sum()), 4),
                "max_dd_r": max_drawdown_r_from_frame(group),
                "geometric_return_pct": geometric_return_pct_from_frame(group),
            }
        )
    return rows


def build_yearly_geometric_returns(trades_all: pd.DataFrame) -> list[dict[str, Any]]:
    if trades_all.empty:
        return []
    if "year" not in trades_all.columns:
        trades_all = trades_all.copy()
        trades_all["year"] = pd.to_datetime(trades_all["signal_time"], utc=True).dt.year.astype(int)
    rows: list[dict[str, Any]] = []
    grouped = trades_all.groupby(
        ["cost_scenario", "window", "profile_kind", "strategy_profile", "profile_label", "year"],
        observed=True,
        sort=True,
    )
    for keys, group in grouped:
        cost_scenario, window, profile_kind, strategy_profile, profile_label, year = keys
        rows.append(
            {
                "cost_scenario": cost_scenario,
                "window": window,
                "profile_kind": profile_kind,
                "strategy_profile": strategy_profile,
                "profile_label": profile_label,
                "year": int(year),
                "trades": int(len(group)),
                "geometric_return_pct": geometric_return_pct_from_frame(group),
                "additive_return_pct": round(float(group["pnl_pct"].sum()), 4),
            }
        )
    return rows


def build_cost_sensitivity(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(summary_rows)
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(["window", "profile_kind", "strategy_profile", "profile_label"], observed=True, sort=True)
    for keys, group in grouped:
        window, profile_kind, strategy_profile, profile_label = keys
        base = group[group["cost_scenario"] == "base"]
        stress = group[group["cost_scenario"] == "stress_x2"]
        if base.empty or stress.empty:
            continue
        base_row = base.iloc[0]
        stress_row = stress.iloc[0]
        rows.append(
            {
                "window": window,
                "profile_kind": profile_kind,
                "strategy_profile": strategy_profile,
                "profile_label": profile_label,
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


def _lookup(rows: list[dict[str, Any]], *, cost: str, window: str, kind: str) -> dict[str, Any]:
    for row in rows:
        if row["cost_scenario"] == cost and row["window"] == window and row["profile_kind"] == kind:
            return row
    raise KeyError(f"Missing row for cost={cost}, window={window}, kind={kind}")


def _lookup_side(rows: list[dict[str, Any]], *, cost: str, window: str, kind: str, side: str) -> dict[str, Any] | None:
    for row in rows:
        if (
            row["cost_scenario"] == cost
            and row["window"] == window
            and row["profile_kind"] == kind
            and row["side"] == side
        ):
            return row
    return None


def build_comparison_decision(
    *,
    summary_rows: list[dict[str, Any]],
    side_rows: list[dict[str, Any]],
    baseline_strategy_profile: str,
    baseline_overlay_profile: str,
    candidate_profile: str,
) -> dict[str, Any]:
    baseline_base_primary = _lookup(summary_rows, cost="base", window="full_2020", kind="baseline_managed")
    candidate_base_primary = _lookup(summary_rows, cost="base", window="full_2020", kind="candidate")
    baseline_stress_primary = _lookup(summary_rows, cost="stress_x2", window="full_2020", kind="baseline_managed")
    candidate_stress_primary = _lookup(summary_rows, cost="stress_x2", window="full_2020", kind="candidate")

    baseline_secondary_long = _lookup_side(
        side_rows, cost="base", window="two_year", kind="baseline_managed", side="LONG"
    )
    candidate_secondary_long = _lookup_side(
        side_rows, cost="base", window="two_year", kind="candidate", side="LONG"
    )
    baseline_primary_short = _lookup_side(
        side_rows, cost="base", window="full_2020", kind="baseline_managed", side="SHORT"
    )
    candidate_primary_short = _lookup_side(
        side_rows, cost="base", window="full_2020", kind="candidate", side="SHORT"
    )

    secondary_long_delta_r = round(
        float((candidate_secondary_long or {}).get("cum_r", 0.0) - (baseline_secondary_long or {}).get("cum_r", 0.0)),
        4,
    )
    primary_short_delta_r = round(
        float((candidate_primary_short or {}).get("cum_r", 0.0) - (baseline_primary_short or {}).get("cum_r", 0.0)),
        4,
    )

    pass_head_to_head_geo = float(candidate_base_primary["geometric_return_pct"]) > float(
        baseline_base_primary["geometric_return_pct"]
    )
    pass_head_to_head_pf = float(candidate_base_primary["profit_factor"]) >= float(
        baseline_base_primary["profit_factor"]
    )
    pass_head_to_head_max_dd = float(candidate_base_primary["max_dd_r"]) <= (
        float(baseline_base_primary["max_dd_r"]) + 2.0
    )
    pass_candidate_stress_geo = float(candidate_stress_primary["geometric_return_pct"]) > 0.0
    pass_candidate_stress_pf = float(candidate_stress_primary["profit_factor"]) >= 1.0
    pass_secondary_long_guard = secondary_long_delta_r >= -2.0

    can_challenge = (
        pass_head_to_head_geo
        and pass_head_to_head_pf
        and pass_head_to_head_max_dd
        and pass_candidate_stress_geo
        and pass_candidate_stress_pf
        and pass_secondary_long_guard
    )
    status = (
        "candidate_can_challenge_mainline"
        if can_challenge
        else "mainline_still_preferred_candidate_kept_in_pool"
    )
    return {
        "baseline_strategy_profile": baseline_strategy_profile,
        "baseline_overlay_profile": baseline_overlay_profile,
        "candidate_profile": candidate_profile,
        "base_full_2020_baseline_geometric_return_pct": round(
            float(baseline_base_primary["geometric_return_pct"]), 4
        ),
        "base_full_2020_candidate_geometric_return_pct": round(
            float(candidate_base_primary["geometric_return_pct"]), 4
        ),
        "base_full_2020_delta_geometric_return_pct": round(
            float(candidate_base_primary["geometric_return_pct"] - baseline_base_primary["geometric_return_pct"]), 4
        ),
        "base_full_2020_baseline_cagr_pct": round(float(baseline_base_primary["cagr_pct"]), 4),
        "base_full_2020_candidate_cagr_pct": round(float(candidate_base_primary["cagr_pct"]), 4),
        "base_full_2020_delta_cagr_pct": round(
            float(candidate_base_primary["cagr_pct"] - baseline_base_primary["cagr_pct"]), 4
        ),
        "base_full_2020_baseline_profit_factor": round(float(baseline_base_primary["profit_factor"]), 4),
        "base_full_2020_candidate_profit_factor": round(float(candidate_base_primary["profit_factor"]), 4),
        "base_full_2020_delta_profit_factor": round(
            float(candidate_base_primary["profit_factor"] - baseline_base_primary["profit_factor"]), 4
        ),
        "base_full_2020_baseline_max_dd_r": round(float(baseline_base_primary["max_dd_r"]), 4),
        "base_full_2020_candidate_max_dd_r": round(float(candidate_base_primary["max_dd_r"]), 4),
        "base_full_2020_delta_max_dd_r": round(
            float(candidate_base_primary["max_dd_r"] - baseline_base_primary["max_dd_r"]), 4
        ),
        "stress_full_2020_baseline_geometric_return_pct": round(
            float(baseline_stress_primary["geometric_return_pct"]), 4
        ),
        "stress_full_2020_candidate_geometric_return_pct": round(
            float(candidate_stress_primary["geometric_return_pct"]), 4
        ),
        "stress_full_2020_baseline_profit_factor": round(float(baseline_stress_primary["profit_factor"]), 4),
        "stress_full_2020_candidate_profit_factor": round(float(candidate_stress_primary["profit_factor"]), 4),
        "base_two_year_long_delta_r": secondary_long_delta_r,
        "base_full_2020_short_delta_r": primary_short_delta_r,
        "pass_head_to_head_geo": pass_head_to_head_geo,
        "pass_head_to_head_pf": pass_head_to_head_pf,
        "pass_head_to_head_max_dd": pass_head_to_head_max_dd,
        "pass_candidate_stress_geo": pass_candidate_stress_geo,
        "pass_candidate_stress_pf": pass_candidate_stress_pf,
        "pass_secondary_long_guard": pass_secondary_long_guard,
        "pool_status": "independent_pool_member",
        "status": status,
        "next_route": "run_mainline_parallel_window_comparison"
        if can_challenge
        else "keep_in_pool_and_search_next_candidate",
    }


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
    side_rows: list[dict[str, Any]],
    decision_row: dict[str, Any],
    baseline_strategy_profile: str,
    baseline_overlay_profile: str,
    candidate_profile: str,
) -> str:
    base_primary = [
        row
        for row in summary_rows
        if row["cost_scenario"] == "base" and row["window"] == "full_2020"
    ]
    base_primary = sorted(base_primary, key=lambda row: row["profile_kind"])
    base_primary_yearly = [
        row
        for row in yearly_rows
        if row["cost_scenario"] == "base" and row["window"] == "full_2020"
    ]
    base_primary_yearly = sorted(base_primary_yearly, key=lambda row: (row["profile_kind"], row["year"]))
    base_primary_side = [
        row
        for row in side_rows
        if row["cost_scenario"] == "base" and row["window"] == "full_2020"
    ]
    base_primary_side = sorted(base_primary_side, key=lambda row: (row["profile_kind"], row["side"]))
    conclusion = (
        "candidate can challenge mainline under current independent-pool gate."
        if decision_row["status"] == "candidate_can_challenge_mainline"
        else "mainline remains preferred, but candidate is retained in independent pool."
    )
    return "\n".join(
        [
            "# Breakout Candidate vs Challenger Managed",
            "",
            f"- baseline managed: `{baseline_strategy_profile}` + `{baseline_overlay_profile}`",
            f"- independent candidate: `{candidate_profile}`",
            "- this comparison is parallel evaluation only; no PnL summation between strategies.",
            "",
            "## Base Full_2020 Summary",
            "",
            markdown_table(
                base_primary,
                [
                    ("profile_kind", "Kind"),
                    ("strategy_profile", "Profile"),
                    ("trades", "Trades"),
                    ("profit_factor", "PF"),
                    ("cum_r", "Cum R"),
                    ("max_dd_r", "MaxDD R"),
                    ("geometric_return_pct", "Geom Return %"),
                    ("cagr_pct", "CAGR %"),
                ],
            ),
            "",
            "## Base Full_2020 Side Summary",
            "",
            markdown_table(
                base_primary_side,
                [
                    ("profile_kind", "Kind"),
                    ("side", "Side"),
                    ("trades", "Trades"),
                    ("profit_factor", "PF"),
                    ("cum_r", "Cum R"),
                    ("geometric_return_pct", "Geom Return %"),
                ],
            ),
            "",
            "## Base Full_2020 Yearly Geometric Returns",
            "",
            markdown_table(
                base_primary_yearly,
                [
                    ("profile_kind", "Kind"),
                    ("year", "Year"),
                    ("trades", "Trades"),
                    ("geometric_return_pct", "Geom Return %"),
                ],
            ),
            "",
            "## Decision",
            "",
            markdown_table(
                [decision_row],
                [
                    ("base_full_2020_baseline_geometric_return_pct", "Baseline Geom %"),
                    ("base_full_2020_candidate_geometric_return_pct", "Candidate Geom %"),
                    ("base_full_2020_candidate_cagr_pct", "Candidate CAGR %"),
                    ("base_full_2020_candidate_profit_factor", "Candidate PF"),
                    ("base_full_2020_candidate_max_dd_r", "Candidate MaxDD"),
                    ("stress_full_2020_candidate_geometric_return_pct", "Stress Candidate Geom %"),
                    ("stress_full_2020_candidate_profit_factor", "Stress Candidate PF"),
                    ("base_two_year_long_delta_r", "Two-Year LONG Delta R"),
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

    primary_start = parse_date(args.primary_start)
    secondary_start = parse_date(args.secondary_start)
    if secondary_start <= primary_start:
        raise ValueError("secondary-start must be later than primary-start")
    requested_end = parse_date(args.end) if args.end else datetime.now(timezone.utc)

    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    history_service = build_service()
    enriched_history = {
        args.baseline_strategy_profile: history_service.prepare_enriched_history(
            exchange=args.exchange,
            market_type=args.market_type,
            symbol=args.symbol,
            strategy_profile=args.baseline_strategy_profile,
            start=primary_start,
            end=requested_end,
        ),
        args.candidate_profile: history_service.prepare_enriched_history(
            exchange=args.exchange,
            market_type=args.market_type,
            symbol=args.symbol,
            strategy_profile=args.candidate_profile,
            start=primary_start,
            end=requested_end,
        ),
    }
    resolved_end = resolve_end_from_history(enriched_history)

    summary_rows: list[dict[str, Any]] = []
    baseline_trade_frames: list[pd.DataFrame] = []
    candidate_trade_frames: list[pd.DataFrame] = []

    for cost_scenario, overrides in COST_SCENARIOS.items():
        baseline_service = build_service(assumption_overrides=overrides)
        candidate_service = make_candidate_service(assumption_overrides=overrides)
        for window in WINDOWS:
            start = secondary_start if window == "two_year" else primary_start
            baseline_summary, baseline_trades = run_managed_baseline_with_helper(
                service=baseline_service,
                symbol=args.symbol,
                baseline_strategy_profile=args.baseline_strategy_profile,
                baseline_overlay_profile=args.baseline_overlay_profile,
                start=start,
                end=resolved_end,
                enriched_frames=enriched_history[args.baseline_strategy_profile],
            )
            candidate_report = candidate_service.run(
                exchange=args.exchange,
                market_type=args.market_type,
                symbols=[args.symbol],
                strategy_profiles=[args.candidate_profile],
                start=start,
                end=resolved_end,
            )
            candidate_summary = candidate_report.overall[0]

            baseline_frame = trades_frame(
                cost_scenario=cost_scenario,
                window=window,
                window_start=start,
                window_end=resolved_end,
                profile_kind="baseline_managed",
                strategy_profile=args.baseline_strategy_profile,
                profile_label=baseline_label(
                    baseline_strategy_profile=args.baseline_strategy_profile,
                    baseline_overlay_profile=args.baseline_overlay_profile,
                ),
                trades=[asdict(item) for item in baseline_trades],
            )
            candidate_frame = trades_frame(
                cost_scenario=cost_scenario,
                window=window,
                window_start=start,
                window_end=resolved_end,
                profile_kind="candidate",
                strategy_profile=args.candidate_profile,
                profile_label=args.candidate_profile,
                trades=[asdict(item) for item in candidate_report.trades],
            )
            baseline_trade_frames.append(baseline_frame)
            candidate_trade_frames.append(candidate_frame)

            summary_rows.append(
                summary_row(
                    cost_scenario=cost_scenario,
                    window=window,
                    window_start=start,
                    window_end=resolved_end,
                    profile_kind="baseline_managed",
                    strategy_profile=args.baseline_strategy_profile,
                    profile_label=baseline_label(
                        baseline_strategy_profile=args.baseline_strategy_profile,
                        baseline_overlay_profile=args.baseline_overlay_profile,
                    ),
                    summary=baseline_summary,
                    trades_frame_obj=baseline_frame,
                )
            )
            summary_rows.append(
                summary_row(
                    cost_scenario=cost_scenario,
                    window=window,
                    window_start=start,
                    window_end=resolved_end,
                    profile_kind="candidate",
                    strategy_profile=args.candidate_profile,
                    profile_label=args.candidate_profile,
                    summary=candidate_summary,
                    trades_frame_obj=candidate_frame,
                )
            )

    baseline_trades_all = pd.concat(baseline_trade_frames, ignore_index=True)
    candidate_trades_all = pd.concat(candidate_trade_frames, ignore_index=True)
    trades_all = pd.concat([baseline_trades_all, candidate_trades_all], ignore_index=True)

    side_rows = build_side_summary(trades_all)
    yearly_rows = build_yearly_geometric_returns(trades_all)
    cost_rows = build_cost_sensitivity(summary_rows)
    decision_row = build_comparison_decision(
        summary_rows=summary_rows,
        side_rows=side_rows,
        baseline_strategy_profile=args.baseline_strategy_profile,
        baseline_overlay_profile=args.baseline_overlay_profile,
        candidate_profile=args.candidate_profile,
    )

    report = build_report(
        summary_rows=summary_rows,
        yearly_rows=yearly_rows,
        side_rows=side_rows,
        decision_row=decision_row,
        baseline_strategy_profile=args.baseline_strategy_profile,
        baseline_overlay_profile=args.baseline_overlay_profile,
        candidate_profile=args.candidate_profile,
    )

    write_csv(output_dir / "summary_all.csv", summary_rows)
    write_csv(output_dir / "side_summary_all.csv", side_rows)
    write_csv(output_dir / "yearly_geometric_returns.csv", yearly_rows)
    write_csv(output_dir / "cost_sensitivity.csv", cost_rows)
    write_csv(output_dir / "comparison_decision.csv", [decision_row])
    baseline_trades_all.to_csv(output_dir / "baseline_trades.csv", index=False)
    candidate_trades_all.to_csv(output_dir / "candidate_trades.csv", index=False)
    (output_dir / "report.md").write_text(report + "\n", encoding="utf-8")
    (output_dir / "meta.json").write_text(
        json.dumps(
            {
                "baseline_strategy_profile": args.baseline_strategy_profile,
                "baseline_overlay_profile": args.baseline_overlay_profile,
                "candidate_profile": args.candidate_profile,
                "resolved_end": resolved_end.isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved breakout candidate vs challenger managed artifacts to {output_dir}")


if __name__ == "__main__":
    main()
