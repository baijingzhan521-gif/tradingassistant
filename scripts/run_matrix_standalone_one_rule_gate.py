from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.logging import configure_logging
from app.services.strategy_service import StrategyService
from scripts.run_trend_pullback_standalone_one_rule_gate import (
    build_cost_sensitivity,
    build_trade_concentration,
    build_trade_frame,
    build_yearly_geometric_returns,
    ensure_output_dir,
    make_service,
    markdown_table,
    parse_date,
    resolve_end_from_history,
    summary_row,
    trade_rows,
    write_csv,
)


SYMBOL = "BTC/USDT:USDT"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "backtests" / "matrix_standalone_one_rule_gate"
BASELINE_PROFILE = "swing_trend_matrix_no_gate_current_entry_v1"
DEFAULT_PROFILES = [
    BASELINE_PROFILE,
    "swing_trend_long_regime_gate_v1",
    "swing_trend_matrix_no_gate_simple_entry_v1",
]
WINDOW_STARTS = {
    "full_2020": "2020-01-01",
    "two_year": "2024-03-19",
}
ACTIVATION_PRECHECK_START = "2024-01-01"
ACTIVATION_PRECHECK_END = "2026-03-01"
ACTIVATION_THRESHOLD = 0.05
COST_SCENARIOS = {
    "base": {},
    "stress_x2": {"taker_fee_bps": 10.0, "slippage_bps": 4.0},
    "stress_x3": {"taker_fee_bps": 15.0, "slippage_bps": 6.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone one-rule promotion gate for swing trend matrix candidates.")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    parser.add_argument("--primary-start", default=WINDOW_STARTS["full_2020"])
    parser.add_argument("--secondary-start", default=WINDOW_STARTS["two_year"])
    parser.add_argument("--end", default=None, help="Defaults to latest available data at runtime.")
    parser.add_argument("--activation-precheck-start", default=ACTIVATION_PRECHECK_START)
    parser.add_argument("--activation-precheck-end", default=ACTIVATION_PRECHECK_END)
    parser.add_argument("--activation-threshold", type=float, default=ACTIVATION_THRESHOLD)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def _strategy_behavior_signature(profile: str) -> tuple[bool, bool]:
    strategy = StrategyService().build_strategy(profile)
    enforce_regime_gate = bool(getattr(strategy, "enforce_regime_gate", True))
    use_current_entry = bool(getattr(strategy, "use_current_entry", True))
    return enforce_regime_gate, use_current_entry


def _build_activation_precheck(
    *,
    service,
    symbol: str,
    profiles: list[str],
    baseline_profile: str,
    precheck_start: datetime,
    precheck_end: datetime,
    enriched_history: dict[str, dict[str, Any]],
    threshold: float,
) -> tuple[list[dict[str, Any]], set[str]]:
    signature_by_profile: dict[str, set[tuple[str, str]]] = {}
    trade_count_by_profile: dict[str, int] = {}
    for profile in profiles:
        _summary, trades = service.run_symbol_strategy_with_enriched_frames(
            symbol=symbol,
            strategy_profile=profile,
            start=precheck_start,
            end=precheck_end,
            enriched_frames=enriched_history[profile],
        )
        signatures = {(str(trade.entry_time), str(trade.side)) for trade in trades}
        signature_by_profile[profile] = signatures
        trade_count_by_profile[profile] = len(trades)

    baseline_signatures = signature_by_profile.get(baseline_profile, set())
    rows: list[dict[str, Any]] = []
    active_candidates: set[str] = set()
    for profile in profiles:
        if profile == baseline_profile:
            continue
        candidate_signatures = signature_by_profile.get(profile, set())
        union = baseline_signatures | candidate_signatures
        intersection = baseline_signatures & candidate_signatures
        same_ratio = (len(intersection) / len(union)) if union else 1.0
        changed_ratio = 1.0 - same_ratio
        passed = changed_ratio >= threshold
        if passed:
            active_candidates.add(profile)
        rows.append(
            {
                "baseline_profile": baseline_profile,
                "candidate_profile": profile,
                "precheck_start": precheck_start.date().isoformat(),
                "precheck_end": precheck_end.date().isoformat(),
                "signature_definition": "(entry_time, side)",
                "baseline_trade_count": int(trade_count_by_profile.get(baseline_profile, 0)),
                "candidate_trade_count": int(trade_count_by_profile.get(profile, 0)),
                "intersection_count": int(len(intersection)),
                "union_count": int(len(union)),
                "same_ratio": round(float(same_ratio), 6),
                "changed_ratio": round(float(changed_ratio), 6),
                "threshold": round(float(threshold), 6),
                "pass_activation_precheck": bool(passed),
            }
        )
    return rows, active_candidates


def _summary_lookup(summary_rows: list[dict[str, Any]], *, cost: str, window: str, profile: str) -> dict[str, Any]:
    for row in summary_rows:
        if row["cost_scenario"] == cost and row["window"] == window and row["strategy_profile"] == profile:
            return row
    raise KeyError(f"missing summary row cost={cost} window={window} profile={profile}")


def _concentration_lookup(
    concentration_rows: list[dict[str, Any]], *, cost: str, window: str, profile: str
) -> dict[str, Any]:
    for row in concentration_rows:
        if row["cost_scenario"] == cost and row["window"] == window and row["strategy_profile"] == profile:
            return row
    raise KeyError(f"missing concentration row cost={cost} window={window} profile={profile}")


def build_promotion_decision(
    *,
    summary_rows: list[dict[str, Any]],
    concentration_rows: list[dict[str, Any]],
    activation_rows: list[dict[str, Any]],
    baseline_profile: str,
    profiles: list[str],
) -> list[dict[str, Any]]:
    baseline_base_full = _summary_lookup(summary_rows, cost="base", window="full_2020", profile=baseline_profile)
    baseline_geo = float(baseline_base_full["geometric_return_pct"])
    activation_map = {
        str(row["candidate_profile"]): bool(row["pass_activation_precheck"])
        for row in activation_rows
    }

    rows: list[dict[str, Any]] = []
    for profile in profiles:
        if profile == baseline_profile:
            continue
        base_full = _summary_lookup(summary_rows, cost="base", window="full_2020", profile=profile)
        stress_x2 = _summary_lookup(summary_rows, cost="stress_x2", window="full_2020", profile=profile)
        stress_x3 = _summary_lookup(summary_rows, cost="stress_x3", window="full_2020", profile=profile)
        concentration = _concentration_lookup(concentration_rows, cost="base", window="full_2020", profile=profile)
        pass_activation_precheck = bool(activation_map.get(profile, False))

        pass_geo_vs_baseline = float(base_full["geometric_return_pct"]) > baseline_geo
        pass_base_pf = float(base_full["profit_factor"]) >= 1.10
        pass_base_max_dd = float(base_full["max_dd_r"]) <= 6.0
        pass_stress_x2_geo = float(stress_x2["geometric_return_pct"]) > 0.0
        pass_stress_x2_pf = float(stress_x2["profit_factor"]) >= 1.00
        pass_stress_x3_geo = float(stress_x3["geometric_return_pct"]) > 0.0
        pass_stress_x3_pf = float(stress_x3["profit_factor"]) >= 1.00
        pass_trades_floor = int(base_full["trades"]) >= 10
        pass_top3_share = float(concentration["top3_trades_pnl_share_pct"]) <= 65.0
        pass_best_year_share = float(concentration["best_year_geometric_pct_share"]) <= 80.0

        promoted = (
            pass_activation_precheck
            and pass_geo_vs_baseline
            and pass_base_pf
            and pass_base_max_dd
            and pass_stress_x2_geo
            and pass_stress_x2_pf
            and pass_stress_x3_geo
            and pass_stress_x3_pf
            and pass_trades_floor
            and pass_top3_share
            and pass_best_year_share
        )
        if promoted:
            status = "promoted_standalone_candidate"
        elif not pass_activation_precheck:
            status = "rejected_inactive_candidate"
        else:
            status = "rejected_fragile_or_unprofitable"

        rows.append(
            {
                "baseline_profile": baseline_profile,
                "candidate_profile": profile,
                "base_full_2020_baseline_geometric_return_pct": round(baseline_geo, 4),
                "base_full_2020_candidate_geometric_return_pct": round(float(base_full["geometric_return_pct"]), 4),
                "base_full_2020_candidate_cagr_pct": round(float(base_full["cagr_pct"]), 4),
                "base_full_2020_candidate_profit_factor": round(float(base_full["profit_factor"]), 4),
                "base_full_2020_candidate_max_dd_r": round(float(base_full["max_dd_r"]), 4),
                "stress_x2_full_2020_candidate_geometric_return_pct": round(float(stress_x2["geometric_return_pct"]), 4),
                "stress_x2_full_2020_candidate_cagr_pct": round(float(stress_x2["cagr_pct"]), 4),
                "stress_x2_full_2020_candidate_profit_factor": round(float(stress_x2["profit_factor"]), 4),
                "stress_x3_full_2020_candidate_geometric_return_pct": round(float(stress_x3["geometric_return_pct"]), 4),
                "stress_x3_full_2020_candidate_cagr_pct": round(float(stress_x3["cagr_pct"]), 4),
                "stress_x3_full_2020_candidate_profit_factor": round(float(stress_x3["profit_factor"]), 4),
                "base_full_2020_candidate_trades": int(base_full["trades"]),
                "base_full_2020_top3_trades_pnl_share_pct": round(float(concentration["top3_trades_pnl_share_pct"]), 4),
                "base_full_2020_best_year_geometric_pct_share": round(
                    float(concentration["best_year_geometric_pct_share"]), 4
                ),
                "pass_activation_precheck": pass_activation_precheck,
                "pass_base_geo_vs_baseline": pass_geo_vs_baseline,
                "pass_base_pf": pass_base_pf,
                "pass_base_max_dd": pass_base_max_dd,
                "pass_stress_x2_geo": pass_stress_x2_geo,
                "pass_stress_x2_pf": pass_stress_x2_pf,
                "pass_stress_x3_geo": pass_stress_x3_geo,
                "pass_stress_x3_pf": pass_stress_x3_pf,
                "pass_trades_floor": pass_trades_floor,
                "pass_top3_share": pass_top3_share,
                "pass_best_year_share": pass_best_year_share,
                "status": status,
                "next_route": (
                    "promote_and_compare_with_switch_baseline"
                    if promoted
                    else "freeze_matrix_and_move_to_intraday_v2_round2"
                ),
            }
        )
    return rows


def select_top_promoted_candidate(decision_rows: list[dict[str, Any]]) -> str | None:
    promoted = [row for row in decision_rows if row["status"] == "promoted_standalone_candidate"]
    if not promoted:
        return None
    promoted.sort(
        key=lambda row: (
            float(row["base_full_2020_candidate_geometric_return_pct"]),
            float(row["base_full_2020_candidate_profit_factor"]),
            -float(row["base_full_2020_candidate_max_dd_r"]),
        ),
        reverse=True,
    )
    return str(promoted[0]["candidate_profile"])


def _zero_stress_summary_row(
    *,
    cost_scenario: str,
    window: str,
    window_start: datetime,
    window_end: datetime,
    strategy_profile: str,
) -> dict[str, Any]:
    return {
        "cost_scenario": cost_scenario,
        "window": window,
        "window_start": window_start.date().isoformat(),
        "window_end": window_end.date().isoformat(),
        "strategy_profile": strategy_profile,
        "trades": 0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "expectancy_r": 0.0,
        "cum_r": 0.0,
        "max_dd_r": 0.0,
        "avg_holding_bars": 0.0,
        "geometric_return_pct": 0.0,
        "additive_return_pct": 0.0,
        "cagr_pct": 0.0,
    }


def build_report(
    *,
    summary_rows: list[dict[str, Any]],
    yearly_rows: list[dict[str, Any]],
    concentration_rows: list[dict[str, Any]],
    activation_rows: list[dict[str, Any]],
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
    activation_rows = sorted(activation_rows, key=lambda row: row["candidate_profile"])
    promoted = [row["candidate_profile"] for row in decision_rows if row["status"] == "promoted_standalone_candidate"]
    conclusion = (
        f"promoted candidates: {', '.join(promoted)}"
        if promoted
        else "no candidate promoted; move to intraday_mtf_v2 round-2."
    )
    return "\n".join(
        [
            "# Matrix Standalone One-Rule Gate",
            "",
            "- scope: standalone promotion gate for matrix orthogonal family",
            "- baseline profile: `swing_trend_matrix_no_gate_current_entry_v1`",
            "- one-rule candidates only: `swing_trend_long_regime_gate_v1`, `swing_trend_matrix_no_gate_simple_entry_v1`",
            "",
            "## Activation Precheck",
            "",
            markdown_table(
                activation_rows,
                [
                    ("candidate_profile", "Candidate"),
                    ("baseline_trade_count", "Baseline Trades"),
                    ("candidate_trade_count", "Candidate Trades"),
                    ("changed_ratio", "Changed Ratio"),
                    ("threshold", "Threshold"),
                    ("pass_activation_precheck", "Pass"),
                ],
            ),
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
                    ("pass_activation_precheck", "Pass Activation"),
                    ("base_full_2020_candidate_geometric_return_pct", "Base Geom %"),
                    ("base_full_2020_candidate_cagr_pct", "Base CAGR %"),
                    ("base_full_2020_candidate_profit_factor", "Base PF"),
                    ("base_full_2020_candidate_max_dd_r", "Base MaxDD R"),
                    ("stress_x2_full_2020_candidate_geometric_return_pct", "StressX2 Geom %"),
                    ("stress_x3_full_2020_candidate_geometric_return_pct", "StressX3 Geom %"),
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


def _validate_profile_pool(profiles: list[str]) -> None:
    if profiles != DEFAULT_PROFILES:
        raise ValueError(
            "matrix one-rule gate requires strict fixed candidate pool: "
            + ", ".join(DEFAULT_PROFILES)
        )

    base_gate, base_entry = _strategy_behavior_signature(BASELINE_PROFILE)
    if base_gate or not base_entry:
        raise ValueError("matrix baseline behavior signature drifted; expected no-gate + current-entry.")
    for profile in profiles[1:]:
        gate, current_entry = _strategy_behavior_signature(profile)
        changed = int(gate != base_gate) + int(current_entry != base_entry)
        if changed != 1:
            raise ValueError(
                f"profile {profile} is not strict one-rule vs baseline behavior signature (changed={changed})."
            )


def main() -> None:
    args = parse_args()
    configure_logging()
    profiles = list(dict.fromkeys(args.profiles))
    _validate_profile_pool(profiles)

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

    activation_start = parse_date(args.activation_precheck_start)
    activation_end_requested = parse_date(args.activation_precheck_end)
    activation_end = min(activation_end_requested, resolved_end)
    if activation_end <= activation_start:
        raise ValueError("activation-precheck-end must be later than activation-precheck-start.")

    base_service = make_service(assumption_overrides=COST_SCENARIOS["base"])
    activation_rows, active_candidates = _build_activation_precheck(
        service=base_service,
        symbol=args.symbol,
        profiles=profiles,
        baseline_profile=BASELINE_PROFILE,
        precheck_start=activation_start,
        precheck_end=activation_end,
        enriched_history=enriched_history,
        threshold=float(args.activation_threshold),
    )

    summary_rows: list[dict[str, Any]] = []
    trade_row_list: list[dict[str, Any]] = []

    for window, (window_start, window_end) in windows.items():
        for profile in profiles:
            summary, trades = base_service.run_symbol_strategy_with_enriched_frames(
                symbol=args.symbol,
                strategy_profile=profile,
                start=window_start,
                end=window_end,
                enriched_frames=enriched_history[profile],
            )
            summary_rows.append(
                summary_row(
                    cost_scenario="base",
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
                    cost_scenario="base",
                    window=window,
                    strategy_profile=profile,
                    trades=trades,
                )
            )

    stress_profiles = [BASELINE_PROFILE, *sorted(active_candidates)] if active_candidates else [BASELINE_PROFILE]
    skipped_profiles = [profile for profile in profiles if profile not in stress_profiles]

    for cost_scenario in ("stress_x2", "stress_x3"):
        service = make_service(assumption_overrides=COST_SCENARIOS[cost_scenario])
        for window, (window_start, window_end) in windows.items():
            for profile in stress_profiles:
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
            for profile in skipped_profiles:
                summary_rows.append(
                    _zero_stress_summary_row(
                        cost_scenario=cost_scenario,
                        window=window,
                        window_start=window_start,
                        window_end=window_end,
                        strategy_profile=profile,
                    )
                )

    trade_frame = build_trade_frame(trade_row_list)
    yearly_rows = build_yearly_geometric_returns(trade_frame)
    cost_rows = build_cost_sensitivity(summary_rows)
    concentration_rows = build_trade_concentration(trade_frame, yearly_rows)
    decision_rows = build_promotion_decision(
        summary_rows=summary_rows,
        concentration_rows=concentration_rows,
        activation_rows=activation_rows,
        baseline_profile=BASELINE_PROFILE,
        profiles=profiles,
    )

    report = build_report(
        summary_rows=summary_rows,
        yearly_rows=yearly_rows,
        concentration_rows=concentration_rows,
        activation_rows=activation_rows,
        decision_rows=decision_rows,
        profiles=profiles,
    )

    write_csv(output_dir / "activation_precheck.csv", activation_rows)
    write_csv(output_dir / "summary_all.csv", summary_rows)
    write_csv(output_dir / "yearly_geometric_returns.csv", yearly_rows)
    write_csv(output_dir / "cost_sensitivity.csv", cost_rows)
    write_csv(output_dir / "trade_concentration.csv", concentration_rows)
    write_csv(output_dir / "promotion_decision.csv", decision_rows)
    (output_dir / "report.md").write_text(report + "\n", encoding="utf-8")

    top_candidate = select_top_promoted_candidate(decision_rows)
    if top_candidate:
        print(f"saved matrix standalone one-rule gate artifacts to {output_dir}; top promoted={top_candidate}")
    else:
        print(f"saved matrix standalone one-rule gate artifacts to {output_dir}; no promoted candidates")


if __name__ == "__main__":
    main()
