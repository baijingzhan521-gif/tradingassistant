from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_range_failure_vs_challenger_managed import ensure_output_dir, parse_date, write_csv
from scripts.post_tp1_managed_replay import build_service
from scripts.run_range_failure_vs_challenger_managed import resolve_end_from_history
from scripts.run_simple_candidate_v2_regime_switch_fixed_calendar import (
    DEFAULT_SWITCH_DATE,
    markdown_table,
    run_regime_switch_case,
)


DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "backtests" / "simple_candidate_v2_regime_switch_robustness"
DEFAULT_PRIMARY_START = "2020-01-01"
PROBE_SWITCH_DATES = (
    "2024-01-19",
    "2024-02-18",
    "2024-03-19",
    "2024-04-18",
    "2024-05-18",
)
ROBUST_COST_SCENARIOS = {
    "base": {},
    "stress_x2": {"taker_fee_bps": 10.0, "slippage_bps": 4.0},
    "stress_x3": {"taker_fee_bps": 15.0, "slippage_bps": 6.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness audit for the fixed-calendar simple_candidate_v2 -> challenger_managed switch.")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--simple-profile", default="swing_trend_simple_candidate_v2")
    parser.add_argument("--challenger-strategy-profile", default="swing_trend_long_regime_short_no_reversal_no_aux_v1")
    parser.add_argument(
        "--challenger-overlay-profile",
        default="be_if_no_extension_within_3bars_after_tp1_and_veto_hold_at_trend_96_98",
    )
    parser.add_argument("--primary-start", default=DEFAULT_PRIMARY_START)
    parser.add_argument("--end", default=None, help="Defaults to latest available data at runtime.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def utc_from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def lookup_row(
    rows: list[dict[str, Any]],
    *,
    cost_scenario: str,
    window: str,
    scenario_kind: str,
    extra: tuple[str, Any] | None = None,
) -> dict[str, Any]:
    for row in rows:
        if row["cost_scenario"] == cost_scenario and row["window"] == window and row["scenario_kind"] == scenario_kind:
            if extra is not None and row.get(extra[0]) != extra[1]:
                continue
            return row
    raise KeyError(f"Missing row for cost={cost_scenario}, window={window}, kind={scenario_kind}")


def build_probe_panel_row(*, switch_date: str, case: dict[str, Any]) -> dict[str, Any]:
    summary_rows = case["summary_rows"]
    decision_row = case["decision_row"]

    baseline = lookup_row(
        summary_rows,
        cost_scenario="base",
        window="full_2020",
        scenario_kind="always_challenger_managed",
    )
    switch = lookup_row(
        summary_rows,
        cost_scenario="base",
        window="full_2020",
        scenario_kind="switch_simple_candidate_v2_then_challenger_managed",
    )
    stress2_baseline = lookup_row(
        summary_rows,
        cost_scenario="stress_x2",
        window="full_2020",
        scenario_kind="always_challenger_managed",
    )
    stress2_switch = lookup_row(
        summary_rows,
        cost_scenario="stress_x2",
        window="full_2020",
        scenario_kind="switch_simple_candidate_v2_then_challenger_managed",
    )
    stress3_baseline = lookup_row(
        summary_rows,
        cost_scenario="stress_x3",
        window="full_2020",
        scenario_kind="always_challenger_managed",
    )
    stress3_switch = lookup_row(
        summary_rows,
        cost_scenario="stress_x3",
        window="full_2020",
        scenario_kind="switch_simple_candidate_v2_then_challenger_managed",
    )
    switch_long = lookup_row(
        case["side_rows"],
        cost_scenario="base",
        window="two_year",
        scenario_kind="switch_simple_candidate_v2_then_challenger_managed",
        extra=("side", "LONG"),
    )

    return {
        "switch_date": switch_date,
        "status": decision_row["status"],
        "base_baseline_geometric_return_pct": baseline["geometric_return_pct"],
        "base_switch_geometric_return_pct": switch["geometric_return_pct"],
        "base_delta_geometric_return_pct": round(float(switch["geometric_return_pct"]) - float(baseline["geometric_return_pct"]), 4),
        "base_baseline_profit_factor": baseline["profit_factor"],
        "base_switch_profit_factor": switch["profit_factor"],
        "base_delta_profit_factor": round(float(switch["profit_factor"]) - float(baseline["profit_factor"]), 4),
        "base_baseline_max_dd_r": baseline["max_dd_r"],
        "base_switch_max_dd_r": switch["max_dd_r"],
        "stress_x2_baseline_geometric_return_pct": stress2_baseline["geometric_return_pct"],
        "stress_x2_switch_geometric_return_pct": stress2_switch["geometric_return_pct"],
        "stress_x2_delta_geometric_return_pct": round(
            float(stress2_switch["geometric_return_pct"]) - float(stress2_baseline["geometric_return_pct"]), 4
        ),
        "stress_x2_baseline_profit_factor": stress2_baseline["profit_factor"],
        "stress_x2_switch_profit_factor": stress2_switch["profit_factor"],
        "stress_x3_baseline_geometric_return_pct": stress3_baseline["geometric_return_pct"],
        "stress_x3_switch_geometric_return_pct": stress3_switch["geometric_return_pct"],
        "stress_x3_delta_geometric_return_pct": round(
            float(stress3_switch["geometric_return_pct"]) - float(stress3_baseline["geometric_return_pct"]), 4
        ),
        "stress_x3_baseline_profit_factor": stress3_baseline["profit_factor"],
        "stress_x3_switch_profit_factor": stress3_switch["profit_factor"],
        "two_year_long_cum_r": switch_long["cum_r"],
        "base_pass": bool(decision_row["pass_base_geo"])
        and bool(decision_row["pass_base_pf"])
        and bool(decision_row["pass_base_max_dd"]),
        "stress_x2_pass": bool(decision_row["pass_stress_geo"]) and bool(decision_row["pass_stress_pf"]),
        "stress_x3_beats_baseline": float(stress3_switch["geometric_return_pct"]) > float(stress3_baseline["geometric_return_pct"]),
        "stress_x3_positive": float(stress3_switch["geometric_return_pct"]) > 0.0,
        "stress_x3_pf_ge_baseline": float(stress3_switch["profit_factor"] or 0.0) >= float(stress3_baseline["profit_factor"] or 0.0),
    }


def build_concentration_rows(*, case: dict[str, Any]) -> list[dict[str, Any]]:
    trade_frame: pd.DataFrame = case["trade_frame"]
    base_full = trade_frame[(trade_frame["cost_scenario"] == "base") & (trade_frame["window"] == "full_2020")]
    rows: list[dict[str, Any]] = []
    for scenario_kind in ("always_challenger_managed", "switch_simple_candidate_v2_then_challenger_managed"):
        sub = base_full[base_full["scenario_kind"] == scenario_kind].copy()
        if sub.empty:
            continue
        total_r = float(sub["pnl_r"].sum())
        ordered = sub.sort_values("pnl_r", ascending=False)
        top3_share = round(float(ordered.head(3)["pnl_r"].sum()) / total_r * 100.0, 2) if total_r else None
        year_r = sub.groupby(sub["signal_time"].dt.year)["pnl_r"].sum()
        month_r = sub.groupby(sub["signal_time"].dt.strftime("%Y-%m"))["pnl_r"].sum()
        quarter_r = sub.groupby(sub["signal_time"].dt.tz_localize(None).dt.to_period("Q").astype(str))["pnl_r"].sum()
        best_year = float(year_r.max()) if not year_r.empty else 0.0
        best_month = float(month_r.max()) if not month_r.empty else 0.0
        best_quarter = float(quarter_r.max()) if not quarter_r.empty else 0.0
        rows.append(
            {
                "scenario_kind": scenario_kind,
                "trades": int(len(sub)),
                "cum_r": round(total_r, 4),
                "top3_trade_pnl_share_pct": top3_share,
                "best_year_pnl_share_pct": round(best_year / total_r * 100.0, 2) if total_r else None,
                "best_month_pnl_share_pct": round(best_month / total_r * 100.0, 2) if total_r else None,
                "best_quarter_pnl_share_pct": round(best_quarter / total_r * 100.0, 2) if total_r else None,
                "positive_years": int((year_r > 0).sum()),
                "negative_years": int((year_r < 0).sum()),
            }
        )
    return rows


def build_robustness_decision(*, probe_rows: list[dict[str, Any]], concentration_rows: list[dict[str, Any]]) -> dict[str, Any]:
    switch_row = next(row for row in concentration_rows if row["scenario_kind"] == "switch_simple_candidate_v2_then_challenger_managed")
    base_top3 = float(switch_row["top3_trade_pnl_share_pct"] or 0.0)
    base_best_year = float(switch_row["best_year_pnl_share_pct"] or 0.0)
    base_best_month = float(switch_row["best_month_pnl_share_pct"] or 0.0)

    all_base = all(row["base_pass"] for row in probe_rows)
    all_stress2 = all(row["stress_x2_pass"] for row in probe_rows)
    all_stress3_positive = all(row["stress_x3_positive"] for row in probe_rows)
    all_stress3_beats = all(row["stress_x3_beats_baseline"] for row in probe_rows)
    all_stress3_pf = all(row["stress_x3_pf_ge_baseline"] for row in probe_rows)

    if all_base and all_stress2 and all_stress3_beats and all_stress3_pf:
        if base_top3 <= 35.0 and base_best_year <= 55.0 and base_best_month <= 35.0:
            status = "robust_under_probe_panel"
        else:
            status = "robust_but_concentrated"
    else:
        status = "date_sensitive_or_cost_fragile"

    risk_flags: list[str] = []
    for row in probe_rows:
        if not row["base_pass"]:
            risk_flags.append(f"{row['switch_date']}:base_fail")
        if not row["stress_x2_pass"]:
            risk_flags.append(f"{row['switch_date']}:stress_x2_fail")
        if not row["stress_x3_positive"]:
            risk_flags.append(f"{row['switch_date']}:stress_x3_negative")
        if not row["stress_x3_beats_baseline"]:
            risk_flags.append(f"{row['switch_date']}:stress_x3_under_baseline")
    if base_top3 > 35.0:
        risk_flags.append("concentration:top3_trade_share_gt_35pct")
    if base_best_year > 55.0:
        risk_flags.append("concentration:best_year_share_gt_55pct")
    if base_best_month > 35.0:
        risk_flags.append("concentration:best_month_share_gt_35pct")

    return {
        "status": status,
        "all_probe_base_pass": all_base,
        "all_probe_stress_x2_pass": all_stress2,
        "all_probe_stress_x3_positive": all_stress3_positive,
        "all_probe_stress_x3_beats_baseline": all_stress3_beats,
        "all_probe_stress_x3_pf_ge_baseline": all_stress3_pf,
        "switch_top3_trade_pnl_share_pct": round(base_top3, 2),
        "switch_best_year_pnl_share_pct": round(base_best_year, 2),
        "switch_best_month_pnl_share_pct": round(base_best_month, 2),
        "risk_flags": ";".join(risk_flags) if risk_flags else "",
    }


def build_report(
    *,
    probe_rows: list[dict[str, Any]],
    concentration_rows: list[dict[str, Any]],
    decision_row: dict[str, Any],
    canonical_case: dict[str, Any],
) -> str:
    probe_rows_sorted = sorted(probe_rows, key=lambda row: row["switch_date"])
    concentration_rows_sorted = sorted(concentration_rows, key=lambda row: row["scenario_kind"])
    canonical_base_year_rows = [
        row
        for row in canonical_case["yearly_rows"]
        if row["cost_scenario"] == "base" and row["window"] == "full_2020"
    ]
    canonical_base_year_rows = sorted(canonical_base_year_rows, key=lambda row: (row["scenario_kind"], int(row["year"])))
    return "\n".join(
        [
            "# Simple Candidate V2 Regime Switch Robustness Audit",
            "",
            f"- canonical switch date: `{DEFAULT_SWITCH_DATE}`",
            "- probe dates are fixed and symmetric around the canonical cut; no date search is performed.",
            "",
            "## Probe Panel",
            "",
            markdown_table(
                probe_rows_sorted,
                [
                    ("switch_date", "Switch Date"),
                    ("status", "Decision"),
                    ("base_delta_geometric_return_pct", "Base ΔGeom %"),
                    ("base_delta_profit_factor", "Base ΔPF"),
                    ("stress_x2_delta_geometric_return_pct", "Stress x2 ΔGeom %"),
                    ("stress_x3_delta_geometric_return_pct", "Stress x3 ΔGeom %"),
                    ("base_pass", "Base Pass"),
                    ("stress_x2_pass", "Stress x2 Pass"),
                    ("stress_x3_positive", "Stress x3 Positive"),
                    ("stress_x3_beats_baseline", "Stress x3 Beats Baseline"),
                ],
            ),
            "",
            "## Concentration",
            "",
            markdown_table(
                concentration_rows_sorted,
                [
                    ("scenario_kind", "Scenario"),
                    ("trades", "Trades"),
                    ("cum_r", "Cum R"),
                    ("top3_trade_pnl_share_pct", "Top 3 Share %"),
                    ("best_year_pnl_share_pct", "Best Year Share %"),
                    ("best_month_pnl_share_pct", "Best Month Share %"),
                    ("best_quarter_pnl_share_pct", "Best Quarter Share %"),
                    ("positive_years", "Positive Years"),
                    ("negative_years", "Negative Years"),
                ],
            ),
            "",
            "## Canonical Yearly Breakdown",
            "",
            markdown_table(
                canonical_base_year_rows,
                [
                    ("scenario_kind", "Scenario"),
                    ("year", "Year"),
                    ("trades", "Trades"),
                    ("profit_factor", "PF"),
                    ("cum_r", "Cum R"),
                    ("max_dd_r", "MaxDD R"),
                    ("geometric_return_pct", "Geom Return %"),
                ],
            ),
            "",
            "## Robustness Decision",
            "",
            markdown_table(
                [decision_row],
                [
                    ("status", "Status"),
                    ("all_probe_base_pass", "All Probe Base Pass"),
                    ("all_probe_stress_x2_pass", "All Probe Stress x2 Pass"),
                    ("all_probe_stress_x3_positive", "All Probe Stress x3 Positive"),
                    ("all_probe_stress_x3_beats_baseline", "All Probe Stress x3 Beats Baseline"),
                    ("switch_top3_trade_pnl_share_pct", "Switch Top3 Share %"),
                    ("switch_best_year_pnl_share_pct", "Switch Best Year Share %"),
                    ("switch_best_month_pnl_share_pct", "Switch Best Month Share %"),
                    ("risk_flags", "Risk Flags"),
                ],
            ),
            "",
            "## Conclusion",
            "",
            (
                "这版切换在更严格的敏感性面板下仍然成立。"
                if decision_row["status"] == "robust_under_probe_panel"
                else "这版切换存在切点敏感、成本脆弱或收益集中度偏高的风险。"
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    primary_start = parse_date(args.primary_start)
    requested_end = parse_date(args.end) if args.end else None
    if requested_end is None:
        requested_end = datetime.now(timezone.utc)

    history_service = build_service()
    shared_history = {
        args.challenger_strategy_profile: history_service.prepare_enriched_history(
            exchange=args.exchange,
            market_type=args.market_type,
            symbol=args.symbol,
            strategy_profile=args.challenger_strategy_profile,
            start=primary_start,
            end=requested_end,
        ),
        args.simple_profile: history_service.prepare_enriched_history(
            exchange=args.exchange,
            market_type=args.market_type,
            symbol=args.symbol,
            strategy_profile=args.simple_profile,
            start=primary_start,
            end=requested_end,
        ),
    }
    resolved_end = resolve_end_from_history(shared_history)

    probe_rows: list[dict[str, Any]] = []
    canonical_case: dict[str, Any] | None = None
    for switch_date_str in PROBE_SWITCH_DATES:
        switch_date = utc_from_iso(switch_date_str)
        case = run_regime_switch_case(
            symbol=args.symbol,
            exchange=args.exchange,
            market_type=args.market_type,
            simple_profile=args.simple_profile,
            challenger_strategy_profile=args.challenger_strategy_profile,
            challenger_overlay_profile=args.challenger_overlay_profile,
            switch_date=switch_date,
            primary_start=primary_start,
            requested_end=requested_end,
            cost_scenarios=ROBUST_COST_SCENARIOS,
            enriched_history=shared_history,
            resolved_end=resolved_end,
        )
        probe_rows.append(build_probe_panel_row(switch_date=switch_date_str, case=case))
        if switch_date_str == DEFAULT_SWITCH_DATE:
            canonical_case = case

    if canonical_case is None:
        raise RuntimeError("Canonical case was not produced. Check probe dates.")

    concentration_rows = build_concentration_rows(case=canonical_case)
    decision_row = build_robustness_decision(probe_rows=probe_rows, concentration_rows=concentration_rows)
    report = build_report(
        probe_rows=probe_rows,
        concentration_rows=concentration_rows,
        decision_row=decision_row,
        canonical_case=canonical_case,
    )

    write_csv(output_dir / "probe_panel.csv", probe_rows)
    write_csv(output_dir / "concentration_summary.csv", concentration_rows)
    write_csv(output_dir / "robustness_decision.csv", [decision_row])
    write_csv(output_dir / "canonical_summary_all.csv", canonical_case["summary_rows"])
    write_csv(output_dir / "canonical_yearly_geometric_returns.csv", canonical_case["yearly_rows"])
    (output_dir / "report.md").write_text(report + "\n", encoding="utf-8")
    print(f"saved regime-switch robustness audit artifacts to {output_dir}")


if __name__ == "__main__":
    main()
