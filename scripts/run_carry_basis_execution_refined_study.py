from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.logging import configure_logging
from app.services.carry_basis_execution_refined_service import CarryBasisExecutionRefinedService
from app.services.carry_basis_research_service import CarryBasisResearchService


INPUT_PATH = ROOT / "artifacts" / "derivatives" / "bybit_btcusdt_hourly_state" / "research_table.csv"
OUTPUT_DIR = ROOT / "artifacts" / "derivatives" / "bybit_btcusdt_carry_basis_execution_refined"
CALIBRATION_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
CALIBRATION_END = datetime(2024, 3, 19, tzinfo=timezone.utc)
EVAL_START = CALIBRATION_END
EVAL_END = datetime(2026, 3, 19, tzinfo=timezone.utc)
FOCUS_CANDIDATES = {"basis_positive", "basis_and_funding_positive"}
HORIZON_HOURS = 168


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
    body = [header, divider]
    for row in rows:
        values: list[str] = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join(body)


def load_frame() -> pd.DataFrame:
    frame = pd.read_csv(INPUT_PATH)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def build_dataset_rows(calibration: pd.DataFrame, evaluation: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "calibration_start": CALIBRATION_START.isoformat(),
            "calibration_end": CALIBRATION_END.isoformat(),
            "eval_start": EVAL_START.isoformat(),
            "eval_end": EVAL_END.isoformat(),
            "calibration_hours": int(len(calibration)),
            "eval_hours": int(len(evaluation)),
            "eval_days": round(float(len(evaluation) / 24.0), 2),
            "funding_events_eval": int(evaluation["funding_rate_event"].notna().sum()),
        }
    ]


def build_monthly_summary(monthly_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not monthly_rows or not summary_rows:
        return []
    summary_frame = pd.DataFrame(summary_rows)
    monthly_frame = pd.DataFrame(monthly_rows)
    rows: list[dict[str, Any]] = []
    for _, row in summary_frame.iterrows():
        subset = monthly_frame[
            (monthly_frame["candidate"] == row["candidate"])
            & (monthly_frame["scenario"] == row["scenario"])
            & (monthly_frame["horizon_hours"] == row["horizon_hours"])
        ].copy()
        rows.append(
            {
                "candidate": row["candidate"],
                "scenario": row["scenario"],
                "capital_mode": row["capital_mode"],
                "annual_opportunity_cost_pct": row["annual_opportunity_cost_pct"],
                "months": int(len(subset)),
                "positive_months": int((subset["net_roc_sum_bps"] > 0).sum()) if not subset.empty else 0,
                "positive_month_rate_pct": round(float((subset["net_roc_sum_bps"] > 0).mean() * 100.0), 2) if not subset.empty else 0.0,
            }
        )
    return rows


def build_delta_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not summary_rows:
        return []
    frame = pd.DataFrame(summary_rows)
    baseline = frame[frame["scenario"] == "legacy_proxy_baseline"].copy()
    if baseline.empty:
        return []
    baseline = baseline.set_index("candidate")
    deltas: list[dict[str, Any]] = []
    for _, row in frame[frame["scenario"] != "legacy_proxy_baseline"].iterrows():
        base_row = baseline.loc[row["candidate"]]
        deltas.append(
            {
                "candidate": row["candidate"],
                "scenario": row["scenario"],
                "capital_mode": row["capital_mode"],
                "annual_opportunity_cost_pct": row["annual_opportunity_cost_pct"],
                "delta_all_in_cost_bps": round(float(row["all_in_cost_bps"] - base_row["all_in_cost_bps"]), 4),
                "delta_annualized_roc_pct": round(float(row["annualized_roc_pct"] - base_row["annualized_roc_pct"]), 4),
                "delta_net_mean_bps": round(float(row["net_mean_bps"] - base_row["net_mean_bps"]), 4),
            }
        )
    return deltas


def build_recommendation(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not summary_rows:
        return {
            "status": "no_results",
            "interpretation": "refined_matrix_empty",
            "recommendation": "stop_here",
        }
    frame = pd.DataFrame(summary_rows)
    realistic = frame[frame["scenario"] == "realistic_base_segregated_4opp"].copy()
    if realistic.empty:
        return {
            "status": "missing_anchor_scenario",
            "interpretation": "cannot_compare",
            "recommendation": "inspect_outputs",
        }
    worst_case = frame[frame["scenario"] == "realistic_base_segregated_8opp"].copy()
    best_realistic = realistic.sort_values("annualized_roc_pct", ascending=False).iloc[0]
    worst_realistic = (
        worst_case.sort_values("annualized_roc_pct", ascending=True).iloc[0]
        if not worst_case.empty
        else realistic.sort_values("annualized_roc_pct", ascending=True).iloc[0]
    )

    if float(best_realistic["annualized_roc_pct"]) <= 0.5:
        interpretation = "edge_collapses_close_to_zero_under_refined_cost_and_capital_assumptions"
        recommendation = "do_not_promote"
    elif float(best_realistic["annualized_roc_pct"]) <= 1.0:
        interpretation = "edge_survives_but_is_too_thin_once_execution_and_capital_are_made_more_realistic"
        recommendation = "treat_as_research_only"
    else:
        interpretation = "edge_survives_only_under_tight_execution_and_capital_assumptions"
        recommendation = "continue_only_if_execution_is_the_main_project"

    return {
        "status": "ok",
        "best_realistic_candidate": best_realistic["candidate"],
        "best_realistic_ann_roc_pct": round(float(best_realistic["annualized_roc_pct"]), 4),
        "worst_realistic_candidate": worst_realistic["candidate"],
        "worst_realistic_ann_roc_pct": round(float(worst_realistic["annualized_roc_pct"]), 4),
        "interpretation": interpretation,
        "recommendation": recommendation,
    }


def build_report(
    dataset_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    monthly_rows: list[dict[str, Any]],
    recommendation_row: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# BTC Carry / Basis Refined Execution Validation",
            "",
            "- 这版只验证两条 168h 候选：`basis_positive`、`basis_and_funding_positive`。",
            "- 这不是实盘成交级回测。`spot` 仍只能用 `index_close` 作为 mid 代理，`perp` 仍只能用 `mark_close` 作为 mid 代理；所谓“更真实”仅意味着显式加入分腿 fee/slippage、机会成本和共池/分池资本口径。",
            "- `pooled` 口径是假设统一资金池里，现货持仓可以充分支持 `15%` 的 perp 初始保证金，因此资本需求约为 `1.0x`；`segregated` 口径是假设现货资金和 perp 保证金分开，因此资本需求为 `1.15x`。这两个口径都是研究假设，不应伪装成真实交易所结算规则。",
            "",
            "## Dataset",
            "",
            markdown_table(
                dataset_rows,
                [
                    ("calibration_start", "Calibration Start"),
                    ("calibration_end", "Calibration End"),
                    ("eval_start", "Eval Start"),
                    ("eval_end", "Eval End"),
                    ("calibration_hours", "Calibration Hours"),
                    ("eval_hours", "Eval Hours"),
                    ("eval_days", "Eval Days"),
                    ("funding_events_eval", "Eval Funding Events"),
                ],
            ),
            "",
            "## Refined Matrix",
            "",
            markdown_table(
                summary_rows,
                [
                    ("candidate", "Candidate"),
                    ("scenario", "Scenario"),
                    ("capital_mode", "Capital"),
                    ("annual_opportunity_cost_pct", "Opp Cost %"),
                    ("trades", "Trades"),
                    ("fee_cost_bps", "Fee bps"),
                    ("slippage_cost_bps", "Slippage bps"),
                    ("opportunity_cost_bps", "Opp Cost bps"),
                    ("all_in_cost_bps", "All-in Cost bps"),
                    ("net_mean_bps", "Net Mean bps"),
                    ("annualized_roc_pct", "Ann ROC %"),
                    ("max_drawdown_pct", "MaxDD %"),
                ],
            ),
            "",
            "## Delta Vs Legacy Hybrid",
            "",
            markdown_table(
                delta_rows,
                [
                    ("candidate", "Candidate"),
                    ("scenario", "Scenario"),
                    ("capital_mode", "Capital"),
                    ("annual_opportunity_cost_pct", "Opp Cost %"),
                    ("delta_all_in_cost_bps", "Delta Cost bps"),
                    ("delta_net_mean_bps", "Delta Net Mean bps"),
                    ("delta_annualized_roc_pct", "Delta Ann ROC %"),
                ],
            ),
            "",
            "## Monthly Stability",
            "",
            markdown_table(
                monthly_rows,
                [
                    ("candidate", "Candidate"),
                    ("scenario", "Scenario"),
                    ("capital_mode", "Capital"),
                    ("annual_opportunity_cost_pct", "Opp Cost %"),
                    ("months", "Months"),
                    ("positive_months", "Positive Months"),
                    ("positive_month_rate_pct", "Positive Month %"),
                ],
            ),
            "",
            "## Decision",
            "",
            markdown_table(
                [recommendation_row],
                [
                    ("status", "Status"),
                    ("best_realistic_candidate", "Best Realistic Candidate"),
                    ("best_realistic_ann_roc_pct", "Best Realistic Ann ROC %"),
                    ("worst_realistic_candidate", "Worst Realistic Candidate"),
                    ("worst_realistic_ann_roc_pct", "Worst Realistic Ann ROC %"),
                    ("interpretation", "Interpretation"),
                    ("recommendation", "Recommendation"),
                ],
            ),
            "",
        ]
    )


def main() -> None:
    configure_logging()
    ensure_output_dir()

    base_service = CarryBasisResearchService()
    refined_service = CarryBasisExecutionRefinedService()
    frame = load_frame()
    calibration = frame[(frame["timestamp"] >= pd.Timestamp(CALIBRATION_START)) & (frame["timestamp"] < pd.Timestamp(CALIBRATION_END))].copy()
    evaluation = frame[(frame["timestamp"] >= pd.Timestamp(EVAL_START)) & (frame["timestamp"] < pd.Timestamp(EVAL_END))].copy()

    edges = base_service.build_calibration_edges(calibration)
    evaluation = base_service.enrich_frame(evaluation, edges)
    candidates = [item for item in base_service.build_candidates() if item.label in FOCUS_CANDIDATES]
    scenarios = refined_service.build_focus_scenarios()

    summary_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for candidate in candidates:
            trades, summary, monthly = refined_service.simulate_sequence(
                evaluation,
                candidate=candidate,
                scenario=scenario,
                horizon=HORIZON_HOURS,
            )
            summary_rows.append(summary)
            trade_rows.extend(trades)
            monthly_rows.extend(monthly)

    summary_rows = sorted(summary_rows, key=lambda row: (row["candidate"], row["scenario"]))
    monthly_summary_rows = build_monthly_summary(monthly_rows, summary_rows)
    delta_rows = build_delta_rows(summary_rows)
    recommendation_row = build_recommendation(summary_rows)
    dataset_rows = build_dataset_rows(calibration, evaluation)

    write_csv(OUTPUT_DIR / "scenario_matrix.csv", summary_rows)
    write_csv(OUTPUT_DIR / "trade_summary.csv", trade_rows)
    write_csv(OUTPUT_DIR / "monthly_summary.csv", monthly_summary_rows)
    write_csv(OUTPUT_DIR / "delta_vs_legacy.csv", delta_rows)
    write_csv(OUTPUT_DIR / "recommendation.csv", [recommendation_row])
    (OUTPUT_DIR / "report.md").write_text(
        build_report(
            dataset_rows,
            summary_rows,
            delta_rows,
            monthly_summary_rows,
            recommendation_row,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
