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
from app.services.carry_basis_execution_service import CarryBasisExecutionService
from app.services.carry_basis_research_service import CarryBasisResearchService


INPUT_PATH = ROOT / "artifacts" / "derivatives" / "bybit_btcusdt_hourly_state" / "research_table.csv"
OUTPUT_DIR = ROOT / "artifacts" / "derivatives" / "bybit_btcusdt_carry_basis_execution"
CALIBRATION_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
CALIBRATION_END = datetime(2024, 3, 19, tzinfo=timezone.utc)
EVAL_START = CALIBRATION_END
EVAL_END = datetime(2026, 3, 19, tzinfo=timezone.utc)


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


def select_highlights(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not summary_rows:
        return []
    frame = pd.DataFrame(summary_rows)
    return (
        frame.sort_values(
            ["annualized_roc_pct", "cumulative_roc_pct", "max_drawdown_pct"],
            ascending=[False, False, True],
        )
        .head(12)
        .to_dict("records")
    )


def build_monthly_summary(monthly_rows: list[dict[str, Any]], best_summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if best_summary is None or not monthly_rows:
        return None
    frame = pd.DataFrame(monthly_rows)
    subset = frame[
        (frame["candidate"] == best_summary["candidate"])
        & (frame["scenario"] == best_summary["scenario"])
        & (frame["horizon_hours"] == best_summary["horizon_hours"])
    ].copy()
    if subset.empty:
        return None
    return {
        "candidate": best_summary["candidate"],
        "scenario": best_summary["scenario"],
        "horizon_hours": int(best_summary["horizon_hours"]),
        "months": int(len(subset)),
        "positive_months": int((subset["net_roc_sum_bps"] > 0).sum()),
        "positive_month_rate_pct": round(float((subset["net_roc_sum_bps"] > 0).mean() * 100.0), 2),
    }


def build_recommendation(best_summary: dict[str, Any] | None, monthly_summary: dict[str, Any] | None) -> dict[str, Any]:
    if best_summary is None:
        return {
            "best_candidate": "none",
            "best_scenario": "none",
            "best_horizon_hours": 0,
            "annualized_roc_pct": 0.0,
            "interpretation": "no_execution_sensitive_candidate",
            "recommendation": "do_not_continue",
        }

    annualized = float(best_summary["annualized_roc_pct"])
    drawdown = float(best_summary["max_drawdown_pct"])
    month_rate = float(monthly_summary["positive_month_rate_pct"]) if monthly_summary else 0.0

    if annualized <= 0.0:
        interpretation = "execution_costs_and_turnover_absorb_edge"
        recommendation = "stop_here"
    elif month_rate < 35.0:
        interpretation = "positive_but_highly_path_dependent"
        recommendation = "continue_only_if_you_accept_execution_project_risk"
    elif drawdown > annualized * 1.5:
        interpretation = "capital_efficiency_too_weak_for_risk_taken"
        recommendation = "do_not_promote_yet"
    else:
        interpretation = "candidate_survives_execution_sensitive_screen"
        recommendation = "continue_to_execution_buildout"

    return {
        "best_candidate": best_summary["candidate"],
        "best_scenario": best_summary["scenario"],
        "best_horizon_hours": int(best_summary["horizon_hours"]),
        "annualized_roc_pct": round(annualized, 4),
        "interpretation": interpretation,
        "recommendation": recommendation,
    }


def build_report(
    dataset_rows: list[dict[str, Any]],
    summary_highlights: list[dict[str, Any]],
    monthly_summary: dict[str, Any] | None,
    recommendation_row: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# BTC Carry / Basis Execution-Sensitive Prototype",
            "",
            "- 这版把最小 carry 研究推进到 execution-sensitive 原型：固定持有期、非重叠持仓、spot/perp 分腿成本、perp 初始保证金、资本利用率和周转。",
            "- 这仍然不是实盘级回测。`spot` 仍用 `index_close` 代理，`perp` 用 `mark_close`；没有订单簿、借贷、真实现货成交、强平和保证金路径。",
            "- maker/taker 成本情景是研究用假设，不是声称当前官方费率。",
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
            "## Top Sequence Results",
            "",
            markdown_table(
                summary_highlights,
                [
                    ("candidate", "Candidate"),
                    ("scenario", "Scenario"),
                    ("horizon_hours", "Horizon h"),
                    ("trades", "Trades"),
                    ("utilization_pct", "Utilization %"),
                    ("round_trips_per_year", "Turns/Yr"),
                    ("gross_notional_turnover_x_per_year", "Notional Turnover x/Yr"),
                    ("net_mean_bps", "Net Mean bps"),
                    ("net_roc_mean_bps", "Net ROC Mean bps"),
                    ("cumulative_roc_pct", "Cum ROC %"),
                    ("annualized_roc_pct", "Ann ROC %"),
                    ("max_drawdown_pct", "MaxDD %"),
                ],
            ),
            "",
            "## Best Monthly Stability",
            "",
            markdown_table(
                [monthly_summary] if monthly_summary else [],
                [
                    ("candidate", "Candidate"),
                    ("scenario", "Scenario"),
                    ("horizon_hours", "Horizon h"),
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
                    ("best_candidate", "Best Candidate"),
                    ("best_scenario", "Best Scenario"),
                    ("best_horizon_hours", "Best Horizon h"),
                    ("annualized_roc_pct", "Ann ROC %"),
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
    exec_service = CarryBasisExecutionService()
    frame = load_frame()
    calibration = frame[(frame["timestamp"] >= pd.Timestamp(CALIBRATION_START)) & (frame["timestamp"] < pd.Timestamp(CALIBRATION_END))].copy()
    evaluation = frame[(frame["timestamp"] >= pd.Timestamp(EVAL_START)) & (frame["timestamp"] < pd.Timestamp(EVAL_END))].copy()

    edges = base_service.build_calibration_edges(calibration)
    evaluation = base_service.enrich_frame(evaluation, edges)
    candidates = base_service.build_candidates()
    scenarios = exec_service.build_scenarios()

    summary_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for candidate in candidates:
            for horizon in (24, 72, 168):
                trades, summary, monthly = exec_service.simulate_sequence(
                    evaluation,
                    candidate=candidate,
                    scenario=scenario,
                    horizon=horizon,
                )
                summary_rows.append(summary)
                trade_rows.extend(trades)
                monthly_rows.extend(monthly)

    best_summary = exec_service.choose_best_summary(summary_rows)
    best_monthly_summary = build_monthly_summary(monthly_rows, best_summary)
    recommendation_row = build_recommendation(best_summary, best_monthly_summary)
    dataset_rows = build_dataset_rows(calibration, evaluation)
    summary_highlights = select_highlights(summary_rows)

    write_csv(OUTPUT_DIR / "dataset_summary.csv", dataset_rows)
    write_csv(OUTPUT_DIR / "execution_summary.csv", summary_rows)
    write_csv(OUTPUT_DIR / "execution_trades.csv", trade_rows)
    write_csv(OUTPUT_DIR / "monthly_summary.csv", monthly_rows)
    if best_monthly_summary:
        write_csv(OUTPUT_DIR / "best_monthly_summary.csv", [best_monthly_summary])
    write_csv(OUTPUT_DIR / "recommendation.csv", [recommendation_row])

    (OUTPUT_DIR / "report.md").write_text(
        build_report(dataset_rows, summary_highlights, best_monthly_summary, recommendation_row),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
