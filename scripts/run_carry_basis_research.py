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
from app.services.carry_basis_research_service import CarryBasisResearchService


INPUT_PATH = ROOT / "artifacts" / "derivatives" / "bybit_btcusdt_hourly_state" / "research_table.csv"
OUTPUT_DIR = ROOT / "artifacts" / "derivatives" / "bybit_btcusdt_carry_basis_research"
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
            "funding_events_eval": int(evaluation["funding_rate_event"].notna().sum()),
        }
    ]


def select_highlights(
    unconditional_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unconditional_highlights = (
        pd.DataFrame(unconditional_rows).sort_values(["leg", "horizon_hours"]).to_dict("records")
        if unconditional_rows
        else []
    )
    candidate_highlights: list[dict[str, Any]] = []
    if candidate_rows:
        frame = pd.DataFrame(candidate_rows)
        candidate_highlights = (
            frame.sort_values(
                ["net_mean_bps_cost_28bps", "gross_mean_bps", "observations"],
                ascending=[False, False, False],
            )
            .head(10)
            .to_dict("records")
        )
    return unconditional_highlights, candidate_highlights


def build_recommendation(best_candidate: dict[str, Any] | None, monthly_summary: dict[str, Any] | None) -> dict[str, Any]:
    if best_candidate is None:
        return {
            "best_candidate": "none",
            "best_horizon_hours": 0,
            "best_net_mean_bps_28bps": 0.0,
            "interpretation": "no_carry_state_found",
            "recommendation": "do_not_continue",
        }

    net_mean = float(best_candidate["net_mean_bps_cost_28bps"])
    monthly_rate = float(monthly_summary["positive_month_rate_pct"]) if monthly_summary else 0.0
    if net_mean <= 0:
        interpretation = "gross_positive_but_net_negative_after_costs"
        recommendation = "do_not_promote_to_strategy"
    elif monthly_rate < 40.0:
        interpretation = "net_positive_but_monthly_stability_weak"
        recommendation = "continue_research_only_if_execution_is_priority"
    else:
        interpretation = "candidate_worthy_of_further_execution_research"
        recommendation = "continue_to_execution_sensitive_carry_prototype"

    return {
        "best_candidate": best_candidate["candidate"],
        "best_horizon_hours": int(best_candidate["horizon_hours"]),
        "best_net_mean_bps_28bps": round(net_mean, 4),
        "interpretation": interpretation,
        "recommendation": recommendation,
    }


def build_report(
    dataset_rows: list[dict[str, Any]],
    unconditional_highlights: list[dict[str, Any]],
    candidate_highlights: list[dict[str, Any]],
    recommendation_row: dict[str, Any],
    monthly_summary_row: dict[str, Any] | None,
) -> str:
    return "\n".join(
        [
            "# BTC Spot-Perp Carry / Basis Minimal Prototype",
            "",
            "- 这不是可部署回测，而是最小研究原型：用 Bybit 小时面板判断 `BTC long spot / short perp` 的 carry/basis 研究值不值得继续。",
            "- `spot` 用 `index_close` 代理，`perp` 用 `mark_close`；所以这版只能看研究值，不能当作成交级策略结果。",
            "- funding 只计建仓后的未来 funding 事件；成本敏感性固定看 `0 / 10 / 20 / 28 bps` round-trip。",
            "- `short spot / long perp` 只作为对称诊断，不代表当前建议直接研究可执行反向 carry。",
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
                    ("funding_events_eval", "Eval Funding Events"),
                ],
            ),
            "",
            "## Unconditional Carry",
            "",
            markdown_table(
                unconditional_highlights,
                [
                    ("leg", "Leg"),
                    ("horizon_hours", "Horizon h"),
                    ("observations", "Obs"),
                    ("gross_mean_bps", "Gross Mean bps"),
                    ("gross_positive_rate_pct", "Gross >0 %"),
                    ("net_mean_bps_cost_10bps", "Net Mean 10bps"),
                    ("net_mean_bps_cost_20bps", "Net Mean 20bps"),
                    ("net_mean_bps_cost_28bps", "Net Mean 28bps"),
                ],
            ),
            "",
            "## Candidate States",
            "",
            markdown_table(
                candidate_highlights,
                [
                    ("candidate", "Candidate"),
                    ("horizon_hours", "Horizon h"),
                    ("observations", "Obs"),
                    ("gross_mean_bps", "Gross Mean bps"),
                    ("gross_positive_rate_pct", "Gross >0 %"),
                    ("net_mean_bps_cost_10bps", "Net Mean 10bps"),
                    ("net_mean_bps_cost_20bps", "Net Mean 20bps"),
                    ("net_mean_bps_cost_28bps", "Net Mean 28bps"),
                    ("net_positive_rate_pct_cost_28bps", "Net >0 % 28bps"),
                ],
            ),
            "",
            "## Best Candidate Monthly Stability",
            "",
            markdown_table(
                [monthly_summary_row] if monthly_summary_row else [],
                [
                    ("candidate", "Candidate"),
                    ("horizon_hours", "Horizon h"),
                    ("cost_bps", "Cost bps"),
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
                    ("best_horizon_hours", "Best Horizon h"),
                    ("best_net_mean_bps_28bps", "Best Net Mean 28bps"),
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

    service = CarryBasisResearchService()
    frame = load_frame()
    calibration = frame[(frame["timestamp"] >= pd.Timestamp(CALIBRATION_START)) & (frame["timestamp"] < pd.Timestamp(CALIBRATION_END))].copy()
    evaluation = frame[(frame["timestamp"] >= pd.Timestamp(EVAL_START)) & (frame["timestamp"] < pd.Timestamp(EVAL_END))].copy()

    edges = service.build_calibration_edges(calibration)
    evaluation = service.enrich_frame(evaluation, edges)

    unconditional_rows = service.summarize_unconditional(evaluation)
    candidates = service.build_candidates()
    candidate_rows = service.summarize_candidates(evaluation, candidates)
    best_candidate = service.choose_best_candidate(candidate_rows, cost_bps=28.0)

    monthly_rows: list[dict[str, Any]] = []
    monthly_summary_row: dict[str, Any] | None = None
    if best_candidate is not None:
        candidate = next(item for item in candidates if item.label == best_candidate["candidate"])
        monthly_rows, monthly_summary_row = service.summarize_monthly_stability(
            evaluation,
            candidate=candidate,
            horizon=int(best_candidate["horizon_hours"]),
            cost_bps=28.0,
        )

    recommendation_row = build_recommendation(best_candidate, monthly_summary_row)
    dataset_rows = build_dataset_rows(calibration, evaluation)
    unconditional_highlights, candidate_highlights = select_highlights(unconditional_rows, candidate_rows)

    write_csv(OUTPUT_DIR / "dataset_summary.csv", dataset_rows)
    write_csv(OUTPUT_DIR / "unconditional_summary.csv", unconditional_rows)
    write_csv(OUTPUT_DIR / "candidate_summary.csv", candidate_rows)
    write_csv(OUTPUT_DIR / "best_candidate_monthly.csv", monthly_rows)
    if monthly_summary_row:
        write_csv(OUTPUT_DIR / "best_candidate_monthly_summary.csv", [monthly_summary_row])
    write_csv(OUTPUT_DIR / "recommendation.csv", [recommendation_row])

    (OUTPUT_DIR / "report.md").write_text(
        build_report(dataset_rows, unconditional_highlights, candidate_highlights, recommendation_row, monthly_summary_row),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
