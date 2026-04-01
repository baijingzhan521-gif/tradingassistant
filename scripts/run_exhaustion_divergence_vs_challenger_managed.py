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
from scripts.post_tp1_managed_replay import PROFILE_SPEC_MAP, build_service
from scripts.run_range_failure_vs_challenger_managed import (
    COST_SCENARIOS,
    DEFAULT_BASELINE_OVERLAY_PROFILE,
    DEFAULT_BASELINE_STRATEGY_PROFILE,
    WINDOWS,
    baseline_label,
    build_cost_sensitivity,
    build_decision,
    ensure_output_dir,
    markdown_table,
    parse_date,
    report_to_rows,
    resolve_end_from_history,
    run_managed_baseline_with_helper,
    summarize_complementarity,
    trades_frame,
    write_csv,
)


SYMBOL = "BTC/USDT:USDT"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "backtests" / "exhaustion_divergence_vs_challenger_managed"
DEFAULT_ALT_PROFILE = "swing_exhaustion_divergence_v1_btc"
WINDOW_STARTS = {
    "two_year": "2024-03-19",
    "full_2020": "2020-01-01",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-run exhaustion divergence takeaways against challenger_managed baseline.")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--baseline-strategy-profile", default=DEFAULT_BASELINE_STRATEGY_PROFILE)
    parser.add_argument("--baseline-overlay-profile", default=DEFAULT_BASELINE_OVERLAY_PROFILE)
    parser.add_argument("--alt-profile", default=DEFAULT_ALT_PROFILE)
    parser.add_argument("--primary-start", default=WINDOW_STARTS["full_2020"])
    parser.add_argument("--secondary-start", default=WINDOW_STARTS["two_year"])
    parser.add_argument("--end", default=None, help="Defaults to latest available data at runtime.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def make_alt_service(*, assumption_overrides: dict[str, Any] | None = None) -> BacktestService:
    assumptions = {
        "exit_profile": "exhaustion_divergence_strategy_defined",
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


def build_report(
    *,
    summary_rows: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
    complementarity_rows: list[dict[str, Any]],
    offset_rows: list[dict[str, Any]],
    decision_row: dict[str, Any],
    baseline_strategy_profile: str,
    baseline_overlay_profile: str,
    alt_profile: str,
) -> str:
    base_summary_rows = [row for row in summary_rows if row["cost_scenario"] == "base"]
    stress_summary_rows = [row for row in summary_rows if row["cost_scenario"] == "stress_x2"]

    if decision_row["status"] == "complementary_watchlist":
        continuation_text = "standalone 和互补性两层都过线，`exhaustion divergence` 暂时保留在 complementary watchlist。"
    elif decision_row["status"] == "rejected_offset":
        continuation_text = "standalone 没崩，但 offset 不够，这条线先不继续扩。"
    else:
        continuation_text = "standalone 自身不够稳，这条互补 alpha 也不继续扩。"

    family_text = (
        "有一定不同 alpha 家族证据，月度相关性没有和主线强同向。"
        if decision_row["different_alpha_family_signal"]
        else "不同 alpha 家族证据不足，月度同向性还是太强。"
    )
    offset_text = (
        f"主窗口 `baseline 亏 / alt 盈` 月份数是 `{decision_row['base_full_2020_baseline_negative_alt_positive_months']}`，累计 offset `{decision_row['base_full_2020_offset_r_sum']:.4f}R`。"
    )
    standalone_text = (
        "standalone 没过 floor。"
        if decision_row["status"] == "rejected_floor"
        else "standalone 先过了 floor。"
    )

    return "\n".join(
        [
            "# Exhaustion Divergence vs Challenger Managed",
            "",
            f"- baseline managed: `{baseline_strategy_profile}` + `{baseline_overlay_profile}`",
            f"- alt profile: `{alt_profile}`",
            "- 这轮仍然固定用 `challenger_managed` 做 baseline，不回退到旧 raw mainline。",
            "- `combined_naive_r` 只作方向参考，不能当真实组合收益。",
            "",
            "## Base Summary",
            "",
            markdown_table(
                base_summary_rows,
                [
                    ("window", "Window"),
                    ("profile_label", "Profile"),
                    ("trades", "Trades"),
                    ("profit_factor", "PF"),
                    ("expectancy_r", "Exp R"),
                    ("cum_r", "Cum R"),
                    ("max_dd_r", "MaxDD R"),
                ],
            ),
            "",
            "## Stress Summary",
            "",
            markdown_table(
                stress_summary_rows,
                [
                    ("window", "Window"),
                    ("profile_label", "Profile"),
                    ("trades", "Trades"),
                    ("profit_factor", "PF"),
                    ("expectancy_r", "Exp R"),
                    ("cum_r", "Cum R"),
                    ("max_dd_r", "MaxDD R"),
                ],
            ),
            "",
            "## Cost Sensitivity",
            "",
            markdown_table(
                cost_rows,
                [
                    ("window", "Window"),
                    ("profile_label", "Profile"),
                    ("base_cum_r", "Base Cum R"),
                    ("stress_cum_r", "Stress Cum R"),
                    ("delta_cum_r", "Stress-Base Cum R"),
                    ("base_profit_factor", "Base PF"),
                    ("stress_profit_factor", "Stress PF"),
                    ("delta_max_dd_r", "Stress-Base MaxDD"),
                ],
            ),
            "",
            "## Complementarity Summary",
            "",
            markdown_table(
                complementarity_rows,
                [
                    ("cost_scenario", "Cost"),
                    ("window", "Window"),
                    ("months", "Months"),
                    ("monthly_corr", "Monthly Corr"),
                    ("baseline_negative_alt_positive_months", "Base Neg / Alt Pos"),
                    ("offset_r_sum", "Offset R"),
                    ("opposite_sign_months", "Opposite Sign Months"),
                    ("combined_naive_r", "Naive Combined R"),
                ],
            ),
            "",
            "## Offset Months",
            "",
            markdown_table(
                offset_rows,
                [
                    ("cost_scenario", "Cost"),
                    ("window", "Window"),
                    ("month", "Month"),
                    ("baseline_r", "Baseline R"),
                    ("alt_r", "Alt R"),
                    ("combined_naive_r", "Naive Combined R"),
                    ("baseline_negative_alt_positive", "Base Neg / Alt Pos"),
                ],
            ),
            "",
            "## Decision",
            "",
            markdown_table(
                [decision_row],
                [
                    ("base_full_2020_alt_profit_factor", "Base PF"),
                    ("base_full_2020_alt_cum_r", "Base Cum R"),
                    ("base_full_2020_alt_max_dd_r", "Base MaxDD"),
                    ("stress_full_2020_alt_profit_factor", "Stress PF"),
                    ("stress_full_2020_alt_cum_r", "Stress Cum R"),
                    ("base_full_2020_monthly_corr", "Monthly Corr"),
                    ("base_full_2020_offset_r_sum", "Offset R"),
                    ("status", "Status"),
                ],
            ),
            "",
            "## Conclusion",
            "",
            f"- {family_text}",
            f"- {offset_text}",
            f"- {standalone_text}",
            f"- {continuation_text}",
        ]
    )


def main() -> None:
    args = parse_args()
    configure_logging()

    primary_start = parse_date(args.primary_start)
    secondary_start = parse_date(args.secondary_start)
    if args.baseline_overlay_profile not in PROFILE_SPEC_MAP:
        raise ValueError(f"Unknown baseline overlay profile: {args.baseline_overlay_profile}")
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
        args.alt_profile: history_service.prepare_enriched_history(
            exchange=args.exchange,
            market_type=args.market_type,
            symbol=args.symbol,
            strategy_profile=args.alt_profile,
            start=primary_start,
            end=requested_end,
        ),
    }
    resolved_end = resolve_end_from_history(enriched_history)

    baseline_summary_rows: list[dict[str, Any]] = []
    alt_summary_rows: list[dict[str, Any]] = []
    baseline_trade_frames: list[pd.DataFrame] = []
    alt_trade_frames: list[pd.DataFrame] = []
    complementarity_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []

    for cost_scenario, overrides in COST_SCENARIOS.items():
        baseline_service = build_service(assumption_overrides=overrides)
        alt_service = make_alt_service(assumption_overrides=overrides)
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
            alt_report = alt_service.run(
                exchange=args.exchange,
                market_type=args.market_type,
                symbols=[args.symbol],
                strategy_profiles=[args.alt_profile],
                start=start,
                end=resolved_end,
            )

            baseline_summary_rows.extend(
                report_to_rows(
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
                    report=type("Report", (), {"overall": [baseline_summary]})(),
                )
            )
            alt_summary_rows.extend(
                report_to_rows(
                    cost_scenario=cost_scenario,
                    window=window,
                    window_start=start,
                    window_end=resolved_end,
                    profile_kind="alt",
                    strategy_profile=args.alt_profile,
                    profile_label=args.alt_profile,
                    report=alt_report,
                )
            )

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
            alt_frame = trades_frame(
                cost_scenario=cost_scenario,
                window=window,
                window_start=start,
                window_end=resolved_end,
                profile_kind="alt",
                strategy_profile=args.alt_profile,
                profile_label=args.alt_profile,
                trades=[asdict(item) for item in alt_report.trades],
            )
            baseline_trade_frames.append(baseline_frame)
            alt_trade_frames.append(alt_frame)

            comp_row, comp_monthly_rows = summarize_complementarity(
                cost_scenario=cost_scenario,
                window=window,
                window_start=start,
                window_end=resolved_end,
                baseline_trades=baseline_frame,
                alt_trades=alt_frame,
            )
            complementarity_rows.append(comp_row)
            monthly_rows.extend(comp_monthly_rows)

    summary_rows = baseline_summary_rows + alt_summary_rows
    cost_rows = build_cost_sensitivity(summary_rows)
    decision_row = build_decision(
        summary_rows=summary_rows,
        complementarity_rows=complementarity_rows,
        baseline_strategy_profile=args.baseline_strategy_profile,
        baseline_overlay_profile=args.baseline_overlay_profile,
        alt_profile=args.alt_profile,
    )
    if decision_row["status"] == "complementary_watchlist":
        decision_row["next_route"] = "hold_exhaustion_watchlist"
    else:
        decision_row["next_route"] = "no_complementary_watchlist"
    offset_rows = [row for row in monthly_rows if row["baseline_negative_alt_positive"]]

    report = build_report(
        summary_rows=summary_rows,
        cost_rows=cost_rows,
        complementarity_rows=complementarity_rows,
        offset_rows=offset_rows,
        decision_row=decision_row,
        baseline_strategy_profile=args.baseline_strategy_profile,
        baseline_overlay_profile=args.baseline_overlay_profile,
        alt_profile=args.alt_profile,
    )
    (output_dir / "report.md").write_text(report + "\n", encoding="utf-8")

    write_csv(output_dir / "summary_all.csv", summary_rows)
    write_csv(output_dir / "cost_sensitivity.csv", cost_rows)
    write_csv(output_dir / "monthly_complementarity.csv", monthly_rows)
    write_csv(output_dir / "complementarity_summary.csv", complementarity_rows)
    write_csv(output_dir / "offset_months.csv", offset_rows)
    write_csv(output_dir / "decision.csv", [decision_row])
    pd.concat(baseline_trade_frames, ignore_index=True).to_csv(output_dir / "baseline_trades.csv", index=False)
    pd.concat(alt_trade_frames, ignore_index=True).to_csv(output_dir / "alt_trades.csv", index=False)
    (output_dir / f"baseline_{args.baseline_strategy_profile}_meta.json").write_text(
        json.dumps(
            {
                "baseline_strategy_profile": args.baseline_strategy_profile,
                "baseline_overlay_profile": args.baseline_overlay_profile,
                "alt_profile": args.alt_profile,
                "resolved_end": resolved_end.isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"saved exhaustion divergence vs challenger managed artifacts to {output_dir}")


if __name__ == "__main__":
    main()
