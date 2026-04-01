from __future__ import annotations

import argparse
import csv
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

from app.backtesting.service import BacktestAssumptions, BacktestReport, BacktestService
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.services.strategy_service import StrategyService
from scripts.post_tp1_managed_replay import PROFILE_SPEC_MAP, build_service, run_profile


SYMBOL = "BTC/USDT:USDT"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "backtests" / "range_failure_vs_challenger_managed"
DEFAULT_BASELINE_STRATEGY_PROFILE = "swing_trend_long_regime_short_no_reversal_no_aux_v1"
DEFAULT_BASELINE_OVERLAY_PROFILE = "be_if_no_extension_within_3bars_after_tp1_and_veto_hold_at_trend_96_98"
DEFAULT_ALT_PROFILE = "swing_range_failure_v1_btc"
WINDOWS = ("two_year", "full_2020")
WINDOW_STARTS = {
    "two_year": "2024-03-19",
    "full_2020": "2020-01-01",
}
COST_SCENARIOS = {
    "base": {},
    "stress_x2": {"taker_fee_bps": 10.0, "slippage_bps": 4.0},
}

BASE_FLOOR_PF_MIN = 0.80
BASE_FLOOR_CUM_R_MIN = -10.0
BASE_FLOOR_MAX_DD_R_MAX = 20.0
STRESS_FLOOR_PF_MIN = 0.75
STRESS_FLOOR_CUM_R_MIN = -15.0
COMPLEMENTARY_MAX_CORR = 0.20
COMPLEMENTARY_MIN_OFFSET_MONTHS = 3
COMPLEMENTARY_MIN_OFFSET_R = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-run range-failure takeaways against challenger_managed baseline.")
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


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def resolve_end_from_history(enriched_history: dict[str, dict[str, pd.DataFrame]]) -> datetime:
    profile_ends: list[pd.Timestamp] = []
    for frames in enriched_history.values():
        profile_ends.append(min(pd.Timestamp(frame["timestamp"].max()) for frame in frames.values()))
    return min(profile_ends).to_pydatetime()


def make_alt_service(*, assumption_overrides: dict[str, Any] | None = None) -> BacktestService:
    assumptions = {
        "exit_profile": "range_failure_strategy_defined",
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


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


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
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def baseline_label(*, baseline_strategy_profile: str, baseline_overlay_profile: str) -> str:
    return f"{baseline_strategy_profile} + {baseline_overlay_profile}"


def alt_label(*, alt_profile: str) -> str:
    return alt_profile


def run_managed_baseline_with_helper(
    *,
    service: BacktestService,
    symbol: str,
    baseline_strategy_profile: str,
    baseline_overlay_profile: str,
    start: datetime,
    end: datetime,
    enriched_frames: dict[str, pd.DataFrame],
):
    if baseline_overlay_profile not in PROFILE_SPEC_MAP:
        raise ValueError(f"Unknown baseline overlay profile: {baseline_overlay_profile}")
    strategy = service.strategy_service.build_strategy(baseline_strategy_profile)
    trigger_tf = str(strategy.window_config["trigger_timeframe"])
    if trigger_tf != "1h":
        raise ValueError(f"Managed replay assumes a 1H trigger timeframe, got {trigger_tf}.")
    spec = PROFILE_SPEC_MAP[baseline_overlay_profile]
    extension_features = pd.DataFrame(
        {
            "high": enriched_frames[trigger_tf]["high"].astype(float),
            "close": enriched_frames[trigger_tf]["close"].astype(float),
        }
    )
    summary, trades, _ = run_profile(
        service=service,
        strategy=strategy,
        symbol=symbol,
        strategy_profile=baseline_strategy_profile,
        spec=spec,
        start=start,
        end=end,
        enriched=enriched_frames,
        extension_features=extension_features,
    )
    return summary, trades


def run_alt_profile(
    *,
    service: BacktestService,
    symbol: str,
    alt_profile: str,
    exchange: str,
    market_type: str,
    start: datetime,
    end: datetime,
) -> BacktestReport:
    return service.run(
        exchange=exchange,
        market_type=market_type,
        symbols=[symbol],
        strategy_profiles=[alt_profile],
        start=start,
        end=end,
    )


def report_to_rows(
    *,
    cost_scenario: str,
    window: str,
    window_start: datetime,
    window_end: datetime,
    profile_kind: str,
    strategy_profile: str,
    profile_label: str,
    report: BacktestReport,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in report.overall:
        rows.append(
            {
                "cost_scenario": cost_scenario,
                "window": window,
                "window_start": window_start.date().isoformat(),
                "window_end": window_end.date().isoformat(),
                "profile_kind": profile_kind,
                "strategy_profile": strategy_profile,
                "profile_label": profile_label,
                "trades": item.total_trades,
                "win_rate_pct": round(item.win_rate, 2),
                "profit_factor": round(item.profit_factor, 4),
                "expectancy_r": round(item.expectancy_r, 4),
                "cum_r": round(item.cumulative_r, 4),
                "max_dd_r": round(item.max_drawdown_r, 4),
                "avg_holding_bars": round(item.avg_holding_bars, 2),
                "tp1_hit_rate_pct": round(item.tp1_hit_rate, 2),
                "tp2_hit_rate_pct": round(item.tp2_hit_rate, 2),
            }
        )
    return rows


def trades_frame(
    *,
    cost_scenario: str,
    window: str,
    window_start: datetime,
    window_end: datetime,
    profile_kind: str,
    strategy_profile: str,
    profile_label: str,
    trades: list[dict[str, Any]],
) -> pd.DataFrame:
    frame = pd.DataFrame(trades)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "cost_scenario",
                "window",
                "window_start",
                "window_end",
                "profile_kind",
                "strategy_profile",
                "profile_label",
                "signal_time",
                "entry_time",
                "exit_time",
                "pnl_r",
                "signal_month",
            ]
        )
    frame["cost_scenario"] = cost_scenario
    frame["window"] = window
    frame["window_start"] = window_start.date().isoformat()
    frame["window_end"] = window_end.date().isoformat()
    frame["profile_kind"] = profile_kind
    frame["strategy_profile"] = strategy_profile
    frame["profile_label"] = profile_label
    frame["signal_time"] = pd.to_datetime(frame["signal_time"], utc=True)
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True)
    frame["signal_month"] = frame["signal_time"].dt.strftime("%Y-%m")
    frame["entry_month"] = frame["entry_time"].dt.strftime("%Y-%m")
    return frame.sort_values("entry_time").reset_index(drop=True)


def summarize_complementarity(
    *,
    cost_scenario: str,
    window: str,
    window_start: datetime,
    window_end: datetime,
    baseline_trades: pd.DataFrame,
    alt_trades: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline_monthly = (
        baseline_trades.groupby("signal_month")["pnl_r"].sum().rename("baseline_r").reset_index()
    )
    alt_monthly = alt_trades.groupby("signal_month")["pnl_r"].sum().rename("alt_r").reset_index()
    monthly = baseline_monthly.merge(alt_monthly, on="signal_month", how="outer").fillna(0.0)
    monthly["cost_scenario"] = cost_scenario
    monthly["window"] = window
    monthly["window_start"] = window_start.date().isoformat()
    monthly["window_end"] = window_end.date().isoformat()
    monthly["combined_naive_r"] = monthly["baseline_r"] + monthly["alt_r"]
    monthly["baseline_negative_alt_positive"] = (monthly["baseline_r"] < 0) & (monthly["alt_r"] > 0)
    monthly["opposite_sign"] = (monthly["baseline_r"] * monthly["alt_r"]) < 0
    monthly = monthly.sort_values("signal_month").reset_index(drop=True)

    corr = None
    if len(monthly) >= 2 and monthly["baseline_r"].std() > 0 and monthly["alt_r"].std() > 0:
        corr = float(monthly["baseline_r"].corr(monthly["alt_r"]))

    offset_months = monthly[monthly["baseline_negative_alt_positive"]].copy()
    summary = {
        "cost_scenario": cost_scenario,
        "window": window,
        "window_start": window_start.date().isoformat(),
        "window_end": window_end.date().isoformat(),
        "months": int(len(monthly)),
        "monthly_corr": round(corr, 4) if corr is not None else None,
        "baseline_negative_alt_positive_months": int(offset_months.shape[0]),
        "offset_r_sum": round(float(offset_months["alt_r"].sum()), 4),
        "opposite_sign_months": int(monthly["opposite_sign"].sum()),
        "combined_naive_r": round(float(monthly["combined_naive_r"].sum()), 4),
    }
    monthly_rows = monthly[
        [
            "cost_scenario",
            "window",
            "window_start",
            "window_end",
            "signal_month",
            "baseline_r",
            "alt_r",
            "combined_naive_r",
            "baseline_negative_alt_positive",
            "opposite_sign",
        ]
    ].rename(columns={"signal_month": "month"}).to_dict("records")
    return summary, monthly_rows


def build_cost_sensitivity(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(summary_rows)
    rows: list[dict[str, Any]] = []
    for (window, strategy_profile, profile_kind), group in frame.groupby(
        ["window", "strategy_profile", "profile_kind"],
        sort=False,
    ):
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
                "profile_kind": profile_kind,
                "profile_label": base_row["profile_label"],
                "base_trades": int(base_row["trades"]),
                "stress_trades": int(stress_row["trades"]),
                "base_cum_r": round(float(base_row["cum_r"]), 4),
                "stress_cum_r": round(float(stress_row["cum_r"]), 4),
                "delta_cum_r": round(float(stress_row["cum_r"] - base_row["cum_r"]), 4),
                "base_profit_factor": round(float(base_row["profit_factor"]), 4),
                "stress_profit_factor": round(float(stress_row["profit_factor"]), 4),
                "delta_profit_factor": round(float(stress_row["profit_factor"] - base_row["profit_factor"]), 4),
                "base_max_dd_r": round(float(base_row["max_dd_r"]), 4),
                "stress_max_dd_r": round(float(stress_row["max_dd_r"]), 4),
                "delta_max_dd_r": round(float(stress_row["max_dd_r"] - base_row["max_dd_r"]), 4),
            }
        )
    return rows


def build_decision(
    *,
    summary_rows: list[dict[str, Any]],
    complementarity_rows: list[dict[str, Any]],
    baseline_strategy_profile: str,
    baseline_overlay_profile: str,
    alt_profile: str,
) -> dict[str, Any]:
    summary_map = {
        (row["cost_scenario"], row["window"], row["profile_kind"]): row
        for row in summary_rows
    }
    complementarity_map = {
        (row["cost_scenario"], row["window"]): row
        for row in complementarity_rows
    }

    base_full = summary_map[("base", "full_2020", "alt")]
    stress_full = summary_map[("stress_x2", "full_2020", "alt")]
    base_secondary = summary_map[("base", "two_year", "alt")]
    comp_full = complementarity_map[("base", "full_2020")]
    comp_secondary = complementarity_map[("base", "two_year")]

    pass_base_pf = float(base_full["profit_factor"]) >= BASE_FLOOR_PF_MIN
    pass_base_cum = float(base_full["cum_r"]) >= BASE_FLOOR_CUM_R_MIN
    pass_base_dd = float(base_full["max_dd_r"]) <= BASE_FLOOR_MAX_DD_R_MAX
    pass_stress_pf = float(stress_full["profit_factor"]) >= STRESS_FLOOR_PF_MIN
    pass_stress_cum = float(stress_full["cum_r"]) >= STRESS_FLOOR_CUM_R_MIN
    floor_pass = pass_base_pf and pass_base_cum and pass_base_dd and pass_stress_pf and pass_stress_cum

    pass_monthly_corr = (
        comp_full["monthly_corr"] is not None and float(comp_full["monthly_corr"]) <= COMPLEMENTARY_MAX_CORR
    )
    pass_offset_months = int(comp_full["baseline_negative_alt_positive_months"]) >= COMPLEMENTARY_MIN_OFFSET_MONTHS
    pass_offset_r = float(comp_full["offset_r_sum"]) >= COMPLEMENTARY_MIN_OFFSET_R
    complementary_pass = pass_monthly_corr and pass_offset_months and pass_offset_r

    if not floor_pass:
        status = "rejected_floor"
    elif not complementary_pass:
        status = "rejected_offset"
    else:
        status = "complementary_watchlist"

    return {
        "baseline_strategy_profile": baseline_strategy_profile,
        "baseline_overlay_profile": baseline_overlay_profile,
        "alt_profile": alt_profile,
        "primary_window": "full_2020",
        "secondary_window": "two_year",
        "base_full_2020_alt_cum_r": round(float(base_full["cum_r"]), 4),
        "base_full_2020_alt_profit_factor": round(float(base_full["profit_factor"]), 4),
        "base_full_2020_alt_max_dd_r": round(float(base_full["max_dd_r"]), 4),
        "stress_full_2020_alt_cum_r": round(float(stress_full["cum_r"]), 4),
        "stress_full_2020_alt_profit_factor": round(float(stress_full["profit_factor"]), 4),
        "base_two_year_alt_cum_r": round(float(base_secondary["cum_r"]), 4),
        "base_full_2020_monthly_corr": comp_full["monthly_corr"],
        "base_full_2020_baseline_negative_alt_positive_months": int(comp_full["baseline_negative_alt_positive_months"]),
        "base_full_2020_offset_r_sum": round(float(comp_full["offset_r_sum"]), 4),
        "base_full_2020_opposite_sign_months": int(comp_full["opposite_sign_months"]),
        "base_two_year_monthly_corr": comp_secondary["monthly_corr"],
        "base_two_year_baseline_negative_alt_positive_months": int(comp_secondary["baseline_negative_alt_positive_months"]),
        "base_two_year_offset_r_sum": round(float(comp_secondary["offset_r_sum"]), 4),
        "pass_base_full_2020_pf_floor": pass_base_pf,
        "pass_base_full_2020_cum_r_floor": pass_base_cum,
        "pass_base_full_2020_max_dd_floor": pass_base_dd,
        "pass_stress_full_2020_pf_floor": pass_stress_pf,
        "pass_stress_full_2020_cum_r_floor": pass_stress_cum,
        "pass_full_2020_monthly_corr_gate": pass_monthly_corr,
        "pass_full_2020_offset_months_gate": pass_offset_months,
        "pass_full_2020_offset_r_gate": pass_offset_r,
        "different_alpha_family_signal": pass_monthly_corr,
        "worth_continuing": status == "complementary_watchlist",
        "status": status,
        "next_route": "exhaustion_divergence" if status != "complementary_watchlist" else "hold_range_failure_watchlist",
    }


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
        continuation_text = "standalone 和互补性两层都过线，`range-failure` 暂时保留在 complementary watchlist。"
    elif decision_row["status"] == "rejected_offset":
        continuation_text = "standalone 没崩，但 offset 不够，`range-failure` 不值得继续深挖，下一条直接转 `exhaustion divergence`。"
    else:
        continuation_text = "standalone 已经差到不值得继续，`range-failure` 直接判出局，下一条直接转 `exhaustion divergence`。"

    family_text = (
        "有一定不同 alpha 家族证据，至少在月度相关性上没有和主线强同向。"
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
            "# Range Failure vs Challenger Managed",
            "",
            f"- baseline managed: `{baseline_strategy_profile}` + `{baseline_overlay_profile}`",
            f"- alt profile: `{alt_profile}`",
            "- 这轮不再拿旧 raw mainline 做对照，baseline 固定为 promotion gate 通过后的 `challenger_managed`。",
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
            alt_report = run_alt_profile(
                service=alt_service,
                symbol=args.symbol,
                alt_profile=args.alt_profile,
                exchange=args.exchange,
                market_type=args.market_type,
                start=start,
                end=resolved_end,
            )
            baseline_report = BacktestReport(
                generated_at=datetime.now(timezone.utc).isoformat(),
                exchange=args.exchange,
                market_type=args.market_type,
                start=start.isoformat(),
                end=resolved_end.isoformat(),
                symbols=[args.symbol],
                strategy_profiles=[args.baseline_strategy_profile],
                assumptions={},
                overall=[baseline_summary],
                by_symbol=[baseline_summary],
                trades=baseline_trades,
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
                    report=baseline_report,
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
                    profile_label=alt_label(alt_profile=args.alt_profile),
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
                profile_label=alt_label(alt_profile=args.alt_profile),
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

    print(f"saved range-failure vs challenger managed artifacts to {output_dir}")


if __name__ == "__main__":
    main()
