from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.logging import configure_logging
from scripts.post_tp1_managed_replay import (
    FLOAT_TOLERANCE,
    PROFILE_SPECS,
    baseline_profile_label,
    build_service,
    precompute_extension_features,
    run_profile,
    summarize_side_rows,
    trades_to_frame,
    validate_baseline_replay,
)

WINDOW_PRESETS = {
    "two_year": ("2024-03-19", "2026-03-19"),
    "full_2020": ("2020-03-19", "2026-03-19"),
}

DEFAULT_BASELINE_PROFILE = "swing_trend_long_regime_gate_v1"
DEFAULT_BASELINE_DIR = "artifacts/backtests/stop_ablation_mainline"
DEFAULT_OUTPUT_DIR = "artifacts/backtests/post_tp1_extension_mainline"
DEFAULT_DUAL_OUTPUT_DIR = "artifacts/backtests/post_tp1_extension_dual_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-TP1 1H extension management matrix.")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument(
        "--windows",
        default="two_year,full_2020",
        help="Comma-separated presets: two_year,full_2020",
    )
    parser.add_argument(
        "--profiles",
        default=",".join(spec.name for spec in PROFILE_SPECS),
        help="Comma-separated extension profile names.",
    )
    parser.add_argument(
        "--strategy-profiles",
        default=DEFAULT_BASELINE_PROFILE,
        help="Comma-separated baseline strategy profiles to replay against the extension matrix.",
    )
    parser.add_argument(
        "--baseline-dir",
        default=DEFAULT_BASELINE_DIR,
        help=(
            "Legacy baseline artifact dir containing *_structure_trades.csv. "
            "If multiple strategy profiles are passed, this only auto-applies to the mainline champion unless overridden."
        ),
    )
    parser.add_argument(
        "--baseline-dirs",
        default=None,
        help="Optional comma-separated strategy_profile=dir mappings for external baseline artifacts.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def sanitize_slug(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return safe or "default"


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_baseline_dir_mapping(raw: str | None) -> dict[str, Path]:
    if not raw:
        return {}
    mapping: dict[str, Path] = {}
    for item in parse_csv_list(raw):
        if "=" not in item:
            raise ValueError(f"Invalid baseline dir mapping: {item!r}. Expected strategy_profile=dir.")
        strategy_profile, directory = item.split("=", 1)
        strategy_profile = strategy_profile.strip()
        directory = directory.strip()
        if not strategy_profile or not directory:
            raise ValueError(f"Invalid baseline dir mapping: {item!r}.")
        mapping[strategy_profile] = resolve_path(directory)
    return mapping


def resolve_baseline_dir_map(
    *,
    legacy_baseline_dir: str | None,
    explicit_mapping_text: str | None,
    strategy_profiles: list[str],
) -> dict[str, Path]:
    mapping = parse_baseline_dir_mapping(explicit_mapping_text)
    if not legacy_baseline_dir:
        return mapping

    legacy_path = resolve_path(legacy_baseline_dir)
    if len(strategy_profiles) == 1:
        mapping.setdefault(strategy_profiles[0], legacy_path)
        return mapping

    if DEFAULT_BASELINE_PROFILE in strategy_profiles:
        mapping.setdefault(DEFAULT_BASELINE_PROFILE, legacy_path)
    return mapping

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
    body: list[str] = []
    for row in rows:
        values: list[str] = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def load_baseline_trades(path: Path) -> pd.DataFrame:
    trades = pd.read_csv(path)
    if trades.empty:
        raise ValueError(f"Baseline trades CSV is empty: {path}")
    for column in ("signal_time", "entry_time", "exit_time"):
        trades[column] = pd.to_datetime(trades[column], utc=True)
    return trades.sort_values("entry_time").reset_index(drop=True)


def resolve_baseline_reference(
    *,
    service: BacktestService,
    args: argparse.Namespace,
    window_name: str,
    start: datetime,
    end: datetime,
    strategy_profile: str,
    enriched: dict[str, pd.DataFrame],
    baseline_dir_map: dict[str, Path],
) -> tuple[pd.DataFrame, str, str | None]:
    baseline_dir = baseline_dir_map.get(strategy_profile)
    if baseline_dir is not None:
        baseline_path = baseline_dir / f"{window_name}_structure_trades.csv"
        if not baseline_path.exists():
            raise FileNotFoundError(f"Missing baseline structure trades for {strategy_profile}: {baseline_path}")
        return load_baseline_trades(baseline_path), "artifact_csv", str(baseline_path)

    _, trades = service.run_symbol_strategy_with_enriched_frames(
        symbol=args.symbol,
        strategy_profile=strategy_profile,
        start=start,
        end=end,
        enriched_frames=enriched,
    )
    return trades_to_frame(trades), "direct_backtest", None


def trade_output_path(
    *,
    output_dir: Path,
    window_name: str,
    baseline_strategy_profile: str,
    spec_name: str,
    legacy_mode: bool,
) -> Path:
    if legacy_mode:
        return output_dir / f"{window_name}_{spec_name}_trades.csv"
    return output_dir / f"{window_name}_{sanitize_slug(baseline_strategy_profile)}_{spec_name}_trades.csv"


def summary_output_path(
    *,
    output_dir: Path,
    window_name: str,
    baseline_strategy_profile: str,
    kind: str,
    legacy_mode: bool,
) -> Path:
    if legacy_mode:
        return output_dir / f"{window_name}_{kind}.csv"
    return output_dir / f"{window_name}_{sanitize_slug(baseline_strategy_profile)}_{kind}.csv"


def build_acceptance_rows(
    *,
    summary_rows: list[dict[str, Any]],
    side_rows: list[dict[str, Any]],
    baseline_strategy_profiles: list[str],
    overlay_profiles: list[str],
    acceptance_window: str,
    secondary_window: str | None,
) -> list[dict[str, Any]]:
    summary_map = {
        (row["baseline_strategy_profile"], row["window"], row["profile"]): row
        for row in summary_rows
    }
    side_map = {
        (row["baseline_strategy_profile"], row["window"], row["profile"], row["side"]): row
        for row in side_rows
    }
    rows: list[dict[str, Any]] = []
    for baseline_strategy_profile in baseline_strategy_profiles:
        baseline_acceptance = summary_map[(baseline_strategy_profile, acceptance_window, "baseline_be_after_tp1")]
        baseline_secondary_long = (
            side_map.get((baseline_strategy_profile, secondary_window, "baseline_be_after_tp1", "LONG"))
            if secondary_window is not None
            else None
        )
        baseline_secondary_long_cum = (
            float(baseline_secondary_long["cum_r"]) if baseline_secondary_long is not None else None
        )
        for overlay_profile in overlay_profiles:
            overlay_acceptance = summary_map[(baseline_strategy_profile, acceptance_window, overlay_profile)]
            overlay_secondary_long = (
                side_map.get((baseline_strategy_profile, secondary_window, overlay_profile, "LONG"))
                if secondary_window is not None
                else None
            )
            overlay_secondary_long_cum = (
                float(overlay_secondary_long["cum_r"]) if overlay_secondary_long is not None else None
            )
            pass_cum = float(overlay_acceptance["cum_r"]) > float(baseline_acceptance["cum_r"])
            pass_pf = float(overlay_acceptance["profit_factor"]) > float(baseline_acceptance["profit_factor"])
            pass_dd = float(overlay_acceptance["max_dd_r"]) <= float(baseline_acceptance["max_dd_r"]) + 2.0
            if baseline_secondary_long_cum is None or overlay_secondary_long_cum is None:
                pass_long_guard = None
                qualified = pass_cum and pass_pf and pass_dd
            else:
                pass_long_guard = overlay_secondary_long_cum >= baseline_secondary_long_cum - 2.0
                qualified = pass_cum and pass_pf and pass_dd and bool(pass_long_guard)
            rows.append(
                {
                    "baseline_strategy_profile": baseline_strategy_profile,
                    "baseline_profile_label": baseline_profile_label(baseline_strategy_profile),
                    "overlay_profile": overlay_profile,
                    "overlay_label": next(spec.label for spec in PROFILE_SPECS if spec.name == overlay_profile),
                    "acceptance_window": acceptance_window,
                    "secondary_window": secondary_window,
                    "baseline_cum_r": round(float(baseline_acceptance["cum_r"]), 4),
                    "overlay_cum_r": round(float(overlay_acceptance["cum_r"]), 4),
                    "delta_cum_r": round(float(overlay_acceptance["cum_r"] - baseline_acceptance["cum_r"]), 4),
                    "baseline_profit_factor": round(float(baseline_acceptance["profit_factor"]), 4),
                    "overlay_profit_factor": round(float(overlay_acceptance["profit_factor"]), 4),
                    "delta_profit_factor": round(float(overlay_acceptance["profit_factor"] - baseline_acceptance["profit_factor"]), 4),
                    "baseline_max_dd_r": round(float(baseline_acceptance["max_dd_r"]), 4),
                    "overlay_max_dd_r": round(float(overlay_acceptance["max_dd_r"]), 4),
                    "delta_max_dd_r": round(float(overlay_acceptance["max_dd_r"] - baseline_acceptance["max_dd_r"]), 4),
                    "baseline_secondary_long_cum_r": round(baseline_secondary_long_cum, 4)
                    if baseline_secondary_long_cum is not None
                    else None,
                    "overlay_secondary_long_cum_r": round(overlay_secondary_long_cum, 4)
                    if overlay_secondary_long_cum is not None
                    else None,
                    "delta_secondary_long_cum_r": round(overlay_secondary_long_cum - baseline_secondary_long_cum, 4)
                    if baseline_secondary_long_cum is not None and overlay_secondary_long_cum is not None
                    else None,
                    "pass_cum_r": pass_cum,
                    "pass_profit_factor": pass_pf,
                    "pass_max_dd": pass_dd,
                    "pass_secondary_long_guard": pass_long_guard,
                    "qualified": qualified,
                }
            )
    return sorted(
        rows,
        key=lambda item: (
            int(bool(item["qualified"])),
            float(item["delta_cum_r"]),
            float(item["delta_profit_factor"]),
            -float(item["delta_max_dd_r"]),
        ),
        reverse=True,
    )


def classify_overlay_results(
    *,
    acceptance_rows: list[dict[str, Any]],
    baseline_strategy_profiles: list[str],
    champion_profile: str,
    challenger_profile: str | None,
) -> dict[str, Any]:
    if not acceptance_rows:
        return {
            "classification": "rejected",
            "selected_overlay_profile": None,
            "selected_overlay_label": None,
            "qualified_baselines": None,
            "reason": "No acceptance rows were generated.",
        }

    def score(rows: list[dict[str, Any]]) -> tuple[float, float, float]:
        return (
            float(sum(row["delta_cum_r"] for row in rows)),
            float(sum(row["delta_profit_factor"] for row in rows)),
            -float(max(row["delta_max_dd_r"] for row in rows)),
        )

    qualified_rows = [row for row in acceptance_rows if row["qualified"]]
    rows_by_overlay: dict[str, list[dict[str, Any]]] = {}
    for row in qualified_rows:
        rows_by_overlay.setdefault(str(row["overlay_profile"]), []).append(row)

    universal_candidates: list[tuple[str, list[dict[str, Any]]]] = []
    challenger_only_candidates: list[tuple[str, list[dict[str, Any]]]] = []
    for overlay_profile, rows in rows_by_overlay.items():
        qualified_baselines = {str(row["baseline_strategy_profile"]) for row in rows}
        if qualified_baselines == set(baseline_strategy_profiles):
            universal_candidates.append((overlay_profile, rows))
        if challenger_profile is not None and qualified_baselines == {challenger_profile}:
            challenger_only_candidates.append((overlay_profile, rows))

    if universal_candidates:
        selected_overlay, selected_rows = max(universal_candidates, key=lambda item: score(item[1]))
        return {
            "classification": "universal_overlay",
            "selected_overlay_profile": selected_overlay,
            "selected_overlay_label": selected_rows[0]["overlay_label"],
            "qualified_baselines": ",".join(sorted({row["baseline_strategy_profile"] for row in selected_rows})),
            "reason": "At least one overlay improved every active baseline under the fixed acceptance rules.",
        }

    if challenger_only_candidates:
        selected_overlay, selected_rows = max(challenger_only_candidates, key=lambda item: score(item[1]))
        return {
            "classification": "challenger_only_overlay",
            "selected_overlay_profile": selected_overlay,
            "selected_overlay_label": selected_rows[0]["overlay_label"],
            "qualified_baselines": ",".join(sorted({row["baseline_strategy_profile"] for row in selected_rows})),
            "reason": "No universal overlay was found, but at least one overlay improved the challenger without clearing the champion.",
        }

    champion_only_rows = [
        row
        for row in qualified_rows
        if row["baseline_strategy_profile"] == champion_profile
    ]
    champion_only_note = (
        " Champion-only improvements were treated as rejected because this phase requires dual-baseline robustness."
        if champion_only_rows and challenger_profile is not None
        else ""
    )
    return {
        "classification": "rejected",
        "selected_overlay_profile": None,
        "selected_overlay_label": None,
        "qualified_baselines": None,
        "reason": f"No overlay cleared the active-baseline standard.{champion_only_note}",
    }


def main() -> None:
    args = parse_args()
    configure_logging()

    selected_windows = parse_csv_list(args.windows)
    selected_profiles = parse_csv_list(args.profiles)
    selected_strategy_profiles = parse_csv_list(args.strategy_profiles)
    profile_map = {spec.name: spec for spec in PROFILE_SPECS}

    unknown_windows = sorted(set(selected_windows) - set(WINDOW_PRESETS))
    if unknown_windows:
        raise ValueError(f"Unsupported windows: {', '.join(unknown_windows)}")
    unknown_profiles = sorted(set(selected_profiles) - set(profile_map))
    if unknown_profiles:
        raise ValueError(f"Unsupported profiles: {', '.join(unknown_profiles)}")
    if not selected_strategy_profiles:
        raise ValueError("At least one strategy profile is required.")

    baseline_dir_map = resolve_baseline_dir_map(
        legacy_baseline_dir=args.baseline_dir,
        explicit_mapping_text=args.baseline_dirs,
        strategy_profiles=selected_strategy_profiles,
    )

    output_dir_value = args.output_dir
    if len(selected_strategy_profiles) > 1 and output_dir_value == DEFAULT_OUTPUT_DIR:
        output_dir_value = DEFAULT_DUAL_OUTPUT_DIR
    output_dir = resolve_path(output_dir_value)
    output_dir.mkdir(parents=True, exist_ok=True)

    legacy_mode = len(selected_strategy_profiles) == 1
    service = build_service()

    all_summary_rows: list[dict[str, Any]] = []
    all_side_rows: list[dict[str, Any]] = []
    all_trade_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    acceptance_rows: list[dict[str, Any]] = []
    baseline_reference_rows: list[dict[str, Any]] = []

    acceptance_window = "full_2020" if "full_2020" in selected_windows else selected_windows[-1]
    secondary_window = "two_year" if "two_year" in selected_windows else None

    report_sections = [
        "# Post-TP1 Extension Matrix",
        "",
        "- 这轮只研究主线管理层 alpha，不再扩 entry/filter。",
        "- 这次直接跑 sequence-aware 完整 backtest。",
        "- 基线是各自 baseline strategy 的 `BE after TP1`，overlay 只改 `LONG` 命中 TP1 之后的持有逻辑。",
        "- 新规则只看 `1H` 本身，不再使用额外 `4H` 管理信息。",
        "- 操作化定义：`LONG` 命中 `TP1` 后，观察后续 `N` 根已收 `1H` K。若这些 bar 里从未刷新 `TP1` 命中那根 bar 的最高点，则在观察窗口结束后把 stop 收到 `BE`；一旦出现新高扩展，则继续 hold 结构止损。",
        "",
        "## Active Baselines",
        "",
        markdown_table(
            [
                {
                    "strategy_profile": strategy_profile,
                    "profile_label": baseline_profile_label(strategy_profile),
                    "baseline_source": "artifact_csv" if strategy_profile in baseline_dir_map else "direct_backtest",
                }
                for strategy_profile in selected_strategy_profiles
            ],
            [
                ("strategy_profile", "Baseline Strategy"),
                ("profile_label", "Label"),
                ("baseline_source", "Baseline Source"),
            ],
        ),
        "",
    ]

    for baseline_strategy_profile in selected_strategy_profiles:
        baseline_label = baseline_profile_label(baseline_strategy_profile)
        report_sections.extend(
            [
                f"## {baseline_label}",
                "",
                f"- baseline strategy: `{baseline_strategy_profile}`",
                "",
            ]
        )

        strategy = service.strategy_service.build_strategy(baseline_strategy_profile)
        trigger_tf = str(strategy.window_config["trigger_timeframe"])
        if trigger_tf != "1h":
            raise ValueError(
                f"Post-TP1 extension matrix assumes a 1H trigger timeframe, got {trigger_tf} for {baseline_strategy_profile}."
            )

        for window_name in selected_windows:
            start_raw, end_raw = WINDOW_PRESETS[window_name]
            start = parse_date(start_raw)
            end = parse_date(end_raw)

            base_frames = service.prepare_history(
                exchange=args.exchange,
                market_type=args.market_type,
                symbol=args.symbol,
                strategy_profile=baseline_strategy_profile,
                start=start,
                end=end,
            )
            enriched = {
                timeframe: service._enrich_frame(strategy, timeframe, frame)
                for timeframe, frame in base_frames.items()
            }
            extension_features = precompute_extension_features(trigger_frame=enriched["1h"])
            baseline_trades, baseline_source, baseline_reference_path = resolve_baseline_reference(
                service=service,
                args=args,
                window_name=window_name,
                start=start,
                end=end,
                strategy_profile=baseline_strategy_profile,
                enriched=enriched,
                baseline_dir_map=baseline_dir_map,
            )

            window_rows: list[dict[str, Any]] = []
            window_trade_rows: list[dict[str, Any]] = []
            baseline_simulated: pd.DataFrame | None = None

            for profile_name in selected_profiles:
                spec = profile_map[profile_name]
                overlay_strategy_profile = f"{baseline_strategy_profile}__extension_condition_{spec.name}"
                summary, trades, annotated_rows = run_profile(
                    service=service,
                    strategy=strategy,
                    symbol=args.symbol,
                    strategy_profile=overlay_strategy_profile,
                    spec=spec,
                    start=start,
                    end=end,
                    enriched=enriched,
                    extension_features=extension_features,
                )
                trade_rows = [
                    {
                        "window": window_name,
                        "baseline_strategy_profile": baseline_strategy_profile,
                        "baseline_profile_label": baseline_label,
                        "baseline_source": baseline_source,
                        "profile": spec.name,
                        "label": spec.label,
                        **row,
                    }
                    for row in annotated_rows
                ]
                write_csv(
                    trade_output_path(
                        output_dir=output_dir,
                        window_name=window_name,
                        baseline_strategy_profile=baseline_strategy_profile,
                        spec_name=spec.name,
                        legacy_mode=legacy_mode,
                    ),
                    trade_rows,
                )
                window_trade_rows.extend(trade_rows)
                all_trade_rows.extend(trade_rows)

                long_df = pd.DataFrame([row for row in trade_rows if row["side"] == "LONG"])
                extension_decision_count = (
                    int(long_df["extension_decision_made"].eq(True).sum()) if not long_df.empty else 0
                )
                extension_confirmed_count = (
                    int(long_df["extension_confirmed"].eq(True).sum()) if not long_df.empty else 0
                )
                row = {
                    "window": window_name,
                    "baseline_strategy_profile": baseline_strategy_profile,
                    "baseline_profile_label": baseline_label,
                    "baseline_source": baseline_source,
                    "profile": spec.name,
                    "label": spec.label,
                    "trades": int(summary.total_trades),
                    "win_rate_pct": round(float(summary.win_rate), 2),
                    "profit_factor": round(float(summary.profit_factor), 4),
                    "expectancy_r": round(float(summary.expectancy_r), 4),
                    "cum_r": round(float(summary.cumulative_r), 4),
                    "max_dd_r": round(float(summary.max_drawdown_r), 4),
                    "avg_holding_bars": round(float(summary.avg_holding_bars), 2),
                    "tp1_hit_rate_pct": round(float(summary.tp1_hit_rate), 2),
                    "tp2_hit_rate_pct": round(float(summary.tp2_hit_rate), 2),
                    "signals_now": int(summary.signals_now),
                    "skipped_entries": int(summary.skipped_entries),
                    "long_trades": int(len(long_df)),
                    "extension_decision_count": extension_decision_count,
                    "extension_confirmed_count": extension_confirmed_count,
                    "extension_confirmed_pct": round(extension_confirmed_count / len(long_df) * 100, 2)
                    if len(long_df)
                    else 0.0,
                }
                window_rows.append(row)
                all_summary_rows.append(row)

                if spec.name == "baseline_be_after_tp1":
                    baseline_simulated = pd.DataFrame(trade_rows)

            if baseline_simulated is None:
                raise ValueError("baseline_be_after_tp1 must be included for validation.")

            mismatches = validate_baseline_replay(baseline_trades, baseline_simulated)
            validation_rows.append(
                {
                    "window": window_name,
                    "baseline_strategy_profile": baseline_strategy_profile,
                    "baseline_profile_label": baseline_label,
                    "baseline_source": baseline_source,
                    "baseline_reference_path": baseline_reference_path,
                    "baseline_trades": int(len(baseline_trades)),
                    "baseline_replay_mismatches": int(len(mismatches)),
                }
            )
            baseline_reference_rows.append(
                {
                    "window": window_name,
                    "baseline_strategy_profile": baseline_strategy_profile,
                    "baseline_profile_label": baseline_label,
                    "baseline_source": baseline_source,
                    "baseline_reference_path": baseline_reference_path,
                    "baseline_trades": int(len(baseline_trades)),
                }
            )
            if mismatches:
                mismatch_name = (
                    f"{window_name}_baseline_validation_mismatches.csv"
                    if legacy_mode
                    else f"{window_name}_{sanitize_slug(baseline_strategy_profile)}_baseline_validation_mismatches.csv"
                )
                mismatch_path = output_dir / mismatch_name
                write_csv(mismatch_path, mismatches)
                raise ValueError(
                    f"Baseline replay validation failed for {baseline_strategy_profile}/{window_name}: see {mismatch_path}"
                )

            side_rows = summarize_side_rows(pd.DataFrame(window_trade_rows))
            all_side_rows.extend(side_rows)
            write_csv(
                summary_output_path(
                    output_dir=output_dir,
                    window_name=window_name,
                    baseline_strategy_profile=baseline_strategy_profile,
                    kind="summary",
                    legacy_mode=legacy_mode,
                ),
                window_rows,
            )
            write_csv(
                summary_output_path(
                    output_dir=output_dir,
                    window_name=window_name,
                    baseline_strategy_profile=baseline_strategy_profile,
                    kind="side_summary",
                    legacy_mode=legacy_mode,
                ),
                side_rows,
            )

            window_rows_sorted = sorted(
                window_rows,
                key=lambda item: (float(item["cum_r"]), float(item["profit_factor"]), -float(item["max_dd_r"])),
                reverse=True,
            )
            side_rows_sorted = sorted(side_rows, key=lambda item: (item["label"], item["side"]))

            report_sections.extend(
                [
                    f"### {window_name}",
                    "",
                    f"- baseline source: `{baseline_source}`",
                    *([f"- baseline reference: `{baseline_reference_path}`"] if baseline_reference_path else []),
                    "",
                    markdown_table(
                        window_rows_sorted,
                        [
                            ("label", "Profile"),
                            ("trades", "Trades"),
                            ("profit_factor", "PF"),
                            ("expectancy_r", "Exp R"),
                            ("cum_r", "Cum R"),
                            ("max_dd_r", "Max DD R"),
                            ("extension_decision_count", "Extension Decisions"),
                            ("extension_confirmed_count", "Extensions"),
                            ("extension_confirmed_pct", "Extension %"),
                        ],
                    ),
                    "",
                    "按方向拆开：",
                    "",
                    markdown_table(
                        side_rows_sorted,
                        [
                            ("label", "Profile"),
                            ("side", "Side"),
                            ("trades", "Trades"),
                            ("cum_r", "Cum R"),
                            ("avg_r", "Avg R"),
                            ("profit_factor", "PF"),
                            ("tp2_hit_rate_pct", "TP2 Hit %"),
                        ],
                    ),
                    "",
                ]
            )

    overlay_profiles = [spec.name for spec in PROFILE_SPECS if spec.name in selected_profiles and spec.name != "baseline_be_after_tp1"]
    acceptance_rows = build_acceptance_rows(
        summary_rows=all_summary_rows,
        side_rows=all_side_rows,
        baseline_strategy_profiles=selected_strategy_profiles,
        overlay_profiles=overlay_profiles,
        acceptance_window=acceptance_window,
        secondary_window=secondary_window,
    )
    classification_row = classify_overlay_results(
        acceptance_rows=acceptance_rows,
        baseline_strategy_profiles=selected_strategy_profiles,
        champion_profile=selected_strategy_profiles[0],
        challenger_profile=selected_strategy_profiles[1] if len(selected_strategy_profiles) > 1 else None,
    )

    report_sections.extend(
        [
            "## Acceptance",
            "",
            markdown_table(
                acceptance_rows,
                [
                    ("baseline_profile_label", "Baseline"),
                    ("overlay_label", "Overlay"),
                    ("delta_cum_r", "Delta Cum R"),
                    ("delta_profit_factor", "Delta PF"),
                    ("delta_max_dd_r", "Delta MaxDD"),
                    ("delta_secondary_long_cum_r", "Delta Secondary LONG"),
                    ("qualified", "Qualified"),
                ],
            ),
            "",
            "## Overlay Classification",
            "",
            markdown_table(
                [classification_row],
                [
                    ("classification", "Classification"),
                    ("selected_overlay_label", "Selected Overlay"),
                    ("qualified_baselines", "Qualified Baselines"),
                    ("reason", "Reason"),
                ],
            ),
            "",
        ]
    )

    write_csv(output_dir / "summary_all.csv", all_summary_rows)
    write_csv(output_dir / "side_summary_all.csv", all_side_rows)
    write_csv(output_dir / "validation.csv", validation_rows)
    write_csv(output_dir / "baseline_references.csv", baseline_reference_rows)
    write_csv(output_dir / "acceptance.csv", acceptance_rows)
    write_csv(output_dir / "overlay_classification.csv", [classification_row])
    write_csv(output_dir / "trades_all.csv", all_trade_rows)
    (output_dir / "report.md").write_text("\n".join(report_sections).strip() + "\n", encoding="utf-8")

    print(f"Saved report: {output_dir / 'report.md'}")
    print(f"Saved summary CSV: {output_dir / 'summary_all.csv'}")
    print(f"Saved side summary CSV: {output_dir / 'side_summary_all.csv'}")
    print(f"Saved validation CSV: {output_dir / 'validation.csv'}")
    print(f"Saved acceptance CSV: {output_dir / 'acceptance.csv'}")


if __name__ == "__main__":
    main()
