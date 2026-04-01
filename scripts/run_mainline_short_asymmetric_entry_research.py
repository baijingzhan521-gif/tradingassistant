from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtesting.service import BacktestAssumptions, BacktestService, BacktestTrade
from app.core.logging import configure_logging
from app.data.exchange_client import get_exchange_client_factory
from app.data.ohlcv_service import OhlcvService
from app.services.strategy_service import StrategyService


EXIT_ASSUMPTIONS = {
    "exit_profile": "short_asymmetry_long_scaled1_3_short_fixed1_5",
    "take_profit_mode": "scaled",
    "scaled_tp1_r": 1.0,
    "scaled_tp2_r": 3.0,
    "long_exit": {"take_profit_mode": "scaled", "scaled_tp1_r": 1.0, "scaled_tp2_r": 3.0},
    "short_exit": {"take_profit_mode": "fixed_r", "fixed_take_profit_r": 1.5},
    "swing_detection_mode": "confirmed",
}
MAINLINE = "swing_trend_long_regime_gate_v1"
CANDIDATE_PROFILES = (
    MAINLINE,
    "swing_trend_long_regime_short_relaxed_trigger_v1",
    "swing_trend_long_regime_short90_free_space_v1",
    "swing_trend_long_regime_short_no_auxiliary_v1",
    "swing_trend_long_regime_short_no_reversal_v1",
    "swing_trend_long_regime_short_no_reversal_no_aux_v1",
)
SHORT_ONLY_CANDIDATES = tuple(profile for profile in CANDIDATE_PROFILES if profile != MAINLINE)
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "backtests" / "mainline_short_asymmetric_entry_research"
PRIMARY_WINDOW = "primary_2020_latest"
SECONDARY_WINDOW = "secondary_2024_latest"
DIFF_YEARS = (2022, 2024, 2026)
MATCH_TOLERANCE = pd.Timedelta(hours=12)

PROFILE_LABELS = {
    MAINLINE: "Mainline",
    "swing_trend_long_regime_short_relaxed_trigger_v1": "Short Relaxed Regained Fast",
    "swing_trend_long_regime_short90_free_space_v1": "Short Relaxed RF @ TS90 + Free Space",
    "swing_trend_long_regime_short_no_auxiliary_v1": "Short No Auxiliary",
    "swing_trend_long_regime_short_no_reversal_v1": "Short No Reversal",
    "swing_trend_long_regime_short_no_reversal_no_aux_v1": "Short No Reversal + No Auxiliary",
}


@dataclass(frozen=True)
class FoldWindow:
    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass(frozen=True)
class WindowSpec:
    name: str
    start: datetime
    end: datetime


@dataclass
class CandidateFoldResult:
    train_summary: dict[str, Any]
    test_summary: dict[str, Any]
    test_trades: list[BacktestTrade]


@dataclass(frozen=True)
class MatchPair:
    main_idx: int
    alt_idx: int
    kind: str
    gap_hours: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research short-only asymmetric entry candidates for the BTC mainline.")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--primary-start", default="2020-01-01")
    parser.add_argument("--secondary-start", default="2024-03-19")
    parser.add_argument("--end", default=None, help="Defaults to latest available data at runtime.")
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--step-days", type=int, default=90)
    parser.add_argument("--scheme", choices=["rolling", "anchored"], default="rolling")
    parser.add_argument("--min-train-trades", type=int, default=12)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def build_service() -> BacktestService:
    return BacktestService(
        ohlcv_service=OhlcvService(get_exchange_client_factory()),
        strategy_service=StrategyService(),
        assumptions=BacktestAssumptions(**EXIT_ASSUMPTIONS),
    )


def generate_folds(
    *,
    start: datetime,
    end: datetime,
    train_days: int,
    test_days: int,
    step_days: int,
    scheme: str,
) -> list[FoldWindow]:
    folds: list[FoldWindow] = []
    anchor_start = start
    train_start = start
    train_end = train_start + timedelta(days=train_days)
    index = 1

    while train_end + timedelta(days=test_days) <= end:
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)
        folds.append(
            FoldWindow(
                index=index,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        index += 1
        if scheme == "anchored":
            train_end = train_end + timedelta(days=step_days)
            train_start = anchor_start
        else:
            train_start = train_start + timedelta(days=step_days)
            train_end = train_start + timedelta(days=train_days)
    return folds


def to_summary_dict(summary) -> dict[str, Any]:
    return {
        "total_trades": int(summary.total_trades),
        "win_rate": float(summary.win_rate),
        "profit_factor": float(summary.profit_factor),
        "expectancy_r": float(summary.expectancy_r),
        "cumulative_r": float(summary.cumulative_r),
        "max_drawdown_r": float(summary.max_drawdown_r),
        "avg_holding_bars": float(summary.avg_holding_bars),
        "tp1_hit_rate": float(summary.tp1_hit_rate),
        "tp2_hit_rate": float(summary.tp2_hit_rate),
        "signals_now": int(summary.signals_now),
        "skipped_entries": int(summary.skipped_entries),
    }


def resolve_end_from_history(enriched_history: dict[str, dict[str, pd.DataFrame]]) -> datetime:
    profile_ends: list[pd.Timestamp] = []
    for frames in enriched_history.values():
        profile_ends.append(min(pd.Timestamp(frame["timestamp"].max()) for frame in frames.values()))
    return min(profile_ends).to_pydatetime()


def summary_row(*, window: str, profile: str, source: str, summary_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        "window": window,
        "source": source,
        "strategy_profile": profile,
        "profile_label": PROFILE_LABELS.get(profile, profile),
        **summary_dict,
    }


def trade_rows_from_trades(
    *,
    window: str,
    source: str,
    profile: str,
    trades: list[BacktestTrade],
    fold: int | None = None,
    selected_candidate: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trade in trades:
        row = asdict(trade)
        row["window"] = window
        row["source"] = source
        row["profile_label"] = PROFILE_LABELS.get(profile, profile)
        row["fold"] = fold
        row["selected_candidate"] = selected_candidate
        rows.append(row)
    return rows


def build_trade_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "window",
                "source",
                "strategy_profile",
                "profile_label",
                "side",
                "signal_time",
                "entry_time",
                "exit_time",
                "bars_held",
                "exit_reason",
                "tp1_hit",
                "tp2_hit",
                "pnl_r",
                "fold",
                "selected_candidate",
            ]
        )
    frame = pd.DataFrame(rows)
    for column in ("signal_time", "entry_time", "exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    frame["year"] = frame["entry_time"].dt.year.astype(int)
    frame["month"] = frame["entry_time"].dt.strftime("%Y-%m")
    return frame


def compute_group_max_drawdown(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    ordered = frame.sort_values("entry_time")
    cumulative = ordered["pnl_r"].cumsum()
    drawdown = cumulative - cumulative.cummax()
    return round(abs(float(drawdown.min())), 4)


def summarize_trade_distribution(frame: pd.DataFrame, *, group_cols: list[str]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(group_cols, sort=True, dropna=False, observed=True)
    for keys, group in grouped:
        keys = (keys,) if not isinstance(keys, tuple) else keys
        row = {column: value for column, value in zip(group_cols, keys)}
        wins = float(group.loc[group["pnl_r"] > 0, "pnl_r"].sum())
        losses = float(-group.loc[group["pnl_r"] < 0, "pnl_r"].sum())
        profit_factor = round(wins / losses, 4) if losses > 0 else None
        row.update(
            {
                "trades": int(len(group)),
                "win_rate_pct": round(float((group["pnl_r"] > 0).mean() * 100.0), 2),
                "expectancy_r": round(float(group["pnl_r"].mean()), 4),
                "cumulative_r": round(float(group["pnl_r"].sum()), 4),
                "profit_factor": profit_factor,
                "max_drawdown_r": compute_group_max_drawdown(group),
                "avg_bars_held": round(float(group["bars_held"].mean()), 2),
            }
        )
        rows.append(row)
    return rows


def rank_train_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda item: (
            float(item["train_cumulative_r"]),
            float(item["train_profit_factor"]),
            -float(item["train_max_drawdown_r"]),
            float(item["train_expectancy_r"]),
            float(item["train_total_trades"]),
        ),
        reverse=True,
    )


def rank_oos_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda item: (
            float(item["cumulative_r"]),
            float(item["profit_factor"]),
            -float(item["max_drawdown_r"]),
            float(item["expectancy_r"]),
        ),
        reverse=True,
    )


def render_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_无数据_"
    head = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(key)) for key, _ in columns) + " |")
    return "\n".join([head, sep, *body])


def run_window(
    *,
    service: BacktestService,
    args: argparse.Namespace,
    window: WindowSpec,
    enriched_history: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, Any]:
    print(
        f"[short-asymmetry] window {window.name} {window.start.date().isoformat()}->{window.end.date().isoformat()}",
        flush=True,
    )
    full_summary_rows: list[dict[str, Any]] = []
    full_trade_rows: list[dict[str, Any]] = []
    candidate_fold_rows: list[dict[str, Any]] = []
    independent_oos_rows: list[dict[str, Any]] = []
    independent_oos_trade_rows: list[dict[str, Any]] = []
    selection_fold_rows: list[dict[str, Any]] = []
    selection_trade_rows: list[dict[str, Any]] = []
    selection_count_rows: list[dict[str, Any]] = []
    fold_result_map: dict[tuple[str, int], CandidateFoldResult] = {}

    folds = generate_folds(
        start=window.start,
        end=window.end,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        scheme=args.scheme,
    )

    for profile in CANDIDATE_PROFILES:
        print(f"[short-asymmetry] full-window {window.name} {profile}", flush=True)
        summary, trades = service.run_symbol_strategy_with_enriched_frames(
            symbol=args.symbol,
            strategy_profile=profile,
            start=window.start,
            end=window.end,
            enriched_frames=enriched_history[profile],
        )
        full_summary_rows.append(
            summary_row(window=window.name, profile=profile, source="full_window", summary_dict=to_summary_dict(summary))
        )
        full_trade_rows.extend(
            trade_rows_from_trades(
                window=window.name,
                source="full_window",
                profile=profile,
                trades=trades,
            )
        )

    for profile in CANDIDATE_PROFILES:
        print(f"[short-asymmetry] independent-oos {window.name} {profile}", flush=True)
        all_test_trades: list[BacktestTrade] = []
        test_signals_now = 0
        test_skipped_entries = 0
        for fold in folds:
            print(
                f"[short-asymmetry] fold {window.name} #{fold.index} {profile} train {fold.train_start.date().isoformat()}->{fold.train_end.date().isoformat()} test {fold.test_start.date().isoformat()}->{fold.test_end.date().isoformat()}",
                flush=True,
            )
            train_summary, _ = service.run_symbol_strategy_with_enriched_frames(
                symbol=args.symbol,
                strategy_profile=profile,
                start=fold.train_start,
                end=fold.train_end,
                enriched_frames=enriched_history[profile],
            )
            test_summary, test_trades = service.run_symbol_strategy_with_enriched_frames(
                symbol=args.symbol,
                strategy_profile=profile,
                start=fold.test_start,
                end=fold.test_end,
                enriched_frames=enriched_history[profile],
            )
            fold_result_map[(profile, fold.index)] = CandidateFoldResult(
                train_summary=to_summary_dict(train_summary),
                test_summary=to_summary_dict(test_summary),
                test_trades=test_trades,
            )
            candidate_fold_rows.append(
                {
                    "window": window.name,
                    "strategy_profile": profile,
                    "profile_label": PROFILE_LABELS.get(profile, profile),
                    "fold": fold.index,
                    "train_start": fold.train_start.date().isoformat(),
                    "train_end": fold.train_end.date().isoformat(),
                    "test_start": fold.test_start.date().isoformat(),
                    "test_end": fold.test_end.date().isoformat(),
                    "train_total_trades": int(train_summary.total_trades),
                    "train_profit_factor": round(float(train_summary.profit_factor), 4),
                    "train_expectancy_r": round(float(train_summary.expectancy_r), 4),
                    "train_cumulative_r": round(float(train_summary.cumulative_r), 4),
                    "train_max_drawdown_r": round(float(train_summary.max_drawdown_r), 4),
                    "test_total_trades": int(test_summary.total_trades),
                    "test_profit_factor": round(float(test_summary.profit_factor), 4),
                    "test_expectancy_r": round(float(test_summary.expectancy_r), 4),
                    "test_cumulative_r": round(float(test_summary.cumulative_r), 4),
                    "test_max_drawdown_r": round(float(test_summary.max_drawdown_r), 4),
                }
            )
            all_test_trades.extend(test_trades)
            test_signals_now += int(test_summary.signals_now)
            test_skipped_entries += int(test_summary.skipped_entries)

        oos_summary = service._summarize_trades(
            trades=all_test_trades,
            strategy_profile=profile,
            symbol=args.symbol,
            signals_now=test_signals_now,
            skipped_entries=test_skipped_entries,
        )
        independent_oos_rows.append(
            summary_row(window=window.name, profile=profile, source="independent_oos", summary_dict=to_summary_dict(oos_summary))
        )
        independent_oos_trade_rows.extend(
            trade_rows_from_trades(
                window=window.name,
                source="independent_oos",
                profile=profile,
                trades=all_test_trades,
            )
        )

    selection_counts = {profile: 0 for profile in CANDIDATE_PROFILES}
    for fold in folds:
        print(f"[short-asymmetry] select {window.name} fold #{fold.index}", flush=True)
        train_rows: list[dict[str, Any]] = []
        for profile in CANDIDATE_PROFILES:
            result = fold_result_map[(profile, fold.index)]
            if result.train_summary["total_trades"] < args.min_train_trades:
                continue
            train_rows.append(
                {
                    "strategy_profile": profile,
                    "profile_label": PROFILE_LABELS.get(profile, profile),
                    **{f"train_{key}": value for key, value in result.train_summary.items()},
                }
            )

        ranked = rank_train_candidates(train_rows)
        chosen = ranked[0] if ranked else None
        if chosen is None:
            selection_fold_rows.append(
                {
                    "window": window.name,
                    "fold": fold.index,
                    "train_start": fold.train_start.date().isoformat(),
                    "train_end": fold.train_end.date().isoformat(),
                    "test_start": fold.test_start.date().isoformat(),
                    "test_end": fold.test_end.date().isoformat(),
                    "selected_candidate": None,
                    "selected_profile_label": None,
                    "train_cumulative_r": None,
                    "train_profit_factor": None,
                    "train_max_drawdown_r": None,
                    "test_total_trades": 0,
                    "test_profit_factor": None,
                    "test_expectancy_r": None,
                    "test_cumulative_r": None,
                    "test_max_drawdown_r": None,
                }
            )
            continue

        chosen_profile = str(chosen["strategy_profile"])
        chosen_result = fold_result_map[(chosen_profile, fold.index)]
        selection_counts[chosen_profile] += 1
        selection_fold_rows.append(
            {
                "window": window.name,
                "fold": fold.index,
                "train_start": fold.train_start.date().isoformat(),
                "train_end": fold.train_end.date().isoformat(),
                "test_start": fold.test_start.date().isoformat(),
                "test_end": fold.test_end.date().isoformat(),
                "selected_candidate": chosen_profile,
                "selected_profile_label": PROFILE_LABELS.get(chosen_profile, chosen_profile),
                "train_cumulative_r": chosen["train_cumulative_r"],
                "train_profit_factor": chosen["train_profit_factor"],
                "train_max_drawdown_r": chosen["train_max_drawdown_r"],
                "test_total_trades": chosen_result.test_summary["total_trades"],
                "test_profit_factor": chosen_result.test_summary["profit_factor"],
                "test_expectancy_r": chosen_result.test_summary["expectancy_r"],
                "test_cumulative_r": chosen_result.test_summary["cumulative_r"],
                "test_max_drawdown_r": chosen_result.test_summary["max_drawdown_r"],
            }
        )
        selection_trade_rows.extend(
            trade_rows_from_trades(
                window=window.name,
                source="selected_oos",
                profile=chosen_profile,
                trades=chosen_result.test_trades,
                fold=fold.index,
                selected_candidate=chosen_profile,
            )
        )

    for profile, count in selection_counts.items():
        selection_count_rows.append(
            {
                "window": window.name,
                "strategy_profile": profile,
                "profile_label": PROFILE_LABELS.get(profile, profile),
                "selected_folds": count,
            }
        )

    selection_trade_objects = [
        BacktestTrade(**{key: row[key] for key in BacktestTrade.__dataclass_fields__.keys()})
        for row in selection_trade_rows
    ]
    selected_summary = service._summarize_trades(
        trades=selection_trade_objects,
        strategy_profile=f"{window.name}_selected_oos",
        symbol=args.symbol,
        signals_now=0,
        skipped_entries=0,
    )
    selected_summary_row = summary_row(
        window=window.name,
        profile="selected_candidate_pool",
        source="selected_oos",
        summary_dict=to_summary_dict(selected_summary),
    )

    return {
        "window": window,
        "folds": folds,
        "full_summary_rows": full_summary_rows,
        "full_trade_rows": full_trade_rows,
        "candidate_fold_rows": candidate_fold_rows,
        "independent_oos_rows": independent_oos_rows,
        "independent_oos_trade_rows": independent_oos_trade_rows,
        "selection_fold_rows": selection_fold_rows,
        "selection_trade_rows": selection_trade_rows,
        "selection_count_rows": selection_count_rows,
        "selected_summary_row": selected_summary_row,
    }


def value_or_floor(value: float | None) -> float:
    return float("-inf") if value is None else float(value)


def build_shortlist(
    *,
    primary_oos_rows: list[dict[str, Any]],
    secondary_full_side_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_profile = {row["strategy_profile"]: row for row in primary_oos_rows}
    mainline = by_profile[MAINLINE]
    long_side_rows = {
        row["strategy_profile"]: row
        for row in secondary_full_side_rows
        if row["side"] == "LONG"
    }
    mainline_long_cum = float(long_side_rows.get(MAINLINE, {}).get("cumulative_r", 0.0))

    rows: list[dict[str, Any]] = []
    for profile in SHORT_ONLY_CANDIDATES:
        primary = by_profile[profile]
        secondary_long = long_side_rows.get(profile)
        secondary_long_cum = float(secondary_long["cumulative_r"]) if secondary_long is not None else 0.0
        pass_cum = float(primary["cumulative_r"]) > float(mainline["cumulative_r"])
        pass_pf = float(primary["profit_factor"]) > float(mainline["profit_factor"])
        pass_dd = float(primary["max_drawdown_r"]) <= float(mainline["max_drawdown_r"]) + 2.0
        pass_long_guard = secondary_long_cum >= mainline_long_cum - 2.0
        rows.append(
            {
                "strategy_profile": profile,
                "profile_label": PROFILE_LABELS.get(profile, profile),
                "primary_oos_cumulative_r": round(float(primary["cumulative_r"]), 4),
                "primary_oos_profit_factor": round(float(primary["profit_factor"]), 4),
                "primary_oos_max_drawdown_r": round(float(primary["max_drawdown_r"]), 4),
                "delta_vs_mainline_cum_r": round(float(primary["cumulative_r"] - mainline["cumulative_r"]), 4),
                "delta_vs_mainline_pf": round(float(primary["profit_factor"] - mainline["profit_factor"]), 4),
                "delta_vs_mainline_max_dd_r": round(float(primary["max_drawdown_r"] - mainline["max_drawdown_r"]), 4),
                "secondary_long_cumulative_r": round(secondary_long_cum, 4),
                "secondary_long_delta_r": round(secondary_long_cum - mainline_long_cum, 4),
                "pass_primary_cum_r": pass_cum,
                "pass_primary_pf": pass_pf,
                "pass_primary_max_dd": pass_dd,
                "pass_secondary_long_guard": pass_long_guard,
                "qualified": pass_cum and pass_pf and pass_dd and pass_long_guard,
            }
        )

    return sorted(
        rows,
        key=lambda item: (
            int(bool(item["qualified"])),
            float(item["primary_oos_cumulative_r"]),
            float(item["primary_oos_profit_factor"]),
            -float(item["primary_oos_max_drawdown_r"]),
        ),
        reverse=True,
    )


def build_ordered_matches(main_df: pd.DataFrame, alt_df: pd.DataFrame) -> list[MatchPair]:
    main_df = main_df.sort_values("signal_time").reset_index(drop=False)
    alt_df = alt_df.sort_values("signal_time").reset_index(drop=False)
    main_times = main_df["signal_time"].tolist()
    alt_times = alt_df["signal_time"].tolist()
    tolerance_seconds = int(MATCH_TOLERANCE.total_seconds())

    def better_score(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
        if left[0] != right[0]:
            return left if left[0] > right[0] else right
        return left if left[1] >= right[1] else right

    @lru_cache(maxsize=None)
    def solve(i: int, j: int) -> tuple[int, int]:
        if i >= len(main_times) or j >= len(alt_times):
            return (0, 0)

        best = better_score(solve(i + 1, j), solve(i, j + 1))
        gap_seconds = abs(int((main_times[i] - alt_times[j]).total_seconds()))
        if gap_seconds <= tolerance_seconds:
            matched = solve(i + 1, j + 1)
            candidate = (matched[0] + 1, matched[1] - gap_seconds)
            best = better_score(best, candidate)
        return best

    def reconstruct(i: int, j: int) -> list[MatchPair]:
        if i >= len(main_times) or j >= len(alt_times):
            return []

        current = solve(i, j)
        gap_seconds = abs(int((main_times[i] - alt_times[j]).total_seconds()))
        if gap_seconds <= tolerance_seconds:
            matched = solve(i + 1, j + 1)
            candidate = (matched[0] + 1, matched[1] - gap_seconds)
            if candidate == current:
                return [
                    MatchPair(
                        main_idx=int(main_df.loc[i, "index"]),
                        alt_idx=int(alt_df.loc[j, "index"]),
                        kind="exact" if gap_seconds == 0 else "near",
                        gap_hours=gap_seconds / 3600,
                    )
                ] + reconstruct(i + 1, j + 1)

        if solve(i + 1, j) == current:
            return reconstruct(i + 1, j)
        return reconstruct(i, j + 1)

    return reconstruct(0, 0)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def run_winner_diff(
    *,
    output_dir: Path,
    trade_rows: list[dict[str, Any]],
    winner_profile: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trades = build_trade_frame(trade_rows)
    trades = trades[trades["strategy_profile"].isin([MAINLINE, winner_profile])].copy()

    pair_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for year in DIFF_YEARS:
        for side in ("LONG", "SHORT"):
            subset = trades[(trades["year"] == year) & (trades["side"] == side)].copy()
            main_df = subset[subset["strategy_profile"] == MAINLINE].sort_values("signal_time")
            alt_df = subset[subset["strategy_profile"] == winner_profile].sort_values("signal_time")
            pairs = build_ordered_matches(main_df, alt_df)
            matched_main = {pair.main_idx for pair in pairs}
            matched_alt = {pair.alt_idx for pair in pairs}

            for pair in pairs:
                main_row = trades.loc[pair.main_idx]
                alt_row = trades.loc[pair.alt_idx]
                pair_rows.append(
                    {
                        "year": year,
                        "side": side,
                        "match_kind": pair.kind,
                        "gap_hours": round(pair.gap_hours, 2),
                        "main_signal_time": main_row["signal_time"].isoformat(),
                        "alt_signal_time": alt_row["signal_time"].isoformat(),
                        "main_pnl_r": round(float(main_row["pnl_r"]), 4),
                        "alt_pnl_r": round(float(alt_row["pnl_r"]), 4),
                        "delta_r": round(float(alt_row["pnl_r"] - main_row["pnl_r"]), 4),
                        "main_exit_reason": main_row["exit_reason"],
                        "alt_exit_reason": alt_row["exit_reason"],
                        "main_trend_strength": int(main_row["trend_strength"]),
                        "alt_trend_strength": int(alt_row["trend_strength"]),
                    }
                )

            unmatched_main = trades.loc[sorted(main_df.index.difference(list(matched_main)))]
            unmatched_alt = trades.loc[sorted(alt_df.index.difference(list(matched_alt)))]
            for origin, frame in [("mainline_only", unmatched_main), ("winner_only", unmatched_alt)]:
                for row in frame.itertuples():
                    unmatched_rows.append(
                        {
                            "year": year,
                            "side": side,
                            "origin": origin,
                            "signal_time": row.signal_time.isoformat(),
                            "entry_time": row.entry_time.isoformat(),
                            "exit_time": row.exit_time.isoformat(),
                            "trend_strength": int(row.trend_strength),
                            "confidence": int(row.confidence),
                            "bars_held": int(row.bars_held),
                            "exit_reason": row.exit_reason,
                            "pnl_r": round(float(row.pnl_r), 4),
                        }
                    )

            pair_subset = [row for row in pair_rows if row["year"] == year and row["side"] == side]
            summary_rows.append(
                {
                    "year": year,
                    "side": side,
                    "main_trades": int(len(main_df)),
                    "winner_trades": int(len(alt_df)),
                    "exact_pairs": int(sum(row["match_kind"] == "exact" for row in pair_subset)),
                    "near_pairs": int(sum(row["match_kind"] == "near" for row in pair_subset)),
                    "mainline_only": int(len(unmatched_main)),
                    "winner_only": int(len(unmatched_alt)),
                    "matched_delta_r": round(float(sum(row["delta_r"] for row in pair_subset)), 4),
                    "mainline_only_pnl_r": round(float(unmatched_main["pnl_r"].sum()), 4),
                    "winner_only_pnl_r": round(float(unmatched_alt["pnl_r"].sum()), 4),
                    "overall_delta_r": round(float(alt_df["pnl_r"].sum() - main_df["pnl_r"].sum()), 4),
                }
            )

    monthly_rows: list[dict[str, Any]] = []
    subset = trades[trades["year"].isin(DIFF_YEARS)].copy()
    if not subset.empty:
        grouped = (
            subset.groupby(["year", "month", "strategy_profile"])
            .agg(trades=("pnl_r", "size"), cumulative_r=("pnl_r", "sum"))
            .reset_index()
        )
        pivot = (
            grouped.pivot(index=["year", "month"], columns="strategy_profile", values="cumulative_r")
            .fillna(0.0)
            .reset_index()
        )
        for row in pivot.itertuples(index=False):
            main_r = float(getattr(row, MAINLINE, 0.0))
            winner_r = float(getattr(row, winner_profile, 0.0))
            monthly_rows.append(
                {
                    "year": int(row.year),
                    "month": row.month,
                    "mainline_pnl_r": round(main_r, 4),
                    "winner_pnl_r": round(winner_r, 4),
                    "diff_r": round(winner_r - main_r, 4),
                }
            )

    write_csv(output_dir / "matched_pairs.csv", pair_rows)
    write_csv(output_dir / "unmatched_trades.csv", unmatched_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "monthly_breakdown.csv", monthly_rows)

    top_positive = sorted(monthly_rows, key=lambda item: item["diff_r"], reverse=True)[:5]
    top_negative = sorted(monthly_rows, key=lambda item: item["diff_r"])[:5]
    report = "\n".join(
        [
            "# Mainline vs Winner Diff",
            "",
            f"- mainline: `{MAINLINE}`",
            f"- winner: `{winner_profile}`",
            f"- years: `{', '.join(str(year) for year in DIFF_YEARS)}`",
            "",
            "## Summary",
            "",
            render_table(
                summary_rows,
                [
                    ("year", "Year"),
                    ("side", "Side"),
                    ("main_trades", "Main Trades"),
                    ("winner_trades", "Winner Trades"),
                    ("matched_delta_r", "Matched Delta R"),
                    ("mainline_only_pnl_r", "Main Only R"),
                    ("winner_only_pnl_r", "Winner Only R"),
                    ("overall_delta_r", "Overall Delta R"),
                ],
            ),
            "",
            "## Best Months",
            "",
            render_table(top_positive, [("month", "Month"), ("mainline_pnl_r", "Mainline"), ("winner_pnl_r", "Winner"), ("diff_r", "Diff")]),
            "",
            "## Worst Months",
            "",
            render_table(top_negative, [("month", "Month"), ("mainline_pnl_r", "Mainline"), ("winner_pnl_r", "Winner"), ("diff_r", "Diff")]),
        ]
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return summary_rows, pair_rows, unmatched_rows, monthly_rows


def fmt_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def build_conclusion(
    *,
    shortlist_rows: list[dict[str, Any]],
    primary_oos_side_rows: list[dict[str, Any]],
    secondary_full_side_rows: list[dict[str, Any]],
    winner_diff_summary: list[dict[str, Any]],
) -> tuple[str, str | None]:
    qualified = [row for row in shortlist_rows if row["qualified"]]
    if not qualified:
        return "SHORT 非对称 entry 未被证明成立。下一条默认转去 `post_tp1 extension/path-hold`，不再继续扩 entry 组合。", None

    winner = qualified[0]
    winner_profile = str(winner["strategy_profile"])
    primary_side_map = {
        (row["strategy_profile"], row["side"]): row
        for row in primary_oos_side_rows
        if row["window"] == PRIMARY_WINDOW
    }
    secondary_long_map = {
        row["strategy_profile"]: row
        for row in secondary_full_side_rows
        if row["window"] == SECONDARY_WINDOW and row["side"] == "LONG"
    }
    main_short = primary_side_map.get((MAINLINE, "SHORT"))
    winner_short = primary_side_map.get((winner_profile, "SHORT"))
    main_long = primary_side_map.get((MAINLINE, "LONG"))
    winner_long = primary_side_map.get((winner_profile, "LONG"))
    winner_secondary_long = float(secondary_long_map.get(winner_profile, {}).get("cumulative_r", 0.0))
    mainline_secondary_long = float(secondary_long_map.get(MAINLINE, {}).get("cumulative_r", 0.0))
    secondary_long_delta = winner_secondary_long - mainline_secondary_long

    short_delta = float(winner_short["cumulative_r"] - main_short["cumulative_r"]) if winner_short and main_short else 0.0
    long_delta = float(winner_long["cumulative_r"] - main_long["cumulative_r"]) if winner_long and main_long else 0.0
    if abs(short_delta) >= abs(long_delta):
        alpha_source = f"增量主要来自 SHORT，主窗口 OOS SHORT 累计多了 {short_delta:.4f}R。"
    else:
        alpha_source = f"增量主要来自 LONG，主窗口 OOS LONG 累计多了 {long_delta:.4f}R。"

    keep_long = (
        f"次级窗口 LONG 基本保住了，secondary full-window LONG 相对 mainline 的差值为 {secondary_long_delta:.4f}R。"
        if secondary_long_delta >= -2.0
        else f"次级窗口 LONG 没保住，secondary full-window LONG 相对 mainline 少了 {abs(secondary_long_delta):.4f}R。"
    )

    matched_delta = sum(float(row["matched_delta_r"]) for row in winner_diff_summary)
    selection_delta = sum(float(row["winner_only_pnl_r"] - row["mainline_only_pnl_r"]) for row in winner_diff_summary)
    if abs(matched_delta) >= abs(selection_delta) * 1.25:
        quality_text = "改善更像来自更好的交易质量，而不是单纯多做了几笔。"
    elif abs(selection_delta) >= abs(matched_delta) * 1.25:
        quality_text = "改善更像来自不同的交易选择，也就是放行/过滤的样本变了。"
    else:
        quality_text = "改善同时来自 matched setup 的收益变化和交易选择变化，两边都有贡献。"

    yearly_rows = [row for row in winner_diff_summary if row["side"] == "SHORT"]
    positive_years = [str(row["year"]) for row in yearly_rows if float(row["overall_delta_r"]) > 0]
    negative_years = [str(row["year"]) for row in yearly_rows if float(row["overall_delta_r"]) < 0]
    phase_text = (
        f"有效年份主要在 {', '.join(positive_years) if positive_years else '无明显正向年份'}；"
        f"退化年份主要在 {', '.join(negative_years) if negative_years else '无明显退化年份'}。"
    )

    conclusion = " ".join([alpha_source, keep_long, quality_text, phase_text])
    return conclusion, winner_profile


def main() -> None:
    args = parse_args()
    configure_logging()

    primary_start = parse_date(args.primary_start)
    secondary_start = parse_date(args.secondary_start)
    requested_end = parse_date(args.end) if args.end else datetime.now(timezone.utc)
    if secondary_start <= primary_start:
        raise ValueError("secondary-start must be later than primary-start")

    service = build_service()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    enriched_history: dict[str, dict[str, pd.DataFrame]] = {}
    for profile in CANDIDATE_PROFILES:
        print(f"[short-asymmetry] preload {profile}", flush=True)
        enriched_history[profile] = service.prepare_enriched_history(
            exchange=args.exchange,
            market_type=args.market_type,
            symbol=args.symbol,
            strategy_profile=profile,
            start=primary_start,
            end=requested_end,
        )

    resolved_end = resolve_end_from_history(enriched_history)
    primary_window = WindowSpec(name=PRIMARY_WINDOW, start=primary_start, end=resolved_end)
    secondary_window = WindowSpec(name=SECONDARY_WINDOW, start=secondary_start, end=resolved_end)

    window_results = [
        run_window(service=service, args=args, window=primary_window, enriched_history=enriched_history),
        run_window(service=service, args=args, window=secondary_window, enriched_history=enriched_history),
    ]

    full_summary_rows = [row for result in window_results for row in result["full_summary_rows"]]
    full_trade_rows = [row for result in window_results for row in result["full_trade_rows"]]
    candidate_fold_rows = [row for result in window_results for row in result["candidate_fold_rows"]]
    independent_oos_rows = [row for result in window_results for row in result["independent_oos_rows"]]
    independent_oos_trade_rows = [row for result in window_results for row in result["independent_oos_trade_rows"]]
    selection_fold_rows = [row for result in window_results for row in result["selection_fold_rows"]]
    selection_trade_rows = [row for result in window_results for row in result["selection_trade_rows"]]
    selection_count_rows = [row for result in window_results for row in result["selection_count_rows"]]
    selected_summary_rows = [result["selected_summary_row"] for result in window_results]

    full_trade_frame = build_trade_frame(full_trade_rows)
    independent_oos_trade_frame = build_trade_frame(independent_oos_trade_rows)
    selection_trade_frame = build_trade_frame(selection_trade_rows)

    full_year_rows = summarize_trade_distribution(full_trade_frame, group_cols=["window", "strategy_profile", "profile_label", "year"])
    full_side_rows = summarize_trade_distribution(full_trade_frame, group_cols=["window", "strategy_profile", "profile_label", "side"])
    independent_oos_year_rows = summarize_trade_distribution(
        independent_oos_trade_frame,
        group_cols=["window", "strategy_profile", "profile_label", "year"],
    )
    independent_oos_side_rows = summarize_trade_distribution(
        independent_oos_trade_frame,
        group_cols=["window", "strategy_profile", "profile_label", "side"],
    )
    selected_oos_side_rows = summarize_trade_distribution(
        selection_trade_frame,
        group_cols=["window", "selected_candidate", "side"],
    )

    primary_oos_rows = [row for row in independent_oos_rows if row["window"] == PRIMARY_WINDOW]
    secondary_full_side_rows = [row for row in full_side_rows if row["window"] == SECONDARY_WINDOW]
    shortlist_rows = build_shortlist(
        primary_oos_rows=primary_oos_rows,
        secondary_full_side_rows=secondary_full_side_rows,
    )

    winner_diff_summary: list[dict[str, Any]] = []
    winner_profile: str | None = None
    qualified = [row for row in shortlist_rows if row["qualified"]]
    if qualified:
        winner_profile = str(qualified[0]["strategy_profile"])
        winner_diff_dir = output_dir / "winner_diff"
        winner_diff_summary, _, _, _ = run_winner_diff(
            output_dir=winner_diff_dir,
            trade_rows=[row for row in independent_oos_trade_rows if row["window"] == PRIMARY_WINDOW],
            winner_profile=winner_profile,
        )

    conclusion_text, winner_profile = build_conclusion(
        shortlist_rows=shortlist_rows,
        primary_oos_side_rows=independent_oos_side_rows,
        secondary_full_side_rows=full_side_rows,
        winner_diff_summary=winner_diff_summary,
    )

    write_csv(output_dir / "full_summary.csv", full_summary_rows)
    write_csv(output_dir / "full_trades.csv", full_trade_rows)
    write_csv(output_dir / "full_year_summary.csv", full_year_rows)
    write_csv(output_dir / "full_side_summary.csv", full_side_rows)
    write_csv(output_dir / "candidate_oos_folds.csv", candidate_fold_rows)
    write_csv(output_dir / "independent_oos_summary.csv", independent_oos_rows)
    write_csv(output_dir / "independent_oos_trades.csv", independent_oos_trade_rows)
    write_csv(output_dir / "independent_oos_year_summary.csv", independent_oos_year_rows)
    write_csv(output_dir / "independent_oos_side_summary.csv", independent_oos_side_rows)
    write_csv(output_dir / "selection_folds.csv", selection_fold_rows)
    write_csv(output_dir / "selection_counts.csv", selection_count_rows)
    write_csv(output_dir / "selected_oos_summary.csv", selected_summary_rows)
    write_csv(output_dir / "selected_oos_trades.csv", selection_trade_rows)
    write_csv(output_dir / "selected_oos_side_summary.csv", selected_oos_side_rows)
    write_csv(output_dir / "shortlist.csv", shortlist_rows)

    report_lines = [
        "# Mainline Short-Asymmetric Entry Research",
        "",
        f"- 标的：`{args.symbol}`",
        f"- 主窗口：`{primary_window.start.date().isoformat()} -> {resolved_end.date().isoformat()}`",
        f"- 次级窗口：`{secondary_window.start.date().isoformat()} -> {resolved_end.date().isoformat()}`",
        f"- Walk-forward：`{args.train_days}d train / {args.test_days}d test / {args.step_days}d step / {args.scheme}`",
        f"- 最低训练交易数：`{args.min_train_trades}`",
        "- 口径：`confirmed swing`, `LONG 1R -> 3R scaled / SHORT 1.5R fixed`",
        "",
        "## Candidate Pool",
        "",
        render_table(
            [{"strategy_profile": profile, "profile_label": PROFILE_LABELS.get(profile, profile)} for profile in CANDIDATE_PROFILES],
            [("strategy_profile", "Profile"), ("profile_label", "Label")],
        ),
        "",
        "## Primary Independent OOS",
        "",
        render_table(
            rank_oos_rows(primary_oos_rows),
            [
                ("profile_label", "Profile"),
                ("total_trades", "Trades"),
                ("profit_factor", "PF"),
                ("expectancy_r", "Exp R"),
                ("cumulative_r", "Cum R"),
                ("max_drawdown_r", "MaxDD R"),
            ],
        ),
        "",
        "## Secondary LONG Guardrail",
        "",
        render_table(
            [row for row in secondary_full_side_rows if row["side"] == "LONG"],
            [
                ("profile_label", "Profile"),
                ("trades", "Trades"),
                ("cumulative_r", "Cum R"),
                ("profit_factor", "PF"),
                ("max_drawdown_r", "MaxDD R"),
            ],
        ),
        "",
        "## Shortlist",
        "",
        render_table(
            shortlist_rows,
            [
                ("profile_label", "Profile"),
                ("primary_oos_cumulative_r", "Primary OOS Cum R"),
                ("primary_oos_profit_factor", "Primary OOS PF"),
                ("primary_oos_max_drawdown_r", "Primary OOS MaxDD"),
                ("secondary_long_delta_r", "Secondary LONG Delta"),
                ("qualified", "Qualified"),
            ],
        ),
        "",
        "## Selection Frequency",
        "",
        render_table(
            selection_count_rows,
            [("window", "Window"), ("profile_label", "Profile"), ("selected_folds", "Selected Folds")],
        ),
        "",
        "## Conclusion",
        "",
        f"- {conclusion_text}",
    ]
    if winner_profile is not None:
        report_lines.extend(
            [
                f"- winner: `{winner_profile}`",
                f"- winner diff report: `{output_dir / 'winner_diff' / 'report.md'}`",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    results_json = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": args.symbol,
        "exchange": args.exchange,
        "market_type": args.market_type,
        "primary_start": primary_window.start.isoformat(),
        "secondary_start": secondary_window.start.isoformat(),
        "resolved_end": resolved_end.isoformat(),
        "train_days": args.train_days,
        "test_days": args.test_days,
        "step_days": args.step_days,
        "scheme": args.scheme,
        "min_train_trades": args.min_train_trades,
        "candidates": list(CANDIDATE_PROFILES),
        "shortlist": shortlist_rows,
        "winner_profile": winner_profile,
        "conclusion": conclusion_text,
    }
    (output_dir / "results.json").write_text(json.dumps(results_json, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved short-asymmetric entry research artifacts to {output_dir}")


if __name__ == "__main__":
    main()
