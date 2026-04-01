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
    "exit_profile": "champion_challenger_confirmation_long_scaled1_3_short_fixed1_5",
    "take_profit_mode": "scaled",
    "scaled_tp1_r": 1.0,
    "scaled_tp2_r": 3.0,
    "long_exit": {"take_profit_mode": "scaled", "scaled_tp1_r": 1.0, "scaled_tp2_r": 3.0},
    "short_exit": {"take_profit_mode": "fixed_r", "fixed_take_profit_r": 1.5},
    "swing_detection_mode": "confirmed",
}
DEFAULT_CHAMPION = "swing_trend_long_regime_gate_v1"
DEFAULT_CHALLENGER = "swing_trend_long_regime_short_no_reversal_no_aux_v1"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "backtests" / "mainline_champion_challenger_confirmation"
PRIMARY_WINDOW = "primary_2020_latest"
SECONDARY_WINDOW = "secondary_2024_latest"
DIFF_YEARS = (2022, 2024, 2026)
MATCH_TOLERANCE = pd.Timedelta(hours=12)
FLOAT_TOLERANCE = 1e-4


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
    champion_idx: int
    challenger_idx: int
    kind: str
    gap_hours: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Confirm mainline champion vs challenger robustness.")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="perpetual")
    parser.add_argument("--champion-profile", default=DEFAULT_CHAMPION)
    parser.add_argument("--challenger-profile", default=DEFAULT_CHALLENGER)
    parser.add_argument("--primary-start", default="2020-01-01")
    parser.add_argument("--secondary-start", default="2024-03-19")
    parser.add_argument("--end", default=None, help="Defaults to latest available data at runtime.")
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--step-days", type=int, default=90)
    parser.add_argument("--scheme", choices=["rolling", "anchored"], default="rolling")
    parser.add_argument("--min-train-trades", type=int, default=12)
    parser.add_argument("--reference-dir", default=None, help="Optional reference artifact dir for replay validation.")
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


def role_label(profile: str, *, champion: str, challenger: str) -> str:
    if profile == champion:
        return "Champion"
    if profile == challenger:
        return "Challenger"
    return profile


def summary_row(
    *,
    window: str,
    profile: str,
    source: str,
    summary_dict: dict[str, Any],
    champion: str,
    challenger: str,
) -> dict[str, Any]:
    return {
        "window": window,
        "source": source,
        "strategy_profile": profile,
        "profile_label": role_label(profile, champion=champion, challenger=challenger),
        **summary_dict,
    }


def trade_rows_from_trades(
    *,
    window: str,
    source: str,
    profile: str,
    trades: list[BacktestTrade],
    champion: str,
    challenger: str,
    fold: int | None = None,
    selected_candidate: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trade in trades:
        row = asdict(trade)
        row["window"] = window
        row["source"] = source
        row["profile_label"] = role_label(profile, champion=champion, challenger=challenger)
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def run_window(
    *,
    service: BacktestService,
    args: argparse.Namespace,
    window: WindowSpec,
    profiles: tuple[str, str],
    enriched_history: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, Any]:
    champion, challenger = profiles
    print(
        f"[champion-challenger] window {window.name} {window.start.date().isoformat()}->{window.end.date().isoformat()}",
        flush=True,
    )
    full_summary_rows: list[dict[str, Any]] = []
    full_trade_rows: list[dict[str, Any]] = []
    independent_oos_rows: list[dict[str, Any]] = []
    independent_oos_trade_rows: list[dict[str, Any]] = []
    selection_fold_rows: list[dict[str, Any]] = []
    selection_count_rows: list[dict[str, Any]] = []
    per_fold_delta_rows: list[dict[str, Any]] = []
    fold_result_map: dict[tuple[str, int], CandidateFoldResult] = {}

    folds = generate_folds(
        start=window.start,
        end=window.end,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        scheme=args.scheme,
    )

    for profile in profiles:
        print(f"[champion-challenger] full-window {window.name} {profile}", flush=True)
        summary, trades = service.run_symbol_strategy_with_enriched_frames(
            symbol=args.symbol,
            strategy_profile=profile,
            start=window.start,
            end=window.end,
            enriched_frames=enriched_history[profile],
        )
        full_summary_rows.append(
            summary_row(
                window=window.name,
                profile=profile,
                source="full_window",
                summary_dict=to_summary_dict(summary),
                champion=champion,
                challenger=challenger,
            )
        )
        full_trade_rows.extend(
            trade_rows_from_trades(
                window=window.name,
                source="full_window",
                profile=profile,
                trades=trades,
                champion=champion,
                challenger=challenger,
            )
        )

    for profile in profiles:
        print(f"[champion-challenger] independent-oos {window.name} {profile}", flush=True)
        all_test_trades: list[BacktestTrade] = []
        test_signals_now = 0
        test_skipped_entries = 0
        for fold in folds:
            print(
                f"[champion-challenger] fold {window.name} #{fold.index} {profile} train {fold.train_start.date().isoformat()}->{fold.train_end.date().isoformat()} test {fold.test_start.date().isoformat()}->{fold.test_end.date().isoformat()}",
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
            summary_row(
                window=window.name,
                profile=profile,
                source="independent_oos",
                summary_dict=to_summary_dict(oos_summary),
                champion=champion,
                challenger=challenger,
            )
        )
        independent_oos_trade_rows.extend(
            trade_rows_from_trades(
                window=window.name,
                source="independent_oos",
                profile=profile,
                trades=all_test_trades,
                champion=champion,
                challenger=challenger,
            )
        )

    selection_counts = {profile: 0 for profile in profiles}
    for fold in folds:
        champion_result = fold_result_map[(champion, fold.index)]
        challenger_result = fold_result_map[(challenger, fold.index)]
        train_rows: list[dict[str, Any]] = []
        for profile in profiles:
            result = fold_result_map[(profile, fold.index)]
            if result.train_summary["total_trades"] < args.min_train_trades:
                continue
            train_rows.append(
                {
                    "strategy_profile": profile,
                    "profile_label": role_label(profile, champion=champion, challenger=challenger),
                    **{f"train_{key}": value for key, value in result.train_summary.items()},
                }
            )

        ranked = rank_train_candidates(train_rows)
        chosen = ranked[0] if ranked else None
        chosen_profile = str(chosen["strategy_profile"]) if chosen is not None else None
        if chosen_profile is not None:
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
                "selected_profile_label": role_label(chosen_profile, champion=champion, challenger=challenger)
                if chosen_profile is not None
                else None,
            }
        )
        per_fold_delta_rows.append(
            {
                "window": window.name,
                "fold": fold.index,
                "train_start": fold.train_start.date().isoformat(),
                "train_end": fold.train_end.date().isoformat(),
                "test_start": fold.test_start.date().isoformat(),
                "test_end": fold.test_end.date().isoformat(),
                "champion_train_total_trades": champion_result.train_summary["total_trades"],
                "challenger_train_total_trades": challenger_result.train_summary["total_trades"],
                "champion_train_profit_factor": round(float(champion_result.train_summary["profit_factor"]), 4),
                "challenger_train_profit_factor": round(float(challenger_result.train_summary["profit_factor"]), 4),
                "champion_train_expectancy_r": round(float(champion_result.train_summary["expectancy_r"]), 4),
                "challenger_train_expectancy_r": round(float(challenger_result.train_summary["expectancy_r"]), 4),
                "champion_train_cumulative_r": round(float(champion_result.train_summary["cumulative_r"]), 4),
                "challenger_train_cumulative_r": round(float(challenger_result.train_summary["cumulative_r"]), 4),
                "champion_train_max_drawdown_r": round(float(champion_result.train_summary["max_drawdown_r"]), 4),
                "challenger_train_max_drawdown_r": round(float(challenger_result.train_summary["max_drawdown_r"]), 4),
                "champion_test_total_trades": champion_result.test_summary["total_trades"],
                "challenger_test_total_trades": challenger_result.test_summary["total_trades"],
                "champion_test_profit_factor": round(float(champion_result.test_summary["profit_factor"]), 4),
                "challenger_test_profit_factor": round(float(challenger_result.test_summary["profit_factor"]), 4),
                "champion_test_expectancy_r": round(float(champion_result.test_summary["expectancy_r"]), 4),
                "challenger_test_expectancy_r": round(float(challenger_result.test_summary["expectancy_r"]), 4),
                "champion_test_cumulative_r": round(float(champion_result.test_summary["cumulative_r"]), 4),
                "challenger_test_cumulative_r": round(float(challenger_result.test_summary["cumulative_r"]), 4),
                "champion_test_max_drawdown_r": round(float(champion_result.test_summary["max_drawdown_r"]), 4),
                "challenger_test_max_drawdown_r": round(float(challenger_result.test_summary["max_drawdown_r"]), 4),
                "train_delta_cumulative_r": round(
                    float(challenger_result.train_summary["cumulative_r"] - champion_result.train_summary["cumulative_r"]),
                    4,
                ),
                "train_delta_profit_factor": round(
                    float(challenger_result.train_summary["profit_factor"] - champion_result.train_summary["profit_factor"]),
                    4,
                ),
                "train_delta_max_drawdown_r": round(
                    float(challenger_result.train_summary["max_drawdown_r"] - champion_result.train_summary["max_drawdown_r"]),
                    4,
                ),
                "test_delta_cumulative_r": round(
                    float(challenger_result.test_summary["cumulative_r"] - champion_result.test_summary["cumulative_r"]),
                    4,
                ),
                "test_delta_profit_factor": round(
                    float(challenger_result.test_summary["profit_factor"] - champion_result.test_summary["profit_factor"]),
                    4,
                ),
                "test_delta_max_drawdown_r": round(
                    float(challenger_result.test_summary["max_drawdown_r"] - champion_result.test_summary["max_drawdown_r"]),
                    4,
                ),
                "selected_candidate": chosen_profile,
                "selected_profile_label": role_label(chosen_profile, champion=champion, challenger=challenger)
                if chosen_profile is not None
                else None,
            }
        )

    for profile, count in selection_counts.items():
        selection_count_rows.append(
            {
                "window": window.name,
                "strategy_profile": profile,
                "profile_label": role_label(profile, champion=champion, challenger=challenger),
                "selected_folds": count,
            }
        )

    return {
        "window": window,
        "full_summary_rows": full_summary_rows,
        "full_trade_rows": full_trade_rows,
        "independent_oos_rows": independent_oos_rows,
        "independent_oos_trade_rows": independent_oos_trade_rows,
        "selection_fold_rows": selection_fold_rows,
        "selection_count_rows": selection_count_rows,
        "per_fold_delta_rows": per_fold_delta_rows,
    }


def build_monthly_delta_summary(
    *,
    trade_rows: list[dict[str, Any]],
    window_name: str,
    champion: str,
    challenger: str,
) -> list[dict[str, Any]]:
    trades = build_trade_frame([row for row in trade_rows if row["window"] == window_name])
    if trades.empty:
        return []
    monthly_rows: list[dict[str, Any]] = []
    scopes = [("overall", trades), ("LONG", trades[trades["side"] == "LONG"]), ("SHORT", trades[trades["side"] == "SHORT"])]
    for scope, scope_frame in scopes:
        if scope_frame.empty:
            continue
        grouped = (
            scope_frame.groupby(["year", "month", "strategy_profile"])
            .agg(trades=("pnl_r", "size"), cumulative_r=("pnl_r", "sum"))
            .reset_index()
        )
        pivot = (
            grouped.pivot(index=["year", "month"], columns="strategy_profile", values="cumulative_r")
            .fillna(0.0)
            .reset_index()
        )
        trade_pivot = (
            grouped.pivot(index=["year", "month"], columns="strategy_profile", values="trades")
            .fillna(0)
            .reset_index()
        )
        merged = pivot.merge(trade_pivot, on=["year", "month"], suffixes=("_r", "_trades"))
        for row in merged.itertuples(index=False):
            champion_r = float(getattr(row, champion + "_r", 0.0))
            challenger_r = float(getattr(row, challenger + "_r", 0.0))
            monthly_rows.append(
                {
                    "window": window_name,
                    "scope": scope,
                    "year": int(row.year),
                    "month": row.month,
                    "champion_trades": int(getattr(row, champion + "_trades", 0)),
                    "challenger_trades": int(getattr(row, challenger + "_trades", 0)),
                    "champion_cumulative_r": round(champion_r, 4),
                    "challenger_cumulative_r": round(challenger_r, 4),
                    "delta_r": round(challenger_r - champion_r, 4),
                }
            )
    return monthly_rows


def build_concentration_summary(monthly_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not monthly_rows:
        return []
    frame = pd.DataFrame(monthly_rows)
    rows: list[dict[str, Any]] = []
    for scope, group in frame.groupby("scope", sort=False):
        positive = group[group["delta_r"] > 0].sort_values("delta_r", ascending=False)
        negative = group[group["delta_r"] < 0].sort_values("delta_r")
        total_positive_delta = float(positive["delta_r"].sum()) if not positive.empty else 0.0
        total_negative_delta = float(negative["delta_r"].sum()) if not negative.empty else 0.0
        top1_share = (float(positive.iloc[0]["delta_r"]) / total_positive_delta * 100.0) if total_positive_delta > 0 else 0.0
        top3_share = (float(positive.head(3)["delta_r"].sum()) / total_positive_delta * 100.0) if total_positive_delta > 0 else 0.0
        yearly = group.groupby("year", as_index=False)["delta_r"].sum().sort_values("delta_r", ascending=False)
        positive_yearly = yearly[yearly["delta_r"] > 0]
        dominant_year = int(positive_yearly.iloc[0]["year"]) if not positive_yearly.empty else None
        dominant_year_delta = float(positive_yearly.iloc[0]["delta_r"]) if not positive_yearly.empty else 0.0
        positive_year_total = float(positive_yearly["delta_r"].sum()) if not positive_yearly.empty else 0.0
        dominant_year_share = (dominant_year_delta / positive_year_total * 100.0) if positive_year_total > 0 else 0.0
        regime_specialist = total_positive_delta > 0 and (top3_share >= 70.0 or dominant_year_share >= 70.0)
        rows.append(
            {
                "scope": scope,
                "positive_months": int((group["delta_r"] > 0).sum()),
                "negative_months": int((group["delta_r"] < 0).sum()),
                "total_delta_r": round(float(group["delta_r"].sum()), 4),
                "total_positive_delta_r": round(total_positive_delta, 4),
                "total_negative_delta_r": round(total_negative_delta, 4),
                "top_positive_month": positive.iloc[0]["month"] if not positive.empty else None,
                "top_positive_month_delta_r": round(float(positive.iloc[0]["delta_r"]), 4) if not positive.empty else None,
                "top1_positive_share_pct": round(top1_share, 2),
                "top3_positive_share_pct": round(top3_share, 2),
                "positive_years": int(len(positive_yearly)),
                "dominant_positive_year": dominant_year,
                "dominant_positive_year_delta_r": round(dominant_year_delta, 4) if dominant_year is not None else None,
                "dominant_positive_year_share_pct": round(dominant_year_share, 2),
                "regime_specialist_tendency": regime_specialist,
            }
        )
    return rows


def build_ordered_matches(champion_df: pd.DataFrame, challenger_df: pd.DataFrame) -> list[MatchPair]:
    champion_df = champion_df.sort_values("signal_time").reset_index(drop=False)
    challenger_df = challenger_df.sort_values("signal_time").reset_index(drop=False)
    champion_times = champion_df["signal_time"].tolist()
    challenger_times = challenger_df["signal_time"].tolist()
    tolerance_seconds = int(MATCH_TOLERANCE.total_seconds())

    def better_score(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
        if left[0] != right[0]:
            return left if left[0] > right[0] else right
        return left if left[1] >= right[1] else right

    @lru_cache(maxsize=None)
    def solve(i: int, j: int) -> tuple[int, int]:
        if i >= len(champion_times) or j >= len(challenger_times):
            return (0, 0)

        best = better_score(solve(i + 1, j), solve(i, j + 1))
        gap_seconds = abs(int((champion_times[i] - challenger_times[j]).total_seconds()))
        if gap_seconds <= tolerance_seconds:
            matched = solve(i + 1, j + 1)
            candidate = (matched[0] + 1, matched[1] - gap_seconds)
            best = better_score(best, candidate)
        return best

    def reconstruct(i: int, j: int) -> list[MatchPair]:
        if i >= len(champion_times) or j >= len(challenger_times):
            return []

        current = solve(i, j)
        gap_seconds = abs(int((champion_times[i] - challenger_times[j]).total_seconds()))
        if gap_seconds <= tolerance_seconds:
            matched = solve(i + 1, j + 1)
            candidate = (matched[0] + 1, matched[1] - gap_seconds)
            if candidate == current:
                return [
                    MatchPair(
                        champion_idx=int(champion_df.loc[i, "index"]),
                        challenger_idx=int(challenger_df.loc[j, "index"]),
                        kind="exact" if gap_seconds == 0 else "near",
                        gap_hours=gap_seconds / 3600,
                    )
                ] + reconstruct(i + 1, j + 1)

        if solve(i + 1, j) == current:
            return reconstruct(i + 1, j)
        return reconstruct(i, j + 1)

    return reconstruct(0, 0)


def run_pair_diff(
    *,
    output_dir: Path,
    trade_rows: list[dict[str, Any]],
    champion: str,
    challenger: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trades = build_trade_frame(trade_rows)
    trades = trades[trades["strategy_profile"].isin([champion, challenger])].copy()

    pair_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for year in DIFF_YEARS:
        for side in ("LONG", "SHORT"):
            subset = trades[(trades["year"] == year) & (trades["side"] == side)].copy()
            champion_df = subset[subset["strategy_profile"] == champion].sort_values("signal_time")
            challenger_df = subset[subset["strategy_profile"] == challenger].sort_values("signal_time")
            pairs = build_ordered_matches(champion_df, challenger_df)
            matched_champion = {pair.champion_idx for pair in pairs}
            matched_challenger = {pair.challenger_idx for pair in pairs}

            for pair in pairs:
                champion_row = trades.loc[pair.champion_idx]
                challenger_row = trades.loc[pair.challenger_idx]
                pair_rows.append(
                    {
                        "year": year,
                        "side": side,
                        "match_kind": pair.kind,
                        "gap_hours": round(pair.gap_hours, 2),
                        "champion_signal_time": champion_row["signal_time"].isoformat(),
                        "challenger_signal_time": challenger_row["signal_time"].isoformat(),
                        "champion_pnl_r": round(float(champion_row["pnl_r"]), 4),
                        "challenger_pnl_r": round(float(challenger_row["pnl_r"]), 4),
                        "delta_r": round(float(challenger_row["pnl_r"] - champion_row["pnl_r"]), 4),
                        "champion_exit_reason": champion_row["exit_reason"],
                        "challenger_exit_reason": challenger_row["exit_reason"],
                        "champion_trend_strength": int(champion_row["trend_strength"]),
                        "challenger_trend_strength": int(challenger_row["trend_strength"]),
                    }
                )

            unmatched_champion = trades.loc[sorted(champion_df.index.difference(list(matched_champion)))]
            unmatched_challenger = trades.loc[sorted(challenger_df.index.difference(list(matched_challenger)))]
            for origin, frame in [("champion_only", unmatched_champion), ("challenger_only", unmatched_challenger)]:
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
                    "champion_trades": int(len(champion_df)),
                    "challenger_trades": int(len(challenger_df)),
                    "exact_pairs": int(sum(row["match_kind"] == "exact" for row in pair_subset)),
                    "near_pairs": int(sum(row["match_kind"] == "near" for row in pair_subset)),
                    "champion_only": int(len(unmatched_champion)),
                    "challenger_only": int(len(unmatched_challenger)),
                    "matched_delta_r": round(float(sum(row["delta_r"] for row in pair_subset)), 4),
                    "champion_only_pnl_r": round(float(unmatched_champion["pnl_r"].sum()), 4),
                    "challenger_only_pnl_r": round(float(unmatched_challenger["pnl_r"].sum()), 4),
                    "overall_delta_r": round(float(challenger_df["pnl_r"].sum() - champion_df["pnl_r"].sum()), 4),
                }
            )

    monthly_rows: list[dict[str, Any]] = []
    subset = trades[trades["year"].isin(DIFF_YEARS)].copy()
    if not subset.empty:
        grouped = (
            subset.groupby(["year", "month", "strategy_profile"])
            .agg(cumulative_r=("pnl_r", "sum"))
            .reset_index()
        )
        pivot = (
            grouped.pivot(index=["year", "month"], columns="strategy_profile", values="cumulative_r")
            .fillna(0.0)
            .reset_index()
        )
        for row in pivot.itertuples(index=False):
            champion_r = float(getattr(row, champion, 0.0))
            challenger_r = float(getattr(row, challenger, 0.0))
            monthly_rows.append(
                {
                    "year": int(row.year),
                    "month": row.month,
                    "champion_pnl_r": round(champion_r, 4),
                    "challenger_pnl_r": round(challenger_r, 4),
                    "diff_r": round(challenger_r - champion_r, 4),
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
            "# Champion vs Challenger Diff",
            "",
            f"- champion: `{champion}`",
            f"- challenger: `{challenger}`",
            f"- years: `{', '.join(str(year) for year in DIFF_YEARS)}`",
            "",
            "## Summary",
            "",
            render_table(
                summary_rows,
                [
                    ("year", "Year"),
                    ("side", "Side"),
                    ("champion_trades", "Champion Trades"),
                    ("challenger_trades", "Challenger Trades"),
                    ("matched_delta_r", "Matched Delta R"),
                    ("champion_only_pnl_r", "Champion Only R"),
                    ("challenger_only_pnl_r", "Challenger Only R"),
                    ("overall_delta_r", "Overall Delta R"),
                ],
            ),
            "",
            "## Best Months",
            "",
            render_table(
                top_positive,
                [("month", "Month"), ("champion_pnl_r", "Champion"), ("challenger_pnl_r", "Challenger"), ("diff_r", "Diff")],
            ),
            "",
            "## Worst Months",
            "",
            render_table(
                top_negative,
                [("month", "Month"), ("champion_pnl_r", "Champion"), ("challenger_pnl_r", "Challenger"), ("diff_r", "Diff")],
            ),
        ]
    )
    (output_dir / "report.md").write_text(report + "\n", encoding="utf-8")
    return summary_rows, pair_rows, unmatched_rows, monthly_rows


def build_acceptance_row(
    *,
    primary_oos_rows: list[dict[str, Any]],
    secondary_full_side_rows: list[dict[str, Any]],
    champion: str,
    challenger: str,
) -> dict[str, Any]:
    by_profile = {row["strategy_profile"]: row for row in primary_oos_rows}
    long_side_rows = {
        row["strategy_profile"]: row
        for row in secondary_full_side_rows
        if row["side"] == "LONG"
    }
    champion_primary = by_profile[champion]
    challenger_primary = by_profile[challenger]
    champion_secondary_long = float(long_side_rows.get(champion, {}).get("cumulative_r", 0.0))
    challenger_secondary_long = float(long_side_rows.get(challenger, {}).get("cumulative_r", 0.0))

    pass_cum = float(challenger_primary["cumulative_r"]) > float(champion_primary["cumulative_r"])
    pass_pf = float(challenger_primary["profit_factor"]) > float(champion_primary["profit_factor"])
    pass_dd = float(challenger_primary["max_drawdown_r"]) <= float(champion_primary["max_drawdown_r"]) + 2.0
    pass_long_guard = challenger_secondary_long >= champion_secondary_long - 2.0
    qualified = pass_cum and pass_pf and pass_dd and pass_long_guard
    return {
        "champion_profile": champion,
        "challenger_profile": challenger,
        "primary_oos_champion_cumulative_r": round(float(champion_primary["cumulative_r"]), 4),
        "primary_oos_challenger_cumulative_r": round(float(challenger_primary["cumulative_r"]), 4),
        "primary_oos_delta_cumulative_r": round(float(challenger_primary["cumulative_r"] - champion_primary["cumulative_r"]), 4),
        "primary_oos_champion_profit_factor": round(float(champion_primary["profit_factor"]), 4),
        "primary_oos_challenger_profit_factor": round(float(challenger_primary["profit_factor"]), 4),
        "primary_oos_delta_profit_factor": round(float(challenger_primary["profit_factor"] - champion_primary["profit_factor"]), 4),
        "primary_oos_champion_max_drawdown_r": round(float(champion_primary["max_drawdown_r"]), 4),
        "primary_oos_challenger_max_drawdown_r": round(float(challenger_primary["max_drawdown_r"]), 4),
        "primary_oos_delta_max_drawdown_r": round(float(challenger_primary["max_drawdown_r"] - champion_primary["max_drawdown_r"]), 4),
        "secondary_long_champion_cumulative_r": round(champion_secondary_long, 4),
        "secondary_long_challenger_cumulative_r": round(challenger_secondary_long, 4),
        "secondary_long_delta_r": round(challenger_secondary_long - champion_secondary_long, 4),
        "pass_primary_cum_r": pass_cum,
        "pass_primary_pf": pass_pf,
        "pass_primary_max_dd": pass_dd,
        "pass_secondary_long_guard": pass_long_guard,
        "status": "challenger_active" if qualified else "challenger_rejected",
    }


def build_reference_validation(
    *,
    reference_dir: Path,
    primary_oos_rows: list[dict[str, Any]],
    independent_oos_side_rows: list[dict[str, Any]],
    champion: str,
    challenger: str,
) -> list[dict[str, Any]]:
    validation_rows: list[dict[str, Any]] = []
    reference_summary = pd.read_csv(reference_dir / "independent_oos_summary.csv")
    reference_side = pd.read_csv(reference_dir / "independent_oos_side_summary.csv")
    actual_summary = {
        row["strategy_profile"]: row
        for row in primary_oos_rows
    }
    actual_side = {
        (row["strategy_profile"], row["side"]): row
        for row in independent_oos_side_rows
        if row["window"] == PRIMARY_WINDOW
    }

    summary_metrics = ("total_trades", "profit_factor", "expectancy_r", "cumulative_r", "max_drawdown_r")
    side_metrics = ("trades", "profit_factor", "expectancy_r", "cumulative_r", "max_drawdown_r")
    for profile in (champion, challenger):
        ref_row = reference_summary[
            (reference_summary["window"] == PRIMARY_WINDOW) & (reference_summary["strategy_profile"] == profile)
        ]
        if ref_row.empty:
            raise ValueError(f"Missing reference summary for {profile}")
        ref_dict = ref_row.iloc[0].to_dict()
        actual_row = actual_summary[profile]
        for metric in summary_metrics:
            actual_value = float(actual_row[metric]) if metric != "total_trades" else int(actual_row[metric])
            reference_value = float(ref_dict[metric]) if metric != "total_trades" else int(ref_dict[metric])
            matches = actual_value == reference_value if metric == "total_trades" else abs(actual_value - reference_value) <= FLOAT_TOLERANCE
            validation_rows.append(
                {
                    "scope": "summary",
                    "strategy_profile": profile,
                    "side": None,
                    "metric": metric,
                    "actual": actual_value,
                    "reference": reference_value,
                    "delta": 0 if metric == "total_trades" else round(actual_value - reference_value, 6),
                    "matches": matches,
                }
            )

        for side in ("LONG", "SHORT"):
            ref_side_row = reference_side[
                (reference_side["window"] == PRIMARY_WINDOW)
                & (reference_side["strategy_profile"] == profile)
                & (reference_side["side"] == side)
            ]
            if ref_side_row.empty:
                raise ValueError(f"Missing reference side summary for {profile}/{side}")
            ref_side_dict = ref_side_row.iloc[0].to_dict()
            actual_side_row = actual_side[(profile, side)]
            for metric in side_metrics:
                actual_value = float(actual_side_row[metric]) if metric != "trades" else int(actual_side_row[metric])
                reference_value = float(ref_side_dict[metric]) if metric != "trades" else int(ref_side_dict[metric])
                matches = actual_value == reference_value if metric == "trades" else abs(actual_value - reference_value) <= FLOAT_TOLERANCE
                validation_rows.append(
                    {
                        "scope": "side_summary",
                        "strategy_profile": profile,
                        "side": side,
                        "metric": metric,
                        "actual": actual_value,
                        "reference": reference_value,
                        "delta": 0 if metric == "trades" else round(actual_value - reference_value, 6),
                        "matches": matches,
                    }
                )
    return validation_rows


def main() -> None:
    args = parse_args()
    configure_logging()

    champion = args.champion_profile
    challenger = args.challenger_profile
    if champion == challenger:
        raise ValueError("champion-profile and challenger-profile must be different.")

    primary_start = parse_date(args.primary_start)
    secondary_start = parse_date(args.secondary_start)
    requested_end = parse_date(args.end) if args.end else datetime.now(timezone.utc)
    if secondary_start <= primary_start:
        raise ValueError("secondary-start must be later than primary-start")

    service = build_service()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles = (champion, challenger)
    enriched_history: dict[str, dict[str, pd.DataFrame]] = {}
    for profile in profiles:
        print(f"[champion-challenger] preload {profile}", flush=True)
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
        run_window(service=service, args=args, window=primary_window, profiles=profiles, enriched_history=enriched_history),
        run_window(service=service, args=args, window=secondary_window, profiles=profiles, enriched_history=enriched_history),
    ]

    full_summary_rows = [row for result in window_results for row in result["full_summary_rows"]]
    full_trade_rows = [row for result in window_results for row in result["full_trade_rows"]]
    independent_oos_rows = [row for result in window_results for row in result["independent_oos_rows"]]
    independent_oos_trade_rows = [row for result in window_results for row in result["independent_oos_trade_rows"]]
    selection_fold_rows = [row for result in window_results for row in result["selection_fold_rows"]]
    selection_count_rows = [row for result in window_results for row in result["selection_count_rows"]]
    per_fold_delta_rows = [row for result in window_results for row in result["per_fold_delta_rows"]]

    full_trade_frame = build_trade_frame(full_trade_rows)
    independent_oos_trade_frame = build_trade_frame(independent_oos_trade_rows)
    full_side_rows = summarize_trade_distribution(full_trade_frame, group_cols=["window", "strategy_profile", "profile_label", "side"])
    independent_oos_side_rows = summarize_trade_distribution(
        independent_oos_trade_frame,
        group_cols=["window", "strategy_profile", "profile_label", "side"],
    )
    independent_oos_year_rows = summarize_trade_distribution(
        independent_oos_trade_frame,
        group_cols=["window", "strategy_profile", "profile_label", "year"],
    )

    primary_oos_rows = [row for row in independent_oos_rows if row["window"] == PRIMARY_WINDOW]
    secondary_full_side_rows = [row for row in full_side_rows if row["window"] == SECONDARY_WINDOW]
    acceptance_row = build_acceptance_row(
        primary_oos_rows=primary_oos_rows,
        secondary_full_side_rows=secondary_full_side_rows,
        champion=champion,
        challenger=challenger,
    )

    monthly_delta_rows = build_monthly_delta_summary(
        trade_rows=independent_oos_trade_rows,
        window_name=PRIMARY_WINDOW,
        champion=champion,
        challenger=challenger,
    )
    concentration_rows = build_concentration_summary(monthly_delta_rows)

    diff_summary_rows, _, _, _ = run_pair_diff(
        output_dir=output_dir / "winner_diff",
        trade_rows=[row for row in independent_oos_trade_rows if row["window"] == PRIMARY_WINDOW],
        champion=champion,
        challenger=challenger,
    )

    reference_validation_rows: list[dict[str, Any]] = []
    if args.reference_dir:
        reference_validation_rows = build_reference_validation(
            reference_dir=Path(args.reference_dir),
            primary_oos_rows=primary_oos_rows,
            independent_oos_side_rows=independent_oos_side_rows,
            champion=champion,
            challenger=challenger,
        )
        if any(not row["matches"] for row in reference_validation_rows):
            write_csv(output_dir / "reference_validation.csv", reference_validation_rows)
            raise ValueError("Reference replay validation failed. See reference_validation.csv")

    overall_concentration = next((row for row in concentration_rows if row["scope"] == "overall"), None)
    short_concentration = next((row for row in concentration_rows if row["scope"] == "SHORT"), None)
    overall_specialist = bool(overall_concentration and overall_concentration["regime_specialist_tendency"])
    short_specialist = bool(short_concentration and short_concentration["regime_specialist_tendency"])
    specialist_text = (
        "诊断标签：存在 `regime-specialist tendency`，增量比较集中在少数月份或少数年份。"
        if overall_specialist or short_specialist
        else "诊断标签：没有明显 `regime-specialist tendency`，增量不算极端集中。"
    )

    status = acceptance_row["status"]
    conclusion = (
        f"challenger 通过中门槛确认，状态为 `{status}`。"
        if status == "challenger_active"
        else f"challenger 没通过中门槛确认，状态为 `{status}`。"
    )

    write_csv(output_dir / "full_summary.csv", full_summary_rows)
    write_csv(output_dir / "full_side_summary.csv", full_side_rows)
    write_csv(output_dir / "independent_oos_summary.csv", independent_oos_rows)
    write_csv(output_dir / "independent_oos_side_summary.csv", independent_oos_side_rows)
    write_csv(output_dir / "independent_oos_year_summary.csv", independent_oos_year_rows)
    write_csv(output_dir / "per_fold_delta.csv", per_fold_delta_rows)
    write_csv(output_dir / "selection_counts.csv", selection_count_rows)
    write_csv(output_dir / "selection_folds.csv", selection_fold_rows)
    write_csv(output_dir / "monthly_delta_summary.csv", monthly_delta_rows)
    write_csv(output_dir / "concentration_summary.csv", concentration_rows)
    write_csv(output_dir / "acceptance.csv", [acceptance_row])
    if reference_validation_rows:
        write_csv(output_dir / "reference_validation.csv", reference_validation_rows)

    report_lines = [
        "# Champion / Challenger Confirmation",
        "",
        f"- 标的：`{args.symbol}`",
        f"- champion：`{champion}`",
        f"- challenger：`{challenger}`",
        f"- 主窗口：`{primary_window.start.date().isoformat()} -> {resolved_end.date().isoformat()}`",
        f"- 次级窗口：`{secondary_window.start.date().isoformat()} -> {resolved_end.date().isoformat()}`",
        f"- Walk-forward：`{args.train_days}d train / {args.test_days}d test / {args.step_days}d step / {args.scheme}`",
        f"- 最低训练交易数：`{args.min_train_trades}`",
        "- 口径：`confirmed swing`, `LONG 1R -> 3R scaled / SHORT 1.5R fixed`",
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
        "## Acceptance",
        "",
        render_table(
            [acceptance_row],
            [
                ("primary_oos_delta_cumulative_r", "Primary Delta Cum R"),
                ("primary_oos_delta_profit_factor", "Primary Delta PF"),
                ("primary_oos_delta_max_drawdown_r", "Primary Delta MaxDD"),
                ("secondary_long_delta_r", "Secondary LONG Delta"),
                ("status", "Status"),
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
        "## Concentration",
        "",
        render_table(
            concentration_rows,
            [
                ("scope", "Scope"),
                ("total_delta_r", "Total Delta R"),
                ("top3_positive_share_pct", "Top3 Positive Share %"),
                ("dominant_positive_year_share_pct", "Dominant Year Share %"),
                ("regime_specialist_tendency", "Regime-Specialist"),
            ],
        ),
        "",
        "## Conclusion",
        "",
        f"- {conclusion}",
        f"- {specialist_text}",
        f"- winner diff report: `{output_dir / 'winner_diff' / 'report.md'}`",
    ]
    if reference_validation_rows:
        report_lines.append(f"- reference validation: `{output_dir / 'reference_validation.csv'}`")
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    results_json = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": args.symbol,
        "exchange": args.exchange,
        "market_type": args.market_type,
        "champion_profile": champion,
        "challenger_profile": challenger,
        "primary_start": primary_window.start.isoformat(),
        "secondary_start": secondary_window.start.isoformat(),
        "resolved_end": resolved_end.isoformat(),
        "train_days": args.train_days,
        "test_days": args.test_days,
        "step_days": args.step_days,
        "scheme": args.scheme,
        "min_train_trades": args.min_train_trades,
        "status": status,
        "acceptance": acceptance_row,
        "specialist_text": specialist_text,
        "reference_validation_rows": reference_validation_rows,
    }
    (output_dir / "results.json").write_text(json.dumps(results_json, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved champion/challenger confirmation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
