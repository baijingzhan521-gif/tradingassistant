from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.core.config import BASE_DIR
from app.core.exceptions import TradingAssistantError
from app.schemas.response import BacktestTradeSnapshot, WorkspaceBacktestTradeBookResponse


class BacktestArtifactService:
    def __init__(self, artifact_dir: Path | None = None) -> None:
        self.artifact_dir = artifact_dir or (BASE_DIR / "artifacts" / "backtests" / "btc_confirmed_swing_ablation")

    def get_best_btc_trade_book(self) -> WorkspaceBacktestTradeBookResponse:
        ranking_csv = self._latest_ranking_csv()
        trades_csv = ranking_csv.with_name(f"{ranking_csv.stem}_trades.csv")
        if not trades_csv.exists():
            raise TradingAssistantError(f"Backtest trades artifact not found: {trades_csv}", status_code=404)

        ranking_df = pd.read_csv(ranking_csv)
        if ranking_df.empty:
            raise TradingAssistantError(f"Backtest ranking artifact is empty: {ranking_csv}", status_code=404)
        best = ranking_df.iloc[0]
        best_profile = str(best["profile"])

        trades_df = pd.read_csv(trades_csv)
        if trades_df.empty:
            raise TradingAssistantError(f"Backtest trades artifact is empty: {trades_csv}", status_code=404)

        profile_column = "profile" if "profile" in trades_df.columns else "strategy_profile"
        filtered = trades_df.loc[trades_df[profile_column] == best_profile].copy()
        if filtered.empty:
            raise TradingAssistantError(
                f"Backtest trades artifact does not contain profile '{best_profile}'", status_code=404
            )

        filtered["signal_time"] = pd.to_datetime(filtered["signal_time"], utc=True)
        filtered["entry_time"] = pd.to_datetime(filtered["entry_time"], utc=True)
        filtered["exit_time"] = pd.to_datetime(filtered["exit_time"], utc=True)
        filtered = filtered.sort_values("entry_time").reset_index(drop=True)
        filtered["sequence"] = range(1, len(filtered) + 1)
        filtered["cumulative_r_after_trade"] = filtered["pnl_r"].cumsum().round(4)

        long_trades = filtered.loc[filtered["side"] == "LONG", "pnl_r"]
        short_trades = filtered.loc[filtered["side"] == "SHORT", "pnl_r"]

        return WorkspaceBacktestTradeBookResponse(
            dataset="confirmed_swing_ablation_latest_best_profile",
            dataset_generated_at=ranking_csv.stem.removeprefix("confirmed_swing_ablation_"),
            symbol=str(filtered.iloc[0]["symbol"]),
            strategy_profile=best_profile,
            swing_detection_mode="confirmed",
            exit_profile_label="LONG scaled 1R -> 3R, SHORT fixed 1.5R",
            total_trades=int(best["total_trades"]),
            profit_factor=float(best["profit_factor"]),
            expectancy_r=float(best["expectancy_r"]),
            cumulative_r=float(best["cumulative_r"]),
            max_drawdown_r=float(best["max_drawdown_r"]),
            long_trades=int(len(long_trades)),
            long_r=float(long_trades.sum()) if not long_trades.empty else 0.0,
            short_trades=int(len(short_trades)),
            short_r=float(short_trades.sum()) if not short_trades.empty else 0.0,
            ranking_source_csv=str(ranking_csv.relative_to(BASE_DIR)),
            trades_source_csv=str(trades_csv.relative_to(BASE_DIR)),
            trades=[
                BacktestTradeSnapshot(
                    sequence=int(row["sequence"]),
                    symbol=str(row["symbol"]),
                    strategy_profile=str(row["strategy_profile"]),
                    side=str(row["side"]),
                    signal_time=row["signal_time"].to_pydatetime(),
                    entry_time=row["entry_time"].to_pydatetime(),
                    exit_time=row["exit_time"].to_pydatetime(),
                    entry_price=float(row["entry_price"]),
                    exit_price=float(row["exit_price"]),
                    stop_price=float(row["stop_price"]),
                    tp1_price=float(row["tp1_price"]),
                    tp2_price=float(row["tp2_price"]),
                    bars_held=int(row["bars_held"]),
                    exit_reason=str(row["exit_reason"]),
                    confidence=int(row["confidence"]),
                    tp1_hit=bool(row["tp1_hit"]),
                    tp2_hit=bool(row["tp2_hit"]),
                    pnl_pct=float(row["pnl_pct"]),
                    pnl_r=float(row["pnl_r"]),
                    cumulative_r_after_trade=float(row["cumulative_r_after_trade"]),
                )
                for _, row in filtered.iterrows()
            ],
        )

    def _latest_ranking_csv(self) -> Path:
        candidates = sorted(
            [
                path
                for path in self.artifact_dir.glob("confirmed_swing_ablation_*.csv")
                if not path.name.endswith("_trades.csv")
            ]
        )
        if not candidates:
            raise TradingAssistantError(
                f"No confirmed swing ablation artifacts found under {self.artifact_dir}",
                status_code=404,
            )
        return candidates[-1]
