from __future__ import annotations

import csv
import sys
from datetime import datetime, timedelta, timezone
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
from app.schemas.common import Bias
from app.services.axis_band_risk_label_study_service import AxisBandRiskLabelStudyService
from app.services.strategy_service import StrategyService


SYMBOL = "BTC/USDT:USDT"
EXCHANGE = "binance"
MARKET_TYPE = "perpetual"
PROFILE = "swing_trend_axis_band_state_note_v1"
BASE_PROFILE = "swing_trend_long_regime_gate_v1"
OUTPUT_DIR = ROOT / "artifacts" / "backtests" / "axis_band_risk_label_study"
INPUT_WINDOWS = {
    "two_year": ROOT
    / "artifacts/backtests/level_aware_confirmation_compare_two_year/backtest_long_scaled1_3_short_fixed1_5_20260323T132150Z_trades.csv",
    "full_2020": ROOT
    / "artifacts/backtests/level_aware_confirmation_compare_full_2020/backtest_long_scaled1_3_short_fixed1_5_20260323T135105Z_trades.csv",
}
HORIZONS = (4, 24)


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
    lines = [header, divider]
    for row in rows:
        rendered: list[str] = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def load_window_trades() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for window_label, path in INPUT_WINDOWS.items():
        frame = pd.read_csv(path, parse_dates=["signal_time", "entry_time", "exit_time"])
        frame = frame[frame["strategy_profile"] == BASE_PROFILE].copy()
        frame["window_label"] = window_label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["window_label", "signal_time"]).reset_index(drop=True)


def make_backtest_service() -> BacktestService:
    assumptions = BacktestAssumptions(
        exit_profile="long_scaled1_3_short_fixed1_5",
        take_profit_mode="scaled",
        long_exit={"take_profit_mode": "scaled", "scaled_tp1_r": 1.0, "scaled_tp2_r": 3.0},
        short_exit={"take_profit_mode": "fixed_r", "fixed_take_profit_r": 1.5},
        swing_detection_mode="confirmed",
        cache_dir="artifacts/backtests/cache",
    )
    return BacktestService(
        ohlcv_service=OhlcvService(get_exchange_client_factory()),
        strategy_service=StrategyService(),
        assumptions=assumptions,
    )


def resolve_index(frame: pd.DataFrame, timestamp: pd.Timestamp) -> int | None:
    left_idx = int(frame["timestamp"].searchsorted(timestamp, side="left"))
    if left_idx < len(frame) and frame.iloc[left_idx]["timestamp"] == timestamp:
        return left_idx
    right_idx = int(frame["timestamp"].searchsorted(timestamp, side="right") - 1)
    if right_idx >= 0 and frame.iloc[right_idx]["timestamp"] <= timestamp:
        return right_idx
    return None


def compute_path_metrics(
    *,
    trigger_frame: pd.DataFrame,
    entry_time: pd.Timestamp,
    side: str,
    entry_price: float,
    stop_price: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    risk_unit = abs(float(entry_price) - float(stop_price))
    if risk_unit <= 0:
        for horizon in HORIZONS:
            row[f"forward_close_r_{horizon}h"] = None
            row[f"forward_mfe_r_{horizon}h"] = None
            row[f"forward_mae_r_{horizon}h"] = None
        return row

    entry_idx = resolve_index(trigger_frame, entry_time)
    if entry_idx is None:
        for horizon in HORIZONS:
            row[f"forward_close_r_{horizon}h"] = None
            row[f"forward_mfe_r_{horizon}h"] = None
            row[f"forward_mae_r_{horizon}h"] = None
        return row

    direction = 1.0 if side == "LONG" else -1.0
    for horizon in HORIZONS:
        end_idx = entry_idx + horizon - 1
        if end_idx >= len(trigger_frame):
            row[f"forward_close_r_{horizon}h"] = None
            row[f"forward_mfe_r_{horizon}h"] = None
            row[f"forward_mae_r_{horizon}h"] = None
            continue

        path = trigger_frame.iloc[entry_idx : end_idx + 1]
        future_close = float(trigger_frame.iloc[end_idx]["close"])
        high_watermark = float(path["high"].max())
        low_watermark = float(path["low"].min())

        if side == "LONG":
            mfe = (high_watermark - entry_price) / risk_unit
            mae = (entry_price - low_watermark) / risk_unit
        else:
            mfe = (entry_price - low_watermark) / risk_unit
            mae = (high_watermark - entry_price) / risk_unit

        row[f"forward_close_r_{horizon}h"] = round(float(direction * ((future_close - entry_price) / risk_unit)), 4)
        row[f"forward_mfe_r_{horizon}h"] = round(float(mfe), 4)
        row[f"forward_mae_r_{horizon}h"] = round(float(mae), 4)
    return row


def enrich_trades_with_risk_labels(trades: pd.DataFrame) -> pd.DataFrame:
    service = make_backtest_service()
    strategy = service.strategy_service.build_strategy(PROFILE)

    start = trades["signal_time"].min().to_pydatetime()
    end = (trades["entry_time"].max() + pd.Timedelta(hours=max(HORIZONS))).to_pydatetime()
    frames = service.prepare_history(
        exchange=EXCHANGE,
        market_type=MARKET_TYPE,
        symbol=SYMBOL,
        strategy_profile=PROFILE,
        start=start,
        end=end,
    )
    enriched = service._prepare_enriched_frames(PROFILE, frames)
    trigger_tf = str(strategy.window_config["trigger_timeframe"])
    setup_tf = str(strategy.window_config["setup_timeframe"])
    trigger_frame = enriched[trigger_tf]

    rows: list[dict[str, Any]] = []
    for trade in trades.itertuples(index=False):
        signal_time = pd.Timestamp(trade.signal_time)
        current_indices: dict[str, int] = {}
        missing = False
        for timeframe in strategy.required_timeframes:
            idx = resolve_index(enriched[timeframe], signal_time)
            if idx is None:
                missing = True
                break
            current_indices[timeframe] = idx
        if missing:
            continue

        prepared = {
            timeframe: service._build_snapshot(strategy, timeframe, enriched[timeframe], current_indices[timeframe])
            for timeframe in current_indices
        }
        try:
            higher_bias = Bias(str(trade.higher_bias))
        except ValueError:
            higher_bias = Bias.BULLISH if trade.side == "LONG" else Bias.BEARISH
        overlay = strategy._compute_axis_band_state_note(higher_bias=higher_bias, setup_ctx=prepared[setup_tf])
        path_metrics = compute_path_metrics(
            trigger_frame=trigger_frame,
            entry_time=pd.Timestamp(trade.entry_time),
            side=str(trade.side),
            entry_price=float(trade.entry_price),
            stop_price=float(trade.stop_price),
        )
        rows.append(
            {
                "window_label": str(trade.window_label),
                "symbol": str(trade.symbol),
                "strategy_profile": str(trade.strategy_profile),
                "side": str(trade.side),
                "signal_time": pd.Timestamp(trade.signal_time),
                "entry_time": pd.Timestamp(trade.entry_time),
                "exit_time": pd.Timestamp(trade.exit_time),
                "higher_bias": str(trade.higher_bias),
                "trend_strength": int(trade.trend_strength),
                "entry_price": float(trade.entry_price),
                "stop_price": float(trade.stop_price),
                "pnl_r": float(trade.pnl_r),
                "bars_held": int(trade.bars_held),
                "exit_reason": str(trade.exit_reason),
                "tp1_hit": bool(trade.tp1_hit),
                "tp2_hit": bool(trade.tp2_hit),
                "risk_group": "active" if overlay["active"] else "inactive",
                "risk_label": overlay["label"] or "none",
                "risk_severity": str(overlay["severity"]),
                "risk_extreme": bool(overlay["extreme"]),
                "axis_distance_vol": float(overlay["axis_distance_vol"]),
                "band_position": float(overlay["band_position"]),
                **path_metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["window_label", "signal_time"]).reset_index(drop=True)


def build_report(
    *,
    trades_with_labels: pd.DataFrame,
    trade_summary: list[dict[str, Any]],
    path_summary: list[dict[str, Any]],
    edge_summary: list[dict[str, Any]],
    monthly_stability_summary: list[dict[str, Any]],
) -> str:
    total_rows = len(trades_with_labels)
    two_year_rows = int((trades_with_labels["window_label"] == "two_year").sum()) if not trades_with_labels.empty else 0
    full_rows = int((trades_with_labels["window_label"] == "full_2020").sum()) if not trades_with_labels.empty else 0

    two_year_edge = [row for row in edge_summary if row["window_label"] == "two_year"]
    full_edge = [row for row in edge_summary if row["window_label"] == "full_2020"]
    stability_focus = [
        row
        for row in monthly_stability_summary
        if row["window_label"] == "full_2020" and row["metric"] in {"pnl_r", "forward_close_r_24h"}
    ]

    verdict_lines: list[str] = []
    for row in edge_summary:
        path_delta = row.get("delta_forward_close_r_24h")
        expectancy_delta = row.get("delta_expectancy_r")
        adverse_delta = row.get("delta_mae_r_24h")
        if path_delta is None or expectancy_delta is None or adverse_delta is None:
            verdict = "样本不足，不能判断"
        elif expectancy_delta < 0 and path_delta < 0 and adverse_delta > 0:
            verdict = "风险标签方向上成立"
        elif expectancy_delta > 0 and path_delta > 0:
            verdict = "更像顺势扩张，不像风险警示"
        else:
            verdict = "方向混合，稳定性不足"
        verdict_lines.append(
            f"- `{row['window_label']} / {row['side']}`: {verdict}。"
            f" Active share {row['active_share_pct']:.2f}%，"
            f" `ΔExp {row['delta_expectancy_r']:.4f}R`，"
            f" `ΔFwd24h {path_delta if path_delta is not None else 'NA'}R`，"
            f" `ΔMAE24h {adverse_delta if adverse_delta is not None else 'NA'}R`。"
        )

    return "\n".join(
        [
            "# Axis/Band 风险标签轻量研究",
            "",
            "## 研究口径",
            "",
            "- 不改主线 entry / trigger / exit，只给主线成交交易在 `signal_time` 打 `pullback_risk / rebound_risk` 标签。",
            "- 风险标签只来自 `4H setup` 上的 `axis_distance_vol + band_position`，口径与 `swing_trend_axis_band_state_note_v1` 完全一致。",
            "- 前瞻路径按 `entry_time` 之后的 `1H` bars 计算：`4h/24h` 使用首 `4/24` 根 `1H` bar 的 close/MFE/MAE，单位统一换算成初始风险 `R`。",
            "- 交易分布严格按 `LONG / SHORT` 分开，不做跨方向混合比较。",
            "",
            "## 样本覆盖",
            "",
            f"- 总标签交易数：`{total_rows}`",
            f"- `two_year` 交易数：`{two_year_rows}`",
            f"- `full_2020` 交易数：`{full_rows}`",
            "",
            "## 交易分布摘要",
            "",
            markdown_table(
                trade_summary,
                [
                    ("window_label", "Window"),
                    ("side", "Side"),
                    ("risk_group", "Group"),
                    ("trades", "Trades"),
                    ("win_rate_pct", "Win %"),
                    ("expectancy_r", "Exp R"),
                    ("cumulative_r", "Cum R"),
                    ("stop_loss_rate_pct", "SL %"),
                    ("tp2_hit_rate_pct", "TP2 %"),
                    ("avg_bars_held", "Bars"),
                ],
            ),
            "",
            "## 路径表现摘要",
            "",
            markdown_table(
                path_summary,
                [
                    ("window_label", "Window"),
                    ("side", "Side"),
                    ("risk_group", "Group"),
                    ("trades", "Trades"),
                    ("mean_forward_close_r_4h", "Mean Close 4h R"),
                    ("mean_forward_close_r_24h", "Mean Close 24h R"),
                    ("mean_mfe_r_24h", "Mean MFE 24h R"),
                    ("mean_mae_r_24h", "Mean MAE 24h R"),
                    ("positive_forward_rate_pct_24h", "Pos 24h %"),
                ],
            ),
            "",
            "## Active vs Inactive 差值",
            "",
            markdown_table(
                edge_summary,
                [
                    ("window_label", "Window"),
                    ("side", "Side"),
                    ("active_trades", "Active"),
                    ("inactive_trades", "Inactive"),
                    ("active_share_pct", "Active %"),
                    ("delta_expectancy_r", "ΔExp R"),
                    ("delta_stop_loss_rate_pct", "ΔSL %"),
                    ("delta_forward_close_r_4h", "ΔClose 4h R"),
                    ("delta_forward_close_r_24h", "ΔClose 24h R"),
                    ("delta_mae_r_24h", "ΔMAE 24h R"),
                ],
            ),
            "",
            "## 月度稳定性摘要",
            "",
            markdown_table(
                stability_focus,
                [
                    ("window_label", "Window"),
                    ("side", "Side"),
                    ("metric", "Metric"),
                    ("overall_active_minus_inactive", "Overall Δ"),
                    ("months_with_both_groups", "Months"),
                    ("same_sign_months", "Same Sign"),
                    ("same_sign_month_rate_pct", "Same Sign %"),
                ],
            ),
            "",
            "## 判断",
            "",
            *verdict_lines,
            "",
            "## 边界",
            "",
            "- 这次研究回答的是“风险标签是不是有解释力”，不是“它该不该直接接进 exit 规则”。",
            "- `two_year` 与 `full_2020` 不是独立样本，前者是后者子窗口，所以所谓稳定性仍然主要是方向性证据。",
            "- 月度稳定性若样本很稀，最多只能当弱信号，不该被当成硬证据。",
            "",
            "## 分窗口差值速览",
            "",
            "### Two-Year",
            "",
            markdown_table(
                two_year_edge,
                [
                    ("side", "Side"),
                    ("active_share_pct", "Active %"),
                    ("delta_expectancy_r", "ΔExp R"),
                    ("delta_forward_close_r_24h", "ΔClose 24h R"),
                    ("delta_mae_r_24h", "ΔMAE 24h R"),
                ],
            ),
            "",
            "### Full 2020",
            "",
            markdown_table(
                full_edge,
                [
                    ("side", "Side"),
                    ("active_share_pct", "Active %"),
                    ("delta_expectancy_r", "ΔExp R"),
                    ("delta_forward_close_r_24h", "ΔClose 24h R"),
                    ("delta_mae_r_24h", "ΔMAE 24h R"),
                ],
            ),
            "",
        ]
    ).strip() + "\n"


def main() -> None:
    configure_logging()
    ensure_output_dir()

    trades = load_window_trades()
    labeled_trades = enrich_trades_with_risk_labels(trades)

    study_service = AxisBandRiskLabelStudyService()
    trade_summary = study_service.summarize_trade_distribution(labeled_trades)
    path_summary = study_service.summarize_path_distribution(labeled_trades)
    edge_summary = study_service.summarize_edge(labeled_trades)
    monthly_rows, monthly_stability_summary = study_service.summarize_monthly_stability(labeled_trades)

    labeled_trades.to_csv(OUTPUT_DIR / "trades_with_risk_labels.csv", index=False)
    write_csv(OUTPUT_DIR / "trade_summary.csv", trade_summary)
    write_csv(OUTPUT_DIR / "path_summary.csv", path_summary)
    write_csv(OUTPUT_DIR / "edge_summary.csv", edge_summary)
    write_csv(OUTPUT_DIR / "monthly_path_diff.csv", monthly_rows)
    write_csv(OUTPUT_DIR / "monthly_stability_summary.csv", monthly_stability_summary)
    (OUTPUT_DIR / "report.md").write_text(
        build_report(
            trades_with_labels=labeled_trades,
            trade_summary=trade_summary,
            path_summary=path_summary,
            edge_summary=edge_summary,
            monthly_stability_summary=monthly_stability_summary,
        ),
        encoding="utf-8",
    )

    print(f"Saved labeled trades CSV: {OUTPUT_DIR / 'trades_with_risk_labels.csv'}")
    print(f"Saved trade summary CSV: {OUTPUT_DIR / 'trade_summary.csv'}")
    print(f"Saved path summary CSV: {OUTPUT_DIR / 'path_summary.csv'}")
    print(f"Saved edge summary CSV: {OUTPUT_DIR / 'edge_summary.csv'}")
    print(f"Saved monthly stability summary CSV: {OUTPUT_DIR / 'monthly_stability_summary.csv'}")
    print(f"Saved report: {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
