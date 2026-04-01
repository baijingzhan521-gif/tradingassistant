from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.services.carry_basis_research_service import CarryCandidate, PRIMARY_HORIZONS


@dataclass(frozen=True)
class ExecutionScenario:
    label: str
    description: str
    spot_entry_fee_bps: float
    spot_exit_fee_bps: float
    perp_entry_fee_bps: float
    perp_exit_fee_bps: float
    perp_initial_margin_ratio: float

    @property
    def roundtrip_cost_bps(self) -> float:
        return self.spot_entry_fee_bps + self.spot_exit_fee_bps + self.perp_entry_fee_bps + self.perp_exit_fee_bps

    @property
    def capital_required_multiple(self) -> float:
        return 1.0 + self.perp_initial_margin_ratio


class CarryBasisExecutionService:
    def build_scenarios(self) -> list[ExecutionScenario]:
        return [
            ExecutionScenario(
                label="conservative_taker_taker_20im",
                description="Spot taker 10/10 bps, perp taker 4/4 bps, 20% perp margin.",
                spot_entry_fee_bps=10.0,
                spot_exit_fee_bps=10.0,
                perp_entry_fee_bps=4.0,
                perp_exit_fee_bps=4.0,
                perp_initial_margin_ratio=0.20,
            ),
            ExecutionScenario(
                label="hybrid_maker_taker_15im",
                description="Spot maker 2/2 bps, perp taker 4/4 bps, 15% perp margin.",
                spot_entry_fee_bps=2.0,
                spot_exit_fee_bps=2.0,
                perp_entry_fee_bps=4.0,
                perp_exit_fee_bps=4.0,
                perp_initial_margin_ratio=0.15,
            ),
            ExecutionScenario(
                label="optimistic_maker_maker_10im",
                description="Spot maker 2/2 bps, perp maker 1/1 bps, 10% perp margin.",
                spot_entry_fee_bps=2.0,
                spot_exit_fee_bps=2.0,
                perp_entry_fee_bps=1.0,
                perp_exit_fee_bps=1.0,
                perp_initial_margin_ratio=0.10,
            ),
        ]

    def simulate_sequence(
        self,
        frame: pd.DataFrame,
        *,
        candidate: CarryCandidate,
        scenario: ExecutionScenario,
        horizon: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
        if horizon not in PRIMARY_HORIZONS:
            raise ValueError(f"Unsupported horizon {horizon}")

        gross_col = f"gross_carry_bps_{horizon}h"
        eligible = candidate.selector(frame).fillna(False)
        trade_rows: list[dict[str, Any]] = []
        cursor = 0
        eval_hours = int(len(frame))
        while cursor < len(frame) - horizon:
            if not bool(eligible.iloc[cursor]):
                cursor += 1
                continue

            row = frame.iloc[cursor]
            gross_bps = row.get(gross_col)
            if pd.isna(gross_bps):
                cursor += 1
                continue

            exit_idx = cursor + horizon
            if exit_idx >= len(frame):
                break

            exit_row = frame.iloc[exit_idx]
            net_bps = float(gross_bps) - scenario.roundtrip_cost_bps
            roc_bps = net_bps / scenario.capital_required_multiple
            trade_rows.append(
                {
                    "candidate": candidate.label,
                    "scenario": scenario.label,
                    "horizon_hours": horizon,
                    "signal_time": row["timestamp"],
                    "exit_time": exit_row["timestamp"],
                    "basis_proxy_bps": float(row["basis_proxy_bps"]),
                    "funding_rate": float(row["funding_rate"]),
                    "gross_carry_bps": round(float(gross_bps), 4),
                    "roundtrip_cost_bps": round(float(scenario.roundtrip_cost_bps), 4),
                    "net_carry_bps": round(net_bps, 4),
                    "capital_required_multiple": round(float(scenario.capital_required_multiple), 4),
                    "net_roc_bps": round(roc_bps, 4),
                    "hold_hours": horizon,
                }
            )
            cursor = exit_idx

        summary = self._summarize_trade_sequence(
            trade_rows=trade_rows,
            candidate=candidate,
            scenario=scenario,
            horizon=horizon,
            eval_hours=eval_hours,
        )
        monthly_rows = self._summarize_monthly(trade_rows, candidate=candidate, scenario=scenario, horizon=horizon)
        return trade_rows, summary, monthly_rows

    def _summarize_trade_sequence(
        self,
        *,
        trade_rows: list[dict[str, Any]],
        candidate: CarryCandidate,
        scenario: ExecutionScenario,
        horizon: int,
        eval_hours: int,
    ) -> dict[str, Any]:
        if not trade_rows:
            return {
                "candidate": candidate.label,
                "scenario": scenario.label,
                "horizon_hours": horizon,
                "trades": 0,
                "active_hours": 0,
                "utilization_pct": 0.0,
                "round_trips_per_year": 0.0,
                "gross_notional_turnover_x_per_year": 0.0,
                "gross_mean_bps": 0.0,
                "net_mean_bps": 0.0,
                "net_roc_mean_bps": 0.0,
                "net_positive_rate_pct": 0.0,
                "cumulative_roc_pct": 0.0,
                "annualized_roc_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "capital_required_multiple": round(float(scenario.capital_required_multiple), 4),
                "roundtrip_cost_bps": round(float(scenario.roundtrip_cost_bps), 4),
            }

        trades = pd.DataFrame(trade_rows).sort_values("signal_time").reset_index(drop=True)
        active_hours = int(trades["hold_hours"].sum())
        eval_days = eval_hours / 24.0 if eval_hours else 0.0
        utilization_pct = (active_hours / eval_hours) * 100.0 if eval_hours else 0.0
        round_trips_per_year = (len(trades) / eval_days) * 365.0 if eval_days else 0.0
        gross_turnover_x_per_year = (len(trades) * 4.0 / scenario.capital_required_multiple / eval_days) * 365.0 if eval_days else 0.0

        equity = [1.0]
        for ret_bps in trades["net_roc_bps"]:
            equity.append(equity[-1] * (1.0 + (float(ret_bps) / 10000.0)))
        equity_series = pd.Series(equity[1:], index=trades.index, dtype="float64")
        drawdown = equity_series / equity_series.cummax() - 1.0
        cumulative_roc_pct = (equity_series.iloc[-1] - 1.0) * 100.0
        annualized_roc_pct = ((equity_series.iloc[-1] ** (365.0 / eval_days)) - 1.0) * 100.0 if eval_days else 0.0

        return {
            "candidate": candidate.label,
            "scenario": scenario.label,
            "horizon_hours": horizon,
            "trades": int(len(trades)),
            "active_hours": active_hours,
            "utilization_pct": round(float(utilization_pct), 2),
            "round_trips_per_year": round(float(round_trips_per_year), 2),
            "gross_notional_turnover_x_per_year": round(float(gross_turnover_x_per_year), 2),
            "gross_mean_bps": round(float(trades["gross_carry_bps"].mean()), 4),
            "net_mean_bps": round(float(trades["net_carry_bps"].mean()), 4),
            "net_roc_mean_bps": round(float(trades["net_roc_bps"].mean()), 4),
            "net_positive_rate_pct": round(float((trades["net_carry_bps"] > 0).mean() * 100.0), 2),
            "cumulative_roc_pct": round(float(cumulative_roc_pct), 4),
            "annualized_roc_pct": round(float(annualized_roc_pct), 4),
            "max_drawdown_pct": round(float(abs(drawdown.min()) * 100.0), 4),
            "capital_required_multiple": round(float(scenario.capital_required_multiple), 4),
            "roundtrip_cost_bps": round(float(scenario.roundtrip_cost_bps), 4),
        }

    def _summarize_monthly(
        self,
        trade_rows: list[dict[str, Any]],
        *,
        candidate: CarryCandidate,
        scenario: ExecutionScenario,
        horizon: int,
    ) -> list[dict[str, Any]]:
        if not trade_rows:
            return []
        trades = pd.DataFrame(trade_rows).copy()
        trades["signal_time"] = pd.to_datetime(trades["signal_time"], utc=True)
        trades["month"] = trades["signal_time"].dt.strftime("%Y-%m")
        grouped = trades.groupby("month", sort=True).agg(
            trades=("net_carry_bps", "size"),
            net_mean_bps=("net_carry_bps", "mean"),
            net_roc_mean_bps=("net_roc_bps", "mean"),
            net_roc_sum_bps=("net_roc_bps", "sum"),
        )
        rows: list[dict[str, Any]] = []
        for month, row in grouped.reset_index().iterrows():
            rows.append(
                {
                    "candidate": candidate.label,
                    "scenario": scenario.label,
                    "horizon_hours": horizon,
                    "month": row["month"],
                    "trades": int(row["trades"]),
                    "net_mean_bps": round(float(row["net_mean_bps"]), 4),
                    "net_roc_mean_bps": round(float(row["net_roc_mean_bps"]), 4),
                    "net_roc_sum_bps": round(float(row["net_roc_sum_bps"]), 4),
                }
            )
        return rows

    def choose_best_summary(self, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not rows:
            return None
        frame = pd.DataFrame(rows).copy()
        frame = frame[frame["trades"] >= 8].copy()
        if frame.empty:
            return None
        frame["selection_score"] = frame["annualized_roc_pct"] - frame["max_drawdown_pct"] * 0.5
        return frame.sort_values(
            ["selection_score", "annualized_roc_pct", "utilization_pct"],
            ascending=False,
        ).iloc[0].to_dict()
