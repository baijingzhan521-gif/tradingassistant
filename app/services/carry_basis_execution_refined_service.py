from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.services.carry_basis_research_service import CarryCandidate, PRIMARY_HORIZONS


HOURS_PER_YEAR = 24.0 * 365.0


@dataclass(frozen=True)
class RefinedExecutionScenario:
    label: str
    description: str
    spot_entry_fee_bps: float
    spot_exit_fee_bps: float
    spot_entry_slippage_bps: float
    spot_exit_slippage_bps: float
    perp_entry_fee_bps: float
    perp_exit_fee_bps: float
    perp_entry_slippage_bps: float
    perp_exit_slippage_bps: float
    perp_initial_margin_ratio: float
    capital_mode: str
    annual_opportunity_cost_pct: float

    @property
    def spot_roundtrip_fee_bps(self) -> float:
        return self.spot_entry_fee_bps + self.spot_exit_fee_bps

    @property
    def perp_roundtrip_fee_bps(self) -> float:
        return self.perp_entry_fee_bps + self.perp_exit_fee_bps

    @property
    def fee_roundtrip_bps(self) -> float:
        return self.spot_roundtrip_fee_bps + self.perp_roundtrip_fee_bps

    @property
    def spot_roundtrip_slippage_bps(self) -> float:
        return self.spot_entry_slippage_bps + self.spot_exit_slippage_bps

    @property
    def perp_roundtrip_slippage_bps(self) -> float:
        return self.perp_entry_slippage_bps + self.perp_exit_slippage_bps

    @property
    def slippage_roundtrip_bps(self) -> float:
        return self.spot_roundtrip_slippage_bps + self.perp_roundtrip_slippage_bps

    @property
    def capital_required_multiple(self) -> float:
        if self.capital_mode == "pooled":
            # Optimistic unified-collateral assumption: the spot inventory fully covers perp IM.
            return 1.0
        if self.capital_mode == "segregated":
            return 1.0 + self.perp_initial_margin_ratio
        raise ValueError(f"Unsupported capital mode: {self.capital_mode}")

    def opportunity_cost_bps(self, horizon_hours: int) -> float:
        return self.annual_opportunity_cost_pct * 100.0 * (float(horizon_hours) / HOURS_PER_YEAR) * self.capital_required_multiple

    def all_in_cost_bps(self, horizon_hours: int) -> float:
        return self.fee_roundtrip_bps + self.slippage_roundtrip_bps + self.opportunity_cost_bps(horizon_hours)


class CarryBasisExecutionRefinedService:
    def build_focus_scenarios(self) -> list[RefinedExecutionScenario]:
        base_common = {
            "spot_entry_fee_bps": 2.0,
            "spot_exit_fee_bps": 2.0,
            "spot_entry_slippage_bps": 3.0,
            "spot_exit_slippage_bps": 3.0,
            "perp_entry_fee_bps": 4.0,
            "perp_exit_fee_bps": 4.0,
            "perp_entry_slippage_bps": 1.0,
            "perp_exit_slippage_bps": 1.0,
            "perp_initial_margin_ratio": 0.15,
        }
        return [
            RefinedExecutionScenario(
                label="legacy_proxy_baseline",
                description="Control: original hybrid 15im model, no slippage, segregated capital, no opportunity cost.",
                spot_entry_fee_bps=2.0,
                spot_exit_fee_bps=2.0,
                spot_entry_slippage_bps=0.0,
                spot_exit_slippage_bps=0.0,
                perp_entry_fee_bps=4.0,
                perp_exit_fee_bps=4.0,
                perp_entry_slippage_bps=0.0,
                perp_exit_slippage_bps=0.0,
                perp_initial_margin_ratio=0.15,
                capital_mode="segregated",
                annual_opportunity_cost_pct=0.0,
            ),
            RefinedExecutionScenario(
                label="realistic_base_pooled_0opp",
                description="Explicit spot/perp slippage, pooled capital, 0% annual opportunity cost.",
                capital_mode="pooled",
                annual_opportunity_cost_pct=0.0,
                **base_common,
            ),
            RefinedExecutionScenario(
                label="realistic_base_segregated_0opp",
                description="Explicit spot/perp slippage, segregated capital, 0% annual opportunity cost.",
                capital_mode="segregated",
                annual_opportunity_cost_pct=0.0,
                **base_common,
            ),
            RefinedExecutionScenario(
                label="realistic_base_pooled_4opp",
                description="Explicit spot/perp slippage, pooled capital, 4% annual opportunity cost.",
                capital_mode="pooled",
                annual_opportunity_cost_pct=4.0,
                **base_common,
            ),
            RefinedExecutionScenario(
                label="realistic_base_segregated_4opp",
                description="Explicit spot/perp slippage, segregated capital, 4% annual opportunity cost.",
                capital_mode="segregated",
                annual_opportunity_cost_pct=4.0,
                **base_common,
            ),
            RefinedExecutionScenario(
                label="realistic_base_pooled_8opp",
                description="Explicit spot/perp slippage, pooled capital, 8% annual opportunity cost.",
                capital_mode="pooled",
                annual_opportunity_cost_pct=8.0,
                **base_common,
            ),
            RefinedExecutionScenario(
                label="realistic_base_segregated_8opp",
                description="Explicit spot/perp slippage, segregated capital, 8% annual opportunity cost.",
                capital_mode="segregated",
                annual_opportunity_cost_pct=8.0,
                **base_common,
            ),
        ]

    def simulate_sequence(
        self,
        frame: pd.DataFrame,
        *,
        candidate: CarryCandidate,
        scenario: RefinedExecutionScenario,
        horizon: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
        if horizon not in PRIMARY_HORIZONS:
            raise ValueError(f"Unsupported horizon {horizon}")

        trade_rows: list[dict[str, Any]] = []
        eligible = candidate.selector(frame).fillna(False)
        funding_sums = self._future_funding_sums(frame, horizon)
        eval_hours = int(len(frame))

        cursor = 0
        while cursor < len(frame) - horizon:
            if not bool(eligible.iloc[cursor]):
                cursor += 1
                continue

            exit_idx = cursor + horizon
            if exit_idx >= len(frame):
                break

            row = frame.iloc[cursor]
            exit_row = frame.iloc[exit_idx]
            funding_sum_bps = funding_sums.iloc[cursor]
            if pd.isna(funding_sum_bps):
                cursor += 1
                continue

            spot_entry_exec = float(row["index_close"]) * (1.0 + scenario.spot_entry_slippage_bps / 10000.0)
            spot_exit_exec = float(exit_row["index_close"]) * (1.0 - scenario.spot_exit_slippage_bps / 10000.0)
            spot_leg_exec_bps = ((spot_exit_exec / spot_entry_exec) - 1.0) * 10000.0

            perp_entry_exec = float(row["mark_close"]) * (1.0 - scenario.perp_entry_slippage_bps / 10000.0)
            perp_exit_exec = float(exit_row["mark_close"]) * (1.0 + scenario.perp_exit_slippage_bps / 10000.0)
            perp_leg_exec_bps = -(((perp_exit_exec / perp_entry_exec) - 1.0) * 10000.0)

            proxy_gross_carry_bps = float(row[f"gross_carry_bps_{horizon}h"])
            execution_gross_carry_bps = spot_leg_exec_bps + perp_leg_exec_bps + float(funding_sum_bps)
            fee_cost_bps = scenario.fee_roundtrip_bps
            net_after_exec_costs_bps = execution_gross_carry_bps - fee_cost_bps
            opportunity_cost_bps = scenario.opportunity_cost_bps(horizon)
            net_after_all_costs_bps = net_after_exec_costs_bps - opportunity_cost_bps
            capital_required_multiple = scenario.capital_required_multiple
            net_roc_bps = net_after_all_costs_bps / capital_required_multiple

            trade_rows.append(
                {
                    "candidate": candidate.label,
                    "scenario": scenario.label,
                    "horizon_hours": horizon,
                    "signal_time": row["timestamp"],
                    "exit_time": exit_row["timestamp"],
                    "basis_proxy_bps": round(float(row["basis_proxy_bps"]), 4),
                    "funding_rate": round(float(row["funding_rate"]), 8),
                    "proxy_gross_carry_bps": round(proxy_gross_carry_bps, 4),
                    "execution_gross_carry_bps": round(execution_gross_carry_bps, 4),
                    "fee_cost_bps": round(fee_cost_bps, 4),
                    "slippage_cost_bps": round(scenario.slippage_roundtrip_bps, 4),
                    "opportunity_cost_bps": round(opportunity_cost_bps, 4),
                    "all_in_cost_bps": round(scenario.all_in_cost_bps(horizon), 4),
                    "net_after_all_costs_bps": round(net_after_all_costs_bps, 4),
                    "capital_mode": scenario.capital_mode,
                    "capital_required_multiple": round(capital_required_multiple, 4),
                    "net_roc_bps": round(net_roc_bps, 4),
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

    def _future_funding_sums(self, frame: pd.DataFrame, horizon: int) -> pd.Series:
        funding_events = frame["funding_rate_event"].fillna(0.0).astype("float64") * 10000.0
        values = funding_events.to_numpy(dtype=float)
        result = [float("nan")] * len(values)
        if len(values) <= horizon:
            return pd.Series(result, index=frame.index, dtype="float64")
        cumulative = pd.Series(values).cumsum().to_numpy(dtype=float)
        for idx in range(len(values) - horizon):
            left = cumulative[idx]
            right = cumulative[idx + horizon] if idx + horizon < len(cumulative) else cumulative[-1]
            result[idx] = right - left
        return pd.Series(result, index=frame.index, dtype="float64")

    def _summarize_trade_sequence(
        self,
        *,
        trade_rows: list[dict[str, Any]],
        candidate: CarryCandidate,
        scenario: RefinedExecutionScenario,
        horizon: int,
        eval_hours: int,
    ) -> dict[str, Any]:
        if not trade_rows:
            return {
                "candidate": candidate.label,
                "scenario": scenario.label,
                "horizon_hours": horizon,
                "capital_mode": scenario.capital_mode,
                "annual_opportunity_cost_pct": round(float(scenario.annual_opportunity_cost_pct), 2),
                "trades": 0,
                "active_hours": 0,
                "utilization_pct": 0.0,
                "round_trips_per_year": 0.0,
                "proxy_gross_mean_bps": 0.0,
                "execution_gross_mean_bps": 0.0,
                "net_mean_bps": 0.0,
                "net_roc_mean_bps": 0.0,
                "cumulative_roc_pct": 0.0,
                "annualized_roc_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "fee_cost_bps": round(float(scenario.fee_roundtrip_bps), 4),
                "slippage_cost_bps": round(float(scenario.slippage_roundtrip_bps), 4),
                "opportunity_cost_bps": round(float(scenario.opportunity_cost_bps(horizon)), 4),
                "all_in_cost_bps": round(float(scenario.all_in_cost_bps(horizon)), 4),
                "capital_required_multiple": round(float(scenario.capital_required_multiple), 4),
            }

        trades = pd.DataFrame(trade_rows).sort_values("signal_time").reset_index(drop=True)
        active_hours = int(trades["hold_hours"].sum())
        eval_days = eval_hours / 24.0 if eval_hours else 0.0
        utilization_pct = (active_hours / eval_hours) * 100.0 if eval_hours else 0.0
        round_trips_per_year = (len(trades) / eval_days) * 365.0 if eval_days else 0.0

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
            "capital_mode": scenario.capital_mode,
            "annual_opportunity_cost_pct": round(float(scenario.annual_opportunity_cost_pct), 2),
            "trades": int(len(trades)),
            "active_hours": active_hours,
            "utilization_pct": round(float(utilization_pct), 2),
            "round_trips_per_year": round(float(round_trips_per_year), 2),
            "proxy_gross_mean_bps": round(float(trades["proxy_gross_carry_bps"].mean()), 4),
            "execution_gross_mean_bps": round(float(trades["execution_gross_carry_bps"].mean()), 4),
            "net_mean_bps": round(float(trades["net_after_all_costs_bps"].mean()), 4),
            "net_roc_mean_bps": round(float(trades["net_roc_bps"].mean()), 4),
            "cumulative_roc_pct": round(float(cumulative_roc_pct), 4),
            "annualized_roc_pct": round(float(annualized_roc_pct), 4),
            "max_drawdown_pct": round(float(abs(drawdown.min()) * 100.0), 4),
            "fee_cost_bps": round(float(scenario.fee_roundtrip_bps), 4),
            "slippage_cost_bps": round(float(scenario.slippage_roundtrip_bps), 4),
            "opportunity_cost_bps": round(float(scenario.opportunity_cost_bps(horizon)), 4),
            "all_in_cost_bps": round(float(scenario.all_in_cost_bps(horizon)), 4),
            "capital_required_multiple": round(float(scenario.capital_required_multiple), 4),
        }

    def _summarize_monthly(
        self,
        trade_rows: list[dict[str, Any]],
        *,
        candidate: CarryCandidate,
        scenario: RefinedExecutionScenario,
        horizon: int,
    ) -> list[dict[str, Any]]:
        if not trade_rows:
            return []
        trades = pd.DataFrame(trade_rows).copy()
        trades["signal_time"] = pd.to_datetime(trades["signal_time"], utc=True)
        trades["month"] = trades["signal_time"].dt.strftime("%Y-%m")
        grouped = trades.groupby("month", sort=True).agg(
            trades=("net_after_all_costs_bps", "size"),
            net_mean_bps=("net_after_all_costs_bps", "mean"),
            net_roc_sum_bps=("net_roc_bps", "sum"),
        )
        rows: list[dict[str, Any]] = []
        for _, row in grouped.reset_index().iterrows():
            rows.append(
                {
                    "candidate": candidate.label,
                    "scenario": scenario.label,
                    "horizon_hours": horizon,
                    "capital_mode": scenario.capital_mode,
                    "annual_opportunity_cost_pct": round(float(scenario.annual_opportunity_cost_pct), 2),
                    "month": row["month"],
                    "trades": int(row["trades"]),
                    "net_mean_bps": round(float(row["net_mean_bps"]), 4),
                    "net_roc_sum_bps": round(float(row["net_roc_sum_bps"]), 4),
                }
            )
        return rows
