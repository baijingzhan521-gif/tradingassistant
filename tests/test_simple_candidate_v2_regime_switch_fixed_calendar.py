from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from app.backtesting.service import BacktestTrade
import scripts.run_simple_candidate_v2_regime_switch_fixed_calendar as module


def make_trade(*, signal_time: str, pnl_pct: float, pnl_r: float, side: str = "LONG") -> BacktestTrade:
    return BacktestTrade(
        symbol="BTC/USDT:USDT",
        strategy_profile="dummy",
        side=side,
        higher_bias="bullish",
        trend_strength=80,
        signal_time=signal_time,
        entry_time=signal_time,
        exit_time=signal_time,
        entry_price=100.0,
        exit_price=101.0,
        stop_price=99.0,
        tp1_price=102.0,
        tp2_price=103.0,
        bars_held=4,
        exit_reason="tp2",
        confidence=70,
        tp1_hit=True,
        tp2_hit=True,
        pnl_pct=pnl_pct,
        pnl_r=pnl_r,
        gross_pnl_quote=1.0,
        fees_quote=0.1,
    )


def test_build_switch_rows_keeps_pre_cutover_simple_and_post_cutover_challenger() -> None:
    switch_date = datetime(2024, 3, 19, tzinfo=timezone.utc)
    simple_trades = [
        make_trade(signal_time="2024-03-18T00:00:00+00:00", pnl_pct=5.0, pnl_r=1.0),
        make_trade(signal_time="2024-03-20T00:00:00+00:00", pnl_pct=5.0, pnl_r=1.0),
    ]
    challenger_trades = [
        make_trade(signal_time="2024-03-18T00:00:00+00:00", pnl_pct=2.0, pnl_r=0.5),
        make_trade(signal_time="2024-03-20T00:00:00+00:00", pnl_pct=2.0, pnl_r=0.5),
    ]

    rows = module.build_switch_rows(
        simple_trades=simple_trades,
        challenger_trades=challenger_trades,
        cost_scenario="base",
        window="full_2020",
        switch_date=switch_date,
    )

    assert len(rows) == 2
    assert rows[0]["segment"] == "pre_cutover"
    assert rows[1]["segment"] == "post_cutover"


def test_build_summary_rows_uses_window_specific_cagr() -> None:
    frame = pd.DataFrame(
        [
            {
                "cost_scenario": "base",
                "window": "full_2020",
                "scenario_kind": "always_simple_candidate_v2",
                "scenario_label": "simple",
                "segment": "pre_cutover",
                "signal_time": pd.Timestamp("2024-01-01T00:00:00Z"),
                "entry_time": pd.Timestamp("2024-01-01T00:00:00Z"),
                "exit_time": pd.Timestamp("2024-01-01T01:00:00Z"),
                "pnl_pct": 10.0,
                "pnl_r": 1.0,
                "bars_held": 4,
                "side": "LONG",
            },
            {
                "cost_scenario": "base",
                "window": "full_2020",
                "scenario_kind": "always_simple_candidate_v2",
                "scenario_label": "simple",
                "segment": "post_cutover",
                "signal_time": pd.Timestamp("2024-06-01T00:00:00Z"),
                "entry_time": pd.Timestamp("2024-06-01T00:00:00Z"),
                "exit_time": pd.Timestamp("2024-06-01T01:00:00Z"),
                "pnl_pct": -5.0,
                "pnl_r": -0.5,
                "bars_held": 4,
                "side": "SHORT",
            },
        ]
    )

    rows = module.build_summary_rows(
        frame,
        {
            "full_2020": (datetime(2020, 1, 1, tzinfo=timezone.utc), datetime(2026, 3, 30, tzinfo=timezone.utc)),
            "two_year": (datetime(2024, 3, 19, tzinfo=timezone.utc), datetime(2026, 3, 30, tzinfo=timezone.utc)),
        },
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["trades"] == 2
    assert row["geometric_return_pct"] != 0.0
    assert "cagr_pct" in row


def test_build_switch_decision_promotes_when_gate_passes() -> None:
    summary_rows = [
        {
            "cost_scenario": "base",
            "window": "full_2020",
            "scenario_kind": "always_challenger_managed",
            "scenario_label": "challenger",
            "trades": 10,
            "profit_factor": 1.10,
            "cum_r": 20.0,
            "max_dd_r": 10.0,
            "geometric_return_pct": 15.0,
            "cagr_pct": 5.0,
            "window_start": "2020-01-01",
            "window_end": "2026-03-30",
        },
        {
            "cost_scenario": "base",
            "window": "full_2020",
            "scenario_kind": "switch_simple_candidate_v2_then_challenger_managed",
            "scenario_label": "switch",
            "trades": 11,
            "profit_factor": 1.20,
            "cum_r": 23.0,
            "max_dd_r": 11.0,
            "geometric_return_pct": 18.0,
            "cagr_pct": 6.0,
            "window_start": "2020-01-01",
            "window_end": "2026-03-30",
        },
        {
            "cost_scenario": "stress_x2",
            "window": "full_2020",
            "scenario_kind": "always_challenger_managed",
            "scenario_label": "challenger",
            "trades": 10,
            "profit_factor": 1.05,
            "cum_r": 18.0,
            "max_dd_r": 10.0,
            "geometric_return_pct": 4.0,
            "cagr_pct": 1.5,
            "window_start": "2020-01-01",
            "window_end": "2026-03-30",
        },
        {
            "cost_scenario": "stress_x2",
            "window": "full_2020",
            "scenario_kind": "switch_simple_candidate_v2_then_challenger_managed",
            "scenario_label": "switch",
            "trades": 11,
            "profit_factor": 1.02,
            "cum_r": 19.0,
            "max_dd_r": 11.0,
            "geometric_return_pct": 2.0,
            "cagr_pct": 0.8,
            "window_start": "2020-01-01",
            "window_end": "2026-03-30",
        },
    ]
    side_rows = [
        {
            "cost_scenario": "base",
            "window": "two_year",
            "scenario_kind": "always_challenger_managed",
            "scenario_label": "challenger",
            "side": "LONG",
            "trades": 5,
            "profit_factor": 1.0,
            "cum_r": 8.0,
            "max_dd_r": 2.0,
            "geometric_return_pct": 4.0,
            "expectancy_r": 1.0,
        },
        {
            "cost_scenario": "base",
            "window": "two_year",
            "scenario_kind": "switch_simple_candidate_v2_then_challenger_managed",
            "scenario_label": "switch",
            "side": "LONG",
            "trades": 5,
            "profit_factor": 1.0,
            "cum_r": 7.0,
            "max_dd_r": 2.0,
            "geometric_return_pct": 3.0,
            "expectancy_r": 0.8,
        },
    ]
    segment_rows = [
        {
            "cost_scenario": "base",
            "window": "full_2020",
            "scenario_kind": "always_challenger_managed",
            "scenario_label": "challenger",
            "segment": "pre_cutover",
            "trades": 4,
            "profit_factor": 1.0,
            "expectancy_r": 0.5,
            "cum_r": 4.0,
            "max_dd_r": 1.0,
            "geometric_return_pct": 2.0,
            "additive_return_pct": 2.0,
        },
        {
            "cost_scenario": "base",
            "window": "full_2020",
            "scenario_kind": "always_challenger_managed",
            "scenario_label": "challenger",
            "segment": "post_cutover",
            "trades": 6,
            "profit_factor": 1.0,
            "expectancy_r": 0.5,
            "cum_r": 6.0,
            "max_dd_r": 1.0,
            "geometric_return_pct": 3.0,
            "additive_return_pct": 3.0,
        },
        {
            "cost_scenario": "base",
            "window": "full_2020",
            "scenario_kind": "switch_simple_candidate_v2_then_challenger_managed",
            "scenario_label": "switch",
            "segment": "pre_cutover",
            "trades": 4,
            "profit_factor": 1.1,
            "expectancy_r": 0.8,
            "cum_r": 5.5,
            "max_dd_r": 1.0,
            "geometric_return_pct": 3.5,
            "additive_return_pct": 3.5,
        },
        {
            "cost_scenario": "base",
            "window": "full_2020",
            "scenario_kind": "switch_simple_candidate_v2_then_challenger_managed",
            "scenario_label": "switch",
            "segment": "post_cutover",
            "trades": 6,
            "profit_factor": 1.0,
            "expectancy_r": 0.5,
            "cum_r": 6.5,
            "max_dd_r": 1.0,
            "geometric_return_pct": 3.2,
            "additive_return_pct": 3.2,
        },
    ]

    decision = module.build_switch_decision(
        summary_rows=summary_rows,
        side_rows=side_rows,
        segment_rows=segment_rows,
        baseline_kind="always_challenger_managed",
        switch_kind="switch_simple_candidate_v2_then_challenger_managed",
    )

    assert decision["status"] == "promoted_calendar_switch_candidate"
    assert decision["pass_base_geo"] is True
    assert decision["pass_secondary_long_guard"] is True
