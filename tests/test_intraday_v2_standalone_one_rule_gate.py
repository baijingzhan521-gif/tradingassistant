from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from app.strategies.intraday_mtf_v2 import DEFAULT_CONFIG as BASE_CONFIG
from app.strategies.intraday_mtf_v2_cooldown10_v1 import DEFAULT_CONFIG as COOLDOWN10_CONFIG
from app.strategies.intraday_mtf_v2_pullback_075_v1 import DEFAULT_CONFIG as PULLBACK075_CONFIG
from app.strategies.intraday_mtf_v2_trend70_v1 import DEFAULT_CONFIG as TREND70_CONFIG
from app.services.strategy_service import StrategyService
from scripts.run_intraday_v2_standalone_one_rule_gate import (
    BASELINE_PROFILE,
    apply_activation_gate_to_decision_rows,
    build_activation_precheck_rows,
)
from scripts.run_trend_pullback_standalone_one_rule_gate import build_promotion_decision


def _flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.update(_flatten(value, name))
        else:
            rows[name] = value
    return rows


def _changed_keys(base: dict[str, Any], candidate: dict[str, Any]) -> set[str]:
    b = _flatten(base)
    c = _flatten(candidate)
    keys = set(b.keys()) | set(c.keys())
    return {k for k in keys if b.get(k) != c.get(k)}


def _summary_row(*, cost: str, profile: str, geo: float, cagr: float, pf: float, dd: float, trades: int) -> dict[str, Any]:
    return {
        "cost_scenario": cost,
        "window": "full_2020",
        "strategy_profile": profile,
        "geometric_return_pct": geo,
        "cagr_pct": cagr,
        "profit_factor": pf,
        "max_dd_r": dd,
        "trades": trades,
    }


def _concentration_row(*, profile: str, top3: float, best_year: float) -> dict[str, Any]:
    return {
        "cost_scenario": "base",
        "window": "full_2020",
        "strategy_profile": profile,
        "top3_trades_pnl_share_pct": top3,
        "best_year_geometric_pct_share": best_year,
    }


def test_intraday_v2_candidates_are_single_rule_vs_baseline() -> None:
    assert _changed_keys(BASE_CONFIG, PULLBACK075_CONFIG) == {"execution.pullback_distance_atr"}
    assert _changed_keys(BASE_CONFIG, TREND70_CONFIG) == {"window.trend_strength_threshold"}
    assert _changed_keys(BASE_CONFIG, COOLDOWN10_CONFIG) == {"backtest.cooldown_bars_after_exit"}


def test_intraday_v2_profiles_are_available_with_same_required_timeframes() -> None:
    service = StrategyService()
    baseline = service.build_strategy(BASELINE_PROFILE)
    for profile in ("intraday_mtf_v2_pullback_075_v1", "intraday_mtf_v2_trend70_v1", "intraday_mtf_v2_cooldown10_v1"):
        candidate = service.build_strategy(profile)
        assert candidate.required_timeframes == baseline.required_timeframes


def test_build_promotion_decision_rejects_when_stress_x3_fails() -> None:
    candidate = "intraday_mtf_v2_trend70_v1"
    summary_rows = [
        _summary_row(cost="base", profile=BASELINE_PROFILE, geo=-1.0, cagr=-0.2, pf=0.99, dd=4.0, trades=800),
        _summary_row(cost="stress_x2", profile=BASELINE_PROFILE, geo=-3.0, cagr=-0.6, pf=0.95, dd=4.5, trades=800),
        _summary_row(cost="stress_x3", profile=BASELINE_PROFILE, geo=-5.0, cagr=-1.0, pf=0.90, dd=5.0, trades=800),
        _summary_row(cost="base", profile=candidate, geo=1.0, cagr=0.2, pf=1.11, dd=5.8, trades=220),
        _summary_row(cost="stress_x2", profile=candidate, geo=0.3, cagr=0.08, pf=1.00, dd=6.1, trades=220),
        _summary_row(cost="stress_x3", profile=candidate, geo=-0.1, cagr=-0.03, pf=0.99, dd=6.4, trades=220),
    ]
    concentration_rows = [_concentration_row(profile=candidate, top3=50.0, best_year=58.0)]

    rows = build_promotion_decision(
        summary_rows=summary_rows,
        concentration_rows=concentration_rows,
        baseline_profile=BASELINE_PROFILE,
        profiles=[BASELINE_PROFILE, candidate],
    )

    assert rows[0]["status"] == "rejected_fragile_or_unprofitable"
    assert rows[0]["pass_stress_x3_geo"] is False


def test_build_activation_precheck_rows_marks_pass_by_changed_ratio() -> None:
    signatures = {
        BASELINE_PROFILE: {("2026-01-01T00:00:00+00:00", "LONG"), ("2026-01-02T00:00:00+00:00", "SHORT")},
        "intraday_mtf_v2_pullback_075_v1": {
            ("2026-01-01T00:00:00+00:00", "LONG"),
            ("2026-01-03T00:00:00+00:00", "LONG"),
        },
    }
    rows, active = build_activation_precheck_rows(
        signatures=signatures,
        baseline_profile=BASELINE_PROFILE,
        profiles=[BASELINE_PROFILE, "intraday_mtf_v2_pullback_075_v1"],
        precheck_start=datetime(2024, 1, 1),
        precheck_end=datetime(2026, 3, 1),
        threshold=0.5,
    )
    assert len(rows) == 1
    assert rows[0]["changed_ratio"] == pytest.approx(0.666667, abs=1e-6)
    assert rows[0]["pass_activation_precheck"] is True
    assert "intraday_mtf_v2_pullback_075_v1" in active


def test_apply_activation_gate_to_decision_rows_overrides_status_when_inactive() -> None:
    decision_rows = [
        {
            "candidate_profile": "intraday_mtf_v2_trend70_v1",
            "status": "promoted_standalone_candidate",
            "next_route": "promote_and_compare_with_switch_baseline",
        }
    ]
    activation_rows = [
        {
            "candidate_profile": "intraday_mtf_v2_trend70_v1",
            "pass_activation_precheck": False,
        }
    ]
    updated = apply_activation_gate_to_decision_rows(decision_rows, activation_rows=activation_rows)
    assert updated[0]["pass_activation_precheck"] is False
    assert updated[0]["status"] == "rejected_inactive_candidate"
    assert updated[0]["next_route"] == "freeze_intraday_v2_and_stop_campaign"


@pytest.mark.parametrize(
    ("top3", "best_year", "expected_flag"),
    [
        (66.0, 60.0, "pass_top3_share"),
        (50.0, 81.0, "pass_best_year_share"),
    ],
)
def test_build_promotion_decision_rejects_on_concentration(
    top3: float, best_year: float, expected_flag: str
) -> None:
    candidate = "intraday_mtf_v2_cooldown10_v1"
    summary_rows = [
        _summary_row(cost="base", profile=BASELINE_PROFILE, geo=-1.0, cagr=-0.2, pf=0.99, dd=4.0, trades=800),
        _summary_row(cost="stress_x2", profile=BASELINE_PROFILE, geo=-3.0, cagr=-0.6, pf=0.95, dd=4.5, trades=800),
        _summary_row(cost="stress_x3", profile=BASELINE_PROFILE, geo=-5.0, cagr=-1.0, pf=0.90, dd=5.0, trades=800),
        _summary_row(cost="base", profile=candidate, geo=1.2, cagr=0.25, pf=1.11, dd=5.8, trades=220),
        _summary_row(cost="stress_x2", profile=candidate, geo=0.4, cagr=0.1, pf=1.01, dd=6.1, trades=220),
        _summary_row(cost="stress_x3", profile=candidate, geo=0.2, cagr=0.05, pf=1.00, dd=6.4, trades=220),
    ]
    concentration_rows = [_concentration_row(profile=candidate, top3=top3, best_year=best_year)]

    rows = build_promotion_decision(
        summary_rows=summary_rows,
        concentration_rows=concentration_rows,
        baseline_profile=BASELINE_PROFILE,
        profiles=[BASELINE_PROFILE, candidate],
    )
    assert rows[0]["status"] == "rejected_fragile_or_unprofitable"
    assert rows[0][expected_flag] is False
