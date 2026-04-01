from __future__ import annotations

from typing import Any

import pytest

from app.strategies.trend_pullback_aux2_v1 import DEFAULT_CONFIG as AUX2_CONFIG
from app.strategies.trend_pullback_pullback_085_v1 import DEFAULT_CONFIG as PULLBACK_085_CONFIG
from app.strategies.trend_pullback_trend70_v1 import DEFAULT_CONFIG as TREND70_CONFIG
from app.strategies.trend_pullback_v1 import DEFAULT_CONFIG as BASE_CONFIG
from app.services.strategy_service import StrategyService
from scripts.run_trend_pullback_standalone_one_rule_gate import (
    BASELINE_PROFILE,
    build_promotion_decision,
)


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


def test_trend_pullback_candidates_are_single_rule_vs_baseline() -> None:
    assert _changed_keys(BASE_CONFIG, PULLBACK_085_CONFIG) == {"execution.pullback_distance_atr"}
    assert _changed_keys(BASE_CONFIG, TREND70_CONFIG) == {"window.trend_strength_threshold"}
    assert _changed_keys(BASE_CONFIG, AUX2_CONFIG) == {"trigger.min_auxiliary_confirmations"}


def test_trend_pullback_profiles_are_available_with_same_required_timeframes() -> None:
    service = StrategyService()
    baseline = service.build_strategy(BASELINE_PROFILE)
    for profile in ("trend_pullback_pullback_085_v1", "trend_pullback_trend70_v1", "trend_pullback_aux2_v1"):
        candidate = service.build_strategy(profile)
        assert candidate.required_timeframes == baseline.required_timeframes


def test_build_promotion_decision_promotes_when_all_constraints_pass() -> None:
    candidate = "trend_pullback_trend70_v1"
    summary_rows = [
        _summary_row(cost="base", profile=BASELINE_PROFILE, geo=10.0, cagr=2.0, pf=1.05, dd=7.0, trades=200),
        _summary_row(cost="stress_x2", profile=BASELINE_PROFILE, geo=5.0, cagr=1.0, pf=1.02, dd=8.0, trades=200),
        _summary_row(cost="stress_x3", profile=BASELINE_PROFILE, geo=1.0, cagr=0.3, pf=1.00, dd=9.0, trades=200),
        _summary_row(cost="base", profile=candidate, geo=12.0, cagr=2.4, pf=1.12, dd=5.8, trades=120),
        _summary_row(cost="stress_x2", profile=candidate, geo=4.0, cagr=1.1, pf=1.01, dd=6.2, trades=120),
        _summary_row(cost="stress_x3", profile=candidate, geo=1.5, cagr=0.6, pf=1.00, dd=6.5, trades=120),
    ]
    concentration_rows = [_concentration_row(profile=candidate, top3=42.0, best_year=60.0)]

    rows = build_promotion_decision(
        summary_rows=summary_rows,
        concentration_rows=concentration_rows,
        baseline_profile=BASELINE_PROFILE,
        profiles=[BASELINE_PROFILE, candidate],
    )

    assert rows[0]["status"] == "promoted_standalone_candidate"
    assert rows[0]["pass_stress_x3_geo"] is True


@pytest.mark.parametrize(
    ("override_field", "override_value", "expected_flag"),
    [
        ("profit_factor", 1.09, "pass_base_pf"),
        ("max_dd_r", 6.1, "pass_base_max_dd"),
        ("trades", 9, "pass_trades_floor"),
    ],
)
def test_build_promotion_decision_rejects_on_basic_gate_failure(
    override_field: str, override_value: float, expected_flag: str
) -> None:
    candidate = "trend_pullback_aux2_v1"
    base_row = _summary_row(cost="base", profile=candidate, geo=12.0, cagr=2.3, pf=1.12, dd=5.9, trades=120)
    base_row[override_field] = override_value
    summary_rows = [
        _summary_row(cost="base", profile=BASELINE_PROFILE, geo=10.0, cagr=2.0, pf=1.05, dd=7.0, trades=200),
        _summary_row(cost="stress_x2", profile=BASELINE_PROFILE, geo=5.0, cagr=1.0, pf=1.02, dd=8.0, trades=200),
        _summary_row(cost="stress_x3", profile=BASELINE_PROFILE, geo=1.0, cagr=0.3, pf=1.00, dd=9.0, trades=200),
        base_row,
        _summary_row(cost="stress_x2", profile=candidate, geo=4.0, cagr=1.1, pf=1.01, dd=6.2, trades=120),
        _summary_row(cost="stress_x3", profile=candidate, geo=1.2, cagr=0.5, pf=1.00, dd=6.5, trades=120),
    ]
    concentration_rows = [_concentration_row(profile=candidate, top3=42.0, best_year=60.0)]

    rows = build_promotion_decision(
        summary_rows=summary_rows,
        concentration_rows=concentration_rows,
        baseline_profile=BASELINE_PROFILE,
        profiles=[BASELINE_PROFILE, candidate],
    )

    assert rows[0]["status"] == "rejected_fragile_or_unprofitable"
    assert rows[0][expected_flag] is False
