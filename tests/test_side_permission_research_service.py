from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.schemas.common import Action
from app.services.side_permission_research_service import SidePermissionResearchService


def write_csv(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    frame = pd.DataFrame(rows)
    path = tmp_path / "permissions.csv"
    frame.to_csv(path, index=False)
    return path


def test_load_permission_csv_validates_required_columns_and_label_consistency(tmp_path: Path) -> None:
    service = SidePermissionResearchService()
    missing_col_path = write_csv(
        tmp_path,
        [
            {
                "timestamp": "2024-03-19T00:00:00Z",
                "permission_label": "allow_long",
                "allow_long": True,
                "model_version": "v1",
            }
        ],
    )
    with pytest.raises(ValueError, match="missing required columns"):
        service.load_permission_csv(missing_col_path)

    mismatched_label_path = write_csv(
        tmp_path,
        [
            {
                "timestamp": "2024-03-19T00:00:00Z",
                "permission_label": "allow_short",
                "allow_long": True,
                "allow_short": False,
                "model_version": "v1",
            }
        ],
    )
    with pytest.raises(ValueError, match="permission_label is inconsistent"):
        service.load_permission_csv(mismatched_label_path)


def test_load_permission_csv_rejects_duplicate_timestamps(tmp_path: Path) -> None:
    service = SidePermissionResearchService()
    path = write_csv(
        tmp_path,
        [
            {
                "timestamp": "2024-03-19T00:00:00Z",
                "permission_label": "allow_long",
                "allow_long": True,
                "allow_short": False,
                "model_version": "v1",
            },
            {
                "timestamp": "2024-03-19T00:00:00Z",
                "permission_label": "allow_none",
                "allow_long": False,
                "allow_short": False,
                "model_version": "v1",
            },
        ],
    )
    with pytest.raises(ValueError, match="duplicate timestamps"):
        service.load_permission_csv(path)


def test_resolve_permission_uses_backward_join_and_missing_is_none(tmp_path: Path) -> None:
    service = SidePermissionResearchService()
    path = write_csv(
        tmp_path,
        [
            {
                "timestamp": "2024-03-19T00:00:00Z",
                "permission_label": "allow_long",
                "allow_long": True,
                "allow_short": False,
                "model_version": "v1",
            },
            {
                "timestamp": "2024-03-19T02:00:00Z",
                "permission_label": "allow_short",
                "allow_long": False,
                "allow_short": True,
                "model_version": "v1",
            },
        ],
    )
    permissions = service.load_permission_csv(path)

    missing = service.resolve_permission(permissions, pd.Timestamp("2024-03-18T23:00:00Z"))
    assert missing is None

    resolved = service.resolve_permission(permissions, pd.Timestamp("2024-03-19T01:00:00Z"))
    assert resolved is not None
    assert resolved.permission_label == "allow_long"
    assert resolved.allow_long is True
    assert resolved.allow_short is False


def test_should_veto_respects_variant_and_side_semantics() -> None:
    service = SidePermissionResearchService()
    permissions = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-03-19T00:00:00Z"),
                "permission_label": "allow_long",
                "allow_long": True,
                "allow_short": False,
                "model_version": "v1",
                "long_score": None,
                "short_score": None,
                "meta_regime": None,
            },
            {
                "timestamp": pd.Timestamp("2024-03-19T01:00:00Z"),
                "permission_label": "allow_none",
                "allow_long": False,
                "allow_short": False,
                "model_version": "v1",
                "long_score": None,
                "short_score": None,
                "meta_regime": None,
            },
        ]
    )
    long_state = service.resolve_permission(permissions, pd.Timestamp("2024-03-19T00:30:00Z"))
    none_state = service.resolve_permission(permissions, pd.Timestamp("2024-03-19T01:30:00Z"))

    assert service.should_veto(variant="baseline_no_permission", action=Action.LONG, permission=long_state) is False
    assert service.should_veto(variant="permission_long_only", action=Action.LONG, permission=long_state) is False
    assert service.should_veto(variant="permission_short_only", action=Action.SHORT, permission=long_state) is True
    assert service.should_veto(variant="permission_full_side_control", action=Action.SHORT, permission=long_state) is True
    assert service.should_veto(variant="permission_full_side_control", action=Action.LONG, permission=none_state) is True


def test_gate_coverage_and_loss_cluster_summaries_are_grouped_correctly() -> None:
    service = SidePermissionResearchService()
    frame = pd.DataFrame(
        [
            {
                "window": "two_year",
                "variant": "permission_full_side_control",
                "permission_covered": True,
                "vetoed": True,
                "signal_time": pd.Timestamp("2024-03-20T00:00:00Z"),
                "side": "LONG",
                "trend_strength": 94,
                "bars_held": 9,
                "pnl_r": -1.0,
                "year": 2024,
                "tp1_hit": False,
                "tp2_hit": False,
                "exit_reason": "stop_loss",
            },
            {
                "window": "two_year",
                "variant": "permission_full_side_control",
                "permission_covered": True,
                "vetoed": False,
                "signal_time": pd.Timestamp("2024-03-21T00:00:00Z"),
                "side": "LONG",
                "trend_strength": 95,
                "bars_held": 12,
                "pnl_r": -0.8,
                "year": 2024,
                "tp1_hit": False,
                "tp2_hit": False,
                "exit_reason": "stop_loss",
            },
            {
                "window": "two_year",
                "variant": "permission_full_side_control",
                "permission_covered": False,
                "vetoed": False,
                "signal_time": pd.Timestamp("2024-03-22T00:00:00Z"),
                "side": "LONG",
                "trend_strength": 97,
                "bars_held": 20,
                "pnl_r": 1.4,
                "year": 2024,
                "tp1_hit": True,
                "tp2_hit": False,
                "exit_reason": "breakeven_after_tp1",
            },
        ]
    )

    coverage_rows = service.summarize_gate_coverage(frame, group_cols=["window", "variant"])
    assert coverage_rows[0]["signals_now"] == 3
    assert coverage_rows[0]["signals_with_state"] == 2
    assert coverage_rows[0]["vetoed_signals"] == 1

    clusters = service.identify_loss_clusters(frame, group_cols=["window", "variant"])
    summary = service.summarize_loss_clusters(clusters, group_cols=["window", "variant"])
    assert len(clusters) == 1
    assert clusters[0]["cluster_length"] == 2
    assert clusters[0]["cluster_cumulative_r"] == -1.8
    assert summary[0]["clusters"] == 1
    assert summary[0]["worst_cluster_cumulative_r"] == -1.8
