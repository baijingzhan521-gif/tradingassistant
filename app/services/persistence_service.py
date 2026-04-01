from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi.encoders import jsonable_encoder
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from app.core.exceptions import PersistenceError
from app.models.analysis_record import AnalysisRecord
from app.schemas.analysis import AnalysisResult
from app.schemas.common import Action, Bias
from app.schemas.request import AnalyzeRequest
from app.schemas.response import (
    AnalysisDiffResponse,
    AnalysisListPagination,
    AnalysisListResponse,
    AnalysisSummary,
    FieldChange,
    SectionDiff,
    TimeframeDiff,
)


class PersistenceService:
    def save_analysis(self, db: Session, request: AnalyzeRequest, analysis: AnalysisResult) -> AnalysisResult:
        payload = jsonable_encoder(analysis.model_dump(mode="json"))
        request_payload = jsonable_encoder(request.model_dump(mode="json"))

        record = AnalysisRecord(
            analysis_id=analysis.analysis_id,
            symbol=analysis.symbol,
            exchange=analysis.exchange,
            market_type=analysis.market_type,
            strategy_profile=analysis.strategy_profile,
            action=analysis.decision.action,
            bias=analysis.decision.bias,
            confidence=analysis.decision.confidence,
            summary=analysis.reasoning.summary,
            request_payload=request_payload,
            result_payload=payload,
        )
        try:
            db.add(record)
            db.commit()
        except Exception as exc:  # pragma: no cover - defensive boundary
            db.rollback()
            raise PersistenceError(f"Failed to persist analysis result: {exc}") from exc
        return analysis

    def get_analysis(self, db: Session, analysis_id: str) -> AnalysisResult:
        record = db.scalar(select(AnalysisRecord).where(AnalysisRecord.analysis_id == analysis_id))
        if not record:
            raise PersistenceError(f"Analysis not found: {analysis_id}")
        return AnalysisResult.model_validate(record.result_payload)

    def compare_analyses(self, db: Session, left_analysis_id: str, right_analysis_id: str) -> AnalysisDiffResponse:
        left_record = self._get_analysis_record(db, left_analysis_id)
        right_record = self._get_analysis_record(db, right_analysis_id)

        left_result = self._validate_analysis_result(left_record)
        right_result = self._validate_analysis_result(right_record)
        left_summary = self._to_summary(left_record)
        right_summary = self._to_summary(right_record)

        decision_diff = self._diff_section(
            "decision",
            left_result.decision.model_dump(mode="json"),
            right_result.decision.model_dump(mode="json"),
        )
        market_regime_diff = self._diff_section(
            "market_regime",
            left_result.market_regime.model_dump(mode="json"),
            right_result.market_regime.model_dump(mode="json"),
        )
        diagnostics_diff = self._diff_section(
            "diagnostics",
            left_result.diagnostics.model_dump(mode="json"),
            right_result.diagnostics.model_dump(mode="json"),
        )

        timeframe_diffs = self._diff_timeframes(left_result, right_result)
        changed_sections = self._changed_sections(
            decision_diff=decision_diff,
            market_regime_diff=market_regime_diff,
            diagnostics_diff=diagnostics_diff,
            timeframe_diffs=timeframe_diffs,
        )

        return AnalysisDiffResponse(
            left=left_summary,
            right=right_summary,
            same_symbol=left_result.symbol == right_result.symbol,
            same_exchange=left_result.exchange == right_result.exchange,
            same_market_type=left_result.market_type == right_result.market_type,
            same_strategy_profile=left_result.strategy_profile == right_result.strategy_profile,
            compared_at=datetime.now(timezone.utc),
            decision=decision_diff,
            market_regime=market_regime_diff,
            diagnostics=diagnostics_diff,
            timeframes=timeframe_diffs,
            changed_sections=changed_sections,
            total_change_count=sum(
                section.change_count for section in (decision_diff, market_regime_diff, diagnostics_diff)
            )
            + sum(diff.change_count for diff in timeframe_diffs),
            summary=self._build_diff_summary(
                left_result=left_result,
                right_result=right_result,
                decision_diff=decision_diff,
                market_regime_diff=market_regime_diff,
                diagnostics_diff=diagnostics_diff,
                timeframe_diffs=timeframe_diffs,
            ),
        )

    def list_analyses(
        self,
        db: Session,
        limit: int = 50,
        offset: int = 0,
        *,
        symbol: Optional[str] = None,
        action: Optional[Action | str] = None,
        bias: Optional[Bias | str] = None,
        strategy_profile: Optional[str] = None,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
    ) -> AnalysisListResponse:
        filters = []
        if symbol:
            filters.append(AnalysisRecord.symbol == symbol.strip().upper())
        if action:
            filters.append(AnalysisRecord.action == self._enum_value(action))
        if bias:
            filters.append(AnalysisRecord.bias == self._enum_value(bias))
        if strategy_profile:
            filters.append(AnalysisRecord.strategy_profile == strategy_profile.strip().lower())
        if from_time:
            filters.append(AnalysisRecord.created_at >= self._normalize_datetime(from_time))
        if to_time:
            filters.append(AnalysisRecord.created_at <= self._normalize_datetime(to_time))

        statement = select(AnalysisRecord)
        total_statement = select(func.count()).select_from(AnalysisRecord)
        if filters:
            statement = statement.where(and_(*filters))
            total_statement = total_statement.where(and_(*filters))

        statement = statement.order_by(AnalysisRecord.created_at.desc(), AnalysisRecord.id.desc()).limit(limit).offset(offset)
        rows = db.scalars(statement).all()
        total = db.scalar(total_statement) or 0

        items = [self._to_summary(row) for row in rows]
        returned = len(items)
        pagination = AnalysisListPagination(
            limit=limit,
            offset=offset,
            total=total,
            returned=returned,
            has_more=offset + returned < total,
            next_offset=(offset + returned) if offset + returned < total else None,
            previous_offset=(max(offset - limit, 0) if offset > 0 else None),
        )
        return AnalysisListResponse(items=items, total=total, pagination=pagination)

    def _to_summary(self, row: AnalysisRecord) -> AnalysisSummary:
        result_payload = row.result_payload or {}
        request_payload = row.request_payload or {}
        market_regime = result_payload.get("market_regime", {})
        decision = result_payload.get("decision", {})
        timestamp = self._parse_datetime(result_payload.get("timestamp")) or row.created_at
        recorded_at = row.created_at
        if recorded_at.tzinfo is None:
            recorded_at = recorded_at.replace(tzinfo=timezone.utc)

        return AnalysisSummary(
            analysis_id=row.analysis_id,
            timestamp=timestamp,
            recorded_at=recorded_at,
            symbol=row.symbol,
            exchange=row.exchange,
            market_type=row.market_type,
            strategy_profile=row.strategy_profile,
            requested_timeframes=list(request_payload.get("timeframes", [])),
            action=row.action,
            bias=row.bias,
            confidence=row.confidence,
            recommended_timing=decision.get("recommended_timing", "skip"),
            higher_timeframe_bias=market_regime.get("higher_timeframe_bias", "neutral"),
            trend_strength=int(market_regime.get("trend_strength", 0)),
            volatility_state=market_regime.get("volatility_state", "normal"),
            is_trend_friendly=bool(market_regime.get("is_trend_friendly", False)),
            summary=row.summary,
        )

    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str):
            normalized = value.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                return None
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
        return None

    @staticmethod
    def _normalize_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _enum_value(value: Any) -> str:
        return getattr(value, "value", str(value))

    def _get_analysis_record(self, db: Session, analysis_id: str) -> AnalysisRecord:
        record = db.scalar(select(AnalysisRecord).where(AnalysisRecord.analysis_id == analysis_id))
        if not record:
            raise PersistenceError(f"Analysis not found: {analysis_id}")
        return record

    def _validate_analysis_result(self, record: AnalysisRecord) -> AnalysisResult:
        try:
            return AnalysisResult.model_validate(record.result_payload)
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise PersistenceError(f"Analysis payload is invalid for analysis_id={record.analysis_id}: {exc}") from exc

    def _diff_section(self, section_name: str, left: dict[str, Any], right: dict[str, Any]) -> SectionDiff:
        changes: list[FieldChange] = []
        self._diff_values(left, right, prefix=section_name, changes=changes)
        highlights = self._section_highlights(section_name, changes)
        return SectionDiff(
            changed=bool(changes),
            change_count=len(changes),
            changed_fields=changes,
            highlights=highlights,
        )

    def _diff_timeframes(self, left_result: AnalysisResult, right_result: AnalysisResult) -> list[TimeframeDiff]:
        timeframe_map = {
            "1d": ("day_1", "day_1"),
            "4h": ("hour_4", "hour_4"),
            "1h": ("hour_1", "hour_1"),
            "15m": ("min_15", "min_15"),
            "3m": ("min_3", "min_3"),
        }
        diffs: list[TimeframeDiff] = []
        for timeframe, (left_attr, right_attr) in timeframe_map.items():
            left_model = getattr(left_result.timeframes, left_attr)
            right_model = getattr(right_result.timeframes, right_attr)
            left_frame = left_model.model_dump(mode="json") if left_model is not None else None
            right_frame = right_model.model_dump(mode="json") if right_model is not None else None
            if left_frame is None and right_frame is None:
                continue
            changes: list[FieldChange] = []
            self._diff_values(left_frame, right_frame, prefix=timeframe, changes=changes)
            if not changes:
                continue
            diffs.append(
                TimeframeDiff(
                    timeframe=timeframe,
                    changed=True,
                    change_count=len(changes),
                    changed_fields=changes,
                    highlights=self._timeframe_highlights(changes),
                    signal_shift=self._signal_shift(changes),
                )
            )
        return diffs

    def _diff_values(self, left: Any, right: Any, *, prefix: str, changes: list[FieldChange]) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            keys = sorted(set(left.keys()) | set(right.keys()))
            for key in keys:
                next_prefix = f"{prefix}.{key}"
                self._diff_values(left.get(key), right.get(key), prefix=next_prefix, changes=changes)
            return

        if isinstance(left, list) and isinstance(right, list):
            if self._normalize_list(left) == self._normalize_list(right):
                return
            changes.append(
                FieldChange(
                    field=prefix,
                    before=jsonable_encoder(left),
                    after=jsonable_encoder(right),
                    added=self._list_difference(right, left),
                    removed=self._list_difference(left, right),
                )
            )
            return

        if left == right:
            return

        change = FieldChange(field=prefix, before=jsonable_encoder(left), after=jsonable_encoder(right))
        if self._is_number(left) and self._is_number(right):
            change.delta = float(right) - float(left)
            if float(left) != 0.0:
                change.pct_change = ((float(right) - float(left)) / abs(float(left))) * 100.0
        changes.append(change)

    def _section_highlights(self, section_name: str, changes: list[FieldChange]) -> list[str]:
        priority = {
            "decision": ["decision.action", "decision.bias", "decision.confidence", "decision.recommended_timing"],
            "market_regime": [
                "market_regime.higher_timeframe_bias",
                "market_regime.trend_strength",
                "market_regime.volatility_state",
                "market_regime.is_trend_friendly",
            ],
            "diagnostics": [
                "diagnostics.score_breakdown.total",
                "diagnostics.score_breakdown.base",
                "diagnostics.vetoes",
                "diagnostics.conflict_signals",
                "diagnostics.uncertainty_notes",
                "diagnostics.trigger_maturity.state",
                "diagnostics.trigger_maturity.score",
            ],
        }
        highlighted = [
            self._format_change(change)
            for change in changes
            if change.field in priority.get(section_name, [])
        ]
        return highlighted[:3] if highlighted else [self._format_change(change) for change in changes[:3]]

    def _timeframe_highlights(self, changes: list[FieldChange]) -> list[str]:
        priority = [
            "trend_bias",
            "ema_alignment",
            "structure_state",
            "trend_score",
            "is_pullback_to_value_area",
            "is_extended",
            "trigger_state",
            "price_vs_ema21_pct",
            "price_vs_ema55_pct",
            "price_vs_ema100_pct",
            "price_vs_ema200_pct",
        ]
        highlighted = [
            self._format_change(change)
            for change in changes
            if any(change.field.endswith(f".{field}") for field in priority)
        ]
        return highlighted[:3] if highlighted else [self._format_change(change) for change in changes[:3]]

    def _signal_shift(self, changes: list[FieldChange]) -> Optional[str]:
        priority = [
            "trend_bias",
            "ema_alignment",
            "structure_state",
            "trend_score",
            "trigger_state",
            "close",
        ]
        for field in priority:
            match = next((change for change in changes if change.field.endswith(f".{field}")), None)
            if match:
                return self._format_change(match)
        return None

    def _changed_sections(
        self,
        *,
        decision_diff: SectionDiff,
        market_regime_diff: SectionDiff,
        diagnostics_diff: SectionDiff,
        timeframe_diffs: list[TimeframeDiff],
    ) -> list[str]:
        changed_sections: list[str] = []
        if decision_diff.changed:
            changed_sections.append("decision")
        if market_regime_diff.changed:
            changed_sections.append("market_regime")
        if diagnostics_diff.changed:
            changed_sections.append("diagnostics")
        changed_sections.extend(diff.timeframe for diff in timeframe_diffs)
        return changed_sections

    def _build_diff_summary(
        self,
        *,
        left_result: AnalysisResult,
        right_result: AnalysisResult,
        decision_diff: SectionDiff,
        market_regime_diff: SectionDiff,
        diagnostics_diff: SectionDiff,
        timeframe_diffs: list[TimeframeDiff],
    ) -> str:
        parts: list[str] = []

        decision_change = self._first_change(decision_diff.changed_fields, ("decision.action", "decision.bias", "decision.confidence", "decision.recommended_timing"))
        if decision_change:
            parts.append(self._format_change(decision_change))

        regime_change = self._first_change(
            market_regime_diff.changed_fields,
            (
                "market_regime.higher_timeframe_bias",
                "market_regime.trend_strength",
                "market_regime.volatility_state",
                "market_regime.is_trend_friendly",
            ),
        )
        if regime_change:
            parts.append(self._format_change(regime_change))

        diagnostics_change = self._first_change(
            diagnostics_diff.changed_fields,
            (
                "diagnostics.score_breakdown.total",
                "diagnostics.trigger_maturity.state",
                "diagnostics.trigger_maturity.score",
                "diagnostics.vetoes",
                "diagnostics.conflict_signals",
            ),
        )
        if diagnostics_change:
            parts.append(self._format_change(diagnostics_change))

        first_timeframe = next((diff for diff in timeframe_diffs if diff.changed_fields), None)
        if first_timeframe is not None:
            major_timeframe_change = self._first_change(
                first_timeframe.changed_fields,
                (
                    f"{first_timeframe.timeframe}.trend_bias",
                    f"{first_timeframe.timeframe}.ema_alignment",
                    f"{first_timeframe.timeframe}.structure_state",
                    f"{first_timeframe.timeframe}.trend_score",
                    f"{first_timeframe.timeframe}.trigger_state",
                ),
            )
            if major_timeframe_change:
                parts.append(f"{first_timeframe.timeframe}: {self._format_change(major_timeframe_change)}")

        if not parts:
            if left_result.symbol != right_result.symbol:
                return "No material technical differences detected, but the compared analyses cover different symbols."
            return "No material differences detected."

        return "; ".join(parts) + "."

    @staticmethod
    def _first_change(changes: list[FieldChange], field_names: tuple[str, ...]) -> Optional[FieldChange]:
        for field_name in field_names:
            match = next((change for change in changes if change.field == field_name), None)
            if match is not None:
                return match
        return changes[0] if changes else None

    @staticmethod
    def _format_change(change: FieldChange) -> str:
        if change.added or change.removed:
            return f"{change.field}: added={change.added}, removed={change.removed}"
        if change.delta is not None:
            if change.pct_change is not None:
                return f"{change.field}: {change.before} -> {change.after} (delta={change.delta:.4f}, {change.pct_change:.2f}%)"
            return f"{change.field}: {change.before} -> {change.after} (delta={change.delta:.4f})"
        return f"{change.field}: {change.before} -> {change.after}"

    @staticmethod
    def _normalize_list(values: list[Any]) -> list[str]:
        return sorted(PersistenceService._normalized_value(value) for value in values)

    @staticmethod
    def _list_difference(left: list[Any], right: list[Any]) -> list[Any]:
        right_normalized = {PersistenceService._normalized_value(value) for value in right}
        return [jsonable_encoder(value) for value in left if PersistenceService._normalized_value(value) not in right_normalized]

    @staticmethod
    def _normalized_value(value: Any) -> str:
        encoded = jsonable_encoder(value)
        return json.dumps(encoded, sort_keys=True, ensure_ascii=False, default=str)

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
