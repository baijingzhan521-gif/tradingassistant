from __future__ import annotations

from dataclasses import asdict, dataclass, field

from app.utils.math_utils import clamp


@dataclass
class ScoreContribution:
    label: str
    points: int
    note: str


@dataclass
class ScoreCard:
    base: int = 50
    contributions: list[ScoreContribution] = field(default_factory=list)

    def add(self, points: int, label: str, note: str) -> None:
        self.contributions.append(ScoreContribution(label=label, points=points, note=note))

    @property
    def total(self) -> int:
        return int(clamp(self.base + sum(item.points for item in self.contributions), 0, 100))

    def as_dict(self) -> dict[str, object]:
        return {
            "base": self.base,
            "total": self.total,
            "contributions": [asdict(item) for item in self.contributions],
        }
