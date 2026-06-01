from __future__ import annotations

from dataclasses import dataclass, field

from .candidates import key_of
from .models import Candidate, Point, PointKey


@dataclass
class PointStore:
    points: dict[PointKey, Point] = field(default_factory=dict)
    eval_weight_by_key: dict[PointKey, float] = field(default_factory=dict)

    def add(self, points: list[Point], eval_weight: float) -> None:
        for point in points:
            previous = self.eval_weight_by_key.get(point.key, -1.0)
            if eval_weight >= previous:
                self.points[point.key] = point
                self.eval_weight_by_key[point.key] = eval_weight

    def get(self, key: PointKey) -> Point | None:
        return self.points.get(key)

    def values(self) -> list[Point]:
        return list(self.points.values())

    def missing_or_lower_weight(
        self, candidates: list[Candidate], eval_weight: float
    ) -> list[Candidate]:
        out = []
        for candidate in candidates:
            key = key_of(candidate)
            if self.eval_weight_by_key.get(key, -1.0) + 1e-9 < eval_weight:
                out.append(candidate)
        return out
