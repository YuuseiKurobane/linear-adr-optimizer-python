from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PointKey = tuple[float, float, float]
Candidate = tuple[float, float, float]


@dataclass(frozen=True)
class Point:
    flat: float
    s_multi: float
    d_multi: float
    total_average_memorized: float
    total_cost: float
    total_iters: int
    memorized_fraction: float
    memorized_cards: float
    memorized_per_minute: float
    safety_checks: int = 0
    interval_flips: int = 0
    hard_shortens: int = 0
    dr_samples: int = 0
    dr_p10: float = 0.0
    dr_mean: float = 0.0
    dr_p90: float = 0.0
    dr_spread: float = 0.0

    @property
    def key(self) -> PointKey:
        return tuple(
            round(float(v), 6) for v in (self.flat, self.s_multi, self.d_multi)
        )

    @property
    def safe(self) -> bool:
        return self.interval_flips == 0 and self.hard_shortens == 0


@dataclass(frozen=True)
class FixedCurveEquivalence:
    efficiency_equivalent_dr: float
    memory_equivalent_dr: float
    efficiency_label: str
    memory_label: str
    spread_floor: float
    spread_label: str
    efficiency_censor: int = 0
    memory_censor: int = 0
    censor_strength: int = 0
    efficiency_surplus: float = 0.0
    memory_surplus: float = 0.0
    surplus_balanced: float = 0.0
    surplus_total: float = 0.0


@dataclass(frozen=True)
class FixedEnvelope:
    points: list[tuple[float, Point]]
    min_dr: float
    max_dr: float
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    x_span: float
    y_span: float


@dataclass
class PhaseDiag:
    name: str
    weight: float
    candidates: int = 0
    evaluated: int = 0
    safe: int = 0
    unsafe: int = 0
    promoted: int = 0
    pareto_extra: int = 0
    elapsed_s: float = 0.0
    notes: dict | None = None


@dataclass
class Phase1Domain:
    center: float
    flat_step: float
    s_step: float
    d_step: float
    flat_low: int
    flat_high: int
    s_high: int
    d_high: int
    init_flat_low: int
    init_flat_high: int
    init_s_high: int
    init_d_high: int
    flat_extra_limit: int
    s_extra_limit: int
    d_extra_limit: int

    def expand(self, directions: set[str], batch: int) -> dict[str, int]:
        changed: dict[str, int] = {}
        if "flat_low" in directions:
            limit = self.init_flat_low - self.flat_extra_limit
            old = self.flat_low
            self.flat_low = max(limit, self.flat_low - batch)
            changed["flat_low"] = old - self.flat_low
        if "flat_high" in directions:
            limit = self.init_flat_high + self.flat_extra_limit
            old = self.flat_high
            self.flat_high = min(limit, self.flat_high + batch)
            changed["flat_high"] = self.flat_high - old
        if "s_high" in directions:
            limit = self.init_s_high + self.s_extra_limit
            old = self.s_high
            self.s_high = min(limit, self.s_high + batch)
            changed["s_high"] = self.s_high - old
        if "d_high" in directions:
            limit = self.init_d_high + self.d_extra_limit
            old = self.d_high
            self.d_high = min(limit, self.d_high + batch)
            changed["d_high"] = self.d_high - old
        return {key: value for key, value in changed.items() if value > 0}


@dataclass(frozen=True)
class SearchResult:
    plot_path: Path
    summary_path: Path
    selected_by_label: dict[str, Point]
    labels_by_key: dict[PointKey, list[str]]
    diagnostics: list[PhaseDiag]
