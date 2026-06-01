from __future__ import annotations

import math
import time
from dataclasses import dataclass
from collections.abc import Iterable

from .candidates import logit
from .config import SearchConfig
from .eval_backend import evaluate_fixed_points
from .models import Candidate, FixedEnvelope, PhaseDiag, Point
from .ranking import classify_ranked, fixed_curve_equivalence, make_fixed_envelope


@dataclass
class FixedCurveManager:
    eval_lib: object
    config: SearchConfig
    target_dr: float
    coarse_points_by_pct: dict[float, Point]
    refined_points_by_pct: dict[float, Point]
    fixed_curve: list[tuple[float, Point]]
    fixed_env: FixedEnvelope
    rough_env: FixedEnvelope
    target_fixed: Point
    diagnostics: list[PhaseDiag]

    @classmethod
    def build(
        cls, eval_lib, config: SearchConfig, target_dr: float
    ) -> "FixedCurveManager":
        manager = cls(
            eval_lib=eval_lib,
            config=config,
            target_dr=target_dr,
            coarse_points_by_pct={},
            refined_points_by_pct={},
            fixed_curve=[],
            fixed_env=None,  # type: ignore[arg-type]
            rough_env=None,  # type: ignore[arg-type]
            target_fixed=None,  # type: ignore[arg-type]
            diagnostics=[],
        )
        manager._evaluate_initial()
        return manager

    def _evaluate_initial(self) -> None:
        start = time.perf_counter()
        coarse_pcts = integer_pcts(
            self.config.fixed_dr_start_pct,
            self.config.fixed_dr_end_pct,
            self.config.fixed_curve_coarse_step_pct,
        )
        self.coarse_points_by_pct.update(
            evaluate_fixed_pcts(
                self.eval_lib,
                coarse_pcts,
                self.config.fixed_curve_coarse_weight,
                self.config.seed + 700,
            )
        )
        self.rough_env = make_fixed_envelope(
            points_from_pct_map(self.coarse_points_by_pct)
        )

        target_pct = self.target_dr * 100.0
        initial_pcts = aligned_dense_pcts(
            target_pct - self.config.fixed_curve_initial_radius_pct,
            target_pct + self.config.fixed_curve_initial_radius_pct,
            self.config,
        )
        initial_pcts.add(round(target_pct, 6))
        self._evaluate_refined_pcts(initial_pcts, self.config.seed + 720)
        self._rebuild()

        elapsed = time.perf_counter() - start
        self.diagnostics.append(
            PhaseDiag(
                name="fixed_curve.initial",
                weight=self.config.fixed_curve_refine_weight,
                candidates=len(coarse_pcts) + len(initial_pcts),
                evaluated=len(coarse_pcts) + len(initial_pcts),
                elapsed_s=elapsed,
                notes={
                    "coarse_weight": self.config.fixed_curve_coarse_weight,
                    "coarse_points": len(coarse_pcts),
                    "refined_weight": self.config.fixed_curve_refine_weight,
                    "refined_points": len(initial_pcts),
                    "curve_points": len(self.fixed_curve),
                    "envelope_points": len(self.fixed_env.points),
                    "end_pct": self.config.fixed_dr_end_pct,
                },
            )
        )
        print(
            f"[fixed curve] coarse={len(coarse_pcts)}@{self.config.fixed_curve_coarse_weight:g} "
            f"refined={len(initial_pcts)}@{self.config.fixed_curve_refine_weight:g} "
            f"envelope={len(self.fixed_env.points)} elapsed={elapsed:.1f}s"
        )

    def _evaluate_refined_pcts(
        self, pcts: Iterable[float], seed: int
    ) -> dict[float, Point]:
        new_pcts = sorted(
            pct
            for pct in {round(float(pct), 6) for pct in pcts}
            if self.config.fixed_dr_start_pct - 1e-9
            <= pct
            <= self.config.fixed_dr_end_pct + 1e-9
            and pct not in self.refined_points_by_pct
        )
        if not new_pcts:
            return {}
        evaluated = evaluate_fixed_pcts(
            self.eval_lib,
            new_pcts,
            self.config.fixed_curve_refine_weight,
            seed,
        )
        self.refined_points_by_pct.update(evaluated)
        return evaluated

    def _rebuild(self) -> None:
        merged = dict(self.coarse_points_by_pct)
        merged.update(self.refined_points_by_pct)
        self.fixed_curve = points_from_pct_map(merged)
        self.fixed_env = make_fixed_envelope(self.fixed_curve)
        self.target_fixed = min(
            self.fixed_curve, key=lambda item: abs(item[0] - self.target_dr)
        )[1]

    def ensure_for_points(
        self,
        points: list[Point],
        phase: str,
        seed: int,
        band_pct: float,
    ) -> None:
        if not points:
            return
        needed = self._needed_refined_pcts(points, band_pct)
        new_needed = sorted(
            pct for pct in needed if pct not in self.refined_points_by_pct
        )
        if not new_needed:
            return
        start = time.perf_counter()
        evaluated = self._evaluate_refined_pcts(new_needed, seed)
        self._rebuild()
        elapsed = time.perf_counter() - start
        diag = PhaseDiag(
            name=f"fixed_curve.adapt.{phase}",
            weight=self.config.fixed_curve_refine_weight,
            candidates=len(new_needed),
            evaluated=len(evaluated),
            elapsed_s=elapsed,
            notes={
                "phase": phase,
                "min_pct": min(new_needed) if new_needed else None,
                "max_pct": max(new_needed) if new_needed else None,
                "refined_total": len(self.refined_points_by_pct),
                "curve_points": len(self.fixed_curve),
            },
        )
        self.diagnostics.append(diag)
        print(
            f"[fixed curve adapt:{phase}] added={len(evaluated)} "
            f"range={min(new_needed):.1f}-{max(new_needed):.1f}% elapsed={elapsed:.1f}s"
        )

    def _needed_refined_pcts(self, points: list[Point], band_pct: float) -> set[float]:
        ranked = classify_ranked(
            points,
            self.target_fixed,
            self.fixed_curve,
            self.fixed_env,
            self.target_dr,
            band_pct,
            self.config,
        )
        candidates: dict[tuple[float, float, float], Point] = {}
        per_bucket = max(1, self.config.fixed_curve_adapt_top_per_bucket)
        for label in ("recommended", "efficiency", "memory", "frontier"):
            for point in ranked[label][:per_bucket]:
                candidates[point.key] = point

        needed: set[float] = set()
        for point in candidates.values():
            rough = fixed_curve_equivalence(point, self.rough_env)
            for dr, censor in (
                (rough.efficiency_equivalent_dr, rough.efficiency_censor),
                (rough.memory_equivalent_dr, rough.memory_censor),
            ):
                if censor != 0:
                    continue
                pct = dr * 100.0
                needed.update(
                    aligned_dense_pcts(
                        pct - self.config.fixed_curve_adapt_margin_pct,
                        pct + self.config.fixed_curve_adapt_margin_pct,
                        self.config,
                    )
                )
        if len(needed) > self.config.fixed_curve_adapt_max_points:
            target_pct = self.target_dr * 100.0
            needed = set(
                sorted(needed, key=lambda pct: abs(pct - target_pct))[
                    : self.config.fixed_curve_adapt_max_points
                ]
            )
        return needed


def integer_pcts(start_pct: float, end_pct: float, step_pct: float) -> list[float]:
    if step_pct <= 0.0:
        raise ValueError("--fixed-curve-coarse-step-pct must be positive")
    start = min(start_pct, end_pct)
    end = max(start_pct, end_pct)
    first = math.ceil(start / step_pct) * step_pct
    values = []
    pct = first
    while pct <= end + 1e-9:
        values.append(round(pct, 6))
        pct += step_pct
    for pct in (start, end):
        rounded = round(pct, 6)
        if 0.0 < rounded < 100.0:
            values.append(rounded)
    return sorted({pct for pct in values if 0.0 < pct < 100.0})


def aligned_dense_pcts(
    start_pct: float, end_pct: float, config: SearchConfig
) -> set[float]:
    step = config.fixed_curve_refine_step_pct
    if step <= 0.0:
        raise ValueError("--fixed-curve-refine-step-pct must be positive")
    lo = max(min(start_pct, end_pct), config.fixed_dr_start_pct)
    hi = min(max(start_pct, end_pct), config.fixed_dr_end_pct)
    if lo > hi:
        return set()
    start = math.floor(lo / step) * step
    end = math.ceil(hi / step) * step
    values = set()
    pct = start
    while pct <= end + 1e-9:
        rounded = round(pct, 6)
        if (
            config.fixed_dr_start_pct - 1e-9
            <= rounded
            <= config.fixed_dr_end_pct + 1e-9
        ):
            values.add(rounded)
        pct += step
    return values


def specs_for_pcts(pcts: Iterable[float]) -> list[tuple[float, Candidate]]:
    specs = []
    for pct in sorted({round(float(pct), 6) for pct in pcts}):
        dr = pct / 100.0
        specs.append((dr, (logit(dr), 0.0, 0.0)))
    return specs


def evaluate_fixed_pcts(
    eval_lib,
    pcts: Iterable[float],
    weight: float,
    seed: int,
) -> dict[float, Point]:
    pcts = sorted({round(float(pct), 6) for pct in pcts})
    specs = specs_for_pcts(pcts)
    points = evaluate_fixed_points(eval_lib, specs, weight, seed)
    return {round(dr * 100.0, 6): point for dr, point in points}


def points_from_pct_map(points_by_pct: dict[float, Point]) -> list[tuple[float, Point]]:
    return [(pct / 100.0, points_by_pct[pct]) for pct in sorted(points_by_pct)]


def should_label_fixed_pct(pct: float, target_pct: float, config: SearchConfig) -> bool:
    if abs(pct - target_pct) <= 1e-6:
        return True
    if (
        abs(pct - config.fixed_dr_start_pct) <= 1e-6
        or abs(pct - config.fixed_dr_end_pct) <= 1e-6
    ):
        return True
    step = config.fixed_dr_label_step_pct
    if step <= 0:
        return False
    offset = (pct - config.fixed_dr_start_pct) / step
    return abs(offset - round(offset)) <= 1e-6
