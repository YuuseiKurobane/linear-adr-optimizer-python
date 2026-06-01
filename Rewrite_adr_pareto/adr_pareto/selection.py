from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING

from .candidates import dedupe_candidates, key_of, snap_candidate
from .config import SearchConfig
from .models import Candidate, Point, PointKey
from .ranking import (
    band_for_dr,
    equivalence_map,
    equivalence_sort_key,
    point_equivalence_key,
)
from .safety import safety_pool

if TYPE_CHECKING:
    from .fixed_curve import FixedCurveManager


def reference_labels(config: SearchConfig) -> dict[PointKey, list[str]]:
    labels: dict[PointKey, list[str]] = defaultdict(list)
    if config.include_original:
        original = snap_candidate(tuple(float(v) for v in config.original))
        labels[key_of(original)].append("Original")
    for idx, values in enumerate(config.inspect_point, start=1):
        point = snap_candidate(tuple(float(v) for v in values))
        labels[key_of(point)].append(f"Inspect {idx}")
    return dict(labels)


def reference_candidates(labels_by_key: dict[PointKey, list[str]]) -> list[Candidate]:
    return [key for key in labels_by_key]


def choose_final_candidates(
    pool: list[Point],
    max_spread_pool: list[Point],
    fixed: FixedCurveManager,
    refs: dict[PointKey, list[str]],
    config: SearchConfig,
) -> tuple[list[Candidate], list[Point]]:
    ranked = _ranked_for_final_shortlist(pool, fixed, config)
    selected: dict[PointKey, Point] = {}

    # These preset-sized buckets are defensive candidates for higher-weight
    # final verification. They do not force final labels after verification.
    limits = {
        "recommended": max(
            config.promote_recommended,
            config.final_shortlist_recommended,
        ),
        "efficiency": max(
            config.promote_efficiency_potential,
            config.final_shortlist_efficiency,
        ),
        "memory": max(
            config.promote_memory_potential,
            config.final_shortlist_memory,
        ),
        "frontier": config.final_shortlist_frontier,
    }
    for label in ("recommended", "efficiency", "memory", "frontier"):
        _add_top(selected, ranked[label], limits[label])

    metrics = equivalence_map(selected.values(), fixed.fixed_env)
    ordered = sorted(
        selected.values(),
        key=lambda point: point_equivalence_key(point, metrics),
        reverse=True,
    )
    candidates = [
        (point.flat, point.s_multi, point.d_multi)
        for point in ordered[: config.final_candidate_limit]
    ]

    max_spread_prefinal = max_spread_points(
        max_spread_pool,
        fixed,
        config,
        config.max_spread_final_candidates,
    )
    candidates.extend(
        (point.flat, point.s_multi, point.d_multi) for point in max_spread_prefinal
    )
    candidates.extend(reference_candidates(refs))
    return dedupe_candidates(candidates, enforce_quadrant=False), max_spread_prefinal


def choose_points(
    final: list[Point],
    fixed: FixedCurveManager,
    config: SearchConfig,
) -> dict[str, Point]:
    pool = safety_pool(final, config)
    if not pool:
        return {}

    metrics = equivalence_map(pool, fixed.fixed_env)
    x0 = fixed.target_fixed.memorized_cards
    y0 = fixed.target_fixed.memorized_per_minute
    x_band = band_for_dr(
        fixed.fixed_curve,
        fixed.target_dr,
        config.final_potential_band_pct,
        "memorized_cards",
    )
    y_band = band_for_dr(
        fixed.fixed_curve,
        fixed.target_dr,
        config.final_potential_band_pct,
        "memorized_per_minute",
    )

    selected: dict[str, Point] = {}
    northeast_pool = [
        point
        for point in pool
        if point.memorized_cards > x0 and point.memorized_per_minute > y0
    ]
    if northeast_pool:
        recommended = max(
            northeast_pool, key=lambda point: point_equivalence_key(point, metrics)
        )
        selected["Recommended"] = recommended

        spread_floor = (
            metrics[recommended.key].spread_floor
            - config.aggressive_calm_regret_pct / 100.0
        )
        aggressive_calm_pool = [
            point
            for point in northeast_pool
            if metrics[point.key].spread_floor >= spread_floor
        ] or [recommended]
        selected["Aggressive"] = max(
            aggressive_calm_pool,
            key=lambda point: (
                point.dr_spread if point.dr_samples else -float("inf"),
                *equivalence_sort_key(metrics[point.key]),
                point.memorized_per_minute,
                point.memorized_cards,
            ),
        )
        selected["Calm"] = min(
            aggressive_calm_pool,
            key=lambda point: (
                point.dr_spread if point.dr_samples else float("inf"),
                -equivalence_sort_key(metrics[point.key])[0],
                -point.memorized_per_minute,
                -point.memorized_cards,
            ),
        )

    efficiency_pool = [
        point for point in pool if x_band[0] <= point.memorized_cards <= x_band[1]
    ] or pool
    selected["Efficiency Potential"] = max(
        efficiency_pool,
        key=lambda point: (
            point.memorized_per_minute,
            *point_equivalence_key(point, metrics),
            -abs(point.memorized_cards - x0),
        ),
    )

    memory_pool = [
        point for point in pool if y_band[0] <= point.memorized_per_minute <= y_band[1]
    ] or pool
    selected["Memory Potential"] = max(
        memory_pool,
        key=lambda point: (
            point.memorized_cards,
            *point_equivalence_key(point, metrics),
            -abs(point.memorized_per_minute - y0),
        ),
    )

    selected["Max Spread"] = max(
        pool,
        key=lambda point: (
            *equivalence_sort_key(metrics[point.key]),
            point.memorized_per_minute,
            point.memorized_cards,
        ),
    )
    return selected


def add_reference_labels(
    selected_by_label: dict[str, Point],
    final: list[Point],
    refs: dict[PointKey, list[str]],
) -> dict[str, Point]:
    final_by_key = {point.key: point for point in final}
    out = dict(selected_by_label)
    for key, labels in refs.items():
        point = final_by_key.get(key)
        if point is None:
            continue
        for label in labels:
            out[label] = point
    return out


def labels_by_key(selected_by_label: dict[str, Point]) -> dict[PointKey, list[str]]:
    grouped: dict[PointKey, list[str]] = defaultdict(list)
    for label, point in selected_by_label.items():
        grouped[point.key].append(label)
    return dict(grouped)


def max_spread_points(
    points: list[Point],
    fixed: FixedCurveManager,
    config: SearchConfig,
    limit: int,
) -> list[Point]:
    pool = safety_pool(_dedupe_points(points), config)
    if not pool or limit <= 0:
        return []
    metrics = equivalence_map(pool, fixed.fixed_env)
    return sorted(
        pool,
        key=lambda point: (
            *equivalence_sort_key(metrics[point.key]),
            point.memorized_per_minute,
            point.memorized_cards,
        ),
        reverse=True,
    )[:limit]


def _ranked_for_final_shortlist(
    pool: list[Point],
    fixed: FixedCurveManager,
    config: SearchConfig,
) -> dict[str, list[Point]]:
    from .ranking import classify_ranked

    return classify_ranked(
        pool,
        fixed.target_fixed,
        fixed.fixed_curve,
        fixed.fixed_env,
        fixed.target_dr,
        config.final_potential_band_pct,
        config,
    )


def _add_top(selected: dict[PointKey, Point], ranked: list[Point], limit: int) -> None:
    count = 0
    for point in ranked:
        if point.key not in selected:
            count += 1
        selected.setdefault(point.key, point)
        if count >= limit:
            break


def _dedupe_points(points: Iterable[Point]) -> list[Point]:
    unique: dict[PointKey, Point] = {}
    for point in points:
        unique[point.key] = point
    return list(unique.values())
