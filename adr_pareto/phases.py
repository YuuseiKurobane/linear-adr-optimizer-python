from __future__ import annotations

import time
from collections.abc import Callable

from .candidates import (
    add_existing_bridge_midpoints,
    bridge_midpoint_neighborhoods,
    dedupe_candidates,
    dedupe_points,
    hypercube_candidates,
    key_of,
    make_phase1_domain,
    phase1_boundary_directions,
    phase1_candidates,
    should_include_hypercube_center,
)
from .config import SearchConfig
from .eval_backend import evaluate_raw, evaluate_search
from .fixed_curve import FixedCurveManager
from .models import Candidate, FixedEnvelope, PhaseDiag, Point, PointKey
from .point_store import PointStore
from .ranking import (
    band_for_dr,
    classify_ranked,
    fixed_curve_equivalence,
    equivalence_sort_key,
    point_equivalence_key,
    select_promotions,
)
from .safety import attach_safety_from_rows, safety_by_key, safety_row_is_safe


def phase_diag(
    name: str, weight: float, start: float, points: list[Point], **notes
) -> PhaseDiag:
    return PhaseDiag(
        name=name,
        weight=weight,
        candidates=len(points),
        evaluated=len(points),
        safe=sum(1 for point in points if point.safe),
        unsafe=sum(1 for point in points if not point.safe),
        elapsed_s=time.perf_counter() - start,
        notes=notes or None,
    )


def evaluate_unique(
    eval_lib,
    candidates: list[Candidate],
    weight: float,
    seed: int,
    config: SearchConfig,
    store: PointStore | None = None,
    enforce_quadrant: bool = True,
) -> list[Point]:
    unique = dedupe_candidates(candidates, enforce_quadrant=enforce_quadrant)
    if store is not None:
        unique = store.missing_or_lower_weight(unique, weight)
    if not unique:
        return []
    points = evaluate_search(eval_lib, unique, weight, seed, config)
    if store is not None:
        store.add(points, weight)
    return points


def evaluate_phase1_unique(
    eval_lib,
    candidates: list[Candidate],
    weight: float,
    seed: int,
    config: SearchConfig,
    store: PointStore,
) -> tuple[list[Point], set[PointKey], dict[str, int | bool]]:
    unique = dedupe_candidates(candidates)
    if store is not None:
        unique = store.missing_or_lower_weight(unique, weight)
    stats: dict[str, int | bool] = {
        "candidate_count": len(unique),
        "evaluated_candidates": len(unique),
        "safety_prescreen": False,
        "screened_unsafe": 0,
    }
    if not unique:
        stats["evaluated_candidates"] = 0
        return [], set(), stats
    if config.ignore_safety or config.legacy_unsafe_plot_display:
        points = evaluate_search(eval_lib, unique, weight, seed, config)
        store.add(points, weight)
        return points, set(), stats

    safety = safety_by_key(eval_lib, unique, config)
    safe_candidates: list[Candidate] = []
    screened_unsafe_keys: set[PointKey] = set()
    for candidate in unique:
        key = key_of(candidate)
        if safety_row_is_safe(safety[key]):
            safe_candidates.append(candidate)
        else:
            screened_unsafe_keys.add(key)

    stats["safety_prescreen"] = True
    stats["screened_unsafe"] = len(screened_unsafe_keys)
    stats["evaluated_candidates"] = len(safe_candidates)
    if not safe_candidates:
        return [], screened_unsafe_keys, stats

    points = attach_safety_from_rows(
        evaluate_raw(eval_lib, safe_candidates, weight, seed), safety
    )
    store.add(points, weight)
    return points, screened_unsafe_keys, stats


def run_phase1(
    eval_lib,
    fixed: FixedCurveManager,
    config: SearchConfig,
    store: PointStore,
) -> tuple[list[Point], list[Point], list[PhaseDiag]]:
    all_points: dict[tuple[float, float, float], Point] = {}
    screened_unsafe_keys: set[PointKey] = set()
    diagnostics: list[PhaseDiag] = []
    domain = make_phase1_domain(config, fixed.target_dr)

    for round_idx in range(config.phase1_expand_rounds + 1):
        start = time.perf_counter()
        candidates = phase1_candidates(domain)
        new_candidates = [
            candidate
            for candidate in candidates
            if key_of(candidate) not in all_points
            and key_of(candidate) not in screened_unsafe_keys
        ]
        new_points, skipped_keys, phase1_stats = evaluate_phase1_unique(
            eval_lib,
            new_candidates,
            config.phase1_eval_weight,
            config.seed + round_idx,
            config,
            store,
        )
        screened_unsafe_keys.update(skipped_keys)
        for point in new_points:
            all_points[point.key] = point

        pool = list(all_points.values())
        promoted, _ = select_promotions(
            pool,
            fixed.target_fixed,
            fixed.fixed_curve,
            fixed.fixed_env,
            fixed.target_dr,
            config,
            config.scout_potential_band_pct,
        )
        directions = phase1_boundary_directions(promoted, domain)
        changed: dict[str, int] = {}
        if (
            config.phase1_expand
            and directions
            and round_idx < config.phase1_expand_rounds
        ):
            changed = domain.expand(directions, max(1, config.phase1_expand_batch))

        diag = phase_diag(
            f"phase1.{round_idx}",
            config.phase1_eval_weight,
            start,
            new_points,
            total_pool=len(pool),
            promoted=len(promoted),
            boundary=sorted(directions),
            expanded=changed,
            screened_unsafe=phase1_stats["screened_unsafe"],
            safety_prescreen=phase1_stats["safety_prescreen"],
        )
        diag.candidates = int(phase1_stats["candidate_count"])
        diag.evaluated = int(phase1_stats["evaluated_candidates"])
        diag.promoted = len(promoted)
        diagnostics.append(diag)
        screened_note = (
            f" screened_unsafe={phase1_stats['screened_unsafe']}"
            if phase1_stats["screened_unsafe"]
            else ""
        )
        print(
            f"[phase 1.{round_idx}] new={len(new_points)}{screened_note} pool={len(pool)} "
            f"promote={len(promoted)} boundary={','.join(sorted(directions)) or '-'} "
            f"elapsed={diag.elapsed_s:.1f}s"
        )
        if not changed:
            break

    final_promoted, _ = select_promotions(
        list(all_points.values()),
        fixed.target_fixed,
        fixed.fixed_curve,
        fixed.fixed_env,
        fixed.target_dr,
        config,
        config.scout_potential_band_pct,
    )
    return list(all_points.values()), final_promoted, diagnostics


def run_refinement_phase(
    *,
    eval_lib,
    phase_name: str,
    centers: list[Point],
    base_pool: list[Point],
    previous_weight: float,
    current_weight: float,
    steps: tuple[float, float, float],
    seed: int,
    adapt_seed: int,
    fixed: FixedCurveManager,
    config: SearchConfig,
    store: PointStore,
    pareto_as_render_only: bool,
) -> tuple[list[Point], list[Point], list[Point], PhaseDiag]:
    start = time.perf_counter()
    include_center = should_include_hypercube_center(current_weight, previous_weight)
    candidates = hypercube_candidates(centers, steps, include_center=include_center)
    points = evaluate_unique(eval_lib, candidates, current_weight, seed, config, store)
    pool = dedupe_points([*base_pool, *points])
    fixed.ensure_for_points(
        pool, phase_name, adapt_seed, config.scout_potential_band_pct
    )
    promoted, render_extra = select_promotions(
        pool,
        fixed.target_fixed,
        fixed.fixed_curve,
        fixed.fixed_env,
        fixed.target_dr,
        config,
        config.scout_potential_band_pct,
        pareto_as_render_only=pareto_as_render_only,
    )
    promoted, bridge_points, bridge_generated = apply_bridge_promotions(
        eval_lib=eval_lib,
        promoted=promoted,
        pool=pool,
        steps=steps,
        weight=current_weight,
        seed=seed + 9100,
        config=config,
        store=store,
    )
    if bridge_points:
        points = dedupe_points([*points, *bridge_points])
        pool = dedupe_points([*pool, *bridge_points])
        fixed.ensure_for_points(
            bridge_points,
            f"{phase_name}.bridge",
            adapt_seed + 20,
            config.scout_potential_band_pct,
        )

    diag = phase_diag(
        f"{phase_name}.hypercube",
        current_weight,
        start,
        points,
        generated=len(candidates),
        include_center=include_center,
        pool=len(pool),
        bridge_generated=bridge_generated,
        bridge_evaluated=len(bridge_points),
    )
    diag.promoted = len(promoted)
    diag.pareto_extra = len(render_extra)
    print(
        f"[{phase_name.replace('phase', 'phase ')}] candidates={len(candidates)} "
        f"new={len(points)} pool={len(pool)} promote={len(promoted)} "
        f"render_extra={len(render_extra)} bridge_eval={len(bridge_points)} "
        f"elapsed={diag.elapsed_s:.1f}s"
    )
    return points, promoted, render_extra, diag


def apply_bridge_promotions(
    *,
    eval_lib,
    promoted: list[Point],
    pool: list[Point],
    steps: tuple[float, float, float],
    weight: float,
    seed: int,
    config: SearchConfig,
    store: PointStore,
) -> tuple[list[Point], list[Point], int]:
    bridged = add_existing_bridge_midpoints(
        promoted, pool, steps, config.bridge_midpoint_limit
    )
    if not config.experimental_bridge_midpoint_neighborhoods:
        return bridged, [], 0

    candidates = bridge_midpoint_neighborhoods(bridged, steps)
    points = evaluate_unique(eval_lib, candidates, weight, seed, config, store)
    if not points:
        return bridged, [], len(candidates)
    return dedupe_points([*bridged, *points]), points, len(candidates)


def phase4_seed_profiles(
    pool: list[Point],
    fixed: FixedCurveManager,
    config: SearchConfig,
) -> list[tuple[str, Point]]:
    ranked = classify_ranked(
        pool,
        fixed.target_fixed,
        fixed.fixed_curve,
        fixed.fixed_env,
        fixed.target_dr,
        config.scout_potential_band_pct,
        config,
    )
    seeds: list[tuple[str, Point]] = []
    for label in ("recommended", "efficiency", "memory", "frontier"):
        for point in ranked[label][: config.phase4_seeds_per_objective]:
            seeds.append((label, point))
    unique: dict[tuple[str, tuple[float, float, float]], tuple[str, Point]] = {}
    for label, point in seeds:
        unique[(label, point.key)] = (label, point)
    return list(unique.values())


def objective_key_factory(
    label: str,
    target_fixed: Point,
    fixed_points: list[tuple[float, Point]],
    env: FixedEnvelope,
    target_dr: float,
    config: SearchConfig,
) -> Callable[[Point], tuple]:
    x_band = band_for_dr(
        fixed_points, target_dr, config.scout_potential_band_pct, "memorized_cards"
    )
    y_band = band_for_dr(
        fixed_points,
        target_dr,
        config.scout_potential_band_pct,
        "memorized_per_minute",
    )
    x0 = target_fixed.memorized_cards
    y0 = target_fixed.memorized_per_minute

    def key(point: Point) -> tuple:
        metric = fixed_curve_equivalence(point, env)
        spread_key = equivalence_sort_key(metric)
        if label == "recommended":
            in_region = point.memorized_cards > x0 and point.memorized_per_minute > y0
            return (
                int(in_region),
                *spread_key,
                point.memorized_per_minute,
                point.memorized_cards,
            )
        if label == "efficiency":
            in_band = x_band[0] <= point.memorized_cards <= x_band[1]
            return (
                int(in_band),
                point.memorized_per_minute,
                *spread_key,
                -abs(point.memorized_cards - x0),
            )
        if label == "memory":
            in_band = y_band[0] <= point.memorized_per_minute <= y_band[1]
            return (
                int(in_band),
                point.memorized_cards,
                *spread_key,
                -abs(point.memorized_per_minute - y0),
            )
        metrics = {point.key: metric}
        return point_equivalence_key(point, metrics)

    return key


def run_micro_hillclimb(
    eval_lib,
    seeds: list[tuple[str, Point]],
    starting_pool: list[Point],
    fixed: FixedCurveManager,
    config: SearchConfig,
    store: PointStore,
) -> tuple[list[Point], PhaseDiag]:
    start = time.perf_counter()
    evaluated: dict[tuple[float, float, float], Point] = {
        point.key: point for point in starting_pool
    }
    visited: dict[tuple[float, float, float], Point] = {}
    steps = (config.phase4_flat_step, config.phase4_s_step, config.phase4_d_step)
    eval_count = 0

    for label, seed_point in seeds:
        current = evaluated[seed_point.key]
        key_fn = objective_key_factory(
            label,
            fixed.target_fixed,
            fixed.fixed_curve,
            fixed.fixed_env,
            fixed.target_dr,
            config,
        )
        visited[current.key] = current
        for step_idx in range(config.phase4_max_steps):
            neighbors = hypercube_candidates([current], steps, include_center=False)
            missing = [
                candidate
                for candidate in neighbors
                if key_of(candidate) not in evaluated
            ]
            new_points = evaluate_unique(
                eval_lib,
                missing,
                config.phase4_eval_weight,
                config.seed + 4000 + eval_count + step_idx,
                config,
                store,
            )
            eval_count += len(new_points)
            for point in new_points:
                evaluated[point.key] = point
                visited[point.key] = point
            neighbor_points = [
                evaluated[key_of(candidate)]
                for candidate in neighbors
                if key_of(candidate) in evaluated
            ]
            best = max([current, *neighbor_points], key=key_fn)
            if key_fn(best) <= key_fn(current):
                break
            current = best
            visited[current.key] = current

    diag = phase_diag(
        "phase4.microhill",
        config.phase4_eval_weight,
        start,
        list(visited.values()),
        seeds=len(seeds),
    )
    return list(visited.values()), diag
