from __future__ import annotations

from collections.abc import Iterable

from .config import SearchConfig
from .models import FixedCurveEquivalence, FixedEnvelope, Point
from .safety import safety_pool


def pareto_frontier(points: list[Point]) -> list[Point]:
    ordered = sorted(
        points, key=lambda p: (-p.memorized_cards, -p.memorized_per_minute)
    )
    frontier = []
    best_y = -float("inf")
    for point in ordered:
        if point.memorized_per_minute > best_y:
            frontier.append(point)
            best_y = point.memorized_per_minute
    return sorted(frontier, key=lambda p: p.memorized_cards)


def format_dr_label(dr: float) -> str:
    pct = dr * 100.0
    if abs(pct - round(pct)) < 1e-6:
        return f"{pct:.0f}%"
    return f"{pct:.1f}%"


def make_fixed_envelope(fixed_points: list[tuple[float, Point]]) -> FixedEnvelope:
    frontier = pareto_frontier([point for _, point in fixed_points])
    dr_by_key = {point.key: dr for dr, point in fixed_points}
    envelope = sorted(
        ((dr_by_key[point.key], point) for point in frontier),
        key=lambda item: item[0],
    )
    envelope.sort(key=lambda item: item[1].memorized_cards)
    if not envelope:
        raise ValueError("fixed DR envelope is empty")
    min_dr = min(dr for dr, _ in envelope)
    max_dr = max(dr for dr, _ in envelope)
    xs = [point.memorized_cards for _, point in envelope]
    ys = [point.memorized_per_minute for _, point in envelope]
    return FixedEnvelope(
        points=envelope,
        min_dr=min_dr,
        max_dr=max_dr,
        min_x=min(xs),
        max_x=max(xs),
        min_y=min(ys),
        max_y=max(ys),
        x_span=max(max(xs) - min(xs), 1e-9),
        y_span=max(max(ys) - min(ys), 1e-9),
    )


def interp_by_dr(
    fixed_points: list[tuple[float, Point]], dr: float, attr: str
) -> float:
    ordered = sorted(fixed_points, key=lambda item: item[0])
    if dr <= ordered[0][0]:
        return float(getattr(ordered[0][1], attr))
    if dr >= ordered[-1][0]:
        return float(getattr(ordered[-1][1], attr))
    for (dr1, p1), (dr2, p2) in zip(ordered, ordered[1:]):
        if dr1 - 1e-12 <= dr <= dr2 + 1e-12:
            if abs(dr2 - dr1) < 1e-12:
                return float(getattr(p1, attr))
            ratio = (dr - dr1) / (dr2 - dr1)
            return float(getattr(p1, attr)) + ratio * (
                float(getattr(p2, attr)) - float(getattr(p1, attr))
            )
    return float(getattr(min(ordered, key=lambda item: abs(item[0] - dr))[1], attr))


def equivalent_dr_for_x(env: FixedEnvelope, x: float) -> tuple[float, int]:
    points = env.points
    if x < env.min_x:
        return env.min_dr, -1
    if x > env.max_x:
        return env.max_dr, 1
    for (dr1, p1), (dr2, p2) in zip(points, points[1:]):
        x1 = p1.memorized_cards
        x2 = p2.memorized_cards
        if min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9:
            if abs(x2 - x1) < 1e-12:
                return (dr1 + dr2) / 2.0, 0
            ratio = (x - x1) / (x2 - x1)
            return dr1 + ratio * (dr2 - dr1), 0
    nearest = min(points, key=lambda item: abs(item[1].memorized_cards - x))
    return nearest[0], 0


def equivalent_dr_for_y(env: FixedEnvelope, y: float) -> tuple[float, int]:
    points = env.points
    if y > env.max_y:
        return env.min_dr, -1
    if y < env.min_y:
        return env.max_dr, 1
    for (dr1, p1), (dr2, p2) in zip(points, points[1:]):
        y1 = p1.memorized_per_minute
        y2 = p2.memorized_per_minute
        if min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9:
            if abs(y2 - y1) < 1e-12:
                return (dr1 + dr2) / 2.0, 0
            ratio = (y - y1) / (y2 - y1)
            return dr1 + ratio * (dr2 - dr1), 0
    nearest = min(points, key=lambda item: abs(item[1].memorized_per_minute - y))
    return nearest[0], 0


def format_equivalent_dr(dr: float, censor: int) -> str:
    prefix = "<" if censor < 0 else ">" if censor > 0 else ""
    return f"{prefix}{dr * 100.0:.2f}%"


def format_spread(spread: float, lower_bound: bool) -> str:
    prefix = ">" if lower_bound else ""
    return f"{prefix}{spread * 100.0:.2f}%"


def fixed_curve_equivalence(point: Point, env: FixedEnvelope) -> FixedCurveEquivalence:
    memory_dr, memory_censor = equivalent_dr_for_x(env, point.memorized_cards)
    efficiency_dr, efficiency_censor = equivalent_dr_for_y(
        env, point.memorized_per_minute
    )
    spread_floor = memory_dr - efficiency_dr
    efficiency_surplus = max(0.0, (point.memorized_per_minute - env.max_y) / env.y_span)
    memory_surplus = max(0.0, (point.memorized_cards - env.max_x) / env.x_span)
    censor_strength = int(efficiency_censor != 0) + int(memory_censor != 0)
    lower_bound = efficiency_censor < 0 or memory_censor > 0
    return FixedCurveEquivalence(
        efficiency_equivalent_dr=efficiency_dr,
        memory_equivalent_dr=memory_dr,
        efficiency_label=format_equivalent_dr(efficiency_dr, efficiency_censor),
        memory_label=format_equivalent_dr(memory_dr, memory_censor),
        spread_floor=spread_floor,
        spread_label=format_spread(spread_floor, lower_bound),
        efficiency_censor=efficiency_censor,
        memory_censor=memory_censor,
        censor_strength=censor_strength,
        efficiency_surplus=efficiency_surplus,
        memory_surplus=memory_surplus,
        surplus_balanced=min(efficiency_surplus, memory_surplus),
        surplus_total=efficiency_surplus + memory_surplus,
    )


def equivalence_map(
    points: Iterable[Point],
    env: FixedEnvelope,
) -> dict[tuple[float, float, float], FixedCurveEquivalence]:
    return {point.key: fixed_curve_equivalence(point, env) for point in points}


def equivalence_sort_key(
    metric: FixedCurveEquivalence,
) -> tuple[float, int, float, float]:
    return (
        metric.spread_floor,
        metric.censor_strength,
        metric.surplus_balanced,
        metric.surplus_total,
    )


def point_equivalence_key(
    point: Point,
    metrics: dict[tuple[float, float, float], FixedCurveEquivalence],
) -> tuple:
    metric = metrics[point.key]
    return (
        *equivalence_sort_key(metric),
        point.memorized_per_minute,
        point.memorized_cards,
    )


def band_for_dr(
    fixed_points: list[tuple[float, Point]],
    target_dr: float,
    band_pct: float,
    attr: str,
) -> tuple[float, float]:
    band = max(0.0, band_pct) / 100.0
    lo_dr = max(0.000001, target_dr - band)
    hi_dr = min(0.999999, target_dr + band)
    lo = interp_by_dr(fixed_points, lo_dr, attr)
    hi = interp_by_dr(fixed_points, hi_dr, attr)
    return min(lo, hi), max(lo, hi)


def classify_ranked(
    points: list[Point],
    target_fixed: Point,
    fixed_points: list[tuple[float, Point]],
    env: FixedEnvelope,
    target_dr: float,
    band_pct: float,
    config: SearchConfig,
) -> dict[str, list[Point]]:
    pool = safety_pool(points, config)
    if not pool:
        return {"recommended": [], "efficiency": [], "memory": [], "frontier": []}

    metrics = equivalence_map(pool, env)
    x0 = target_fixed.memorized_cards
    y0 = target_fixed.memorized_per_minute
    x_band = band_for_dr(fixed_points, target_dr, band_pct, "memorized_cards")
    y_band = band_for_dr(fixed_points, target_dr, band_pct, "memorized_per_minute")

    recommended_pool = [
        point
        for point in pool
        if point.memorized_cards > x0 and point.memorized_per_minute > y0
    ] or pool
    efficiency_pool = [
        point for point in pool if x_band[0] <= point.memorized_cards <= x_band[1]
    ] or pool
    memory_pool = [
        point for point in pool if y_band[0] <= point.memorized_per_minute <= y_band[1]
    ] or pool

    recommended = sorted(
        recommended_pool,
        key=lambda point: point_equivalence_key(point, metrics),
        reverse=True,
    )
    efficiency = sorted(
        efficiency_pool,
        key=lambda point: (
            point.memorized_per_minute,
            *point_equivalence_key(point, metrics),
            -abs(point.memorized_cards - x0),
        ),
        reverse=True,
    )
    memory = sorted(
        memory_pool,
        key=lambda point: (
            point.memorized_cards,
            *point_equivalence_key(point, metrics),
            -abs(point.memorized_per_minute - y0),
        ),
        reverse=True,
    )
    frontier = sorted(
        pareto_frontier(pool),
        key=lambda point: point_equivalence_key(point, metrics),
        reverse=True,
    )
    return {
        "recommended": recommended,
        "efficiency": efficiency,
        "memory": memory,
        "frontier": frontier,
    }


def add_top(
    selected: dict[tuple[float, float, float], Point], ranked: list[Point], limit: int
) -> None:
    count = 0
    for point in ranked:
        if point.key not in selected:
            count += 1
        selected.setdefault(point.key, point)
        if count >= limit:
            break


def select_promotions(
    points: list[Point],
    target_fixed: Point,
    fixed_points: list[tuple[float, Point]],
    env: FixedEnvelope,
    target_dr: float,
    config: SearchConfig,
    band_pct: float,
    include_pareto_extra: bool = True,
    pareto_as_render_only: bool = False,
) -> tuple[list[Point], list[Point]]:
    ranked = classify_ranked(
        points, target_fixed, fixed_points, env, target_dr, band_pct, config
    )
    selected: dict[tuple[float, float, float], Point] = {}
    add_top(selected, ranked["recommended"], config.promote_recommended)
    add_top(selected, ranked["efficiency"], config.promote_efficiency_potential)
    add_top(selected, ranked["memory"], config.promote_memory_potential)

    extras: dict[tuple[float, float, float], Point] = {}
    if include_pareto_extra:
        count = 0
        for point in ranked["frontier"]:
            if point.key in selected or point.key in extras:
                continue
            extras[point.key] = point
            count += 1
            if count >= config.promote_pareto_extra:
                break

    if pareto_as_render_only:
        return list(selected.values()), list(extras.values())
    selected.update(extras)
    return list(selected.values()), []
