from __future__ import annotations

import math
from collections.abc import Iterable

from .config import SearchConfig
from .models import Candidate, Phase1Domain, Point, PointKey


def key_of(candidate: Candidate) -> PointKey:
    return tuple(round(float(x), 6) for x in candidate)


def snap_value(value: float) -> float:
    return round(float(value), 6)


def snap_candidate(candidate: Candidate) -> Candidate:
    return tuple(snap_value(v) for v in candidate)


def in_quadrant(candidate: Candidate) -> bool:
    _, s_multi, d_multi = candidate
    return s_multi >= -1e-9 and d_multi <= 1e-9


def dedupe_candidates(
    candidates: Iterable[Candidate],
    enforce_quadrant: bool = True,
) -> list[Candidate]:
    unique: dict[PointKey, Candidate] = {}
    for candidate in candidates:
        snapped = snap_candidate(candidate)
        if enforce_quadrant and not in_quadrant(snapped):
            continue
        unique[key_of(snapped)] = snapped
    return list(unique.values())


def dedupe_points(points: Iterable[Point]) -> list[Point]:
    unique: dict[PointKey, Point] = {}
    for point in points:
        unique[point.key] = point
    return list(unique.values())


def make_phase1_domain(config: SearchConfig, target_dr: float) -> Phase1Domain:
    center = round(logit(target_dr), 3)
    flat_half = int(config.phase1_flat_half_steps)
    s_high = int(round(config.phase1_s_max / config.phase1_s_step))
    d_high = int(round(abs(config.phase1_d_min) / config.phase1_d_step))
    flat_count = flat_half * 2 + 1
    s_count = s_high + 1
    d_count = d_high + 1
    factor = max(0.0, config.phase1_expand_overflow_factor)
    return Phase1Domain(
        center=center,
        flat_step=config.phase1_flat_step,
        s_step=config.phase1_s_step,
        d_step=config.phase1_d_step,
        flat_low=-flat_half,
        flat_high=flat_half,
        s_high=s_high,
        d_high=d_high,
        init_flat_low=-flat_half,
        init_flat_high=flat_half,
        init_s_high=s_high,
        init_d_high=d_high,
        flat_extra_limit=int(math.ceil(flat_count * factor)),
        s_extra_limit=int(math.ceil(s_count * factor)),
        d_extra_limit=int(math.ceil(d_count * factor)),
    )


def phase1_candidates(domain: Phase1Domain) -> list[Candidate]:
    out = []
    for flat_idx in range(domain.flat_low, domain.flat_high + 1):
        flat = domain.center + flat_idx * domain.flat_step
        for s_idx in range(0, domain.s_high + 1):
            s_multi = s_idx * domain.s_step
            for d_idx in range(0, domain.d_high + 1):
                d_multi = -d_idx * domain.d_step
                out.append(snap_candidate((flat, s_multi, d_multi)))
    return out


def phase1_index_of(point: Point, domain: Phase1Domain) -> tuple[int, int, int]:
    flat_idx = round((point.flat - domain.center) / domain.flat_step)
    s_idx = round(point.s_multi / domain.s_step)
    d_idx = round(-point.d_multi / domain.d_step)
    return flat_idx, s_idx, d_idx


def phase1_boundary_directions(promoted: list[Point], domain: Phase1Domain) -> set[str]:
    directions: set[str] = set()
    for point in promoted:
        flat_idx, s_idx, d_idx = phase1_index_of(point, domain)
        if flat_idx <= domain.flat_low:
            directions.add("flat_low")
        if flat_idx >= domain.flat_high:
            directions.add("flat_high")
        if s_idx >= domain.s_high:
            directions.add("s_high")
        if d_idx >= domain.d_high:
            directions.add("d_high")
    return directions


def logit(dr: float) -> float:
    dr = min(max(float(dr), 1e-6), 1.0 - 1e-6)
    return math.log(dr / (1.0 - dr))


def hypercube_candidates(
    centers: Iterable[Point],
    steps: tuple[float, float, float],
    include_center: bool,
) -> list[Candidate]:
    out = []
    offsets = (-1, 0, 1)
    for center in centers:
        for flat_offset in offsets:
            for s_offset in offsets:
                for d_offset in offsets:
                    if (
                        not include_center
                        and flat_offset == 0
                        and s_offset == 0
                        and d_offset == 0
                    ):
                        continue
                    out.append(
                        (
                            center.flat + flat_offset * steps[0],
                            center.s_multi + s_offset * steps[1],
                            center.d_multi + d_offset * steps[2],
                        )
                    )
    return dedupe_candidates(out)


def should_include_hypercube_center(
    current_weight: float, previous_weight: float
) -> bool:
    return current_weight > previous_weight + 1e-9


def add_existing_bridge_midpoints(
    promoted: list[Point],
    pool: list[Point],
    steps: tuple[float, float, float],
    limit: int,
) -> list[Point]:
    if limit <= 0:
        return promoted
    by_key = {point.key: point for point in pool}
    selected = {point.key: point for point in promoted}
    added = 0
    for idx, a in enumerate(promoted):
        for b in promoted[idx + 1 :]:
            midpoint = _qualifying_midpoint(a, b, steps)
            if midpoint is None:
                continue
            midpoint_key = key_of(midpoint)
            if midpoint_key in by_key and midpoint_key not in selected:
                selected[midpoint_key] = by_key[midpoint_key]
                added += 1
                if added >= limit:
                    return list(selected.values())
    return list(selected.values())


def bridge_midpoint_neighborhoods(
    promoted: list[Point],
    steps: tuple[float, float, float],
) -> list[Candidate]:
    out: list[Candidate] = []
    for idx, a in enumerate(promoted):
        for b in promoted[idx + 1 :]:
            midpoint = _qualifying_midpoint(a, b, steps)
            if midpoint is None:
                continue
            bridge_axis = _bridge_axis(a, b, steps)
            if bridge_axis is None:
                continue
            face_axes = [axis for axis in range(3) if axis != bridge_axis]
            for offset_a in (-1, 0, 1):
                for offset_b in (-1, 0, 1):
                    values = [midpoint[0], midpoint[1], midpoint[2]]
                    values[face_axes[0]] += offset_a * steps[face_axes[0]]
                    values[face_axes[1]] += offset_b * steps[face_axes[1]]
                    out.append((values[0], values[1], values[2]))
    return dedupe_candidates(out)


def _qualifying_midpoint(
    a: Point,
    b: Point,
    steps: tuple[float, float, float],
) -> Candidate | None:
    av = (a.flat, a.s_multi, a.d_multi)
    bv = (b.flat, b.s_multi, b.d_multi)
    axis = _bridge_axis(a, b, steps)
    if axis is None:
        return None
    return snap_candidate(tuple((av[i] + bv[i]) / 2.0 for i in range(3)))


def _bridge_axis(a: Point, b: Point, steps: tuple[float, float, float]) -> int | None:
    av = (a.flat, a.s_multi, a.d_multi)
    bv = (b.flat, b.s_multi, b.d_multi)
    diffs = [bv[i] - av[i] for i in range(3)]
    varying = [idx for idx, diff in enumerate(diffs) if abs(diff) > 1e-9]
    if len(varying) != 1:
        return None
    axis = varying[0]
    if abs(abs(diffs[axis]) - 2.0 * steps[axis]) > 1e-6:
        return None
    return axis
