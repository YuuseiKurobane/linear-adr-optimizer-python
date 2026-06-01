from __future__ import annotations

from dataclasses import replace
from typing import Any

from .candidates import key_of
from .config import SearchConfig
from .models import Candidate, Point, PointKey


SafetyRow = tuple[Any, ...]
SafetyByKey = dict[PointKey, SafetyRow]


def safety_by_key(
    eval_lib, candidates: list[Candidate], config: SearchConfig
) -> SafetyByKey:
    if config.ignore_safety or not candidates:
        return {}
    safety_rows = eval_lib.safety_many(
        candidates, config.safety_s_max, config.safety_checks
    )
    return {
        key_of((flat, s_multi, d_multi)): row
        for row in safety_rows
        for flat, s_multi, d_multi in [row[:3]]
    }


def safety_row_is_safe(row: SafetyRow) -> bool:
    return int(row[4]) == 0 and int(row[5]) == 0


def attach_safety_from_rows(points: list[Point], safety: SafetyByKey) -> list[Point]:
    if not safety or not points:
        return points
    return [
        replace(
            point,
            safety_checks=int(safety[point.key][3]),
            interval_flips=int(safety[point.key][4]),
            hard_shortens=int(safety[point.key][5]),
        )
        for point in points
    ]


def attach_safety(eval_lib, points: list[Point], config: SearchConfig) -> list[Point]:
    if config.ignore_safety or not points:
        return points
    candidates = [(p.flat, p.s_multi, p.d_multi) for p in points]
    return attach_safety_from_rows(points, safety_by_key(eval_lib, candidates, config))


def safety_pool(points: list[Point], config: SearchConfig) -> list[Point]:
    if config.ignore_safety:
        return points
    safe = [point for point in points if point.safe]
    return safe or points
