from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .config import SearchConfig, json_safe_config
from .models import PhaseDiag, Point, PointKey


def latest_export(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(path.glob("adr-input-*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No adr-input-*.jsonl files found in {path}")
    return candidates[-1]


def load_export_row(path: Path, preset: str) -> dict:
    export_path = latest_export(path)
    rows = []
    with export_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {export_path}")

    needle = preset.casefold()

    def preset_name(row: dict) -> str:
        return row.get("deck_preset", {}).get("name", "")

    def deck_names(row: dict) -> list[str]:
        return [deck.get("name", "") for deck in row.get("decks", [])]

    passes = (
        lambda row: preset_name(row).casefold() == needle,
        lambda row: any(name.casefold() == needle for name in deck_names(row)),
        lambda row: needle in preset_name(row).casefold(),
        lambda row: any(needle in name.casefold() for name in deck_names(row)),
    )
    for matcher in passes:
        matches = [row for row in rows if matcher(row)]
        if len(matches) == 1:
            matches[0]["_export_path"] = str(export_path)
            return matches[0]
        if len(matches) > 1:
            available = ", ".join(preset_name(row) or "?" for row in matches)
            raise ValueError(
                f"Preset/deck selector {preset!r} is ambiguous. Matches: {available}. "
                "Use the exact deck_preset.name."
            )

    available = ", ".join(preset_name(row) or "?" for row in rows)
    raise ValueError(
        f"Preset/deck selector {preset!r} not found. Available presets: {available}"
    )


def write_summary(
    path: Path,
    *,
    row: dict,
    target_dr: float,
    config: SearchConfig,
    selected_by_label: dict[str, Point],
    labels_by_key: dict[PointKey, list[str]],
    selected_metrics: dict,
    fixed_curve_points: list[tuple[float, Point]],
    fixed_curve_refined_points: dict[float, Point],
    fixed_curve_envelope: list[tuple[float, Point]],
    final_frontier: list[Point],
    phase3_render_extra: list[Point],
    max_spread_prefinal: list[Point],
    diagnostics: list[PhaseDiag],
    plot_layers: dict[str, list[Point]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "export": row["_export_path"],
                "preset": row.get("deck_preset", {}),
                "target_dr": target_dr,
                "args": json_safe_config(config),
                "selected": {
                    label: asdict(point) for label, point in selected_by_label.items()
                },
                "labels_by_point": {
                    ",".join(f"{value:.6f}" for value in key): labels
                    for key, labels in sorted(labels_by_key.items())
                },
                "selected_fixed_curve_metrics": {
                    label: asdict(selected_metrics[point.key])
                    for label, point in selected_by_label.items()
                    if point.key in selected_metrics
                },
                "fixed_curve_points": [
                    {"dr": dr, "point": asdict(point)}
                    for dr, point in fixed_curve_points
                ],
                "fixed_curve_refined_points": [
                    {"dr": pct / 100.0, "point": asdict(point)}
                    for pct, point in sorted(fixed_curve_refined_points.items())
                ],
                "fixed_curve_envelope": [
                    {"dr": dr, "point": asdict(point)}
                    for dr, point in fixed_curve_envelope
                ],
                "final_frontier": [asdict(point) for point in final_frontier],
                "phase3_render_extra": [asdict(point) for point in phase3_render_extra],
                "max_spread_prefinal": [asdict(point) for point in max_spread_prefinal],
                "plot_layers": {
                    key: [asdict(point) for point in points]
                    for key, points in (plot_layers or {}).items()
                },
                "diagnostics": [asdict(diag) for diag in diagnostics],
            },
            handle,
            indent=2,
        )
