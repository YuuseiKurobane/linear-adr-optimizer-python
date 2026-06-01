from __future__ import annotations

from .config import SearchConfig
from .models import FixedCurveEquivalence, PhaseDiag, Point


def print_intro(row: dict, target_dr: float, config: SearchConfig) -> None:
    print(f"Loaded export: {row['_export_path']}")
    print(f"Preset: {row.get('deck_preset', {}).get('name')}")
    print(f"Quality preset: {config.quality_preset}")
    print(
        "Using simulation defaults: "
        f"days={config.days}, deck_size={config.deck_size}, learn_limit={config.learn_limit}"
    )
    print(f"Target DR: {target_dr:.4f}")
    print(f"Safety checks/filtering: {'skipped' if config.ignore_safety else 'on'}")
    if not config.ignore_safety:
        phase1_safety = (
            "legacy display; unsafe points are simulated and plotted"
            if config.legacy_unsafe_plot_display
            else "pre-screen; unsafe points are skipped before simulation"
        )
        print(f"Phase 1 unsafe handling: {phase1_safety}")
    print(f"Original final verification: {'on' if config.include_original else 'off'}")
    print(
        "Aggressive/Calm: northeast-only, "
        f"spread regret <= {config.aggressive_calm_regret_pct:.2f} percentage points"
    )
    print(
        "Aggressive: largest final p90-p10 DR spread; Calm: smallest final p90-p10 DR spread"
    )
    print(
        "Experimental bridge midpoint neighborhoods: "
        f"{'on' if config.experimental_bridge_midpoint_neighborhoods else 'off'}"
    )


def print_point(
    label: str,
    point: Point,
    metrics: dict[tuple[float, float, float], FixedCurveEquivalence],
    config: SearchConfig,
) -> None:
    metric = metrics[point.key]
    dr = (
        f" dr_mean={point.dr_mean:.4f} dr_band={point.dr_spread * 100.0:.2f}% "
        f"dr_n={point.dr_samples}"
        if point.dr_samples
        else ""
    )
    safety = (
        "safety=skipped"
        if config.ignore_safety
        else f"safe={point.safe} flips={point.interval_flips} hard_shortens={point.hard_shortens}"
    )
    print(
        f"{label:22} flat={point.flat:.4f} s={point.s_multi:.4f} d={point.d_multi:.4f} "
        f"x={point.memorized_cards:.2f} y={point.memorized_per_minute:.4f} "
        f"eff={metric.efficiency_label} mem={metric.memory_label} spread={metric.spread_label} "
        f"{dr} {safety}"
    )


def format_point_block(
    label: str,
    point: Point,
    metrics: dict[tuple[float, float, float], FixedCurveEquivalence],
    config: SearchConfig,
) -> str:
    metric = metrics[point.key]
    if point.dr_samples:
        dr_line = f"dr:{point.dr_mean:.4f} band={point.dr_spread * 100.0:.2f}%"
    else:
        dr_line = "dr:n/a band=n/a"
    spread_prefix = ">" if metric.spread_label.startswith(">") else ""
    spread = f"{spread_prefix}{metric.spread_floor * 100.0:.3f}%"
    return (
        f"{label}\n"
        f"flat={point.flat:.3f}, s={point.s_multi:.3f}, d={point.d_multi:.3f}\n"
        f"{dr_line}\n"
        f"eff:{metric.efficiency_label} mem={metric.memory_label} spread:{spread}"
    )


def print_results(
    selected_by_label: dict[str, Point],
    selected_metrics: dict[tuple[float, float, float], FixedCurveEquivalence],
    diagnostics: list[PhaseDiag],
    plot_path,
    summary_path,
    elapsed_s: float,
    config: SearchConfig,
) -> None:
    print()
    for label in (
        "Recommended",
        "Aggressive",
        "Calm",
        "Efficiency Potential",
        "Memory Potential",
        "Max Spread",
        "Original",
    ):
        if label in selected_by_label:
            print_point(label, selected_by_label[label], selected_metrics, config)
    for label in sorted(k for k in selected_by_label if k.startswith("Inspect ")):
        print_point(label, selected_by_label[label], selected_metrics, config)
    if "Recommended" not in selected_by_label:
        print(
            "Recommended            none: no final candidate was strictly northeast of the fixed target DR point"
        )

    print()
    print("Diagnostics:")
    for diag in diagnostics:
        skipped = ""
        if diag.notes and diag.notes.get("screened_unsafe"):
            skipped = f" skipped_unsafe={diag.notes['screened_unsafe']}"
        print(
            f"  {diag.name:28} weight={diag.weight:g} evaluated={diag.evaluated} "
            f"promoted={diag.promoted} extra={diag.pareto_extra}{skipped} "
            f"elapsed={diag.elapsed_s:.1f}s"
        )
    print()
    print(f"Plot: {plot_path}")
    print(f"Summary: {summary_path}")
    print(f"Elapsed: {elapsed_s:.1f}s")
