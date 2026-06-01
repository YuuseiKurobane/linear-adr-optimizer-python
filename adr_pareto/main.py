from __future__ import annotations

import time
from pathlib import Path

from .candidates import dedupe_points
from .config import SearchConfig, parse_args
from .eval_backend import attach_dr_summary, make_eval_lib, parse_dr
from .export_io import load_export_row, write_summary
from .fixed_curve import FixedCurveManager
from .models import SearchResult
from .phases import (
    evaluate_unique,
    phase4_seed_profiles,
    phase_diag,
    run_micro_hillclimb,
    run_phase1,
    run_refinement_phase,
)
from .point_store import PointStore
from .ranking import equivalence_map, pareto_frontier, select_promotions
from .reporting import format_point_block, print_intro, print_results
from .safety import safety_pool
from .selection import (
    add_reference_labels,
    choose_final_candidates,
    choose_points,
    labels_by_key,
    reference_labels,
)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


def run(config: SearchConfig) -> SearchResult:
    row = load_export_row(config.export, config.preset)
    target_dr = parse_dr(
        config.target_dr if config.target_dr is not None else row["desired_retention"]
    )
    refs = reference_labels(config)

    print_intro(row, target_dr, config)

    eval_lib = make_eval_lib(row, config)
    global_start = time.perf_counter()
    store = PointStore()
    fixed = FixedCurveManager.build(eval_lib, config, target_dr)

    phase1, phase1_promoted, phase1_diags = run_phase1(eval_lib, fixed, config, store)
    fixed.ensure_for_points(
        phase1, "phase1", config.seed + 760, config.scout_potential_band_pct
    )
    phase1_promoted, _ = select_promotions(
        phase1,
        fixed.target_fixed,
        fixed.fixed_curve,
        fixed.fixed_env,
        fixed.target_dr,
        config,
        config.scout_potential_band_pct,
    )

    phase2, phase2_promoted, _, diag2 = run_refinement_phase(
        eval_lib=eval_lib,
        phase_name="phase2",
        centers=phase1_promoted,
        base_pool=[],
        previous_weight=config.phase1_eval_weight,
        current_weight=config.phase2_eval_weight,
        steps=(config.phase2_flat_step, config.phase2_s_step, config.phase2_d_step),
        seed=config.seed + 2000,
        adapt_seed=config.seed + 770,
        fixed=fixed,
        config=config,
        store=store,
        pareto_as_render_only=False,
    )

    phase3_new, phase3_promoted, phase3_render_extra, diag3 = run_refinement_phase(
        eval_lib=eval_lib,
        phase_name="phase3",
        centers=phase2_promoted,
        base_pool=phase2,
        previous_weight=config.phase2_eval_weight,
        current_weight=config.phase3_eval_weight,
        steps=(config.phase3_flat_step, config.phase3_s_step, config.phase3_d_step),
        seed=config.seed + 3000,
        adapt_seed=config.seed + 780,
        fixed=fixed,
        config=config,
        store=store,
        pareto_as_render_only=True,
    )
    phase3_pool = dedupe_points([*phase2, *phase3_new])

    phase4_seeds = phase4_seed_profiles(phase3_promoted, fixed, config)
    phase4, diag4 = run_micro_hillclimb(
        eval_lib, phase4_seeds, phase3_pool, fixed, config, store
    )
    fixed.ensure_for_points(
        phase4, "phase4", config.seed + 790, config.scout_potential_band_pct
    )
    print(
        f"[phase 4] seeds={len(phase4_seeds)} visited={len(phase4)} elapsed={diag4.elapsed_s:.1f}s"
    )

    start = time.perf_counter()
    remote_pool = dedupe_points([*phase3_promoted, *phase4])
    all_computed_pool = dedupe_points(
        [*phase1, *phase2, *phase3_new, *phase4, *phase3_render_extra]
    )
    fixed.ensure_for_points(
        remote_pool, "prefinal", config.seed + 795, config.final_potential_band_pct
    )
    final_candidates, max_spread_prefinal = choose_final_candidates(
        remote_pool,
        all_computed_pool,
        fixed,
        refs,
        config,
    )
    fixed.ensure_for_points(
        max_spread_prefinal,
        "maxspread",
        config.seed + 797,
        config.final_potential_band_pct,
    )
    final = evaluate_unique(
        eval_lib,
        final_candidates,
        config.final_eval_weight,
        config.seed + 5000,
        config,
        store,
        enforce_quadrant=False,
    )
    fixed.ensure_for_points(
        final, "final", config.seed + 799, config.final_potential_band_pct
    )
    final = attach_dr_summary(eval_lib, final, config.final_eval_weight, config)
    diag5 = phase_diag(
        "phase5.final",
        config.final_eval_weight,
        start,
        final,
        generated=len(final_candidates),
    )
    print(
        f"[phase 5] candidates={len(final_candidates)} evaluated={len(final)} "
        f"elapsed={diag5.elapsed_s:.1f}s"
    )

    selected = choose_points(final, fixed, config)
    selected = add_reference_labels(selected, final, refs)
    grouped_labels = labels_by_key(selected)
    selected_metrics = equivalence_map(selected.values(), fixed.fixed_env)

    point_only_label = _point_only_label(config)
    if point_only_label is not None:
        txt_path = write_point_only(
            point_only_label,
            selected,
            selected_metrics,
            config,
            row,
        )
        print()
        print(txt_path.read_text(encoding="utf-8").rstrip())
        diagnostics = [*fixed.diagnostics, *phase1_diags, diag2, diag3, diag4, diag5]
        return SearchResult(
            plot_path=txt_path,
            summary_path=txt_path,
            selected_by_label=selected,
            labels_by_key=grouped_labels,
            diagnostics=diagnostics,
        )

    from .plotting import plot_results, write_plot_html

    plot_path = plot_results(
        phase1=phase1,
        phase2=phase2,
        phase3=phase3_new,
        phase4=phase4,
        phase3_render_extra=phase3_render_extra,
        final=final,
        fixed=fixed,
        selected_by_label=selected,
        labels_by_key=grouped_labels,
        selected_metrics=selected_metrics,
        config=config,
        row=row,
    )

    diagnostics = [*fixed.diagnostics, *phase1_diags, diag2, diag3, diag4, diag5]
    summary_path = plot_path.with_suffix(".json")
    write_summary(
        summary_path,
        row=row,
        target_dr=target_dr,
        config=config,
        selected_by_label=selected,
        labels_by_key=grouped_labels,
        selected_metrics=selected_metrics,
        fixed_curve_points=fixed.fixed_curve,
        fixed_curve_refined_points=fixed.refined_points_by_pct,
        fixed_curve_envelope=fixed.fixed_env.points,
        final_frontier=pareto_frontier(safety_pool(final, config)),
        phase3_render_extra=phase3_render_extra,
        max_spread_prefinal=max_spread_prefinal,
        diagnostics=diagnostics,
        plot_layers={
            "phase1_safe": [point for point in phase1 if point.safe],
            "phase1_unsafe": [point for point in phase1 if not point.safe],
            "phase2": phase2,
            "phase3": phase3_new,
            "phase4": phase4,
            "phase3_render_extra": phase3_render_extra,
        },
    )
    write_plot_html(plot_path, summary_path)

    elapsed = time.perf_counter() - global_start
    print_results(
        selected,
        selected_metrics,
        diagnostics,
        plot_path,
        summary_path,
        elapsed,
        config,
    )
    return SearchResult(
        plot_path=plot_path,
        summary_path=summary_path,
        selected_by_label=selected,
        labels_by_key=grouped_labels,
        diagnostics=diagnostics,
    )


def _point_only_label(config: SearchConfig) -> str | None:
    if config.recommended_only:
        return "Recommended"
    if config.aggressive_only:
        return "Aggressive"
    if config.calm_only:
        return "Calm"
    return None


def write_point_only(
    label: str,
    selected_by_label: dict[str, object],
    selected_metrics: dict,
    config: SearchConfig,
    row: dict,
) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    preset_name = row.get("deck_preset", {}).get("name", "preset")
    safe_name = "".join(
        ch if ch.isalnum() or ch in ".-_" else "_" for ch in preset_name
    )
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path = config.output_dir / f"adr_{safe_name}_{label.lower()}_{timestamp}.txt"
    point = selected_by_label.get(label)
    if point is None:
        text = f"{label}\nnone\n"
    else:
        text = format_point_block(label, point, selected_metrics, config) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
