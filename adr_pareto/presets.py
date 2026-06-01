from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import SearchConfig


DEFAULT_QUALITY_PRESET = "medium-high"
FULL_HORIZON_VALUES = {
    "days": 1825,
    "deck_size": 10000,
    "learn_limit": 10,
}


@dataclass(frozen=True)
class PresetSpec:
    name: str
    label: str
    description: str
    recommended_by_default: bool
    values: dict[str, Any]


QUALITY_PRESETS: dict[str, PresetSpec] = {
    "potato": PresetSpec(
        name="potato",
        label="Potato",
        description="Emergency coarse full-horizon search; normally not recommended.",
        recommended_by_default=False,
        values={
            **FULL_HORIZON_VALUES,
            "phase1_eval_weight": 300.0,
            "phase2_eval_weight": 600.0,
            "phase3_eval_weight": 600.0,
            "phase4_eval_weight": 600.0,
            "final_eval_weight": 12000.0,
            "fixed_curve_coarse_weight": 3000.0,
            "fixed_curve_refine_weight": 30000.0,
            "fixed_curve_coarse_step_pct": 4.0,
            "fixed_curve_refine_step_pct": 1.0,
            "fixed_curve_initial_radius_pct": 0.4,
            "fixed_curve_adapt_margin_pct": 0.2,
            "fixed_curve_adapt_top_per_bucket": 1,
            "fixed_curve_adapt_max_points": 12,
            "phase1_flat_step": 0.08,
            "phase1_flat_half_steps": 3,
            "phase1_s_step": 0.05,
            "phase1_s_max": 0.25,
            "phase1_d_step": 0.05,
            "phase1_d_min": -0.20,
            "phase1_expand_rounds": 0,
            "promote_recommended": 6,
            "promote_efficiency_potential": 3,
            "promote_memory_potential": 3,
            "promote_pareto_extra": 8,
            "phase4_seeds_per_objective": 1,
            "phase4_max_steps": 1,
            "final_candidate_limit": 32,
            "max_spread_final_candidates": 2,
            "final_shortlist_recommended": 24,
            "final_shortlist_efficiency": 12,
            "final_shortlist_memory": 12,
            "final_shortlist_frontier": 16,
            "safety_checks": 3000,
        },
    ),
    "lite": PresetSpec(
        name="lite",
        label="Lite",
        description="Fast normal run for hard decks, high target DR, or weaker computers.",
        recommended_by_default=True,
        values={
            **FULL_HORIZON_VALUES,
            "phase1_eval_weight": 600.0,
            "phase2_eval_weight": 1200.0,
            "phase3_eval_weight": 1200.0,
            "phase4_eval_weight": 1200.0,
            "final_eval_weight": 60000.0,
            "fixed_curve_coarse_weight": 5000.0,
            "fixed_curve_refine_weight": 30000.0,
            "fixed_curve_coarse_step_pct": 2.5,
            "fixed_curve_refine_step_pct": 0.5,
            "fixed_curve_initial_radius_pct": 0.7,
            "fixed_curve_adapt_margin_pct": 0.25,
            "fixed_curve_adapt_top_per_bucket": 3,
            "fixed_curve_adapt_max_points": 30,
            "phase1_flat_step": 0.06,
            "phase1_flat_half_steps": 6,
            "phase1_s_step": 0.035,
            "phase1_s_max": 0.28,
            "phase1_d_step": 0.035,
            "phase1_d_min": -0.21,
            "phase1_expand_rounds": 0,
            "promote_recommended": 12,
            "promote_efficiency_potential": 6,
            "promote_memory_potential": 6,
            "promote_pareto_extra": 20,
            "phase4_seeds_per_objective": 1,
            "phase4_max_steps": 2,
            "final_candidate_limit": 60,
            "max_spread_final_candidates": 4,
            "final_shortlist_recommended": 50,
            "final_shortlist_efficiency": 25,
            "final_shortlist_memory": 25,
            "final_shortlist_frontier": 35,
            "safety_checks": 3000,
        },
    ),
    "medium": PresetSpec(
        name="medium",
        label="Medium",
        description="Balanced full-horizon run for harder decks with less breadth than Medium-High.",
        recommended_by_default=True,
        values={
            **FULL_HORIZON_VALUES,
            "phase1_eval_weight": 2000.0,
            "phase2_eval_weight": 4000.0,
            "phase3_eval_weight": 4000.0,
            "phase4_eval_weight": 4000.0,
            "final_eval_weight": 100000.0,
            "fixed_curve_coarse_weight": 10000.0,
            "fixed_curve_refine_weight": 80000.0,
            "fixed_curve_coarse_step_pct": 1.5,
            "fixed_curve_refine_step_pct": 0.3,
            "fixed_curve_initial_radius_pct": 1.0,
            "fixed_curve_adapt_margin_pct": 0.2,
            "fixed_curve_adapt_top_per_bucket": 6,
            "fixed_curve_adapt_max_points": 60,
            "phase1_flat_step": 0.05,
            "phase1_flat_half_steps": 6,
            "phase1_s_step": 0.025,
            "phase1_s_max": 0.275,
            "phase1_d_step": 0.025,
            "phase1_d_min": -0.225,
            "phase1_expand_rounds": 1,
            "promote_recommended": 28,
            "promote_efficiency_potential": 14,
            "promote_memory_potential": 14,
            "promote_pareto_extra": 50,
            "phase4_seeds_per_objective": 3,
            "phase4_max_steps": 4,
            "final_candidate_limit": 100,
            "max_spread_final_candidates": 8,
            "final_shortlist_recommended": 90,
            "final_shortlist_efficiency": 50,
            "final_shortlist_memory": 50,
            "final_shortlist_frontier": 70,
            "safety_checks": 3000,
        },
    ),
    "medium-high": PresetSpec(
        name="medium-high",
        label="Medium-High",
        description="Current baseline behavior.",
        recommended_by_default=True,
        values={
            **FULL_HORIZON_VALUES,
            "phase1_eval_weight": 2000.0,
            "phase2_eval_weight": 4000.0,
            "phase3_eval_weight": 4000.0,
            "phase4_eval_weight": 4000.0,
            "final_eval_weight": 200000.0,
            "fixed_curve_coarse_weight": 10000.0,
            "fixed_curve_refine_weight": 80000.0,
            "fixed_curve_coarse_step_pct": 1.0,
            "fixed_curve_refine_step_pct": 0.2,
            "fixed_curve_initial_radius_pct": 1.0,
            "fixed_curve_adapt_margin_pct": 0.2,
            "fixed_curve_adapt_top_per_bucket": 8,
            "fixed_curve_adapt_max_points": 80,
            "phase1_flat_step": 0.04,
            "phase1_flat_half_steps": 8,
            "phase1_s_step": 0.02,
            "phase1_s_max": 0.26,
            "phase1_d_step": 0.02,
            "phase1_d_min": -0.20,
            "phase1_expand_rounds": 8,
            "promote_recommended": 50,
            "promote_efficiency_potential": 25,
            "promote_memory_potential": 25,
            "promote_pareto_extra": 100,
            "phase4_seeds_per_objective": 6,
            "phase4_max_steps": 8,
            "final_candidate_limit": 180,
            "max_spread_final_candidates": 12,
            "final_shortlist_recommended": 120,
            "final_shortlist_efficiency": 70,
            "final_shortlist_memory": 70,
            "final_shortlist_frontier": 100,
            "safety_checks": 3000,
        },
    ),
    "high": PresetSpec(
        name="high",
        label="High",
        description="Higher-confidence run when the user can wait.",
        recommended_by_default=True,
        values={
            **FULL_HORIZON_VALUES,
            "phase1_eval_weight": 8000.0,
            "phase2_eval_weight": 20000.0,
            "phase3_eval_weight": 50000.0,
            "phase4_eval_weight": 50000.0,
            "final_eval_weight": 500000.0,
            "fixed_curve_coarse_weight": 20000.0,
            "fixed_curve_refine_weight": 160000.0,
            "fixed_curve_coarse_step_pct": 1.0,
            "fixed_curve_refine_step_pct": 0.2,
            "fixed_curve_initial_radius_pct": 1.2,
            "fixed_curve_adapt_margin_pct": 0.2,
            "fixed_curve_adapt_top_per_bucket": 10,
            "fixed_curve_adapt_max_points": 110,
            "phase1_flat_step": 0.04,
            "phase1_flat_half_steps": 9,
            "phase1_s_step": 0.02,
            "phase1_s_max": 0.28,
            "phase1_d_step": 0.02,
            "phase1_d_min": -0.22,
            "phase1_expand_rounds": 8,
            "promote_recommended": 65,
            "promote_efficiency_potential": 35,
            "promote_memory_potential": 35,
            "promote_pareto_extra": 140,
            "phase4_seeds_per_objective": 8,
            "phase4_max_steps": 10,
            "final_candidate_limit": 360,
            "max_spread_final_candidates": 16,
            "final_shortlist_recommended": 120,
            "final_shortlist_efficiency": 70,
            "final_shortlist_memory": 70,
            "final_shortlist_frontier": 120,
            "safety_checks": 5000,
        },
    ),
}


def preset_names() -> tuple[str, ...]:
    return tuple(QUALITY_PRESETS)


def apply_quality_preset(config: SearchConfig, preset_name: str) -> SearchConfig:
    try:
        preset = QUALITY_PRESETS[preset_name]
    except KeyError as exc:
        valid = ", ".join(preset_names())
        raise ValueError(
            f"Unknown quality preset {preset_name!r}; valid presets: {valid}"
        ) from exc
    return replace(config, quality_preset=preset.name, **preset.values)
