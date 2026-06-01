from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_DIR = REPO_ROOT / "exports"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_ORIGINAL = (1.57, 0.135, -0.085)


@dataclass(frozen=True)
class SearchConfig:
    quality_preset: str = "medium-high"
    export: Path = DEFAULT_EXPORT_DIR
    preset: str = "Yuusei"
    target_dr: float | None = None
    days: int = 1825
    deck_size: int = 10000
    learn_limit: int = 10
    seed: int = 1234
    threads: int = 0
    matplotlib: bool = False
    recommended_only: bool = False
    aggressive_only: bool = False
    calm_only: bool = False

    phase1_eval_weight: float = 2000.0
    phase2_eval_weight: float = 4000.0
    phase3_eval_weight: float = 4000.0
    phase4_eval_weight: float = 4000.0
    final_eval_weight: float = 30000.0
    dr_prune_weight: float = 1.0

    phase1_flat_step: float = 0.04
    phase1_flat_half_steps: int = 8
    phase1_s_step: float = 0.02
    phase1_s_max: float = 0.26
    phase1_d_step: float = 0.02
    phase1_d_min: float = -0.20
    phase1_expand: bool = True
    phase1_expand_rounds: int = 8
    phase1_expand_batch: int = 2
    phase1_expand_overflow_factor: float = 2.0

    phase2_flat_step: float = 0.02
    phase2_s_step: float = 0.01
    phase2_d_step: float = 0.01
    phase3_flat_step: float = 0.01
    phase3_s_step: float = 0.005
    phase3_d_step: float = 0.005
    phase4_flat_step: float = 0.002
    phase4_s_step: float = 0.001
    phase4_d_step: float = 0.001
    phase4_seeds_per_objective: int = 6
    phase4_max_steps: int = 8

    promote_recommended: int = 50
    promote_efficiency_potential: int = 25
    promote_memory_potential: int = 25
    promote_pareto_extra: int = 100
    bridge_midpoint_limit: int = 50
    experimental_bridge_midpoint_neighborhoods: bool = False
    final_candidate_limit: int = 180
    max_spread_final_candidates: int = 12
    final_shortlist_recommended: int = 120
    final_shortlist_efficiency: int = 70
    final_shortlist_memory: int = 70
    final_shortlist_frontier: int = 100

    scout_potential_band_pct: float = 0.3
    final_potential_band_pct: float = 0.1
    aggressive_calm_regret_pct: float = 0.50

    safety_s_max: float = 1000.0
    safety_checks: int = 3000
    ignore_safety: bool = False
    legacy_unsafe_plot_display: bool = False

    include_original: bool = False
    original: tuple[float, float, float] = DEFAULT_ORIGINAL
    inspect_point: tuple[tuple[float, float, float], ...] = ()

    fixed_dr_start_pct: float = 60.0
    fixed_dr_end_pct: float = 96.0
    fixed_curve_coarse_weight: float = 10000.0
    fixed_curve_refine_weight: float = 80000.0
    fixed_curve_coarse_step_pct: float = 1.0
    fixed_curve_refine_step_pct: float = 0.2
    fixed_curve_initial_radius_pct: float = 1.0
    fixed_curve_adapt_margin_pct: float = 0.2
    fixed_curve_adapt_top_per_bucket: int = 8
    fixed_curve_adapt_max_points: int = 80
    fixed_dr_label_step_pct: float = 10.0

    output_dir: Path = DEFAULT_OUTPUT_DIR


def build_parser() -> argparse.ArgumentParser:
    from .presets import DEFAULT_QUALITY_PRESET, preset_names

    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Modular staged search for linear FSRS-ADR Pareto points.",
    )
    parser.add_argument(
        "--quality-preset",
        choices=preset_names(),
        default=DEFAULT_QUALITY_PRESET,
        help="Speed/accuracy preset. CLI defaults to medium-high; automatic recommendation belongs in the GUI.",
    )
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--preset", default="Yuusei")
    parser.add_argument("--target-dr", type=float, default=None)
    parser.add_argument("--days", type=int, default=1825)
    parser.add_argument("--deck-size", type=int, default=10000)
    parser.add_argument("--learn-limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Rust worker threads. 0 uses all available CPU cores.",
    )
    parser.add_argument(
        "--matplotlib",
        action="store_true",
        help="Write the legacy Matplotlib PNG instead of the default Plotly HTML.",
    )
    point_only = parser.add_mutually_exclusive_group()
    point_only.add_argument(
        "--recommended-only",
        action="store_true",
        help="Write only the Recommended TXT result artifact; terminal progress and final point still print.",
    )
    point_only.add_argument(
        "--aggressive-only",
        action="store_true",
        help="Write only the Aggressive TXT result artifact; terminal progress and final point still print.",
    )
    point_only.add_argument(
        "--calm-only",
        action="store_true",
        help="Write only the Calm TXT result artifact; terminal progress and final point still print.",
    )

    parser.add_argument("--phase1-eval-weight", type=float, default=2000.0)
    parser.add_argument("--phase2-eval-weight", type=float, default=4000.0)
    parser.add_argument("--phase3-eval-weight", type=float, default=4000.0)
    parser.add_argument("--phase4-eval-weight", type=float, default=4000.0)
    parser.add_argument("--final-eval-weight", type=float, default=30000.0)
    parser.add_argument("--dr-prune-weight", type=float, default=1.0)

    parser.add_argument("--phase1-flat-step", type=float, default=0.04)
    parser.add_argument("--phase1-flat-half-steps", type=int, default=8)
    parser.add_argument("--phase1-s-step", type=float, default=0.02)
    parser.add_argument("--phase1-s-max", type=float, default=0.26)
    parser.add_argument("--phase1-d-step", type=float, default=0.02)
    parser.add_argument("--phase1-d-min", type=float, default=-0.20)
    parser.add_argument(
        "--phase1-expand", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--phase1-expand-rounds", type=int, default=8)
    parser.add_argument("--phase1-expand-batch", type=int, default=2)
    parser.add_argument(
        "--phase1-expand-overflow-factor",
        type=float,
        default=2.0,
        help="Maximum extra coarse-grid indices per expandable side, as a multiple of the initial axis count.",
    )

    parser.add_argument("--phase2-flat-step", type=float, default=0.02)
    parser.add_argument("--phase2-s-step", type=float, default=0.01)
    parser.add_argument("--phase2-d-step", type=float, default=0.01)
    parser.add_argument("--phase3-flat-step", type=float, default=0.01)
    parser.add_argument("--phase3-s-step", type=float, default=0.005)
    parser.add_argument("--phase3-d-step", type=float, default=0.005)
    parser.add_argument("--phase4-flat-step", type=float, default=0.002)
    parser.add_argument("--phase4-s-step", type=float, default=0.001)
    parser.add_argument("--phase4-d-step", type=float, default=0.001)
    parser.add_argument("--phase4-seeds-per-objective", type=int, default=6)
    parser.add_argument("--phase4-max-steps", type=int, default=8)

    parser.add_argument("--promote-recommended", type=int, default=50)
    parser.add_argument("--promote-efficiency-potential", type=int, default=25)
    parser.add_argument("--promote-memory-potential", type=int, default=25)
    parser.add_argument("--promote-pareto-extra", type=int, default=100)
    parser.add_argument(
        "--bridge-midpoint-limit",
        type=int,
        default=50,
        help="Maximum already-evaluated bridge midpoints promoted by the conservative bridge pass.",
    )
    parser.add_argument(
        "--experimental-bridge-midpoint-neighborhoods",
        action="store_true",
        help="Also evaluate deduped 9-point face neighborhoods around qualifying bridge midpoints.",
    )
    parser.add_argument("--final-candidate-limit", type=int, default=180)
    parser.add_argument("--max-spread-final-candidates", type=int, default=12)
    parser.add_argument("--final-shortlist-recommended", type=int, default=120)
    parser.add_argument("--final-shortlist-efficiency", type=int, default=70)
    parser.add_argument("--final-shortlist-memory", type=int, default=70)
    parser.add_argument("--final-shortlist-frontier", type=int, default=100)

    parser.add_argument(
        "--scout-potential-band-pct",
        type=float,
        default=0.3,
        help="Early adaptive fixed-curve band in DR percentage points.",
    )
    parser.add_argument(
        "--final-potential-band-pct",
        type=float,
        default=0.1,
        help="Final potential-label band in DR percentage points.",
    )
    parser.add_argument(
        "--aggressive-calm-regret-pct",
        type=float,
        default=0.50,
        help="Aggressive/Calm may trail Recommended fixed-curve spread by this many DR percentage points.",
    )

    parser.add_argument("--safety-s-max", type=float, default=1000.0)
    parser.add_argument("--safety-checks", type=int, default=3000)
    parser.add_argument(
        "--ignore-safety",
        action="store_true",
        help="Skip safety checks entirely and do not filter promotion, selection, or final frontier by safety.",
    )
    parser.add_argument(
        "--legacy-unsafe-plot-display",
        action="store_true",
        help="Restore old Phase 1 behavior by fully simulating unsafe candidates so they can appear on the plot.",
    )

    parser.add_argument("--include-original", action="store_true")
    parser.add_argument("--original", nargs=3, type=float, default=DEFAULT_ORIGINAL)
    parser.add_argument(
        "--inspect-point", nargs=3, type=float, action="append", default=[]
    )

    parser.add_argument("--fixed-dr-start-pct", type=float, default=60.0)
    parser.add_argument("--fixed-dr-end-pct", type=float, default=96.0)
    parser.add_argument("--fixed-curve-coarse-weight", type=float, default=10000.0)
    parser.add_argument("--fixed-curve-refine-weight", type=float, default=80000.0)
    parser.add_argument("--fixed-curve-coarse-step-pct", type=float, default=1.0)
    parser.add_argument("--fixed-curve-refine-step-pct", type=float, default=0.2)
    parser.add_argument("--fixed-curve-initial-radius-pct", type=float, default=1.0)
    parser.add_argument("--fixed-curve-adapt-margin-pct", type=float, default=0.2)
    parser.add_argument("--fixed-curve-adapt-top-per-bucket", type=int, default=8)
    parser.add_argument("--fixed-curve-adapt-max-points", type=int, default=80)
    parser.add_argument("--fixed-dr-label-step-pct", type=float, default=10.0)

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def parse_args(argv: list[str] | None = None) -> SearchConfig:
    from .presets import DEFAULT_QUALITY_PRESET, apply_quality_preset

    namespace = build_parser().parse_args(argv)
    raw_args = list(sys.argv[1:] if argv is None else argv)
    explicit_dests = _explicit_dests(build_parser(), raw_args)
    values = vars(namespace)
    values["original"] = tuple(float(v) for v in values["original"])
    values["inspect_point"] = tuple(
        tuple(float(v) for v in point) for point in values["inspect_point"]
    )
    quality_preset = values.get("quality_preset", DEFAULT_QUALITY_PRESET)
    config = apply_quality_preset(SearchConfig(), quality_preset)
    overrides = {
        key: value
        for key, value in values.items()
        if key in explicit_dests and key != "quality_preset"
    }
    return replace(config, **overrides)


def json_safe_config(config: SearchConfig) -> dict:
    out = asdict(config)
    out["export"] = str(config.export)
    out["output_dir"] = str(config.output_dir)
    out["inspect_point"] = [list(point) for point in config.inspect_point]
    out["original"] = list(config.original)
    return out


def _explicit_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    option_to_dest = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
        if action.dest != argparse.SUPPRESS
    }
    explicit: set[str] = set()
    for token in argv:
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest is not None:
            explicit.add(dest)
    return explicit
