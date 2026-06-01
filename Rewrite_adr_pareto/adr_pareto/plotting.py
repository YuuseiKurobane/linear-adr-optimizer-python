from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from .config import SearchConfig
from .fixed_curve import FixedCurveManager, should_label_fixed_pct
from .models import FixedCurveEquivalence, Point, PointKey
from .ranking import format_dr_label, pareto_frontier
from .safety import safety_pool


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def label_text(
    labels: list[str],
    point: Point,
    metrics: dict[PointKey, FixedCurveEquivalence],
) -> str:
    metric = metrics[point.key]
    dr_line = (
        f"dr={point.dr_mean:.4f} band={point.dr_spread * 100.0:.2f}%"
        if point.dr_samples
        else "dr=n/a band=n/a"
    )
    return (
        f"{' / '.join(labels)}\n"
        f"flat={point.flat:.3f} s={point.s_multi:.3f} d={point.d_multi:.3f}\n"
        f"{dr_line}\n"
        f"eff={metric.efficiency_label} mem={metric.memory_label} spread={metric.spread_label}"
    )


def plot_results(
    *,
    phase1: list[Point],
    phase2: list[Point],
    phase3: list[Point],
    phase4: list[Point],
    phase3_render_extra: list[Point],
    final: list[Point],
    fixed: FixedCurveManager,
    selected_by_label: dict[str, Point],
    labels_by_key: dict[PointKey, list[str]],
    selected_metrics: dict[PointKey, FixedCurveEquivalence],
    config: SearchConfig,
    row: dict,
) -> Path:
    if not config.matplotlib:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        preset_name = _safe_name(row.get("deck_preset", {}).get("name", "preset"))
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return config.output_dir / f"adr_pareto_{preset_name}_{timestamp}.html"
    return _plot_results_matplotlib(
        phase1=phase1,
        phase2=phase2,
        phase3=phase3,
        phase4=phase4,
        phase3_render_extra=phase3_render_extra,
        final=final,
        fixed=fixed,
        selected_by_label=selected_by_label,
        labels_by_key=labels_by_key,
        selected_metrics=selected_metrics,
        config=config,
        row=row,
    )


def write_plot_html(plot_path: Path, summary_path: Path) -> None:
    if plot_path.suffix.casefold() != ".html":
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    web_root = os.path.relpath(PACKAGE_ROOT / "web", start=plot_path.parent).replace(
        "\\", "/"
    )
    summary_json = _script_json(summary)
    source_json = _script_json(summary_path.name)
    plot_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ADR Pareto Plot</title>
  <link rel="stylesheet" href="{web_root}/adr_plot.css?v=rewrite-1">
</head>
<body>
  <main class="app-shell">
    <header class="toolbar">
      <div class="title-block">
        <h1>ADR Pareto Plot</h1>
        <p id="summary-title">Loading summary...</p>
      </div>
      <form id="summary-form" class="summary-form">
        <input id="summary-path" name="summary" type="text" autocomplete="off" spellcheck="false" aria-label="Summary JSON path">
        <button type="submit">Load</button>
        <label class="file-button">
          <span>Open JSON</span>
          <input id="summary-file" type="file" accept="application/json,.json">
        </label>
      </form>
    </header>
    <section class="plot-panel">
      <div class="plot-frame">
        <div id="plot" class="plot" aria-label="ADR Pareto Plot"></div>
        <aside id="result-box" class="result-box" aria-label="ADR plot labels"></aside>
      </div>
      <div id="status" class="status" role="status"></div>
    </section>
  </main>
  <script>
    window.ADR_INITIAL_SUMMARY = {summary_json};
    window.ADR_INITIAL_SOURCE = {source_json};
  </script>
  <script src="{web_root}/vendor/plotly-3.5.1.min.js"></script>
  <script src="{web_root}/adr_plot.js?v=rewrite-1"></script>
</body>
</html>
""",
        encoding="utf-8",
    )


def _script_json(value) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _plot_results_matplotlib(
    *,
    phase1: list[Point],
    phase2: list[Point],
    phase3: list[Point],
    phase4: list[Point],
    phase3_render_extra: list[Point],
    final: list[Point],
    fixed: FixedCurveManager,
    selected_by_label: dict[str, Point],
    labels_by_key: dict[PointKey, list[str]],
    selected_metrics: dict[PointKey, FixedCurveEquivalence],
    config: SearchConfig,
    row: dict,
) -> Path:
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt

    config.output_dir.mkdir(parents=True, exist_ok=True)
    preset_name = _safe_name(row.get("deck_preset", {}).get("name", "preset"))
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output = config.output_dir / f"adr_pareto_{preset_name}_{timestamp}.png"

    plt.figure(figsize=(13.5, 8.3))
    layers = [
        ("Phase 1 safe", [p for p in phase1 if p.safe], "#6baed6", "o", 13, 0.16),
        ("Phase 1 unsafe", [p for p in phase1 if not p.safe], "#cc4c4c", "x", 11, 0.12),
        ("Phase 2 refine", phase2, "#9ecae1", "o", 13, 0.22),
        ("Phase 3 refine", phase3, "#74c476", "o", 13, 0.24),
        ("Phase 4 hillclimb", phase4, "#fd8d3c", "o", 20, 0.55),
        ("Phase 3 frontier render-only", phase3_render_extra, "#756bb1", "D", 18, 0.35),
    ]
    for label, points, color, marker, size, alpha in layers:
        if points:
            plt.scatter(
                [p.memorized_cards for p in points],
                [p.memorized_per_minute for p in points],
                s=size,
                alpha=alpha,
                c=color,
                marker=marker,
                label=label,
            )

    final_frontier = pareto_frontier(safety_pool(final, config))
    if final_frontier:
        plt.plot(
            [p.memorized_cards for p in final_frontier],
            [p.memorized_per_minute for p in final_frontier],
            c="#111111",
            lw=2.2,
            label="Final verified ADR frontier"
            + (" (safety skipped)" if config.ignore_safety else ""),
        )

    fixed_curve = sorted(fixed.fixed_curve, key=lambda item: item[0])
    plt.plot(
        [p.memorized_cards for _, p in fixed_curve],
        [p.memorized_per_minute for _, p in fixed_curve],
        c="#737373",
        lw=1.4,
        alpha=0.82,
        label="Fixed DR curve",
    )
    _plot_fixed_curve_labels(plt, path_effects, fixed_curve, fixed.target_dr, config)
    _plot_selected_points(plt, selected_by_label, labels_by_key, selected_metrics)

    plt.title(
        f"FSRS-ADR Pareto Search: {row.get('deck_preset', {}).get('name', 'preset')} "
        f"target DR {fixed.target_dr:.3f}"
    )
    plt.xlabel("Average memorized cards")
    plt.ylabel("Average memorized cards per daily minute")
    plt.grid(True, alpha=0.25)
    plt.margins(x=0.04, y=0.14)
    plt.legend(
        loc="best",
        fontsize=7,
        labelspacing=0.55,
        handlelength=1.7,
        handletextpad=0.7,
        borderpad=0.55,
        markerscale=0.95,
    )
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()
    return output


def _plot_fixed_curve_labels(
    plt,
    path_effects,
    fixed_curve: list[tuple[float, Point]],
    target_dr: float,
    config: SearchConfig,
) -> None:
    target_pct = target_dr * 100.0
    for dr, point in fixed_curve:
        pct = dr * 100.0
        if not should_label_fixed_pct(pct, target_pct, config):
            continue
        is_target = abs(pct - target_pct) <= 1e-6
        fixed_label = (
            f"Target {format_dr_label(dr)}" if is_target else format_dr_label(dr)
        )
        annotation = plt.annotate(
            fixed_label,
            (point.memorized_cards, point.memorized_per_minute),
            xytext=(7, -14 if is_target else 5),
            textcoords="offset points",
            fontsize=7,
            color="#4d4d4d",
            va="top" if is_target else "bottom",
            arrowprops={
                "arrowstyle": "-",
                "color": "#666666",
                "lw": 0.65,
                "alpha": 0.85,
            },
            zorder=4,
        )
        annotation.set_path_effects(
            [path_effects.withStroke(linewidth=2.6, foreground="white", alpha=0.78)]
        )


def _plot_selected_points(
    plt,
    selected_by_label: dict[str, Point],
    labels_by_key: dict[PointKey, list[str]],
    selected_metrics: dict[PointKey, FixedCurveEquivalence],
) -> None:
    colors = {
        "Recommended": "#d4a017",
        "Efficiency Potential": "#2ca25f",
        "Memory Potential": "#756bb1",
        "Aggressive": "#e6550d",
        "Calm": "#00897b",
        "Max Spread": "#1f78b4",
        "Original": "#d62728",
    }
    point_by_key = {point.key: point for point in selected_by_label.values()}
    for key, labels in labels_by_key.items():
        point = point_by_key[key]
        primary = _primary_label(labels)
        color = colors.get(primary, "#238b45")
        plt.scatter(
            [point.memorized_cards],
            [point.memorized_per_minute],
            s=48 if "Recommended" in labels else 36,
            c=color,
            edgecolors="#ffffff",
            linewidths=0.7,
            zorder=7,
            label=label_text(labels, point, selected_metrics),
        )


def _primary_label(labels: list[str]) -> str:
    priority = [
        "Recommended",
        "Aggressive",
        "Calm",
        "Max Spread",
        "Efficiency Potential",
        "Memory Potential",
        "Original",
    ]
    for label in priority:
        if label in labels:
            return label
    return labels[0]


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", value.strip(), flags=re.UNICODE)
    return cleaned.strip("_") or "preset"
