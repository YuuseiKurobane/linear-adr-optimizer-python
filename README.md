Original concept by 1DWalker: [srs-simulator on the `fsrs-sa` branch](https://github.com/1DWalker/srs-simulator/tree/fsrs-sa).

# Linear ADR Optimizer Python

This repository is a standalone Python and Rust rewrite of the ADR Pareto search workflow from the FSRS-SA simulator concept. It searches for linear Adaptive Desired Retention parameters for FSRS-style scheduling, compares those candidates against fixed desired-retention baselines, and writes plot/report artifacts for choosing practical ADR settings.

The project is intended for command-line experiments now and for future Anki add-on integration later.

## What It Simulates

FSRS schedules reviews using memory state such as stability and difficulty. A fixed desired retention (fixed DR) uses one target retention value for all cards. ADR varies desired retention by card state using a linear formula:

```text
flat + s_multi * ln(stability) + d_multi * difficulty
```

The optimizer evaluates many `(flat, s_multi, d_multi)` candidates. For each candidate, the Rust simulator estimates long-run review workload, memorized-card count, efficiency, safety behavior, and final DR spread. The Python search then compares candidates to the fixed-DR curve and chooses labels such as `Recommended`, `Aggressive`, `Calm`, `Efficiency Potential`, and `Memory Potential`.

The default simulation horizon is 1825 days with a deck size of 10000 cards and a daily learn limit of 10. Quality presets change search effort and verification weight, not the simulated horizon.

## Repository Layout

- `adr_pareto/`: Python package for CLI parsing, staged search, candidate ranking, safety checks, fixed-DR comparison, report generation, and Plotly web output.
- `adr_pareto/web/`: static Plotly renderer assets for HTML output and future add-on bundling.
- `rust/`: Rust simulator worker used by Python for fast batch evaluation.
- `helper/`: optional location for prebuilt helper binaries downloaded from GitHub Actions artifacts.
- `exports/`: Anki export JSONL input files.
- `outputs/`: generated HTML, JSON, PNG, and TXT result files.
- `adr_pareto_search.py`: compatibility launcher for `python adr_pareto_search.py ...`.
- `ADR_GUI.txt`: notes for the future GUI/add-on workflow.
- `removed_functions_porting_rust.txt`: notes about behavior intentionally removed during the rewrite.
- `next_tasks_todo.txt`: project maintenance tasks that are not part of normal usage.

## Inputs

The optimizer reads `adr-input-*.jsonl` export files from `exports/`. These files can be created with an Anki add-on:

[TODO: INSERT ANKI ADD-ON LINK HERE]

Pass a specific JSONL file with `--export`, or pass a directory containing `adr-input-*.jsonl`. If a directory is used, the newest matching file is selected.

The `--preset` argument selects a row by deck preset or deck name. Exact matches are preferred before partial matches.

## Build The Rust Helper

Build the helper once before running a full search:

```powershell
cd C:\Users\admin\Documents\Codex\linear-adr-optimizer-python\rust
cargo build --release --bin adr-simulator-helper
```

Python automatically looks for the helper in:

- `rust/target/release/`
- `rust/target/debug/`
- `helper/`
- `helper/<platform-artifact-name>/`

You can override helper discovery with the `ADR_SIMULATOR_HELPER` environment variable.

## Run The Search

From the repository root:

```powershell
cd C:\Users\admin\Documents\Codex\linear-adr-optimizer-python
python -m adr_pareto --export ".\exports\adr-input-20260528-213412.jsonl" --preset "Yuusei" --target-dr 85
```

The wrapper script is equivalent:

```powershell
python adr_pareto_search.py --export ".\exports\adr-input-20260528-213412.jsonl" --preset "Yuusei" --target-dr 85
```

If `--export` is omitted, the CLI uses `exports/` and picks the newest `adr-input-*.jsonl` file:

```powershell
python -m adr_pareto --preset "Yuusei" --target-dr 85
```

Use `--matplotlib` to generate a PNG instead of the default standalone HTML plot:

```powershell
python -m adr_pareto --preset "Yuusei" --target-dr 85 --matplotlib
```

## Useful CLI Flags

- `--export`: JSONL file or directory containing `adr-input-*.jsonl`.
- `--preset`: deck preset or deck name selector.
- `--target-dr`: target desired retention. Values above `1.0` are interpreted as percentages, so `85` means `0.85`.
- `--quality-preset`: search effort preset: `potato`, `lite`, `medium`, `medium-high`, or `high`.
- `--days`, `--deck-size`, `--learn-limit`: simulation horizon and deck-growth controls.
- `--seed`: base seed for reproducible staged search.
- `--threads`: Rust worker thread count. `0` uses all available CPU cores.
- `--matplotlib`: write a PNG plot instead of the default HTML plot.
- `--recommended-only`, `--aggressive-only`, `--calm-only`: write only one TXT result.
- `--ignore-safety`: skip safety checks and safety filtering.

## Search Flow

1. Load the selected export row and target desired retention.
2. Build a fixed-DR curve where `s_multi=0` and `d_multi=0`.
3. Run a coarse phase 1 search near the target DR.
4. Promote promising points for recommended-like, efficiency, memory, and Pareto-frontier behavior.
5. Refine promoted neighborhoods in phase 2 and phase 3.
6. Run phase 4 micro-hillclimbs around strong candidates.
7. Build a final shortlist and evaluate it at higher weight.
8. Attach final DR summaries, choose labels, and write outputs.

## Outputs

Normal runs write to `outputs/`:

- `adr_pareto_<preset>_<timestamp>.html`
- `adr_pareto_<preset>_<timestamp>.json`

With `--matplotlib`, the plot is written as:

- `adr_pareto_<preset>_<timestamp>.png`

Point-only modes write TXT files named like:

- `adr_<preset>_<label>_<timestamp>.txt`

The JSON summary includes selected labels, grouped labels by coordinate, fixed-curve points, final frontier points, phase diagnostics, reference points, and the parsed configuration.