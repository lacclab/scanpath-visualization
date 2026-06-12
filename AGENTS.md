# AGENTS.md — Scanpath Studio

Architectural map for AI agents (Claude, Copilot) modifying this code.

## Project

Interactive Streamlit workbench for **eye-tracking-while-reading** scanpath
visualization. Targeted at reading-research / NLP audiences. Distributed as the
PyPI package `scanpath-studio`; deployable to Streamlit Community
Cloud via `streamlit_app.py` at the repo root.

Demo corpus: 3 participants × 2 articles × {Adv, Ele} from **OneStop Eye
Movements** (Berzak, Malmaud, Shubi, Meiri, Lion, Levy, *Scientific Data* 2025;
[doi:10.1038/s41597-025-06272-2](https://doi.org/10.1038/s41597-025-06272-2);
docs at <https://lacclab.github.io/OneStop-Eye-Movements/>), shipped under
`sample_data/` as both CSV and Parquet. Linguistic features
(`gpt2_surprisal`, `wordfreq_frequency`, `subtlex_frequency`, `universal_pos`,
`ptb_pos`, `dependency_relation`, etc.) are preserved.

## Architecture

```text
scanpath_studio/
├─ app.py            entry point: page config, data load, trial filters, dispatch to tabs
├─ tabs.py           tab implementations (Interactive, Reading Measures, Animation, Raw Data, Statistics)
├─ controls.py       sidebar viz controls + column-mapping override UI + trial-filter panel
├─ data.py           schema inference, normalization, filtering (incl. condition/annotation trial filters), sample loaders
├─ measures.py       canonical reading measures (FFD, FPRT, RPD, TFD, regressions) + geometry helpers (line clustering, in-text test)
├─ plots.py          Plotly figure builders (scanpath, animation, comparison, bar, histogram); background color, out-of-text + by-line fixation options
├─ export.py         configurable bulk-export module (PNG/SVG/JSON/CSV/Parquet/mega-table)
├─ annotations.py    per-trial favorites/tags/notes (session state) + JSON import/export
├─ synthetic.py      hand-built ground-truth trial (shared by tests + the "Synthetic test trial" data source)
├─ utils.py          trial-combo construction, trial-selection UI, comparison helpers
├─ constants.py      palette, defaults, citation metadata
├─ styles.py         injected CSS
├─ api.py            headless public API (load/normalize, plot_scanpath, animate_scanpath, save_figure)
├─ cli.py            console entry: `run` launches the app, `render` builds figures headless via api.py
├─ __main__.py       `python -m scanpath_studio` → cli.main
├─ __init__.py       exposes __version__, main(), and lazy re-exports of the api.py surface
└─ sample_data/      bundled demo corpus (CSV + Parquet)
```

### Pipeline

```text
uploaded/sample table(s) → infer_*_schema → normalize_* → canonical columns
                       → filter_data → build_combo_options → tab renderers
                       → make_*_figure / compute_word_metrics / bulk_export
```

### Canonical columns

After `normalize_words` / `normalize_fixations`:

- **Words**: `participant_id`, `trial_id`, `paragraph_id` (and `unique_*` when
  present), `word_id`, `text`, `line_idx`, `x`, `y`, `width`, `height`. Plus
  EyeLink IA columns (`IA_*` renamed to `first_fixation_ms`, etc.) and
  linguistic features when shipped.
- **Fixations**: `participant_id`, `trial_id`, `paragraph_id`, `x`, `y`,
  `duration_ms`, `timestamp_ms`, `word_id`, `pass_index`, `saccade_type`,
  `saccade_amplitude`, `eye`, `noise_flag`, `order_in_trial`.

### Reading measures

`measures.py` computes per (participant, trial, word):
`first_fixation_ms` (FFD), `first_pass_gaze_duration_ms` (FPRT),
`regression_path_duration_ms` (RPD / go-past), `total_fixation_duration_ms`
(TFD), `n_fixations`, `skip_flag`, `regression_in_flag`,
`regression_out_flag`, `first_fix_x/y`. Fixations are enriched with
`saccade_amplitude`, `progression`, `is_regression`. Pre-computed IA values on
the words table take precedence over computed ones.

### Areas of interest (AOIs)

AOIs (word interest areas) are **not computed** by the app — they come directly
from the data's word bounding boxes, supplied either as `(x, y, width, height)`
or as EyeLink's `(IA_LEFT, IA_RIGHT, IA_TOP, IA_BOTTOM)` (which `normalize_words`
converts to `x/y/width/height`). The only thing derived from geometry is the
**fixation→word assignment** in `measures.assign_fixations_to_words`: bounding-box
containment, then nearest word-center within 50 px, else `word_id = NaN`. That
assignment feeds the reading measures and the "out-of-text" flag
(`measures.fixation_in_text_mask`); "color by line" derives visual lines from
word-box `y` clustering (`measures.cluster_word_lines`) because `line_idx` is
often a constant in IA exports.

### Trial annotations & filtering

`annotations.py` keeps per-trial favorites / tags / notes in session state
(keyed by `(participant_id, trial_id)`), with a pure serialize/deserialize core
and JSON download/restore in the sidebar. `controls.sidebar_trial_filters` +
`data.filter_trials` / `data.filter_to_keys` narrow the trial pool by condition
(Hunting/Gathering via `question_preview`, difficulty, repeated reading,
correctness) and by annotation state (favorites / tags) before `build_combo_options`.

## Build / Lint / Test

```bash
# Install in editable mode
pip install -e ".[test]"

# Run app
streamlit run streamlit_app.py
uv run streamlit run streamlit_app.py

# Tests
pytest                              # all 114 tests
pytest tests/test_measures.py       # one file
pytest --cov=scanpath_studio --cov-report=term

# Lint
ruff check --exclude other_vis .
ruff check --select I --fix --exclude other_vis .
ruff format --exclude other_vis .

# Regenerate bundled sample data (needs the full OneStop CSVs under sample_data/OneStop/)
python -m scanpath_studio.update_sample_data
```

CI on GitHub Actions runs pytest on Python 3.11/3.12/3.13/3.14 plus ruff
lint+format checks on every pull request.

## Code style

- `from __future__ import annotations` at the top of every Python file.
- snake_case functions/variables, PascalCase classes, UPPER_SNAKE_CASE
  constants. Files: snake_case.py.
- Sorted imports via ruff (`--select I`). stdlib → third-party → local.
- Use `pd.DataFrame` and `np.ndarray` as type hints directly.
- Add return type hints to public functions.
- Streamlit: `@st.cache_data` for expensive loaders. Use `st.warning` (not
  `st.toast`) for per-rerender warnings.
- Plotting: y-axis inverted (`y_range = [max, min]`) for screen coordinates.
  Word boxes + word-level heatmap use `layout.shapes`. Sacccades are a SINGLE
  trace with `None` separators (never one-trace-per-saccade).
- Centralized palette / sizing in `constants.py`. Marker sizes come from
  `plots._compute_marker_sizes` so single-trial and comparison figures render
  identically.

## Testing patterns

- `tests/conftest.py` exposes `sample_words_df`, `sample_fixations_df`,
  `normalized_words_df`, `normalized_fixations_df`, `sample_raw_gaze_df`.
- `tests/test_measures.py` covers FFD, FPRT, RPD, TFD, skip, regressions on a
  synthetic 4-word layout.
- `tests/synthetic_data.py` is a fully-specified 6-word / 2-line trial with
  hand-traced `EXPECTED` values (incl. a regression and one out-of-text
  fixation); `tests/test_synthetic.py` asserts every measure and geometry
  helper exactly. `tests/test_annotations.py` / `tests/test_filters.py` cover
  the annotation core and trial filtering.
- `tests/test_smoke.py` exercises the full pipeline (load → infer → normalize
  → plot) against the bundled sample, including a perf regression that asserts
  saccades collapse to a single trace.
- `tests/test_apptest.py` uses `streamlit.testing.v1.AppTest` to boot the
  whole app and verify title rendering + no `st.error` calls.
- `tests/test_export.py` checks zip structure, CSV/Parquet selection, and
  progress callback behavior.

## Adding a new column convention

Update the candidate lists in `data.py` (e.g. `WORD_X_CANDIDATES`,
`FIX_SACCADE_AMPLITUDE_CANDIDATES`). `pick_column` walks the list and picks the
first existing column.

## Adding a new reading measure

1. Compute it in `measures.compute_per_word_measures` per trial.
2. Add it to `metric_map` in `data.normalize_words` if it can come pre-computed
   from EyeLink IA columns.
3. Surface it in `controls.preferred_color_fields` (if useful for coloring) and
   in `tabs._MEASURE_OPTIONS` (so the bar plot picker shows it).
4. Add a test under `tests/test_measures.py`.

## Adding a new figure type

1. Add a `make_*_figure` function in `plots.py` using the helpers
   `_compute_axis_ranges`, `_compute_marker_sizes`, `_saccade_segments`,
   `_add_word_label_trace`.
2. Wire it into a tab via `tabs.py` with a Plotly chart call.
3. Add a smoke test in `tests/test_smoke.py` that builds the figure against
   the bundled sample.

## Releasing

1. Bump `version` in `pyproject.toml` and `scanpath_studio/__init__.py`.
2. Commit; tag with `v<version>`; push the tag.
3. The `Publish to PyPI` GitHub Actions workflow builds the wheel + sdist and
   publishes via PyPI Trusted Publishing (requires `pypi` environment set up
   on GitHub with the project name `scanpath-studio`).
