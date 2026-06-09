# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.16.1] - 2026-06-09

### Internal
- **Slack release notifications.** A successful PyPI publish now posts a message
  to the lab Slack (`#scanpath-studio`) — including these release notes — via a
  webhook in the `publish.yml` workflow. `scripts/changelog_notes.py` renders the
  matching changelog section to Slack mrkdwn. No changes to the packaged app.

## [0.16.0] - 2026-06-09

### Added
- **Export the animated scanpath as GIF or MP4** (in addition to the existing
  interactive HTML). The Animated Scanpath tab gains an export-format selector;
  GIF/MP4 rasterize every animation frame through Kaleido (the same headless
  Chrome the PNG/SVG/PDF export uses) and encode them — Pillow for GIF,
  imageio-ffmpeg (bundled ffmpeg, no system package) for MP4. The clip
  reproduces the on-screen Play exactly: every frame held for the average
  duration so the runtime equals the quoted playback time, the slider's
  "Elapsed: X.Xs" readout re-drawn as a per-frame annotation, and the
  play/slider chrome stripped. A single browser is kept warm across frames
  (`kaleido.start_sync_server` → `calc_fig_sync`), so rendering is ~0.1–0.25 s
  per frame instead of a ~10 s cold start each. A progress bar tracks the
  render, the result is cached in session state (so the download button
  survives reruns), and long readings can be capped to a fixed frame count
  (duration preserved) to keep export quick. MP4 is far smaller than GIF and is
  recommended for long readings. New module `scanpath_studio/animation_export.py`
  (+ `tests/test_animation_export.py`).

## [0.15.1] - 2026-06-06

### Changed
- **Multiple Comparison tab layout.** The model-generated scanpath grid now
  renders directly beneath the real scanpath, with the similarity-score table
  below it (previously the table came first). The grid is the visual payload
  compared against the real scanpath, so it now sits adjacent to it.
- **Cite Levenshtein (1966) as the source of NLD.** The NLD metric description
  now credits the underlying edit distance to Levenshtein (1966) and notes
  Eyettention (Deng et al. 2023) as a user of the same normalization.
- **Internal: de-duplicated shared geometry/timing helpers** (no behaviour
  change). Fixation→word-id bounding-box assignment now lives in a single
  `measures._assign_word_ids_single` (used by both
  `measures.assign_fixations_to_words` and
  `similarity.assign_single_trial_word_ids`); the recorded-vs-synthetic
  timestamp heuristic lives in `measures.rebased_fixation_onsets` (used by both
  the similarity time-curve and the animation clock); and
  `model_scanpaths._ordered_word_rows` now reuses `measures.cluster_word_lines`
  for its line-clustering fallback. The 50 px line-misregistration tolerance and
  the 0.5 real-timestamp threshold are now single module constants
  (`LINE_MISREGISTRATION_PX`, `REAL_TIMESTAMP_DWELL_FRAC`). The animation clock's
  threshold was reconciled to compare against the full summed durations (matching
  the documented intent and the similarity time-curve).

### Removed
- The two explanatory captions under the similarity table (the header-arrow
  legend and the per-metric description block). The table keeps its direction
  arrows and best-model highlight.

## [0.15.0] - 2026-06-06

### Added
- **Multiple Comparison tab.** Shows a real scanpath on top, then a grid of
  model-generated scanpaths over the same text, and a similarity table scoring
  each model against the real reading. Until real model outputs are connected,
  the model scanpaths are reproducible, reading-like **synthetic placeholders**
  (deterministic per trial; a 🎲 Regenerate button re-rolls them). The number of
  models is inferred from the data; a "Grid columns" control sets the layout.
- **Scanpath similarity metrics** (`scanpath_studio/similarity.py`): **NLD**
  (Normalized Levenshtein Distance on the word-index/AOI sequence — the metric
  reported by Eyettention) computed for real, with ScanMatch / MultiMatch /
  Scasim registered as labeled placeholders. Table headers carry a direction
  arrow (↓ lower-is-better / ↑ higher-is-better) and highlight the best model.
- **Fixation-index range slider** to window which fixations are drawn and scored.
- **Metric-convergence plots**: NLD vs cumulative fixation index and NLD vs
  elapsed reading time, one line per model, computed over the full reading.

## [0.14.0] - 2026-06-06

### Changed
- **Renamed the project to Scanpath Studio.** The PyPI distribution is now
  `scanpath-studio` (was `scanpath-visualization-app`), the import package is
  `scanpath_studio` (was `scanpath_visualization_app`), the console entry point
  is `scanpath-studio`, and the repository moved to `lacclab/scanpath-studio`.
  Update imports and reinstall: `pip install scanpath-studio`.
- `requirements.txt` now uses compatible-release pins (`~=`) so the Streamlit
  Cloud demo stays on a known-good minor without drifting.

### Added
- Project metadata and docs: `CHANGELOG.md`, `CITATION.cff` (GitHub "Cite this
  repository"), and `CONTRIBUTING.md`.
- A demo GIF generator (`scripts/make_demo_gif.py`) and README animation/screenshot.

### Fixed
- Release pipeline no longer double-fires: dropped the redundant
  `release: published` trigger and added `skip-existing: true`, so a re-run can't
  fail on an already-published version.

### Internal
- Single-source the version (`pyproject.toml` reads `scanpath_studio.__version__`
  dynamically), so a release bumps one file.
- CI cancels superseded in-progress runs (`concurrency`).

## [0.13.0] - 2026-06-06

### Added
- **Interpolated fixation heatmap** — a smooth, word-box-independent density over the
  fixations themselves (duration-weighted when the metric is duration, blurred with a
  numpy-only separable Gaussian). Selected via a new "Heatmap style" radio
  (Word boxes | Interpolated).
- **Saccade direction arrows** — an arrowhead at each saccade midpoint, rotated to gaze
  direction (accounts for the reversed, screen-space y-axis).

## [0.12.0] - 2026-06-06

### Added
- **Simultaneous second scanpath** in the Animated Scanpath tab — two readings of the
  same text co-animated on a shared real-time clock (per-reading rebased `timestamp_ms`),
  with blue/red trails, per-scanpath saccades and current-fixation highlights, a legend,
  and an elapsed-time slider. Opt-in via an "Overlay a second scanpath" toggle with an
  independent trial picker; defaults to another reading of the same text (preferring the
  same participant).

### Changed
- Unified the two animation builders into a single `make_scanpath_animation`; the quoted
  playback time now equals the real animation runtime (one timeline source).
- Lowered the per-frame floor from 50 ms to ~16 ms (one 60 fps frame), making the speed
  slider far more effective at high speeds.
- **Interactive Plot cosmetics:** moved the trial-metadata field picker into the sidebar;
  folded the reading-time / word-count / fixation-count stats into the trial summary table
  as rows; replaced the out-of-box caption with a "Fixations in word boxes → N / N" row;
  single-click "Download HTML" (headless Chrome via Kaleido now only spins up for
  PNG/SVG/PDF).

## [0.11.0] - 2026-06-03

### Added
- **Trial annotations** — per-trial favorites, tags, and notes held in session state, with
  JSON download/restore.
- **Trial filtering & grouping** — sidebar panel to filter by condition (Hunting/Gathering,
  difficulty, repeated reading, correctness) and by annotation state (favorites/tags).
- **Questions/answers panel** — reading regime, selected answer + correctness, and
  answer/distractor spans annotated with whether each was fixated.
- **Plot options** — background color (incl. gray), highlight + count out-of-text
  fixations, and color fixations by line.
- Trial metadata shown per-trial in the comparison view.
- **Synthetic ground-truth test trial** — a hand-built trial shared by the test suite and a
  new "Synthetic test trial" data source, so the visualization can be checked against
  documented expected values.

### Fixed
- Static image export (PNG/SVG/PDF) on Streamlit Cloud, which failed because Kaleido v1
  needs Chrome — ship `packages.txt` (chromium), add a browser-free HTML save format, and
  give a clearer error message.
- Bundled raw-gaze sample always filtering to 0 rows (it was recorded for a participant
  absent from the demo) — synthesize a raw-gaze path from a real bundled trial and add a
  regression guard.

### Changed
- Test suite grown from 85 to 114 tests.

[0.14.0]: https://github.com/lacclab/scanpath-studio/releases/tag/v0.14.0
[0.13.0]: https://github.com/lacclab/scanpath-studio/releases/tag/v0.13.0
[0.12.0]: https://github.com/lacclab/scanpath-studio/releases/tag/v0.12.0
[0.11.0]: https://github.com/lacclab/scanpath-studio/releases/tag/v0.11.0
