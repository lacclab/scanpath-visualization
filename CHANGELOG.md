# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Guided setup wizard for uploads.** Uploading now walks you through a
  step-by-step flow — upload your tables (in a collapsible box that tucks away
  once files are in), then map the trial id, participants, texts, and the few
  required fields. Live counts ("✓ N trials detected — make sure this is the
  number you expect", participants, texts) confirm each step. After loading it
  collapses into a compact **Data & mapping** panel you can re-open to tweak.
- **Reusable named datasets.** Finishing an upload saves it (under a name you
  choose) as a first-class data source. Switching between it, the bundled demo,
  and other datasets is instant — no re-uploading or re-mapping. Add another via
  the sidebar's **➕ Add data** button.
- **Optional participant & text.** Datasets without a participant column (a
  single anonymous reader) or without a text/passage column now load and
  visualize — the app fills sensible defaults and hides the Participant / Text
  selectors when a dimension has only one value. The wizard maps **Participants**
  and **Texts** in their own steps (mirroring the Trial id: one shared picker, an
  opt-in per-table override, and several columns composable into one id).
- **Choose the text-highlight column.** A new *Highlight words by* picker chooses
  which per-word column (the OneStop answer span by default, or any boolean
  column in your data) is marked on the reading text.
- **Pick the saccade colour.** A colour picker under *Saccades* sets the saccade
  line and arrow colour (two-trial comparisons keep their per-scanpath colours).
- **Saved configs record their export date** (and the data source + app version),
  shown when you restore one — including a confirmation line under the wizard's
  *Restore a saved setup* box naming where the loaded setup came from.
- **Keep only the fields you need.** The wizard auto-detects reading measures,
  linguistic features and condition columns and lets you drop the ones you don't
  need; everything unmapped is pruned before processing, which is the main
  speed-up on wide datasets (hundreds of columns).
- **Dynamic trial filters.** The **Filter trials** panel now offers a value
  picker for whichever condition fields your dataset actually has (and which you
  chose to keep), instead of a fixed set of OneStop-specific conditions.
- **Display calibration in the loading flow.** The **Experimental Setup**
  (monitor size, font, line spacing, background) is now part of the wizard, so
  the scanpath is true-to-scale from the first render; it stays adjustable from
  the sidebar afterwards.
- **PoTeC loader.** `sps.load_potec(root, download=True)` /
  `scanpath-studio render --potec` load the Potsdam Textbook Corpus end-to-end
  — its filename-encoded ids and separate character-AoI coordinates can't go
  through the generic upload flow. An in-app **Public datasets** source (with
  a dataset registry built for more corpora) ships feature-flagged off
  (`SCANPATH_PUBLIC_DATASETS=1` to preview); it will be enabled in a future
  release.

### Changed
- **Much faster on large datasets.** Switching trial, participant, or settings
  on a big dataset (hundreds of columns, many trials) is now near-instant
  instead of taking minutes — heavy work is cached and thinned, with a visible
  loading spinner while the first render builds.
- **One Visualization controls panel.** The former *Advanced styling* expander
  is merged into **Visualization controls**, grouped by layer (Fixations,
  Saccades, Text, Heatmap) with thin separators; per-layer size/colour/colorscale
  controls sit under each layer, and the fixation/heatmap colour-range sliders
  show whenever they apply (no longer gated behind *Show color bars*).
- **Clearer trial-id mapping.** One shared **Trial ID** picker applies to every
  table by default (with an opt-in *Different trial-id columns per table* toggle
  that now keeps the columns you already picked instead of clearing them), and
  defaults to composing the participant and text ids when both exist. If the
  per-table trial counts disagree the wizard says so.
- **Tidier wizard.** The required-field steps are renamed **Column mapping:
  Fixations** and **Column mapping: Text & Interest Areas** (each collapsible); a
  single **Fields to keep** list replaces the split *Optional fields* / *Extra
  columns* pickers; the four upload boxes are narrower; and uploaded data now
  defaults to a 2560×1440 monitor.
- **Integer colour ranges.** The fixation- and heatmap-colour range sliders read
  as whole numbers instead of long decimals.
- **Refreshed the welcome tour** to match the current app (the *Experimental
  Setup* / merged *Visualization controls* panels, the guided upload).
- **Simpler data-source picker.** "Use bundled demo" is now **Bundled Demo**;
  the synthetic trial is no longer offered as a fresh source; a grayed-out
  **Public Datasets** entry previews what's coming.
- **Clearer Bulk Export controls.** The whole-dataset and filtered scopes are
  now both **All** / **All filtered trials** options inside the *Trials to
  include* picker (no separate checkbox), and the Scope section ends with a live
  "*N of M trials will be exported*" count. Figures are listed one per row
  (PDF → SVG → PNG), default to **PDF + Config** only, the plot-config checkbox
  is renamed **Config** with a short explanation, and the PNG-scale stepper is
  compact and only shown when PNG is ticked.
- **Faster bulk figure export.** Rasterizing PNG/SVG/PDF for many trials now
  reuses one persistent Kaleido browser for the whole batch instead of
  cold-starting Chrome per trial — quicker on large exports and no more
  per-trial "Resorting to unclean kill browser." log noise.

### Fixed
- **Colour bars no longer squash the scanpath.** Turning on *Show color bars* or
  colouring fixations by a categorical field used to shrink the plot and throw
  off its aspect ratio (the reading text then overflowed its word boxes). The
  colour bar / legend now sits in reserved margin, so the plot keeps its true
  scale either way.
- Dropped the stale **Reading regime** line from the Text & question panel, and
  fixed the per-trial annotations note that pointed at an *Annotations* sidebar
  panel that no longer exists (it lives in **💾 Save & restore**).

## [0.19.1] - 2026-06-14

### Added
- **Zipped tables.** Upload boxes now accept `.zip` archives (e.g.
  `data.csv.zip`) wrapping any supported format (csv/tsv/parquet/feather); a
  multi-member zip is concatenated like a multi-file upload.

### Changed
- Raised the max upload size from 500 MB to 5000 MB.

### Fixed
- **Save & restore** no longer crashes when files are uploaded — the upload
  widgets were being swept into the config's column mapping, which isn't
  JSON-serializable.
- Highlighted text in **Trial metadata** and **Paragraph & question** (critical
  / distractor spans, difficulty / preview rows) is now readable in dark mode —
  the light highlight backgrounds now pin a dark text color instead of
  inheriting the theme's light one.

## [0.19.0] - 2026-06-13

### Added
- **Dark mode.** Ships a polished dark theme for the app chrome (☰ →
  **Settings → Appearance**, or follows your OS). The scanpath plot stays light
  in both themes, so it always reproduces the experiment's stimulus faithfully.
- **Raw-gaze-only datasets.** Upload just a raw-gaze table (no words or
  fixations) and visualize the gaze trace — the Scanpath Visualization tab draws
  the time-coloured gaze scatter, the trial picker and Data Statistics work off
  the raw gaze, and the fixation-only views (animation, Generations) show a
  "needs a fixations table" note.
- **Upload a config to restore it.** The sidebar *💾 Save & restore* panel gains a
  JSON uploader that re-applies a previously downloaded config — layers, coloring,
  sizing, text/highlighting, canvas, axes, trial selection, and annotations —
  silently skipping anything that doesn't fit the loaded data.

### Changed
- **Reorganized layout & usability pass.** The **Animated Scanpath** tab folds
  into the main tab (now **Scanpath Visualization**) as an **Animate** checkbox;
  bulk export moves to its own **Bulk Export** tab (with an *export the whole
  dataset* option); **Multiple Comparison** is renamed **Generations (WIP)**.
  In the side panel: a **Trial Info** header (showing the compared trial's info
  too), the trial selectors below it, annotations above a collapsible metadata
  table, and an **Export** toggle. Participant-mode selection now offers trial
  index / text / trial id. Defaults flip to **Fixation index off, saccade
  arrows on**; "Color fixations by line" is now a `line` option in *Color
  fixations by*; Text-highlighting and Heatmap-style options grey out when their
  layer is off. The animation's playback speed, info box, and play / pause /
  restart + scrub controls all move into one place (speed + info under the
  Animate toggle, transport below the plot); default playback is now ×4. The
  monitor/font controls (renamed **Experimental Setup**) move under 📂 Data.
- **Plot configuration + Annotations merged into 💾 Save & restore.** One sidebar
  panel saves the full figure configuration, all annotations, **and** the data
  source + column mapping + app version to a single JSON file (and restores the
  settings + annotations) — capturing more state than before (text sizing,
  highlighting, background).

### Fixed
- **Animated scanpath text is now true-to-scale.** Its transport controls used
  to shrink the equal-aspect plot, leaving the word labels oversized relative to
  the boxes; the figure now reserves the control space without shrinking the
  plot, so the animation matches the static view exactly.
- **The tutorial is now a guided spotlight tour.** The welcome opens centered
  like a modal over a closed sidebar — appearing as soon as the page opens,
  not after the first full render; the following steps open the sidebar and
  drop to a corner card that scrolls the relevant panel into view and pulses
  an outline around it. The previous all-dialog style remains one constant
  away (`tour.TOUR_STYLE = "dialog"`).
- **README trimmed and refreshed** — a concise rewrite (264 → 157 lines)
  reflecting the current five-tab app, with a regenerated hero GIF and
  screenshot.
- **Simpler, more general column mapping.** The Words/IA word-box mapping is now
  a single coordinate-format selector (edges ↔ origin+size) plus four fields
  instead of eight, and column auto-detection matches names case- and
  separator-insensitively (`IA_LEFT`, `ia_left` and `Participant ID` all
  resolve). Required-field markers now match what the loader actually needs.
- **Search-engine discoverability.** Keyword-rich page title and tagline, so the
  `<title>` and the social/search description Streamlit Cloud serves to crawlers
  are no longer brand-only.
- **Mapping panels grouped under each upload box.** On the Upload source each
  table's column-mapping panel now sits directly beneath its own upload box
  (Words/IA, Fixations, Raw gaze), raw-gaze mapping is a first-class peer, and
  every field has a `?` tooltip describing what it is and how it's used.

### Fixed
- Animated scanpath order numbers no longer glide in from the top-left corner —
  they now snap on at their fixation. The labels render in a constant-length
  text trace, so a new number turns on in place instead of a fresh node
  flashing at the (0,0) origin before placement.
- The tutorial's Skip/Done buttons now close the dialog instantly instead of
  leaving it on screen for the ~10 s full-app rerun.

### CI
- **Leaner AppTest suite.** Data-independent integration tests now boot the tiny
  synthetic trial instead of the bundled demo (~10x cheaper per boot, in a
  single run), cutting the AppTest file ~4x. The bundled-demo render still gets
  guardrail coverage.
- **Parallel test runs.** `pytest -n auto` (via `pytest-xdist`) fans the suite
  across the runner's cores, roughly cutting the AppTest-dominated wall-clock to
  a third.
- **Faster dependency install.** The test job installs with `uv` (cached)
  instead of `pip`, trimming the install step that now dominates each run.
- Test on Python 3.14 (the version Streamlit Cloud runs); CI now runs on pull
  requests only. Added a supported-Python-versions badge to the README.
- Added a pull request template (`.github/pull_request_template.md`) with a
  summary/verification prompt and a checklist mirroring the CONTRIBUTING and CI
  checks (tests, ruff, `[Unreleased]` changelog, dependency manifests).
- The publish workflow now creates a GitHub release (with the matching CHANGELOG
  section as the body) on every `v*` tag, alongside the PyPI publish and Slack
  post; `scripts/changelog_notes.py` gained a `--format markdown` mode for it.

## [0.18.0] - 2026-06-11

### Added
- **Flexible dataset support.** Load multi-file datasets (several files, a list,
  or a glob — concatenated with a `source_file` tag), single-report datasets
  (words-only or fixations-only), stimulus-level word boxes (broadcast across
  participants), and AOI-sequence fixations (placed at word/character-box
  centers). TSV inputs are now read directly. The upload panel takes several
  files per table, and either table alone.
- **First-visit tutorial.** A welcome dialog walks through the app's main
  surfaces on first entry (suppressed for embeds and deep links); replay it
  anytime via **🎓 Show tutorial** at the bottom of the sidebar.

### Changed
- **Raw data is shown while the column mapping is incomplete.** A missing
  required column no longer halts the whole app — the uploaded tables render in
  the Raw Data tab so you can see the columns and finish the mapping.
- **About popover in the header.** The LaCC Lab / Code pill links are replaced
  by a single ℹ️ About toggle with credits, the code link, citation guidance,
  and more from the lab.

### Fixed
- The animated scanpath now honours the fixation colour options (Color
  fixations by / by line, colorscale, colour range, colour bar) like the static
  plot; previously they only affected the image. Colours are pinned to the full
  trial's range so they stay stable during playback. Dual-overlay replays keep
  the flat A/B colours.

## [0.17.0] - 2026-06-11

### Added
- **Headless CLI:** `scanpath-studio render` builds one trial's figure straight
  to `.html`/`.png`/`.svg`/`.pdf` (or `--animate` for the HTML replay) without
  launching the app — `--sample` or `--words`/`--fixations`, `--list-trials`,
  per-layer `--no-*` toggles. Bare `scanpath-studio` still launches the app.
- **Public Python API:** `import scanpath_studio as sps` →
  `load_scanpath_data`, `load_sample_data`, `list_trials`,
  `compute_word_metrics`, `plot_scanpath`, `animate_scanpath`, `save_figure` —
  the app's canonical figures, programmatically.

## [0.16.3] - 2026-06-11

### Added
- **Composite trial ids are spelled out in the trial header.** Each remaining
  part gets its own labeled line (e.g. `Repeated reading trial: False`) next to
  Participant / Text, instead of only the joined id.

## [0.16.2] - 2026-06-11

### Added
- **Composite trial IDs in the column mapping.** The *Trial ID* row of every
  Column mapping panel (Words/IA, Fixations, Raw gaze) is now a multiselect:
  pick several columns and the app builds a unique trial ID on the fly by
  joining their values with `_` — for datasets with no precomputed
  unique-trial column (e.g. OneStop-style participant + paragraph +
  repeated-reading). A multi-column choice is authoritative: it overrides any
  raw `unique_trial_id` column and skips the repeated-reading `_r2` suffix
  fallback. Selections that reference columns missing from a newly uploaded
  file are dropped and re-proposed automatically.
- **Composite trials are selectable by their parts.** When the trial id is
  composite, *Select trials by → Trial* breaks it into one cascading selector
  per component (e.g. Text → Participant → repeated-reading) — each narrowed by
  the previous picks — instead of a single opaque `a_b_c` dropdown, mirroring
  the existing Text / Participant modes. Single-column trial ids keep the plain
  unique-trial dropdown.

### Fixed
- **Single-trial data no longer breaks "Select trials by → Participant".**
  With one trial per participant (a one-trial upload, the synthetic source, or
  filters narrowing to one), the Participant picker rendered a one-option
  `st.select_slider`, which crashes the Streamlit frontend (`RangeError: min
  (0) is equal/bigger than max (0)`) and blanks the tab. The picker now shows
  the lone trial as static text. Affected every tab's trial picker (Interactive
  Plot, Animated Scanpath + its overlay, Multiple Comparison, Data Statistics).

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

[0.19.0]: https://github.com/lacclab/scanpath-studio/releases/tag/v0.19.0
[0.14.0]: https://github.com/lacclab/scanpath-studio/releases/tag/v0.14.0
[0.13.0]: https://github.com/lacclab/scanpath-studio/releases/tag/v0.13.0
[0.12.0]: https://github.com/lacclab/scanpath-studio/releases/tag/v0.12.0
[0.11.0]: https://github.com/lacclab/scanpath-studio/releases/tag/v0.11.0
