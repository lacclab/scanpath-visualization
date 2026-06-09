# scanpath_studio

Streamlit workbench for exploring eye-tracking-while-reading scanpaths: word boxes, fixations, saccades, density heatmaps, and per-word reading measures (FFD/FPRT/RPD/TFD, regressions). Top-level [README](../README.md) covers install/release.

## Run

```bash
streamlit run scanpath_studio/app.py        # dev
python -m scanpath_studio                    # packaged
scanpath-studio                                  # console entry point
```

`__main__.py` wraps `streamlit.web.cli` and forwards `sys.argv`. Default port 8501.

## Modules

- [app.py](app.py) — orchestrator: page config, URL-preset parsing (`?source=onestop&participant=pid&trial=N&show_saccades=1`), sidebar render, data load, filter, tab dispatch.
- [data.py](data.py) — schema inference for EyeLink/Gazepoint/snake_case columns (`propose_*_schema`), normalization, `load_onestop_server_bundle` (fast-path per-pid Parquet, falls back to full CSV.zip), `compute_word_metrics`.
- [plots.py](plots.py) — Plotly figure builders: `make_scanpath_figure` (core layered: words + fixations + saccades + heatmap), `make_comparison_figure`, `make_scanpath_animation`, bar/histogram figs. **True-to-scale text:** word labels are sized in *data* space (`_word_label_font_px`: `box_height / line_spacing`, capped by box width via `_width_fit_font`) then converted to screen px with `_display_scale`, so the reading text always fills the real line slot instead of being a fixed font size. `line_spacing` (default 3 = OneStop's one-blank-line-above-and-below) and `scale_text_to_boxes` thread through every spatial builder.
- [measures.py](measures.py) — `assign_fixations_to_words` (bbox containment + nearest-word within 50 px), then per-word FFD/FPRT/RPD/TFD/skip/regression. Used only when pre-aggregated columns (e.g. `IA_FIRST_FIXATION_DURATION`) are absent. Also geometry helpers `cluster_word_lines`, `fixation_in_text_mask`, `assign_fixation_lines` powering the out-of-text + color-by-line plot options.
- [tabs.py](tabs.py) — four tab renderers: `render_single_trial_tab` (main plot + export), `render_animation_tab` (replay), `render_raw_data_tab` (paginated tables with CSV/Parquet download), `render_data_statistics_tab` (summary stats).
- [controls.py](controls.py) — field-spec dicts (`WORD_FIELD_SPECS`, etc.), `column_mapping_ui`, `sidebar_controls` (viz toggles, colorscales, marker size ranges, background color, out-of-text/by-line toggles), `sidebar_trial_filters` (condition + annotation trial filtering).
- [export.py](export.py) — `ExportOptions` dataclass + `bulk_export` (zip per-trial figs + tabular data across filtered trials).
- [animation_export.py](animation_export.py) — rasterize the animated-scanpath `go.Figure` to GIF/MP4 (`export_animation`). Applies each `go.Frame` onto a controls-stripped copy of the base figure, renders every frame through one warm Kaleido browser (`start_sync_server`/`calc_fig_sync` — per-call `to_image` cold-starts Chrome at ~10 s each), then encodes: Pillow for GIF, imageio-ffmpeg (60 fps + per-frame repeats via error diffusion → exact runtime) for MP4. Uniform per-frame duration matches the on-screen Play, so clip runtime == quoted playback time.
- [annotations.py](annotations.py) — per-trial favorites/tags/notes in session state keyed by `(participant_id, trial_id)`; pure serialize/deserialize core + sidebar JSON download/restore + per-trial editor (`render_trial_annotations`).
- [synthetic.py](synthetic.py) — a hand-built 6-word / 2-line ground-truth trial with documented `EXPECTED` measures. Single source of truth shared by `tests/` (via `tests/synthetic_data.py`) and the in-app **"Synthetic test trial"** data source (`load_synthetic_data`), so the viz can be eyeballed against known values.
- [onestop_shard.py](onestop_shard.py) — one-shot prep: shards the ~15 GB OneStop lacclab CSV exports into per-pid Parquet under `$ONESTOP_DATA_DIR/by_pid/{ia,fixations}/<pid>.parquet`. Run this once per data refresh.
- [utils.py](utils.py) — trial selection UI, trial stats, combo-option builders, friendly labels.
- [constants.py](constants.py), [styles.py](styles.py) — colors/scales/sizes, custom CSS.
- [update_sample_data.py](update_sample_data.py) — regenerate the bundled 3-pid demo from full CSVs; also `synthesize_raw_gaze` / `--raw-gaze-only` to (re)build the demo raw-gaze overlay sample.

## Data flow

1. Load: upload, bundled `sample_data/`, the synthetic test trial, or OneStop server bundle (`$ONESTOP_DATA_DIR` set).
2. Infer schema → normalize column names to canonical form.
3. Compute reading measures if pre-aggregated columns missing.
4. Filter by participant/trial/paragraph (sidebar).
5. Dispatch to one of four tabs.
6. Optional bulk export.

## OneStop integration

`onestop_shard.py` solves cold-start: full CSV load takes ~3 min, per-pid Parquet shard loads in ~1 sec. When `$ONESTOP_DATA_DIR` is set AND `?participant=<pid>` is in URL, `load_onestop_server_bundle` reads only that pid's shard. Without a pid, it falls back to the full CSV.zip. An external review app embeds this app via iframe with deep-link URL params.

## Gotchas

- URL deep-link presets are applied before widgets render (in `app.py`); changing widget keys breaks deep-links.
- Schema auto-detection prefers EyeLink names over Gazepoint when both match — override via sidebar `column_mapping_ui` if needed.
- Pre-aggregated reading-measure columns (EyeLink IA_* columns) win over computed `measures.py` outputs; the recomputation is a fallback, not the primary path.
- When the source data lacks a `unique_trial_id` column (e.g. OneStop L2 shards), `normalize_*` falls back to `unique_paragraph_id` and disambiguates repeated readings by ranking on `TRIAL_INDEX` — the 2nd reading gets a `_r2` suffix on `trial_id`. Without this, both readings of a repeated-reading trial collapse into one scanpath.
- Bulk export iterates *all* filtered trials — narrow the filter before clicking export on the full OneStop set.
- Static image export (PNG/SVG/PDF) **and the animation GIF/MP4 export** go through Kaleido v1, which needs a Chrome/Chromium binary. Locally run `plotly_get_chrome -y` once; on Streamlit Cloud it's installed via the repo-root `packages.txt` (`chromium`). The **HTML** save format is a browser-free fallback (`fig.to_html`) when no Chrome is available — for the animation tab, HTML is also the only browser-free export (GIF/MP4 raise `AnimationExportError`, surfaced as a warning pointing back to HTML). MP4 needs no system ffmpeg: the `imageio[ffmpeg]` extra bundles the binary.
- The bundled `raw_gaze.{csv,parquet}` is **synthesized** from one real bundled trial's fixations (`update_sample_data.synthesize_raw_gaze`) — OneStop ships no sample-level gaze. It must stay keyed to a (participant, trial) present in the bundled fixations or the app filters it to 0 rows (regression-guarded in `tests/test_raw_gaze_sample.py`). Regenerate with `python -m scanpath_studio.update_sample_data --raw-gaze-only`.
- AOIs are taken from the data's word boxes, not computed. Only fixation→word assignment is derived (`measures.assign_fixations_to_words`); "out-of-text" = a fixation inside no box; "by-line" infers lines from word-box `y` because `line_idx` is usually constant.
- **Don't render the spatial plots with `st.plotly_chart`.** Streamlit pins the chart *width* to the column while keeping the layout *height*, which re-lays-out Plotly to an unknown scale and leaves the px-sized word labels mismatched to the boxes (the true-to-scale text breaks). The single/animation/comparison plots go through `tabs._render_true_scale_chart`, which embeds the figure's own HTML at its exact pixel size and CSS-scales the whole block uniformly to the column width. Figure size is capped (`plots._DISPLAY_MAX_*`) so it fits a research display; PNG export bumps `scale` to stay crisp (SVG/PDF are vector). The embed loads plotly from CDN (needs network) and re-mounts each rerun.
- Annotations (favorites/tags/notes) live in `st.session_state` only — persistence is via the sidebar JSON download/restore. They don't survive a hard refresh unless re-imported, and on the shared cloud demo they're per-session. Annotation-based trial filters reflect a star/tag on the *next* rerun (the editor writes the store after the sidebar filter reads it).
