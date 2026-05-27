# scanpath_visualization_app

Streamlit workbench for exploring eye-tracking-while-reading scanpaths: word boxes, fixations, saccades, density heatmaps, and per-word reading measures (FFD/FPRT/RPD/TFD, regressions). Top-level [README](../README.md) covers install/release.

## Run

```bash
streamlit run scanpath_visualization_app/app.py        # dev
python -m scanpath_visualization_app                    # packaged
scanpath-visualization                                  # console entry point
```

`__main__.py` wraps `streamlit.web.cli` and forwards `sys.argv`. Default port 8501.

## Modules

- [app.py](app.py) — orchestrator: page config, URL-preset parsing (`?source=onestop&participant=pid&trial=N&show_saccades=1`), sidebar render, data load, filter, tab dispatch.
- [data.py](data.py) — schema inference for EyeLink/Gazepoint/snake_case columns (`propose_*_schema`), normalization, `load_onestop_server_bundle` (fast-path per-pid Parquet, falls back to full CSV.zip), `compute_word_metrics`.
- [plots.py](plots.py) — Plotly figure builders: `make_scanpath_figure` (core layered: words + fixations + saccades + heatmap), `make_comparison_figure`, `make_scanpath_animation`, bar/histogram figs.
- [measures.py](measures.py) — `assign_fixations_to_words` (bbox containment + nearest-word within 50 px), then per-word FFD/FPRT/RPD/TFD/skip/regression. Used only when pre-aggregated columns (e.g. `IA_FIRST_FIXATION_DURATION`) are absent.
- [tabs.py](tabs.py) — four tab renderers: `render_single_trial_tab` (main plot + export), `render_animation_tab` (replay), `render_raw_data_tab` (paginated tables with CSV/Parquet download), `render_data_statistics_tab` (summary stats).
- [controls.py](controls.py) — field-spec dicts (`WORD_FIELD_SPECS`, etc.), `column_mapping_ui`, `sidebar_controls` (viz toggles, colorscales, marker size ranges).
- [export.py](export.py) — `ExportOptions` dataclass + `bulk_export` (zip per-trial figs + tabular data across filtered trials).
- [onestop_shard.py](onestop_shard.py) — one-shot prep: shards the ~15 GB OneStop lacclab CSV exports into per-pid Parquet under `$ONESTOP_DATA_DIR/by_pid/{ia,fixations}/<pid>.parquet`. Run this once per data refresh.
- [utils.py](utils.py) — trial selection UI, trial stats, combo-option builders, friendly labels.
- [constants.py](constants.py), [styles.py](styles.py) — colors/scales/sizes, custom CSS.
- [update_sample_data.py](update_sample_data.py) — regenerate the bundled 3-pid demo from full CSVs.

## Data flow

1. Load: upload, bundled `sample_data/`, or OneStop server bundle (`$ONESTOP_DATA_DIR` set).
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
- PNG export needs `kaleido`; run `plotly_get_chrome -y` once per machine.
