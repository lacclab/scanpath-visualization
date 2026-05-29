# Scanpath Visualization

Interactive Streamlit workbench for exploring **eye-tracking-while-reading**
scanpaths. The tool renders word boxes, fixations, saccades, density heatmaps,
side-by-side trial comparisons, scarf plots, and per-word reading measures.
It computes the canonical reading measures (FFD, FPRT/gaze duration, RPD/go-past,
TFD/dwell, regressions) from first principles when the input only carries raw
fixations, and **surfaces NLP-relevant linguistic features** (GPT-2 surprisal,
word frequency, POS) shipped with the OneStop corpus.

> **Authors:** Omer Shubi, LACC Lab (Technion).
> **Demo corpus:** [OneStop Eye Movements][onestop-paper] (Berzak, Malmaud,
> Shubi, Meiri, Lion, Levy, *Scientific Data* 2025) — a 360-participant
> English eye-tracking dataset with multiple reading regimes. A subset of
> three participants reading two articles across both Adv and Ele difficulty
> levels is bundled with the wheel.

[onestop-paper]: https://www.nature.com/articles/s41597-025-06272-2

---

## Quick start

### Try the live demo

A hosted instance is available on Streamlit Community Cloud (deployment URL
to be filled in once the GitHub repo is public).

### Install from PyPI

```bash
pip install scanpath-visualization-app
scanpath-visualization              # launches the Streamlit UI
# or
python -m scanpath_visualization_app
```

### Run from source

```bash
git clone https://github.com/lacclab/scanpath-visualization.git
cd scanpath-visualization

# Option 1 — uv (fastest)
uv sync
uv run streamlit run streamlit_app.py

# Option 2 — pip in editable mode
pip install -e ".[test]"
streamlit run streamlit_app.py

# Option 3 — conda / mamba
mamba env create -f environment.yml
mamba activate scanpath-visualization
streamlit run streamlit_app.py
```

Tested on Python 3.11 – 3.13.

---

## What the tool does

The app organizes work into five tabs:

| Tab | Purpose |
|-----|---------|
| **Interactive Plot** | Layered scanpath view (word boxes, fixations colored by any feature, saccades, density heatmap, optional raw-gaze trail). Trial picker by trial / by text / by participant. Optional overlay or side-by-side comparison of two trials. |
| **Reading Measures** | Scarf plot (per-fixation timeline), per-IA measure bar plot (FFD / FPRT / RPD / TFD / surprisal / frequency / ...), fixation-duration distribution. |
| **Animated Scanpath** | Frame-by-frame replay where each frame's duration is the actual fixation duration divided by playback speed. |
| **Raw Data** | Paginated tables for words, fixations, and raw gaze, each with per-table CSV + Parquet download buttons. |
| **Data Statistics** | Reading-research summaries: mean fixation duration, mean saccade amplitude, regression rate, reading speed (wpm), plus distributions of trials/fixations/words across the filtered set. |

---

## Computed reading measures

When the input table doesn't include EyeLink IA-aggregated columns (or you
upload raw fixations from a non-EyeLink pipeline), the tool computes the
following per (participant, trial, word) from fixations + word bounding boxes:

| Column | Reading-research name | Definition (Rayner 1998; Inhoff & Radach 1998) |
|--------|------------------------|------------------------------------------------|
| `first_fixation_ms` | FFD — first fixation duration | duration of the very first fixation that lands on the word |
| `first_pass_gaze_duration_ms` | FPRT / gaze duration | sum of fixations from first entry until first leave (forward or backward) |
| `regression_path_duration_ms` | RPD / go-past time | sum of all fixations from first entry until the eye first lands strictly past the word |
| `total_fixation_duration_ms` | TFD / dwell | sum of all fixations on the word |
| `n_fixations` | fixation count | total fixations on the word |
| `skip_flag` | skip | True if the word never received a first-pass fixation |
| `regression_in_flag` | regression in | True if a fixation arrived from a later word |
| `regression_out_flag` | regression out | True if a fixation on the word was followed by a fixation on an earlier word |
| `saccade_amplitude` | saccade amplitude (px) | euclidean distance from previous fixation |

If your table already carries pre-aggregated measures (e.g. EyeLink
`IA_FIRST_FIXATION_DURATION`), those values win — computation is a fallback.

---

## Bulk export

The Interactive tab has a configurable **Bulk export** panel that bundles
artifacts for **every filtered trial** into a single zip:

```text
scanpath_export_<timestamp>.zip
├─ README.md
├─ per_trial/
│  └─ <participant>__<trial>/
│     ├─ figure.png            (raster, scale 1–4×)
│     ├─ figure.svg            (vector — paper figures)
│     ├─ plot_config.json      (reproducible plot settings)
│     ├─ fixations.csv         (or .parquet)
│     └─ measures.csv          (per-word measures, csv/parquet)
└─ aggregate/
   ├─ all_fixations.csv        (long-form across trials)
   └─ all_measures.csv
```

Each artifact has its own checkbox so you can export only what you need.

---

## Data expectations

The app auto-detects column names from common eye-tracking export conventions.
Upload **CSV, Parquet, or Feather** tables. Both file formats work for the
words/IA file, the fixations file, and the optional raw-gaze file.

**Required columns** (auto-detected from EyeLink / Gazepoint / snake_case):

| Table | Required | Common candidates |
|-------|----------|-------------------|
| Words/IA | participant, trial, word ID, bounding box | `participant_id` / `subject_id`; `unique_trial_id` / `trial_id`; `IA_ID` / `word_id`; either `(x, y, width, height)` or `(IA_LEFT, IA_RIGHT, IA_TOP, IA_BOTTOM)` |
| Fixations | participant, trial, x, y, duration | `CURRENT_FIX_X/Y/DURATION` (EyeLink), `FPOGX/FPOGY` (Gazepoint), or snake_case |
| Raw gaze (optional) | participant, trial, x, y | snake_case or `FPOGX/FPOGY` |

After upload, the sidebar shows **Column mapping** expanders — you can override
any auto-detected mapping if your columns don't match the common candidates.

---

## Development

```bash
# Install with test deps
pip install -e ".[test]"

# Run all tests
pytest

# Lint + format
ruff check --exclude other_vis .
ruff format --exclude other_vis .

# Regenerate bundled sample data (requires the full OneStop CSVs locally)
python -m scanpath_visualization_app.update_sample_data
```

See [AGENTS.md](AGENTS.md) for the architectural overview Claude / Copilot use
when modifying this code.

---

## Citation

If you use this tool, please cite **both** the demo paper (placeholder —
replace once published) **and** the OneStop corpus:

```bibtex
@inproceedings{shubi2026scanpath,
  title     = {Scanpath Visualization: An Interactive Workbench for Reading
               Eye-Tracking Data},
  author    = {Shubi, Omer and others},
  booktitle = {Proceedings of the *CL Conference, System Demonstrations},
  year      = {2026},
}

@article{berzak2025onestop,
  title     = {{OneStop}: A 360-Participant {E}nglish Eye Tracking Dataset
               with Different Reading Regimes},
  author    = {Berzak, Yevgeni and Malmaud, Jonathan and Shubi, Omer
               and Meiri, Yoav and Lion, Ella and Levy, Roger},
  journal   = {Scientific Data},
  year      = {2025},
  publisher = {Nature Publishing Group},
  doi       = {10.1038/s41597-025-06272-2},
  url       = {https://www.nature.com/articles/s41597-025-06272-2},
}
```

The bundled demo data is a subset of [OneStop Eye Movements][onestop-corpus],
used here under its original license. The corpus documentation is at
<https://lacclab.github.io/OneStop-Eye-Movements/>.

[onestop-corpus]: https://github.com/lacclab/OneStop-Eye-Movements

---

## License

MIT — see [LICENSE](LICENSE).
