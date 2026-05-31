# Scanpath Visualization

[![PyPI](https://img.shields.io/pypi/v/scanpath-visualization-app.svg)](https://pypi.org/project/scanpath-visualization-app/)
[![Live demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://scanpath-visualization.streamlit.app)
[![CI](https://github.com/lacclab/scanpath-visualization/actions/workflows/ci.yml/badge.svg)](https://github.com/lacclab/scanpath-visualization/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interactive workbench for visualizing **eye-tracking-while-reading** data.
Drop in a trial and see the scanpath the way the reader saw it: words at their
true on-screen positions, fixations and saccades layered on top, a density
heatmap, side-by-side trial comparisons, and animated replay — all tunable, and
all exportable as publication-ready figures.

It is **dataset-agnostic** (auto-detects EyeLink / Gazepoint / snake-case
columns) and ships with a small [OneStop][onestop-paper] demo so you can try it
with zero setup.

> **Authors:** Omer Shubi & Keren Gruteke Klein — LACC Lab, Technion.

![The Scanpath Visualization app](https://raw.githubusercontent.com/lacclab/scanpath-visualization/main/assets/app_screenshot.png)

---

## Try it

**Live demo (zero install):** <https://scanpath-visualization.streamlit.app>

**Or run locally:**

```bash
pip install scanpath-visualization-app
scanpath-visualization      # launches the app in your browser
```

---

## What you can visualize

The plot is built from layers you can toggle independently:

- **Text** — every word drawn at the exact pixel coordinates the participant saw.
- **Fixations** — where the eye paused, sized and **colored by any column** in your data (duration, GPT-2 surprisal, word frequency, …).
- **Saccades** — the jumps between fixations; backward jumps (regressions) stand out.
- **Areas of interest** — word bounding boxes that tie each fixation to a word.
- **Heatmap** — the trial aggregated into a word-level measure (total fixation duration, fixation count, …).

On top of the layered view:

- **Animated replay** — watch the scanpath unfold fixation by fixation, at real or scaled speed.
- **Compare two trials** — overlaid on one canvas or side-by-side (e.g. ordinary vs. information-seeking reading, first vs. repeated reading, L1 vs. L2).
- **Critical-span highlight** — mark a region of interest (e.g. an answer span) by color or border to see at a glance whether it was read.
- **Fully customizable** — map any field to color, size, or axes; every toggle, palette, and scale is independent.

---

## The four tabs

| Tab | What's there |
|-----|--------------|
| **Interactive Plot** | The layered scanpath view, trial picker (by trial / text / participant), trial metadata, and two-trial comparison. |
| **Animated Scanpath** | Frame-by-frame replay; each frame lasts the actual fixation duration ÷ playback speed. |
| **Raw Data** | Paginated word, fixation, and raw-gaze tables, each with CSV + Parquet download. |
| **Data Statistics** | Summary stats (mean fixation duration, saccade amplitude, regression rate, reading speed), a fixation-duration distribution, and a per-word reading-measure bar plot. |

---

## Reading measures from raw fixations

If your data only carries raw fixations, the app computes the canonical
per-word measures itself (pre-aggregated EyeLink columns, if present, take
precedence):

| Measure | Definition |
|---------|------------|
| **FFD** — first fixation duration | duration of the first fixation to land on the word |
| **FPRT** / gaze duration | sum of fixations from first entry until the eye first leaves |
| **RPD** / go-past time | sum of fixations from first entry until the eye first moves past the word |
| **TFD** / dwell | sum of all fixations on the word |
| fixation count, skip, regression in/out, saccade amplitude | standard reading-research flags and counts |

Definitions follow Rayner (1998) and Inhoff & Radach (1998).

---

## Your data

Upload **CSV, Parquet, or Feather** tables for words/AoIs, fixations, and
(optionally) raw gaze. Columns are auto-detected from common EyeLink,
Gazepoint, and snake-case conventions; a sidebar **Column mapping** panel lets
you override any guess.

## Bulk export

One panel exports artifacts for **every filtered trial** into a single zip —
per-trial PNG + SVG figures, the exact plot settings (`plot_config.json`),
fixations, and per-word measures, plus aggregated tables across trials. Ideal
for paper figures or building an image dataset of scanpaths for vision models.

---

## Run from source

```bash
git clone https://github.com/lacclab/scanpath-visualization.git
cd scanpath-visualization
pip install -e ".[test]"          # or: uv sync
streamlit run streamlit_app.py
```

Tested on Python 3.11–3.13. Run the tests with `pytest`; lint with
`ruff check --exclude other_vis .`. See [AGENTS.md](AGENTS.md) for an
architectural overview.

---

## Citation

A system-demo paper is in preparation — **citation TBD**.

If you use the bundled demo data, please cite the OneStop corpus:

```bibtex
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

The bundled demo is a subset of [OneStop Eye Movements][onestop-corpus], used
under its original license ([docs][onestop-docs]).

[onestop-paper]: https://www.nature.com/articles/s41597-025-06272-2
[onestop-corpus]: https://github.com/lacclab/OneStop-Eye-Movements
[onestop-docs]: https://lacclab.github.io/OneStop-Eye-Movements/

---

## License

MIT — see [LICENSE](LICENSE).
