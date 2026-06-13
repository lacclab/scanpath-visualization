# Scanpath Studio

[![PyPI](https://img.shields.io/pypi/v/scanpath-studio.svg)](https://pypi.org/project/scanpath-studio/)
[![Python versions](https://img.shields.io/pypi/pyversions/scanpath-studio.svg)](https://pypi.org/project/scanpath-studio/)
[![Live demo](https://img.shields.io/badge/Live_demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://scanpath-studio.streamlit.app)
[![CI](https://github.com/lacclab/scanpath-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/lacclab/scanpath-studio/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interactive workbench for visualizing **eye-tracking-while-reading** data.
Drop in a trial and see the scanpath the way the reader saw it — words at their
true on-screen positions, with fixations, saccades, a density heatmap, and
animated replay layered on top, all exportable as publication-ready figures.

It is **dataset-agnostic** (auto-detects EyeLink / Gazepoint / snake-case
columns) and ships with a small [OneStop][onestop-paper] demo, so you can try it
with zero setup.

> **Authors:** Omer Shubi, Keren Gruteke Klein, and others (TBD) — LACC Lab, Technion.

![A reading scanpath replayed fixation by fixation](https://raw.githubusercontent.com/lacclab/scanpath-studio/main/assets/scanpath_animation.gif)

*A scanpath replayed fixation by fixation over the text the reader saw.*

## Try it

**Live demo (zero install):** <https://scanpath-studio.streamlit.app>

```bash
pip install scanpath-studio
scanpath-studio      # launches the app in your browser
```

## What it does

The scanpath plot is built from layers you toggle independently:

- **Text** drawn at the exact pixel coordinates the participant saw.
- **Fixations** sized and **colored by any column** in your data (duration, GPT-2 surprisal, word frequency, …).
- **Saccades**, with backward jumps (regressions) standing out.
- **Areas of interest** (word boxes from your data) and a word-level **heatmap** (total fixation duration, count, …).

On top of that:

- **Animated replay** — watch the scanpath unfold at real or scaled speed; export as interactive HTML, GIF, or MP4.
- **Compare readings** — overlay two trials on one canvas or place them side by side (e.g. ordinary vs. information-seeking, first vs. repeated, L1 vs. L2).
- **Critical-span, out-of-text & by-line** highlights — mark an answer span, flag fixations outside every word box, or color fixations by text line.
- **Triage** — star, tag, and annotate trials; save and restore everything as a JSON sidecar.
- **Bulk export** — one zip of per-trial PNG + SVG figures, plot settings, and tabular data across every filtered trial.

![Two readers of the same paragraph, animated on a shared real-time clock](https://raw.githubusercontent.com/lacclab/scanpath-studio/main/assets/demo_dual_scanpath.gif)

*Overlay a second reading to compare two readers of the same text on a shared clock.*

The app is organized into five tabs:

| Tab | What's there |
|-----|--------------|
| **Scanpath Visualization** | The layered scanpath with the trial picker, metadata, and two-trial comparison. Tick **Animate** to replay it frame by frame (export HTML / GIF / MP4). |
| **Generations (WIP)** | A real scanpath vs. several model-generated ones over the same text, scored by similarity *(model outputs are reproducible placeholders for now)*. |
| **Raw Data** | Paginated word / fixation / raw-gaze tables, each with CSV + Parquet download. |
| **Data Statistics** | Summary stats (fixation duration, saccade amplitude, regression rate, …) and per-word / distribution plots. |
| **Bulk Export** | One zip of per-trial figures, plot settings, and tabular data across the filtered trials — or the whole dataset. |

![The Scanpath Studio app](https://raw.githubusercontent.com/lacclab/scanpath-studio/main/assets/app_screenshot.png)

## Your data

Upload **CSV, TSV, Parquet, or Feather** tables for words/AoIs, fixations, and
(optionally) raw gaze. Columns are auto-detected from common EyeLink, Gazepoint,
and snake-case conventions; a sidebar **Column mapping** panel overrides any
guess. The loader bends to fit real corpora — many files per table (concatenated
with a `source_file` tag), a single report (words- or fixations-only),
stimulus-level word boxes broadcast across readers, and AoI-sequence fixations
placed at word/character-box centers.

If your data carries only raw fixations, the app computes the canonical per-word
measures itself — **FFD**, **FPRT** (gaze duration), **RPD** (go-past), **TFD**
(dwell), plus skips and regressions, following Rayner (1998) and Inhoff & Radach
(1998). Pre-aggregated EyeLink columns, when present, take precedence.

A ready-made [**PoTeC**](https://github.com/DiLi-Lab/PoTeC) loader (Potsdam
Textbook Corpus) exercises that flexible pipeline end to end:

```python
import scanpath_studio as sps

words, fixations = sps.load_potec("data/PoTeC", download=True)   # ~45 MB on first call
fig = sps.plot_scanpath(words, fixations, "0", "b0", canvas_size=(1680, 1050))
```

## Command line & Python API

Everything the app draws is also available headless — same pipeline, same figure.

```bash
scanpath-studio render --sample --list-trials              # what's available
scanpath-studio render --sample -o scanpath.html           # interactive HTML
scanpath-studio render --words ia.csv --fixations fix.csv -p p1 -t t3 -o figure.png
scanpath-studio render --sample --animate -o replay.html   # animated replay
```

```python
import scanpath_studio as sps

words, fixations = sps.load_scanpath_data("ia.csv", "fixations.csv")  # paths, globs, or lists; either table optional
sps.list_trials(words, fixations)
fig = sps.plot_scanpath(words, fixations, "p1", "t3")     # every layer toggle is a kwarg
sps.save_figure(fig, "scanpath.png")                       # .html / .png / .svg / .pdf
measures = sps.compute_word_metrics(words, fixations)      # FFD / FPRT / RPD / TFD …
```

HTML export is browser-free; PNG/SVG/PDF/GIF/MP4 go through Kaleido (run
`plotly_get_chrome -y` once). See `scanpath-studio render --help` for all flags.

## Run from source

```bash
git clone https://github.com/lacclab/scanpath-studio.git
cd scanpath-studio
pip install -e ".[test]"          # or: uv sync
streamlit run streamlit_app.py
```

Tested on Python 3.11–3.14. Run the tests with `pytest`; see
[AGENTS.md](AGENTS.md) for an architectural overview.

## Citation

A system-demo paper is in preparation — **citation TBD**. Until then, cite the
software via GitHub's **"Cite this repository"** button (generated from
[`CITATION.cff`](CITATION.cff)).

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

## License

MIT — see [LICENSE](LICENSE).
