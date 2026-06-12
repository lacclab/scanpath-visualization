# Scanpath Studio

[![PyPI](https://img.shields.io/pypi/v/scanpath-studio.svg)](https://pypi.org/project/scanpath-studio/)
[![Live demo](https://img.shields.io/badge/Live_demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://scanpath-studio.streamlit.app)
[![CI](https://github.com/lacclab/scanpath-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/lacclab/scanpath-studio/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interactive workbench for visualizing **eye-tracking-while-reading** data.
Drop in a trial and see the scanpath the way the reader saw it: words at their
true on-screen positions, fixations and saccades layered on top, a density
heatmap, side-by-side trial comparisons, and animated replay — all tunable, and
all exportable as publication-ready figures.

It is **dataset-agnostic** (auto-detects EyeLink / Gazepoint / snake-case
columns) and ships with a small [OneStop][onestop-paper] demo so you can try it
with zero setup.

> **Authors:** Omer Shubi, Keren Gruteke Klein, and others (TBD) — LACC Lab, Technion.

![A reading scanpath replayed fixation by fixation](https://raw.githubusercontent.com/lacclab/scanpath-studio/main/assets/scanpath_animation.gif)

*A scanpath replayed fixation by fixation over the text the reader saw (bundled OneStop demo).*

---

## Try it

**Live demo (zero install):** <https://scanpath-studio.streamlit.app>

**Or run locally:**

```bash
pip install scanpath-studio
scanpath-studio      # launches the app in your browser
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

- **Animated replay** — watch the scanpath unfold fixation by fixation, at real or scaled speed; export it as interactive HTML or a self-playing **GIF / MP4** clip for slides and papers.
- **Compare two trials** — overlaid on one canvas or side-by-side (e.g. ordinary vs. information-seeking reading, first vs. repeated reading, L1 vs. L2).
- **Critical-span highlight** — mark a region of interest (e.g. an answer span) by color or border to see at a glance whether it was read.
- **Out-of-text & by-line** — flag fixations that land outside every word box, or color fixations by the text line they fall on.
- **Fully customizable** — map any field to color, size, or axes; set the plot background (white or a neutral gray); every toggle, palette, and scale is independent.

![Two readers of the same paragraph, animated on a shared real-time clock](https://raw.githubusercontent.com/lacclab/scanpath-studio/main/assets/demo_dual_scanpath.gif)

*Overlay a second reading to compare two readers of the same text on a shared real-time clock.*

---

## The four tabs

| Tab | What's there |
|-----|--------------|
| **Interactive Plot** | The layered scanpath view, trial picker (by trial / text / participant), trial metadata, and two-trial comparison. |
| **Animated Scanpath** | Frame-by-frame replay; each frame lasts the actual fixation duration ÷ playback speed. Export as interactive HTML, GIF, or MP4. |
| **Raw Data** | Paginated word, fixation, and raw-gaze tables, each with CSV + Parquet download. |
| **Data Statistics** | Summary stats (mean fixation duration, saccade amplitude, regression rate, reading speed), a fixation-duration distribution, and a per-word reading-measure bar plot. |

![The Scanpath Studio app](https://raw.githubusercontent.com/lacclab/scanpath-studio/main/assets/app_screenshot.png)

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

## Triage your trials

Filter the trial pool by **condition** — information-seeking *Hunting* vs.
ordinary *Gathering* reading, difficulty, first vs. repeated reading, answer
correctness — or by your own annotations. **Star** favorites, **tag** trials
(e.g. "To exclude"), and jot per-trial **notes**; download everything as a JSON
sidecar and restore it in a later session.

---

## Your data

Upload **CSV, TSV, Parquet, or Feather** tables for words/AoIs, fixations, and
(optionally) raw gaze. Columns are auto-detected from common EyeLink,
Gazepoint, and snake-case conventions; a sidebar **Column mapping** panel lets
you override any guess. No single column uniquely identifies a trial? Map
**Trial ID** to several columns (e.g. participant + paragraph + repeated
reading) and a combined unique trial ID is built on the fly.

**Areas of interest** come straight from your word boxes — given as
`(x, y, width, height)` or EyeLink's `IA_LEFT/RIGHT/TOP/BOTTOM` — the app never
invents them. Fixations are tied to words by bounding-box containment (with a
small nearest-word fallback); fixations that miss every box are flagged
*out-of-text*.

Real corpora come in many shapes, so the loader bends to fit:

- **One file per participant or text.** Drop in several files at once (or pass a
  glob / list of paths to the API and CLI) and they're concatenated, with each
  row tagged by its `source_file` so filename-encoded metadata isn't lost.
- **Only one report.** Have just an interest-area report, or just fixations?
  Load either one alone — the missing layer is simply skipped, and a words-only
  table still draws a heatmap from its own pre-aggregated reading measures.
- **Stimulus-level AoIs.** Word boxes given once per *text* (no participant
  column) are broadcast across every reader of that text.
- **Fixations as word/AoI sequences.** No pixel coordinates, only "which word"?
  Fixations are placed at the matching word-box centers (or, for character-level
  AoIs like PoTeC's, at the fixated character's box).

[**PoTeC**](https://github.com/DiLi-Lab/PoTeC) (Potsdam Textbook Corpus — 75
readers × 12 German textbook texts, one fixation file per reading and
stimulus-level AoIs) loads as a worked example of all four:

```python
import scanpath_studio as sps

words, fixations = sps.load_potec("data/PoTeC", download=True)   # ~45 MB on first call
fig = sps.plot_scanpath(words, fixations, "0", "b0", canvas_size=(1680, 1050))
```

or `scanpath-studio render --potec data/PoTeC -p 0 -t b0 -o potec.png`.

> Heads-up: PoTeC's raw files **can't** be loaded through the generic upload
> flow — its trial/word ids live in filenames and fixation coordinates come from
> a separate character-AoI file. The dedicated loader handles that join. An
> in-app **Public datasets** source built on the same loaders is feature-flagged
> off for now and will appear in a future release.

When you upload your own tables and a required column can't be auto-detected,
the app no longer stops — it shows your raw tables in the **Raw Data** tab so you
can see the column names and finish the **Column mapping** in the sidebar.

## Bulk export

One panel exports artifacts for **every filtered trial** into a single zip —
per-trial PNG + SVG figures, the exact plot settings (`plot_config.json`),
fixations, and per-word measures, plus aggregated tables across trials. Ideal
for paper figures or building an image dataset of scanpaths for vision models.

---

## Command line & Python API

Everything the app draws is also available headless — same pipeline, same
canonical figure.

**CLI** — render a trial straight to a file:

```bash
scanpath-studio render --sample --list-trials         # what's available
scanpath-studio render --sample -o scanpath.html      # interactive HTML
scanpath-studio render --words ia.csv --fixations fixations.csv \
    -p participant_1 -t trial_3 --no-heatmap -o figure.png
scanpath-studio render --fixations 'fixations/*.tsv' -o scanpath.png   # multi-file, fixations-only
scanpath-studio render --potec data/PoTeC -p 0 -t b0 -o potec.png      # PoTeC corpus
scanpath-studio render --sample --animate -o replay.html
```

HTML output is browser-free; PNG/SVG/PDF go through Kaleido (install Chrome
once with `plotly_get_chrome -y`). See `scanpath-studio render --help` for the
full set of layer toggles. `scanpath-studio` on its own still launches the app,
forwarding any extra args to `streamlit run` (e.g. `--server.port 8502`).

**Python API** — the same canonical figures programmatically:

```python
import scanpath_studio as sps

words, fixations = sps.load_scanpath_data("ia.csv", "fixations.csv")
sps.list_trials(words, fixations)                # (participant, trial) combos
fig = sps.plot_scanpath(words, fixations, "participant_1", "trial_3")
sps.save_figure(fig, "scanpath.html")            # or .png/.svg/.pdf
anim = sps.animate_scanpath(words, fixations, "participant_1", "trial_3")
measures = sps.compute_word_metrics(words, fixations)  # FFD/FPRT/RPD/TFD…
```

`sps.load_sample_data()` returns the bundled demo, and `plot_scanpath` /
`animate_scanpath` accept every layer toggle and style option the app exposes
(`show_heatmap=False`, `color_by="pass_index"`, …).

`load_scanpath_data` also takes glob patterns or lists of paths, and either
table may be omitted for single-report datasets:

```python
# one fixation file per participant, no separate IA report
words, fixations = sps.load_scanpath_data(fixations="fixations/*.tsv")
# or a ready-made loader for PoTeC's multi-file, stimulus-AoI layout
words, fixations = sps.load_potec("data/PoTeC", readers=[0, 1], texts=["b0"])
```

---

## Run from source

```bash
git clone https://github.com/lacclab/scanpath-studio.git
cd scanpath-studio
pip install -e ".[test]"          # or: uv sync
streamlit run streamlit_app.py
```

Tested on Python 3.11–3.13. Run the tests with `pytest`; lint with
`ruff check --exclude other_vis .`. See [AGENTS.md](AGENTS.md) for an
architectural overview.

---

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

---

## License

MIT — see [LICENSE](LICENSE).
