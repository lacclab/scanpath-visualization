
# Scanpath Visualization

## Introduction

This folder contains code and resources for visualizing scanpaths.

## New: Scanpath Visualization Workbench

An interactive Streamlit app inspired by the 14/4 brainstorming notes (`Brainstorming with our Friends 8a7bc24f13c949db84e67807703a89d0.md`). It layers text, fixations, saccades, heatmaps, and aggregates.

**Run it**

```bash
# Using conda/mamba (recommended)
conda env create -f environment.yml
conda activate scanpath-visualization
# or with mamba (faster)
mamba env create -f environment.yml
mamba activate scanpath-visualization

streamlit run scanpath_visualization_app/app.py

# Alternative: using pip
pip install -e .
streamlit run scanpath_visualization_app/app.py
```

The app loads bundled demo data from `app/sample_data/` by default so you can explore immediately.
Sample data is stored as `ia.feather` (word layout) and `fixations.feather`.

**Bring your own data (feather only)**

- Words/IA feather: app auto-detects common columns such as `participant_id`/`subject_id`, `trial_id`/`unique_paragraph_id`, `IA_ID`/`word_id`, `IA_LABEL`/`text`, and either `(x, y, width, height)` or `(IA_LEFT, IA_RIGHT, IA_TOP, IA_BOTTOM)`.
- Fixations feather: detects `participant_id`/`subject_id`, `trial_id`/`unique_paragraph_id`, `CURRENT_FIX_DURATION`/`duration_ms`, `CURRENT_FIX_X`/`CURRENT_FIX_Y`, plus optional `CURRENT_FIX_START`, `IA_ID`, `pass_index`/`reread`, `saccade_type`, `eye`, `noise_flag`.

Coordinate systems should match (screen pixels). Filters cover participants, trials, passes, saccade types, eyes, and noise flags. You can overlay two trials, view temporal traces, and export word-level metrics as CSV.

## Installation

To install the necessary dependencies, run the following command:

```bash
conda env create -f exact_environment.yml # TODO add a clean environment file
```

## Streamlit App for Reread Scanpath Comparison in OneStop Eye Movements

```bash
cd comparative-reread-streamlit-viz
```

### Producing Scanpath Plots

`Rscript "create_rr_scanpath_plots.R"`

### Running the Streamlit App

`streamlit run scanpath_visualizer.py`

### Credit

This code was highly based on the code in this repository: <https://github.com/tmalsburg/scanpath>

### Example

<img src="comparative-reread-streamlit-viz/page_example.png" alt="Scanpath Example" width="600"/>

## Arcplot Scanpath Visualization

The file `arcplot-viz.ipynb` contains example code to create an arcplot scanpath visualization using OneStop Eye Movements

### Example

<img src="arcplot-example.png" alt="Arcplot Example" width="800"/>
