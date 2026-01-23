# Scanpath Visualization App

Streamlit workbench for exploring reading scanpaths: layered word boxes, fixations, saccades, density heatmaps, comparisons, and word-level measures. Bundled demo data lets users try it instantly.

## Installation

```bash
pip install scanpath-visualization-app
```

Run the app after installation:

```bash
scanpath-visualization
# or
python -m scanpath_visualization_app
```

## Running from source

### Using conda/mamba (recommended)

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate scanpath-visualization
# or with mamba (faster)
mamba env create -f environment.yml
mamba activate scanpath-visualization

# Run the app
streamlit run scanpath_visualization_app/app.py
# or
python -m scanpath_visualization_app
```

### Using pip (alternative)

```bash
python -m pip install -e .
streamlit run scanpath_visualization_app/app.py
# or
python -m scanpath_visualization_app
```

Tested on Python 3.9 through 3.13 with the latest Streamlit/Plotly/Pandas/Numpy/PyArrow releases.

## Data expectations

Upload Feather tables for words/IA and fixations. Columns are auto-detected using common names (participant/trial IDs, IA/word IDs, text labels, bounding boxes, fixation duration/timestamps/x/y). Missing required fields trigger friendly errors; optional fields drive coloring, filters, and tooltips. Sample Feather files ship with the package under `sample_data/`.

## Packaging & release

- Build artifacts: `python -m build` (produces `dist/` wheel + sdist).
- Verify package data: ensure `sample_data/*.csv` appear in the wheel.
- Upload to PyPI/TestPyPI with `twine upload dist/*`.
