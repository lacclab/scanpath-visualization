# Copilot Instructions for Scanpath Visualization

## Project Overview
This repository contains tools for visualizing eye-tracking scanpaths over text. The main component is a **Streamlit-based interactive workbench** (`scanpath-visualization/`) that renders fixations, saccades, word boxes, and density heatmaps. Legacy/experimental visualizations live in `other_vis/`.

## Architecture

### Main App (`scanpath-visualization/scanpath_visualization_app/`)
| Module | Purpose |
|--------|---------|
| `app.py` | Streamlit entry point; orchestrates tabs, filtering, trial selection |
| `data.py` | Schema inference (`infer_word_schema`, `infer_fix_schema`) and DataFrame normalization |
| `plots.py` | Plotly figure builders (`make_scanpath_figure`, `make_comparison_figure`) |
| `controls.py` | Sidebar UI components for visualization settings |
| `constants.py` | Shared defaults (`FONT_FAMILY`, `DEFAULT_FIGURE_SIZE`) |

**Data flow**: CSV upload → schema inference → normalization → filtering → figure generation → Streamlit render

### Column Auto-Detection Pattern
The app uses `pick_column()` with candidate lists to detect user data columns. When adding new column support:
```python
# In data.py - add candidates in priority order
word_id = pick_column(words, ["word_id", "IA_ID", "ia_id", "ia_index"])
```

## Development Commands

```bash
# Setup & run (recommended: conda/mamba)
conda env create -f environment.yml
conda activate scanpath-visualization
# or with mamba (faster)
mamba env create -f environment.yml
mamba activate scanpath-visualization

streamlit run scanpath_visualization_app/app.py
# or
scanpath-visualization  # or: python -m scanpath_visualization_app

# Alternative: using pip
pip install -e .
streamlit run scanpath_visualization_app/app.py

# Build for PyPI
python -m build
twine upload dist/*
```

## Key Conventions

### Imports
The app supports two import modes for flexibility:
- **Package mode**: `from .constants import ...` (when installed)
- **Direct run mode**: Falls back to `sys.path` manipulation for `streamlit run app.py`

### DataFrame Normalization
All user data is normalized to canonical column names before plotting:
- Words: `participant_id`, `trial_id`, `word_id`, `text`, `x`, `y`, `width`, `height`
- Fixations: `participant_id`, `trial_id`, `x`, `y`, `duration_ms`, `timestamp_ms`, `pass_index`

### Streamlit Caching
Use `@st.cache_data` for expensive data operations (see `load_sample_data()` in data.py).

### Plotly Figures
- Canvas size computed from data bounds with padding
- Shapes array used for word boxes and heatmap overlays
- Y-axis is inverted (`y_range = [max, min]`) for screen coordinates

## File Structure Patterns

- Sample data lives in `scanpath_visualization_app/sample_data/*.csv`
- Package metadata in `pyproject.toml` (setuptools backend)
- Include sample data via `MANIFEST.in` and `[tool.setuptools.package-data]`

## Testing New Features

1. Test with bundled demo data first (`sample_data/`)
2. Verify column auto-detection handles missing optional fields gracefully
3. Check both spatial (`x`/`y`) and temporal (`timestamp_ms`) axis modes

## Legacy Code (`other_vis/`)
Contains experimental matplotlib/R-based visualizations. Not actively maintained but may provide reference implementations for new features.
