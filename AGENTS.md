# AGENTS.md - Scanpath Visualization App

## Overview

This is a Streamlit workbench for visualizing eye-tracking scanpaths over text (word boxes + fixations + saccades + heatmaps + comparisons). The primary product lives in `scanpath_visualization_app/`. Legacy experiments are in `other_vis/`.

## Architecture

- **Entry/UI**: `scanpath_visualization_app/app.py` (tabs, uploads, trial selection, filtering, calls plotting)
- **Data handling**: `scanpath_visualization_app/data.py` (schema inference + normalization + filtering + metrics)
- **Plotting**: `scanpath_visualization_app/plots.py` (Plotly figure builders)
- **Controls/defaults**: `scanpath_visualization_app/controls.py`, `scanpath_visualization_app/constants.py`

Pipeline: uploaded CSVs → `infer_*_schema()` → `normalize_*()` to canonical columns → filters/metrics → `make_*_figure()` → Streamlit render.

---

## Build/Lint/Test Commands

### Running the App

```bash
# Development (from project root)
streamlit run scanpath_visualization_app/app.py

# Fast dev setup with uv
uv sync
uv run streamlit run scanpath_visualization_app/app.py

# Packaged installation
pip install -e .
scanpath-visualization
# or
python -m scanpath_visualization_app
```

### Running Tests

```bash
# All tests
pytest

# Single test file
pytest tests/test_data.py

# Single test
pytest tests/test_data.py::TestPickColumn::test_pick_column_found

# Run with coverage
pytest --cov=scanpath_visualization_app --cov-report=html

# Run only fast tests (exclude slow)
pytest -m "not slow"
```

### Linting & Formatting

```bash
# Check and fix linting issues (exclude other_vis/)
ruff check --fix --exclude other_vis .

# Fix import order only
ruff check --select I --fix --exclude other_vis .

# Format code
ruff format --exclude other_vis .

# Full lint + format pipeline
ruff check --fix --exclude other_vis .
ruff check --select I --fix --exclude other_vis .
ruff format --exclude other_vis .
```

---

## Code Style Guidelines

### General

- **No comments** unless absolutely necessary for understanding
- Use `from __future__ import annotations` at the top of all Python files
- Keep legacy `other_vis/` excluded from all tooling

### Imports

- Use ruff for import sorting (`ruff check --select I --fix`)
- Order: stdlib → third-party → local (relative imports)
- Use `importlib.resources` for package data access

### Naming Conventions

- **Functions/variables**: `snake_case` (e.g., `infer_word_schema`, `normalized_words_df`)
- **Classes**: `PascalCase` (e.g., `TestPickColumn`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_FIGURE_SIZE`)
- **Files**: `snake_case.py`

### Type Hints

- Use `typing` module for type hints: `Optional[str]`, `Dict[str, Any]`, `Tuple[int, int]`
- Add return type hints to all functions
- Use `pd.DataFrame` and `np.ndarray` as type hints directly (no quotes)

### Error Handling

- Use friendly error messages via `st.error()` for Streamlit validation
- Return `None` or empty DataFrames on failure rather than raising
- Use `try/except` for file loading with graceful fallbacks

### Streamlit Patterns

- Use `@st.cache_data` decorator for expensive data loading functions
- Keep the import mode pattern intact: support both package imports (`from .data import ...`) and fallback "direct run" path
- Schema inference functions use `pick_column(df, candidates)` with priority-ordered candidate lists

### Data Processing

- Use pandas for tabular data manipulation
- Use numpy for numerical operations
- Normalize all input data to canonical column names:
  - Words: `participant_id`, `trial_id`, `paragraph_id`, `word_id`, `text`, `line_idx`, `x`, `y`, `width`, `height`
  - Fixations: `participant_id`, `trial_id`, `paragraph_id`, `x`, `y`, `duration_ms`, `timestamp_ms`, `word_id`, `pass_index`, `saccade_type`, `eye`, `noise_flag`

### Plotting

- Use Plotly for all visualizations
- Screen coordinates use inverted y-axis (`y_range = [max, min]`)
- Word boxes and heatmaps use `layout.shapes`
- Use `go.Figure` for all figures

### Testing

- Use pytest with fixtures defined in `tests/conftest.py`
- Mock Streamlit (`@patch("scanpath_visualization_app.data.st")`) in tests that call schema inference
- Use descriptive test class names: `TestPickColumn`, `TestInferWordSchema`, etc.
- Test both success and failure paths

---

## Column Auto-Detection

Schema inference uses `pick_column(df, candidates)` with priority-ordered candidate lists. When adding support for new upstream column names, update the relevant `infer_*_schema` candidate lists in `scanpath_visualization_app/data.py`.

---

## Package Data

Sample data files (CSV) are included in the package under `scanpath_visualization_app/sample_data/`. Access via:

```python
import importlib.resources as resources
data_root = resources.files(PACKAGE_NAME).joinpath("sample_data")
```

---

## Dependencies

- **Runtime**: streamlit, pandas, plotly, numpy, pyarrow, kaleido, watchdog
- **Dev/Test**: pytest, pytest-cov

Python 3.11+ required.
