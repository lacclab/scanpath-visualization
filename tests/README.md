# Test Suite for Scanpath Studio

This directory contains the test suite for the scanpath visualization Streamlit app.

## Running Tests

### Install test dependencies

```bash
# Using conda/mamba (recommended)
conda env create -f environment.yml
conda activate scanpath-studio
# or with mamba (faster)
mamba env create -f environment.yml
mamba activate scanpath-studio

# Using pip (alternative)
pip install -r requirements.txt
# Or install in development mode with test dependencies:
pip install -e ".[test]"
```

### Run all tests

```bash
pytest
```

### Run with coverage

```bash
pytest --cov=scanpath_studio --cov-report=html
```

### Run specific test files

```bash
pytest tests/test_data.py
pytest tests/test_plots.py
pytest tests/test_app.py
```

### Run specific test classes or functions

```bash
pytest tests/test_data.py::TestNormalizeWords
pytest tests/test_data.py::TestNormalizeWords::test_normalize_words_with_box_coordinates
```

## Test Structure

- `conftest.py`: Pytest fixtures and configuration
- `test_data.py`: Tests for data loading, normalization, and filtering functions
- `test_plots.py`: Tests for plotting functions
- `test_app.py`: Tests for app utility functions

## Test Coverage

The test suite covers:

1. **Data Processing** (`test_data.py`):
   - Schema inference (words, fixations, raw gaze)
   - Data normalization
   - Data filtering
   - Canvas size computation
   - Word metrics computation

2. **Plotting** (`test_plots.py`):
   - Word box generation
   - Scanpath figure creation
   - Animation figure creation
   - Comparison figure creation

3. **App Utilities** (`test_app.py`):
   - Combo options building
   - Canvas size clamping
   - Trial statistics computation
   - Metadata gathering
   - Comparison options building

## Notes

- Tests use mocking for Streamlit functions (`st.error`, `st.stop`, etc.) since they require a Streamlit runtime
- Fixtures in `conftest.py` provide sample data for testing
- Tests are designed to be fast and isolated
