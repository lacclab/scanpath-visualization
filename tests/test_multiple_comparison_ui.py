"""Tests for the Multiple Comparison tab's table-formatting / help-text helpers.

These are pure (Styler / string) helpers, so they're unit-testable without
spinning up the full Streamlit app — guarding the best-model highlight direction
and the placeholder formatting against regressions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scanpath_studio.similarity import METRICS
from scanpath_studio.tabs import (
    _best_model_indices,
    _style_similarity_table,
)

_PLACEHOLDER_LABELS = [m.label for m in METRICS if m.fn is None]


def _sample_table() -> pd.DataFrame:
    columns = ["Model"] + [m.label for m in METRICS]
    row1 = {"Model": "Model 1", "NLD": 0.30}
    row2 = {"Model": "Model 2", "NLD": 0.70}
    for label in _PLACEHOLDER_LABELS:
        row1[label] = np.nan
        row2[label] = np.nan
    return pd.DataFrame([row1, row2], columns=columns)


def test_best_model_indices_picks_min_for_nld():
    table = _sample_table()
    best = _best_model_indices(table)
    # NLD is lower-is-better -> Model 1 (row 0) is best.
    assert best["NLD"] == 0
    # Placeholder columns get no entry.
    for label in _PLACEHOLDER_LABELS:
        assert label not in best


def test_best_model_indices_ignores_all_nan_columns():
    table = _sample_table()
    table["NLD"] = np.nan
    assert _best_model_indices(table) == {}


def test_style_table_formats_placeholders_and_highlights_best():
    table = _sample_table()
    html = _style_similarity_table(table).to_html()
    # Placeholder NaN cells render as the em-dash.
    assert "—" in html
    # The best (min-NLD) cell is tinted green.
    assert "#d4edda" in html
    # Real values are formatted to 3 decimals.
    assert "0.300" in html and "0.700" in html


def test_style_table_headers_carry_direction_arrows():
    table = _sample_table()
    html = _style_similarity_table(table).to_html()
    # NLD is lower-is-better -> down arrow in its header.
    assert "NLD ↓" in html
    # The higher-is-better placeholders (ScanMatch, MultiMatch) get an up arrow.
    assert "↑" in html
