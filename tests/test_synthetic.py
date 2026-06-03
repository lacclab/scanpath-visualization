"""Exact-value tests against the hand-built synthetic trial.

Every expectation here is hand-traced in ``tests/synthetic_data.py``; if a
measure or geometry helper changes behavior, these break loudly with a known
ground truth rather than an approximate one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scanpath_visualization_app.measures import (
    assign_fixation_lines,
    assign_fixations_to_words,
    cluster_word_lines,
    compute_per_word_measures,
    fixation_in_text_mask,
)
from tests.synthetic_data import (
    EXPECTED,
    make_synthetic_fixations,
    make_synthetic_words,
)


def test_cluster_word_lines_matches_layout():
    words = make_synthetic_words()
    lines = cluster_word_lines(words)
    assert list(lines.loc[words.index]) == EXPECTED["word_line"]


def test_fixation_to_word_assignment():
    words = make_synthetic_words()
    fix = make_synthetic_fixations()
    assigned = assign_fixations_to_words(fix, words)["word_id"].tolist()
    expected = EXPECTED["fixation_word_id"]
    assert len(assigned) == len(expected)
    for got, exp in zip(assigned, expected):
        if exp is np.nan or (isinstance(exp, float) and np.isnan(exp)):
            assert pd.isna(got)
        else:
            assert got == exp


def test_in_text_mask_and_out_of_text_count():
    words = make_synthetic_words()
    fix = make_synthetic_fixations()
    mask = fixation_in_text_mask(fix, words)
    assert list(mask) == EXPECTED["in_text"]
    assert int((~mask).sum()) == EXPECTED["out_of_text_count"]


def test_fixation_line_assignment():
    words = make_synthetic_words()
    fix = make_synthetic_fixations()
    lines = assign_fixation_lines(fix, words)
    assert [int(v) for v in lines] == EXPECTED["fixation_line"]


def test_per_word_measures_exact():
    words = make_synthetic_words()
    fix = make_synthetic_fixations()
    measures = compute_per_word_measures(fix, words).set_index("word_id")

    for col in (
        "first_fixation_ms",
        "first_pass_gaze_duration_ms",
        "regression_path_duration_ms",
        "total_fixation_duration_ms",
    ):
        for wid, exp in EXPECTED[col].items():
            assert float(measures.loc[wid, col]) == float(exp), (col, wid)

    for wid, exp in EXPECTED["n_fixations"].items():
        assert int(measures.loc[wid, "n_fixations"]) == exp, ("n_fixations", wid)

    for col in ("skip_flag", "regression_in_flag", "regression_out_flag"):
        for wid, exp in EXPECTED[col].items():
            assert bool(measures.loc[wid, col]) == exp, (col, wid)


def test_synthetic_source_through_app_pipeline():
    """The 'Synthetic test trial' data source must survive infer + normalize
    (the app's load path) and still yield the ground-truth measures."""
    from scanpath_visualization_app.data import (
        compute_word_metrics,
        infer_fix_schema,
        infer_word_schema,
        normalize_fixations,
        normalize_words,
    )
    from scanpath_visualization_app.synthetic import load_synthetic_data

    words, fixations = load_synthetic_data()
    words_n = normalize_words(words, infer_word_schema(words))
    fixations_n = normalize_fixations(fixations, infer_fix_schema(fixations))
    assert words_n["trial_id"].nunique() == 1
    measures = compute_word_metrics(words_n, fixations_n).set_index("word_id")
    for wid, exp in EXPECTED["first_fixation_ms"].items():
        assert float(measures.loc[wid, "first_fixation_ms"]) == float(exp)
    for wid, exp in EXPECTED["total_fixation_duration_ms"].items():
        assert float(measures.loc[wid, "total_fixation_duration_ms"]) == float(exp)
