"""Tests for the synthetic model-scanpath generator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scanpath_studio.model_scanpaths import (
    DEFAULT_N_MODELS,
    MAX_MODELS,
    _MAX_DUR_MS,
    _MIN_DUR_MS,
    _ordered_word_rows,
    generate_model_scanpath,
    generate_model_scanpaths,
    MODEL_PROFILES,
)
from scanpath_studio.plots import make_scanpath_figure
from scanpath_studio.similarity import aoi_sequence
from tests.synthetic_data import make_synthetic_words

_FIX_COLUMNS = {
    "participant_id",
    "trial_id",
    "text_id",
    "x",
    "y",
    "duration_ms",
    "timestamp_ms",
    "fixation_id",
    "word_id",
    "pass_index",
    "order_in_trial",
    "eye",
    "noise_flag",
    "saccade_type",
}


def test_ordered_word_rows_uses_word_id_when_complete():
    # With a complete word_id, reading order is simply word_id order.
    words = pd.DataFrame(
        {
            "text": ["c", "a", "b"],
            "word_id": [2, 0, 1],
            "x": [200, 0, 100],
            "y": [0, 0, 0],
            "width": [40, 40, 40],
            "height": [30, 30, 30],
        }
    )
    ordered = _ordered_word_rows(words)
    assert list(ordered["word_id"]) == [0, 1, 2]
    assert list(ordered["text"]) == ["a", "b", "c"]


def test_ordered_word_rows_falls_back_to_line_clustering():
    # No usable word_id -> reading order is inferred from geometry: rows are
    # clustered into visual lines by y (via the shared measures.cluster_word_lines
    # helper), then read left-to-right. Input order is deliberately scrambled.
    words = pd.DataFrame(
        {
            "text": ["A", "B", "C", "D", "E"],
            "x": [200, 0, 100, 50, 150],
            "y": [0, 0, 0, 100, 100],  # line 0 (y=0): A,B,C; line 1 (y=100): D,E
            "width": [40, 40, 40, 40, 40],
            "height": [30, 30, 30, 30, 30],
        }
    )
    ordered = _ordered_word_rows(words)
    # Line 0 left-to-right (B@0, C@100, A@200), then line 1 (D@50, E@150).
    assert list(ordered["text"]) == ["B", "C", "A", "D", "E"]


def test_generate_returns_canonical_columns():
    words = make_synthetic_words()
    fix = generate_model_scanpath(
        words, MODEL_PROFILES[0], model_index=0, reference_trial_id="t"
    )
    assert _FIX_COLUMNS.issubset(set(fix.columns))
    assert len(fix) > 0
    # participant id is the model name; trial id mirrors the reference trial.
    assert set(fix["participant_id"]) == {"Model 1"}
    assert set(fix["trial_id"]) == {"t"}


def test_durations_and_timestamps_well_formed():
    words = make_synthetic_words()
    fix = generate_model_scanpath(
        words, MODEL_PROFILES[2], model_index=2, reference_trial_id="t"
    )
    dur = fix["duration_ms"].to_numpy()
    assert (dur >= _MIN_DUR_MS).all()
    assert (dur <= _MAX_DUR_MS).all()
    ts = fix["timestamp_ms"].to_numpy()
    # timestamps strictly increase (gap + positive duration each step)
    assert np.all(np.diff(ts) > 0)
    # order_in_trial is 1..k contiguous
    assert list(fix["order_in_trial"]) == list(range(1, len(fix) + 1))


def test_word_ids_are_within_text():
    words = make_synthetic_words()
    valid = set(words["word_id"].tolist())
    fix = generate_model_scanpath(
        words, MODEL_PROFILES[5], model_index=5, reference_trial_id="t"
    )
    assert set(fix["word_id"].dropna().tolist()).issubset(valid)


def test_determinism_same_seed():
    words = make_synthetic_words()
    a = generate_model_scanpaths(words, n_models=3, reference_trial_id="t", nonce=0)
    b = generate_model_scanpaths(words, n_models=3, reference_trial_id="t", nonce=0)
    assert list(a.keys()) == list(b.keys())
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_nonce_changes_scanpath():
    words = make_synthetic_words()
    a = generate_model_scanpath(
        words, MODEL_PROFILES[0], model_index=0, reference_trial_id="t", nonce=0
    )
    b = generate_model_scanpath(
        words, MODEL_PROFILES[0], model_index=0, reference_trial_id="t", nonce=1
    )
    # Different seed -> the landing coordinates differ (length may coincide).
    differ = (len(a) != len(b)) or not np.allclose(
        a["x"].to_numpy()[: min(len(a), len(b))],
        b["x"].to_numpy()[: min(len(a), len(b))],
    )
    assert differ


def test_different_trial_id_changes_seed():
    words = make_synthetic_words()
    a = generate_model_scanpath(
        words, MODEL_PROFILES[0], model_index=0, reference_trial_id="trial_a"
    )
    b = generate_model_scanpath(
        words, MODEL_PROFILES[0], model_index=0, reference_trial_id="trial_b"
    )
    differ = (len(a) != len(b)) or not np.allclose(
        a["x"].to_numpy()[: min(len(a), len(b))],
        b["x"].to_numpy()[: min(len(a), len(b))],
    )
    assert differ


def test_generate_all_respects_count_and_cap():
    words = make_synthetic_words()
    default = generate_model_scanpaths(words, reference_trial_id="t")
    assert len(default) == DEFAULT_N_MODELS

    capped = generate_model_scanpaths(
        words, n_models=MAX_MODELS + 5, reference_trial_id="t"
    )
    assert len(capped) == MAX_MODELS

    at_least_one = generate_model_scanpaths(words, n_models=1, reference_trial_id="t")
    assert len(at_least_one) == 1


def test_empty_words_yields_empty_frame():
    words = make_synthetic_words().iloc[0:0]
    fix = generate_model_scanpath(
        words, MODEL_PROFILES[0], model_index=0, reference_trial_id="t"
    )
    assert fix.empty
    assert _FIX_COLUMNS.issubset(set(fix.columns))


def test_generated_scanpath_has_nonempty_aoi():
    words = make_synthetic_words()
    fix = generate_model_scanpath(
        words, MODEL_PROFILES[0], model_index=0, reference_trial_id="t"
    )
    assert len(aoi_sequence(fix, words)) > 0


def test_generated_scanpath_renders_in_figure():
    """A generated frame must drop straight into the figure builder."""
    words = make_synthetic_words()
    fix = generate_model_scanpath(
        words, MODEL_PROFILES[0], model_index=0, reference_trial_id="t"
    )
    fig = make_scanpath_figure(
        words,
        fix,
        canvas_width=400,
        canvas_height=300,
        base_font_size=12,
        font_family="monospace",
        x_field="x",
        y_field="y",
        show_words=True,
        show_word_labels=False,
        show_fixations=True,
        show_order=False,
        show_saccades=True,
        show_heatmap=False,
        color_by="duration_ms",
        heatmap_metric=None,
        marker_size_range=(8, 24),
        order_font_size=10,
        order_font_color="#000000",
        show_colorbars=False,
        fixation_color_range=None,
        heatmap_range=None,
    )
    assert fig is not None
    assert len(fig.data) > 0
