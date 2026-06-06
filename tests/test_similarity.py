"""Tests for scanpath similarity metrics (Levenshtein / NLD / AOI sequence)."""

from __future__ import annotations

import numpy as np

from scanpath_studio.model_scanpaths import MODEL_PROFILES, generate_model_scanpath
from scanpath_studio.similarity import (
    METRICS,
    aoi_sequence,
    compute_similarity_table,
    levenshtein,
    nld_by_fixation_index,
    nld_by_time,
    normalized_levenshtein,
)
from tests.synthetic_data import make_synthetic_fixations, make_synthetic_words


def test_levenshtein_basic_cases():
    assert levenshtein([1, 2, 3], [1, 2, 3]) == 0
    assert levenshtein([], []) == 0
    assert levenshtein([], [1, 2]) == 2
    assert levenshtein([1, 2, 3], []) == 3
    # one deletion
    assert levenshtein([1, 2, 3], [1, 3]) == 1
    # one substitution
    assert levenshtein([1, 2, 3], [1, 9, 3]) == 1
    # symmetry
    assert levenshtein([1, 2, 3, 4], [4, 3, 2, 1]) == levenshtein(
        [4, 3, 2, 1], [1, 2, 3, 4]
    )


def test_normalized_levenshtein_bounds_and_values():
    assert normalized_levenshtein([], []) == 0.0
    assert normalized_levenshtein([1], []) == 1.0
    assert normalized_levenshtein([1, 2, 3], [1, 2, 3]) == 0.0
    # one substitution over length 4 -> 0.25
    assert normalized_levenshtein([1, 2, 3, 4], [1, 2, 3, 5]) == 0.25
    for a, b in [([1, 2], [3, 4, 5]), ([1], [1, 2, 3, 4]), ([], [9])]:
        val = normalized_levenshtein(a, b)
        assert 0.0 <= val <= 1.0


def test_aoi_sequence_matches_hand_traced_words():
    """The synthetic trial's fixations land on word ids [0,0,1,2,1,3,4,(out),5];
    the out-of-text fixation is dropped, leaving [0,0,1,2,1,3,4,5]."""
    words = make_synthetic_words()
    fix = make_synthetic_fixations()
    assert aoi_sequence(fix, words) == [0, 0, 1, 2, 1, 3, 4, 5]


def test_aoi_sequence_is_id_agnostic():
    """A scanpath carrying foreign participant/trial ids still maps onto the
    given word boxes (real model scanpaths use synthetic ids)."""
    words = make_synthetic_words()
    fix = make_synthetic_fixations()
    relabelled = fix.copy()
    relabelled["participant_id"] = "Model 1"
    relabelled["trial_id"] = "model_trial"
    assert aoi_sequence(relabelled, words) == aoi_sequence(fix, words)


def test_similarity_table_shape_and_placeholders():
    words = make_synthetic_words()
    real = make_synthetic_fixations()
    # Two "models": an exact copy (NLD 0) and a reversed sequence (NLD > 0).
    reversed_fix = real.iloc[::-1].reset_index(drop=True)
    reversed_fix["timestamp_ms"] = range(len(reversed_fix))
    models = {"copy": real.copy(), "reversed": reversed_fix}

    table = compute_similarity_table(real, models, words)

    assert list(table["Model"]) == ["copy", "reversed"]
    expected_cols = ["Model"] + [m.label for m in METRICS]
    assert list(table.columns) == expected_cols

    # NLD is real, in [0, 1]; the exact copy scores 0.
    assert table.loc[0, "NLD"] == 0.0
    # Pin the full real-data path (fixations -> AOI -> edit distance) to a
    # hand-traced value rather than a tautological [0, 1] bound:
    # real AOI = [0,0,1,2,1,3,4,5], reversed = [5,4,3,1,2,1,0,0] -> NLD 6/8 = 0.75.
    assert table.loc[1, "NLD"] == 0.75

    # The three placeholder metrics are all NaN.
    placeholder_labels = [m.label for m in METRICS if m.fn is None]
    assert len(placeholder_labels) == 3
    for label in placeholder_labels:
        assert table[label].isna().all()


def test_metrics_registry_has_one_real_metric():
    real = [m for m in METRICS if m.fn is not None]
    assert len(real) == 1
    assert real[0].label == "NLD"
    assert real[0].lower_is_better is True


def test_empty_inputs_do_not_crash():
    words = make_synthetic_words()
    empty = make_synthetic_fixations().iloc[0:0]
    assert aoi_sequence(empty, words) == []
    # Two empty sequences are identical, not undefined: NLD is 0.0 (not NaN).
    nld = normalized_levenshtein([], [])
    assert nld == 0.0
    assert not np.isnan(nld)


# --- Cumulative convergence curves -------------------------------------------


def test_nld_by_fixation_index_identical_is_zero():
    words = make_synthetic_words()
    fix = make_synthetic_fixations()
    ks, nlds = nld_by_fixation_index(fix, fix, words)
    assert ks == list(range(1, len(fix) + 1))
    assert all(v == 0.0 for v in nlds)


def test_nld_by_fixation_index_bounds_and_alignment():
    words = make_synthetic_words()
    real = make_synthetic_fixations()
    model = generate_model_scanpath(
        words, MODEL_PROFILES[3], model_index=3, reference_trial_id="t"
    )
    ks, nlds = nld_by_fixation_index(real, model, words)
    assert len(ks) == len(nlds) > 0
    assert ks[0] == 1
    assert ks[-1] == max(len(real), len(model))
    assert all(0.0 <= v <= 1.0 for v in nlds)


def test_nld_by_time_identical_is_zero_and_sorted():
    words = make_synthetic_words()
    fix = make_synthetic_fixations()
    xs, nlds = nld_by_time(fix, fix, words)
    assert len(xs) == len(nlds) > 0
    assert xs == sorted(xs)
    assert all(v == 0.0 for v in nlds)


def test_nld_by_time_bounds():
    words = make_synthetic_words()
    real = make_synthetic_fixations()
    model = generate_model_scanpath(
        words, MODEL_PROFILES[5], model_index=5, reference_trial_id="t"
    )
    xs, nlds = nld_by_time(real, model, words)
    assert len(xs) == len(nlds) > 0
    assert xs == sorted(xs)
    assert all(0.0 <= v <= 1.0 for v in nlds)


def test_curves_handle_empty_and_one_sided():
    words = make_synthetic_words()
    empty = make_synthetic_fixations().iloc[0:0]
    real = make_synthetic_fixations()
    assert nld_by_fixation_index(empty, empty, words) == ([], [])
    assert nld_by_time(empty, empty, words) == ([], [])
    # One side empty: the curve spans the non-empty scanpath, every prefix is
    # maximally different (NLD 1.0).
    ks, nlds = nld_by_fixation_index(real, empty, words)
    assert ks == list(range(1, len(real) + 1))
    assert all(v == 1.0 for v in nlds)
