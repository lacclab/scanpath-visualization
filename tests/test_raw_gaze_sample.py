"""Tests for the synthesized raw-gaze demo sample (update_sample_data)."""

from __future__ import annotations

import pandas as pd

from scanpath_studio.update_sample_data import (
    _RAW_GAZE_COLUMNS,
    synthesize_raw_gaze,
)


def _fixations():
    return pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p1", "p1", "p1"],
            "unique_trial_id": ["p1_T1", "p1_T1", "p1_T1", "p1_T2", "p1_T2"],
            "CURRENT_FIX_X": [100, 200, 300, 100, 200],
            "CURRENT_FIX_Y": [100, 100, 100, 300, 300],
            "CURRENT_FIX_DURATION": [200, 150, 180, 100, 120],
            "CURRENT_FIX_START": [0, 400, 800, 0, 300],
        }
    )


def _ia(trial="p1_T1"):
    return pd.DataFrame({"participant_id": ["p1"], "unique_trial_id": [trial]})


def test_synthesize_keys_to_shared_trial():
    rg = synthesize_raw_gaze(_fixations(), _ia("p1_T1"), seed=1)
    assert list(rg.columns) == _RAW_GAZE_COLUMNS
    assert not rg.empty
    # Only the trial present in BOTH tables is used.
    assert set(rg["unique_trial_id"]) == {"p1_T1"}
    assert set(rg["participant_id"]) == {"p1"}
    # Gaze points stay within the trial's spatial extent (+ jitter).
    assert rg["x"].between(50, 350).all()
    assert rg["y"].between(50, 150).all()


def test_synthesize_empty_without_shared_trial():
    rg = synthesize_raw_gaze(_fixations(), _ia("p1_NOPE"), seed=1)
    assert rg.empty
    assert list(rg.columns) == _RAW_GAZE_COLUMNS


def test_synthesize_is_deterministic():
    a = synthesize_raw_gaze(_fixations(), _ia("p1_T1"), seed=7)
    b = synthesize_raw_gaze(_fixations(), _ia("p1_T1"), seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_synthesize_handles_missing_columns():
    bad_fix = pd.DataFrame({"participant_id": ["p1"], "unique_trial_id": ["p1_T1"]})
    rg = synthesize_raw_gaze(bad_fix, _ia("p1_T1"))
    assert rg.empty


def test_bundled_raw_gaze_overlaps_a_bundled_trial():
    """Regression guard: the shipped raw_gaze sample must cover a (participant,
    trial) that exists in the shipped fixations — otherwise the app filters it
    to 0 rows (the original bug)."""
    import pytest

    from scanpath_studio.data import load_sample_data, load_sample_raw_gaze

    _, fixations = load_sample_data()
    raw_gaze = load_sample_raw_gaze()
    if raw_gaze.empty:
        pytest.skip("no bundled raw-gaze sample")
    assert {"participant_id", "unique_trial_id"}.issubset(raw_gaze.columns)
    fix_keys = set(
        zip(
            fixations["participant_id"].astype(str),
            fixations["unique_trial_id"].astype(str),
        )
    )
    rg_keys = set(
        zip(
            raw_gaze["participant_id"].astype(str),
            raw_gaze["unique_trial_id"].astype(str),
        )
    )
    assert rg_keys & fix_keys, "bundled raw gaze overlaps no bundled trial"
