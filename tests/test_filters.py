"""Tests for trial-level filtering helpers in data.py."""

from __future__ import annotations

import pandas as pd

from scanpath_visualization_app.data import filter_to_keys, filter_trials


def _words():
    return pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2", "p2"],
            "trial_id": ["a", "b", "a", "b"],
            "question_preview": [True, False, True, False],
            "difficulty_level": ["Adv", "Ele", "Adv", "Ele"],
        }
    )


def _fixations():
    # Two fixations per trial.
    return pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p1", "p1", "p2", "p2", "p2", "p2"],
            "trial_id": ["a", "a", "b", "b", "a", "a", "b", "b"],
            "question_preview": [True, True, False, False, True, True, False, False],
            "difficulty_level": [
                "Adv",
                "Adv",
                "Ele",
                "Ele",
                "Adv",
                "Adv",
                "Ele",
                "Ele",
            ],
        }
    )


def test_filter_trials_by_participant():
    w, f = filter_trials(_words(), _fixations(), participants=["p1"])
    assert set(w["participant_id"]) == {"p1"}
    assert set(f["participant_id"]) == {"p1"}


def test_filter_trials_by_metadata_hunting():
    # question_preview True == Hunting.
    w, f = filter_trials(_words(), _fixations(), metadata={"question_preview": {True}})
    assert set(w["trial_id"]) == {"a"}  # only the Hunting trials
    assert set(f["trial_id"]) == {"a"}
    assert (w["question_preview"]).all()


def test_filter_trials_combined():
    w, f = filter_trials(
        _words(),
        _fixations(),
        participants=["p2"],
        metadata={"difficulty_level": {"Ele"}},
    )
    assert set(zip(w["participant_id"], w["trial_id"])) == {("p2", "b")}
    assert set(zip(f["participant_id"], f["trial_id"])) == {("p2", "b")}


def test_filter_trials_noop_when_empty_selection():
    w, f = filter_trials(_words(), _fixations(), participants=None, metadata={})
    assert len(w) == 4 and len(f) == 8


def test_filter_to_keys():
    keep = {("p1", "a"), ("p2", "b")}
    w, f = filter_to_keys(_words(), _fixations(), keep)
    assert set(zip(w["participant_id"], w["trial_id"])) == keep
    assert set(zip(f["participant_id"], f["trial_id"])) == keep
