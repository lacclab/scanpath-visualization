"""Tests for app.py utility functions."""

import pandas as pd

from scanpath_studio.app import (
    _build_comparison_options,
    build_combo_options,
    compute_trial_stats,
    gather_trial_metadata,
)
from scanpath_studio.data import compute_canvas_size


class TestBuildComboOptions:
    """Tests for build_combo_options function."""

    def test_build_combo_options_basic(self):
        fixations = pd.DataFrame(
            {
                "participant_id": ["p1", "p1", "p2"],
                "trial_id": ["t1", "t2", "t1"],
                "paragraph_id": ["para1", "para1", "para1"],
            }
        )
        combos, labels, label_to_combo = build_combo_options(fixations)
        assert len(combos) == 3
        assert len(labels) == 3
        assert len(label_to_combo) == 3
        assert "participant_id" in combos.columns
        assert "trial_id" in combos.columns
        assert "paragraph_id" in combos.columns

    def test_build_combo_options_with_unique_ids(self):
        fixations = pd.DataFrame(
            {
                "participant_id": ["p1", "p1"],
                "trial_id": ["t1", "t1"],
                "unique_trial_id": ["ut1", "ut1"],
                "unique_paragraph_id": ["up1", "up1"],
            }
        )
        combos, labels, label_to_combo = build_combo_options(fixations)
        assert "unique_trial_id" in combos.columns or "trial_id" in combos.columns

    def test_build_combo_options_with_trial_index(self):
        fixations = pd.DataFrame(
            {
                "participant_id": ["p1", "p1"],
                "trial_id": ["t1", "t2"],
                "paragraph_id": ["para1", "para1"],
                "TRIAL_INDEX": [1, 2],
            }
        )
        combos, labels, label_to_combo = build_combo_options(fixations)
        assert len(combos) == 2


class TestComputeCanvasSize:
    """Tests for compute_canvas_size (replaces removed clamp_canvas_size)."""

    def test_canvas_size_uses_data_extent(
        self, normalized_words_df, normalized_fixations_df
    ):
        width, height = compute_canvas_size(
            normalized_words_df, normalized_fixations_df
        )
        # The fixture has words extending to x=350, y=100 — expect width≥350
        assert width >= 350
        assert height >= 100

    def test_canvas_size_floor(self):
        empty = pd.DataFrame()
        width, height = compute_canvas_size(empty, empty)
        assert width >= 100
        assert height >= 100


class TestComputeTrialStats:
    """Tests for compute_trial_stats function."""

    def test_compute_trial_stats_basic(
        self, normalized_words_df, normalized_fixations_df
    ):
        stats = compute_trial_stats(normalized_words_df, normalized_fixations_df)
        assert "total_reading_time_ms" in stats
        assert "total_reading_time_s" in stats
        assert "word_count" in stats
        assert "fixation_count" in stats
        assert stats["word_count"] == len(normalized_words_df)
        assert stats["fixation_count"] == len(normalized_fixations_df)

    def test_compute_trial_stats_with_dwell_time(self):
        words = pd.DataFrame(
            {
                "participant_id": ["p1"],
                "trial_id": ["t1"],
                "trial_dwell_time_ms": [5000],
            }
        )
        fixations = pd.DataFrame(
            {
                "participant_id": ["p1", "p1"],
                "trial_id": ["t1", "t1"],
                "duration_ms": [200, 250],
            }
        )
        stats = compute_trial_stats(words, fixations)
        assert stats["total_reading_time_ms"] == 5000

    def test_compute_trial_stats_empty_fixations(self, normalized_words_df):
        empty_fixations = pd.DataFrame()
        stats = compute_trial_stats(normalized_words_df, empty_fixations)
        assert stats["fixation_count"] == 0
        assert stats["total_reading_time_ms"] == 0


class TestGatherTrialMetadata:
    """Tests for gather_trial_metadata function."""

    def test_gather_trial_metadata_single_value(
        self, normalized_words_df, normalized_fixations_df
    ):
        normalized_words_df["difficulty_level"] = ["Adv", "Adv", "Adv"]
        metadata = gather_trial_metadata(
            normalized_words_df, normalized_fixations_df, ["difficulty_level"]
        )
        assert len(metadata) == 1
        assert metadata.iloc[0]["Field"] == "difficulty_level"
        assert "Adv" in str(metadata.iloc[0]["Value"])

    def test_gather_trial_metadata_numeric(
        self, normalized_words_df, normalized_fixations_df
    ):
        normalized_fixations_df["duration_ms"] = [200, 250, 180]
        metadata = gather_trial_metadata(
            normalized_words_df, normalized_fixations_df, ["duration_ms"]
        )
        assert len(metadata) == 1
        assert "mean" in str(metadata.iloc[0]["Value"]).lower()

    def test_gather_trial_metadata_missing_field(
        self, normalized_words_df, normalized_fixations_df
    ):
        metadata = gather_trial_metadata(
            normalized_words_df, normalized_fixations_df, ["nonexistent_field"]
        )
        assert len(metadata) == 0

    def test_gather_trial_metadata_multiple_fields(
        self, normalized_words_df, normalized_fixations_df
    ):
        normalized_words_df["difficulty_level"] = ["Adv", "Adv", "Adv"]
        normalized_fixations_df["pass_index"] = [1, 1, 1]
        metadata = gather_trial_metadata(
            normalized_words_df,
            normalized_fixations_df,
            ["difficulty_level", "pass_index"],
        )
        assert len(metadata) == 2


class TestBuildComparisonOptions:
    """Tests for _build_comparison_options function."""

    def test_build_comparison_options_text_mode(self):
        combos = pd.DataFrame(
            {
                "participant_id": ["p1", "p1", "p2"],
                "trial_id": ["t1", "t2", "t1"],
                "paragraph_id": ["para1", "para1", "para1"],
            }
        )
        options = _build_comparison_options(combos, "Text", "p1", "t1", "para1")
        assert len(options) > 0
        # Should prioritize same participant, same text
        assert any("p1" in opt[2] for opt in options)

    def test_build_comparison_options_participant_mode(self):
        combos = pd.DataFrame(
            {
                "participant_id": ["p1", "p2", "p3"],
                "trial_id": ["t1", "t1", "t1"],
                "paragraph_id": ["para1", "para1", "para2"],
            }
        )
        options = _build_comparison_options(combos, "Participant", "p1", "t1", "para1")
        assert len(options) > 0
        # Should prioritize same text, different participants
        assert any("para1" in opt[2] and "p2" in opt[2] for opt in options)

    def test_build_comparison_options_none_mode(self):
        combos = pd.DataFrame(
            {
                "participant_id": ["p1", "p2"],
                "trial_id": ["t1", "t1"],
                "paragraph_id": ["para1", "para2"],
            }
        )
        options = _build_comparison_options(combos, "None", "p1", "t1", None)
        assert len(options) > 0
        # Should not include the primary trial
        assert not any(opt[0] == "p1" and opt[1] == "t1" for opt in options)

    def test_build_comparison_options_with_unique_paragraph_id(self):
        combos = pd.DataFrame(
            {
                "participant_id": ["p1", "p2"],
                "trial_id": ["t1", "t1"],
                "unique_paragraph_id": ["up1", "up1"],
            }
        )
        options = _build_comparison_options(combos, "Text", "p1", "t1", "up1")
        assert len(options) > 0
