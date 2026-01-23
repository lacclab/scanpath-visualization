"""Tests for data.py module."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from scanpath_visualization_app.data import (
    pick_column,
    infer_word_schema,
    infer_fix_schema,
    infer_raw_gaze_schema,
    normalize_words,
    normalize_fixations,
    normalize_raw_gaze,
    filter_data,
    filter_raw_gaze,
    compute_canvas_size,
    compute_word_metrics,
    default_filters,
)


class TestPickColumn:
    """Tests for pick_column function."""

    def test_pick_column_found(self):
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        assert pick_column(df, ["col1", "col2"]) == "col1"
        assert pick_column(df, ["col2", "col1"]) == "col2"

    def test_pick_column_not_found(self):
        df = pd.DataFrame({"col1": [1, 2]})
        assert pick_column(df, ["col3", "col4"]) is None

    def test_pick_column_empty_candidates(self):
        df = pd.DataFrame({"col1": [1, 2]})
        assert pick_column(df, []) is None


class TestInferWordSchema:
    """Tests for infer_word_schema function."""

    @patch("scanpath_visualization_app.data.st")
    def test_infer_word_schema_success(self, mock_st):
        df = pd.DataFrame({
            "participant_id": ["p1"],
            "trial_id": ["t1"],
            "IA_ID": [1],
            "IA_LEFT": [100],
            "IA_RIGHT": [150],
            "IA_TOP": [50],
            "IA_BOTTOM": [100],
            "IA_LABEL": ["word"],
        })
        schema = infer_word_schema(df)
        assert schema is not None
        assert schema["participant"] == "participant_id"
        assert schema["trial"] == "trial_id"
        assert schema["word_id"] == "IA_ID"
        assert schema["left"] == "IA_LEFT"

    @patch("scanpath_visualization_app.data.st")
    def test_infer_word_schema_missing_required(self, mock_st):
        df = pd.DataFrame({"col1": [1]})
        schema = infer_word_schema(df)
        assert schema is None

    @patch("scanpath_visualization_app.data.st")
    def test_infer_word_schema_missing_coordinates(self, mock_st):
        df = pd.DataFrame({
            "participant_id": ["p1"],
            "trial_id": ["t1"],
            "IA_ID": [1],
        })
        schema = infer_word_schema(df)
        assert schema is None


class TestInferFixSchema:
    """Tests for infer_fix_schema function."""

    @patch("scanpath_visualization_app.data.st")
    def test_infer_fix_schema_success(self, mock_st):
        df = pd.DataFrame({
            "participant_id": ["p1"],
            "trial_id": ["t1"],
            "CURRENT_FIX_X": [100],
            "CURRENT_FIX_Y": [200],
            "CURRENT_FIX_DURATION": [250],
        })
        schema = infer_fix_schema(df)
        assert schema is not None
        assert schema["participant"] == "participant_id"
        assert schema["x"] == "CURRENT_FIX_X"
        assert schema["y"] == "CURRENT_FIX_Y"
        assert schema["duration"] == "CURRENT_FIX_DURATION"

    @patch("scanpath_visualization_app.data.st")
    def test_infer_fix_schema_missing_required(self, mock_st):
        df = pd.DataFrame({"col1": [1]})
        schema = infer_fix_schema(df)
        assert schema is None


class TestInferRawGazeSchema:
    """Tests for infer_raw_gaze_schema function."""

    @patch("scanpath_visualization_app.data.st")
    def test_infer_raw_gaze_schema_success(self, mock_st):
        df = pd.DataFrame({
            "participant_id": ["p1"],
            "trial_id": ["t1"],
            "x": [100],
            "y": [200],
            "timestamp": [0],
        })
        schema = infer_raw_gaze_schema(df)
        assert schema is not None
        assert schema["participant"] == "participant_id"
        assert schema["trial"] == "trial_id"
        assert schema["x"] == "x"
        assert schema["y"] == "y"

    @patch("scanpath_visualization_app.data.st")
    def test_infer_raw_gaze_schema_missing_required(self, mock_st):
        df = pd.DataFrame({"col1": [1]})
        schema = infer_raw_gaze_schema(df)
        assert schema is None


class TestNormalizeWords:
    """Tests for normalize_words function."""

    def test_normalize_words_with_box_coordinates(self):
        df = pd.DataFrame({
            "participant_id": ["p1", "p1"],
            "trial_id": ["t1", "t1"],
            "IA_ID": [1, 2],
            "IA_LEFT": [100, 200],
            "IA_RIGHT": [150, 250],
            "IA_TOP": [50, 50],
            "IA_BOTTOM": [100, 100],
            "IA_LABEL": ["word1", "word2"],
        })
        schema = {
            "participant": "participant_id",
            "trial": "trial_id",
            "word_id": "IA_ID",
            "left": "IA_LEFT",
            "right": "IA_RIGHT",
            "top": "IA_TOP",
            "bottom": "IA_BOTTOM",
            "text": "IA_LABEL",
        }
        result = normalize_words(df, schema)
        assert "participant_id" in result.columns
        assert "trial_id" in result.columns
        assert "word_id" in result.columns
        assert "x" in result.columns
        assert "y" in result.columns
        assert "width" in result.columns
        assert "height" in result.columns
        assert result["x"].iloc[0] == 100
        assert result["width"].iloc[0] == 50

    def test_normalize_words_with_xywh_coordinates(self):
        df = pd.DataFrame({
            "participant_id": ["p1"],
            "trial_id": ["t1"],
            "word_id": [1],
            "x": [100],
            "y": [50],
            "width": [50],
            "height": [50],
        })
        schema = {
            "participant": "participant_id",
            "trial": "trial_id",
            "word_id": "word_id",
            "x": "x",
            "y": "y",
            "width": "width",
            "height": "height",
        }
        result = normalize_words(df, schema)
        assert result["x"].iloc[0] == 100
        assert result["width"].iloc[0] == 50


class TestNormalizeFixations:
    """Tests for normalize_fixations function."""

    def test_normalize_fixations_basic(self):
        df = pd.DataFrame({
            "participant_id": ["p1", "p1"],
            "trial_id": ["t1", "t1"],
            "CURRENT_FIX_X": [100, 200],
            "CURRENT_FIX_Y": [150, 250],
            "CURRENT_FIX_DURATION": [250, 300],
            "CURRENT_FIX_START": [0, 250],
        })
        schema = {
            "participant": "participant_id",
            "trial": "trial_id",
            "x": "CURRENT_FIX_X",
            "y": "CURRENT_FIX_Y",
            "duration": "CURRENT_FIX_DURATION",
            "timestamp": "CURRENT_FIX_START",
        }
        result = normalize_fixations(df, schema)
        assert "participant_id" in result.columns
        assert "trial_id" in result.columns
        assert "x" in result.columns
        assert "y" in result.columns
        assert "duration_ms" in result.columns
        assert "timestamp_ms" in result.columns
        assert "order_in_trial" in result.columns
        assert result["x"].iloc[0] == 100
        assert result["duration_ms"].iloc[0] == 250

    def test_normalize_fixations_without_timestamp(self):
        df = pd.DataFrame({
            "participant_id": ["p1", "p1"],
            "trial_id": ["t1", "t1"],
            "CURRENT_FIX_X": [100, 200],
            "CURRENT_FIX_Y": [150, 250],
            "CURRENT_FIX_DURATION": [250, 300],
        })
        schema = {
            "participant": "participant_id",
            "trial": "trial_id",
            "x": "CURRENT_FIX_X",
            "y": "CURRENT_FIX_Y",
            "duration": "CURRENT_FIX_DURATION",
        }
        result = normalize_fixations(df, schema)
        assert "timestamp_ms" in result.columns
        # Should auto-generate timestamps
        assert result["timestamp_ms"].iloc[0] == 0
        assert result["timestamp_ms"].iloc[1] == 1


class TestNormalizeRawGaze:
    """Tests for normalize_raw_gaze function."""

    def test_normalize_raw_gaze_basic(self):
        df = pd.DataFrame({
            "participant_id": ["p1", "p1"],
            "trial_id": ["t1", "t1"],
            "x": [100, 200],
            "y": [150, 250],
            "timestamp": [0, 1],
        })
        schema = {
            "participant": "participant_id",
            "trial": "trial_id",
            "x": "x",
            "y": "y",
            "timestamp": "timestamp",
        }
        result = normalize_raw_gaze(df, schema)
        assert "participant_id" in result.columns
        assert "trial_id" in result.columns
        assert "x" in result.columns
        assert "y" in result.columns
        assert "timestamp_ms" in result.columns


class TestFilterData:
    """Tests for filter_data function."""

    def test_filter_data_by_participants(self, normalized_words_df, normalized_fixations_df):
        filters = {"participants": ["p1"], "trials": ["t1"]}
        words_filtered, fixations_filtered = filter_data(
            normalized_words_df, normalized_fixations_df, filters
        )
        assert all(words_filtered["participant_id"] == "p1")
        assert all(fixations_filtered["participant_id"] == "p1")

    def test_filter_data_by_trials(self, normalized_words_df, normalized_fixations_df):
        filters = {"participants": ["p1", "p2"], "trials": ["t1"]}
        words_filtered, fixations_filtered = filter_data(
            normalized_words_df, normalized_fixations_df, filters
        )
        assert all(words_filtered["trial_id"] == "t1")
        assert all(fixations_filtered["trial_id"] == "t1")

    def test_filter_data_by_pass_index(self, normalized_fixations_df):
        normalized_fixations_df["pass_index"] = [1, 1, 2]
        filters = {"participants": ["p1"], "trials": ["t1"], "pass_indices": [1]}
        words_df = pd.DataFrame({
            "participant_id": ["p1"],
            "trial_id": ["t1"],
        })
        words_filtered, fixations_filtered = filter_data(
            words_df, normalized_fixations_df, filters
        )
        assert all(fixations_filtered["pass_index"] == 1)

    def test_filter_data_exclude_noise(self, normalized_fixations_df):
        normalized_fixations_df["noise_flag"] = [False, False, True]
        filters = {"participants": ["p1"], "trials": ["t1"], "include_noise": False}
        words_df = pd.DataFrame({
            "participant_id": ["p1"],
            "trial_id": ["t1"],
        })
        words_filtered, fixations_filtered = filter_data(
            words_df, normalized_fixations_df, filters
        )
        assert all(~fixations_filtered["noise_flag"])


class TestFilterRawGaze:
    """Tests for filter_raw_gaze function."""

    def test_filter_raw_gaze_basic(self, sample_raw_gaze_df):
        result = filter_raw_gaze(sample_raw_gaze_df, ["p1"], ["t1"])
        assert len(result) == 5
        assert all(result["participant_id"] == "p1")

    def test_filter_raw_gaze_empty(self):
        empty_df = pd.DataFrame()
        result = filter_raw_gaze(empty_df, ["p1"], ["t1"])
        assert result.empty

    def test_filter_raw_gaze_no_match(self, sample_raw_gaze_df):
        result = filter_raw_gaze(sample_raw_gaze_df, ["p2"], ["t2"])
        assert len(result) == 0


class TestComputeCanvasSize:
    """Tests for compute_canvas_size function."""

    def test_compute_canvas_size(self, normalized_words_df, normalized_fixations_df):
        width, height = compute_canvas_size(normalized_words_df, normalized_fixations_df)
        assert isinstance(width, int)
        assert isinstance(height, int)
        assert width >= 100
        assert height >= 100


class TestComputeWordMetrics:
    """Tests for compute_word_metrics function."""

    def test_compute_word_metrics_basic(self, normalized_words_df, normalized_fixations_df):
        normalized_words_df["first_fixation_ms"] = [200, 250, 180]
        normalized_words_df["n_fixations"] = [1, 2, 1]
        result = compute_word_metrics(normalized_words_df, normalized_fixations_df)
        assert "participant_id" in result.columns
        assert "trial_id" in result.columns
        assert "word_id" in result.columns
        assert "first_fixation_ms" in result.columns


class TestDefaultFilters:
    """Tests for default_filters function."""

    def test_default_filters_basic(self, normalized_words_df, normalized_fixations_df):
        filters = default_filters(normalized_words_df, normalized_fixations_df)
        assert "participants" in filters
        assert "trials" in filters
        assert isinstance(filters["participants"], list)
        assert isinstance(filters["trials"], list)

    def test_default_filters_with_pass_index(self, normalized_words_df, normalized_fixations_df):
        normalized_fixations_df["pass_index"] = [1, 1, 2]
        filters = default_filters(normalized_words_df, normalized_fixations_df)
        assert "pass_indices" in filters

    def test_default_filters_with_saccade_type(self, normalized_words_df, normalized_fixations_df):
        normalized_fixations_df["saccade_type"] = ["RIGHT", "LEFT", "RIGHT"]
        filters = default_filters(normalized_words_df, normalized_fixations_df)
        assert "saccade_types" in filters
