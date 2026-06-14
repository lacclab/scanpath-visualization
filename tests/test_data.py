"""Tests for data.py module."""

from unittest.mock import patch

import pandas as pd

from scanpath_studio.data import (
    compute_canvas_size,
    compute_word_metrics,
    default_filters,
    filter_data,
    filter_raw_gaze,
    infer_fix_schema,
    infer_raw_gaze_schema,
    infer_word_schema,
    normalize_fixations,
    normalize_raw_gaze,
    normalize_words,
    pick_column,
    propose_word_schema,
    trial_id_series,
    trial_mapping_columns,
)
from scanpath_studio.controls import (
    BOX_FORMAT_EDGES,
    BOX_FORMAT_ORIGIN,
    _default_box_format,
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

    def test_pick_column_case_insensitive(self):
        df = pd.DataFrame({"Participant_ID": ["p1"], "IA_LEFT": [1]})
        assert pick_column(df, ["participant_id"]) == "Participant_ID"
        assert pick_column(df, ["ia_left"]) == "IA_LEFT"

    def test_pick_column_separator_insensitive(self):
        df = pd.DataFrame({"participant id": ["p1"], "Word-Id": [1]})
        assert pick_column(df, ["participant_id"]) == "participant id"
        assert pick_column(df, ["word_id"]) == "Word-Id"

    def test_pick_column_priority_order_preserved(self):
        # The first candidate that matches anything wins, even case-insensitively.
        df = pd.DataFrame({"X": [1], "left": [2]})
        assert pick_column(df, ["x", "left"]) == "X"
        assert pick_column(df, ["left", "x"]) == "left"

    def test_pick_column_leftmost_duplicate_wins(self):
        # Two columns folding to the same key — the leftmost original wins.
        df = pd.DataFrame({"IA_LEFT": [1], "ia_left": [2]})
        assert pick_column(df, ["ia_left"]) == "IA_LEFT"


class TestProposeWordSchemaMatching:
    """propose_word_schema resolves real-world column-name variants."""

    def test_mixed_case_and_separators(self):
        df = pd.DataFrame(
            {
                "Participant ID": ["p1"],
                "TRIAL-ID": ["t1"],
                "Ia Id": [1],
                "IA Left": [100],
                "IA Right": [150],
                "IA Top": [50],
                "IA Bottom": [100],
            }
        )
        schema = propose_word_schema(df)
        assert schema["participant"] == "Participant ID"
        assert schema["trial"] == "TRIAL-ID"
        assert schema["word_id"] == "Ia Id"
        assert schema["left"] == "IA Left"
        assert schema["right"] == "IA Right"
        assert schema["top"] == "IA Top"
        assert schema["bottom"] == "IA Bottom"


class TestDefaultBoxFormat:
    """_default_box_format picks the encoding auto-detect actually found."""

    def test_edges_when_edges_detected(self):
        proposed = {
            "left": "IA_LEFT",
            "right": "IA_RIGHT",
            "top": "IA_TOP",
            "bottom": "IA_BOTTOM",
            "x": None,
            "y": None,
            "width": None,
            "height": None,
        }
        assert _default_box_format(proposed) == BOX_FORMAT_EDGES

    def test_origin_when_origin_detected(self):
        proposed = {
            "left": None,
            "right": None,
            "top": None,
            "bottom": None,
            "x": "x",
            "y": "y",
            "width": "width",
            "height": "height",
        }
        assert _default_box_format(proposed) == BOX_FORMAT_ORIGIN

    def test_defaults_to_edges_when_nothing_detected(self):
        assert _default_box_format({}) == BOX_FORMAT_EDGES


class TestInferWordSchema:
    """Tests for infer_word_schema function."""

    @patch("scanpath_studio.data.st")
    def test_infer_word_schema_success(self, mock_st):
        df = pd.DataFrame(
            {
                "participant_id": ["p1"],
                "trial_id": ["t1"],
                "IA_ID": [1],
                "IA_LEFT": [100],
                "IA_RIGHT": [150],
                "IA_TOP": [50],
                "IA_BOTTOM": [100],
                "IA_LABEL": ["word"],
            }
        )
        schema = infer_word_schema(df)
        assert schema is not None
        assert schema["participant"] == "participant_id"
        assert schema["trial"] == "trial_id"
        assert schema["word_id"] == "IA_ID"
        assert schema["left"] == "IA_LEFT"

    @patch("scanpath_studio.data.st")
    def test_infer_word_schema_missing_required(self, mock_st):
        df = pd.DataFrame({"col1": [1]})
        schema = infer_word_schema(df)
        assert schema is None

    @patch("scanpath_studio.data.st")
    def test_infer_word_schema_missing_coordinates(self, mock_st):
        df = pd.DataFrame(
            {
                "participant_id": ["p1"],
                "trial_id": ["t1"],
                "IA_ID": [1],
            }
        )
        schema = infer_word_schema(df)
        assert schema is None


class TestInferFixSchema:
    """Tests for infer_fix_schema function."""

    @patch("scanpath_studio.data.st")
    def test_infer_fix_schema_success(self, mock_st):
        df = pd.DataFrame(
            {
                "participant_id": ["p1"],
                "trial_id": ["t1"],
                "CURRENT_FIX_X": [100],
                "CURRENT_FIX_Y": [200],
                "CURRENT_FIX_DURATION": [250],
            }
        )
        schema = infer_fix_schema(df)
        assert schema is not None
        assert schema["participant"] == "participant_id"
        assert schema["x"] == "CURRENT_FIX_X"
        assert schema["y"] == "CURRENT_FIX_Y"
        assert schema["duration"] == "CURRENT_FIX_DURATION"

    @patch("scanpath_studio.data.st")
    def test_infer_fix_schema_missing_required(self, mock_st):
        df = pd.DataFrame({"col1": [1]})
        schema = infer_fix_schema(df)
        assert schema is None


class TestInferRawGazeSchema:
    """Tests for infer_raw_gaze_schema function."""

    @patch("scanpath_studio.data.st")
    def test_infer_raw_gaze_schema_success(self, mock_st):
        df = pd.DataFrame(
            {
                "participant_id": ["p1"],
                "trial_id": ["t1"],
                "x": [100],
                "y": [200],
                "timestamp": [0],
            }
        )
        schema = infer_raw_gaze_schema(df)
        assert schema is not None
        assert schema["participant"] == "participant_id"
        assert schema["trial"] == "trial_id"
        assert schema["x"] == "x"
        assert schema["y"] == "y"

    @patch("scanpath_studio.data.st")
    def test_infer_raw_gaze_schema_missing_required(self, mock_st):
        df = pd.DataFrame({"col1": [1]})
        schema = infer_raw_gaze_schema(df)
        assert schema is None


class TestNormalizeWords:
    """Tests for normalize_words function."""

    def test_normalize_words_with_box_coordinates(self):
        df = pd.DataFrame(
            {
                "participant_id": ["p1", "p1"],
                "trial_id": ["t1", "t1"],
                "IA_ID": [1, 2],
                "IA_LEFT": [100, 200],
                "IA_RIGHT": [150, 250],
                "IA_TOP": [50, 50],
                "IA_BOTTOM": [100, 100],
                "IA_LABEL": ["word1", "word2"],
            }
        )
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
        df = pd.DataFrame(
            {
                "participant_id": ["p1"],
                "trial_id": ["t1"],
                "word_id": [1],
                "x": [100],
                "y": [50],
                "width": [50],
                "height": [50],
            }
        )
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
        df = pd.DataFrame(
            {
                "participant_id": ["p1", "p1"],
                "trial_id": ["t1", "t1"],
                "CURRENT_FIX_X": [100, 200],
                "CURRENT_FIX_Y": [150, 250],
                "CURRENT_FIX_DURATION": [250, 300],
                "CURRENT_FIX_START": [0, 250],
            }
        )
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
        df = pd.DataFrame(
            {
                "participant_id": ["p1", "p1"],
                "trial_id": ["t1", "t1"],
                "CURRENT_FIX_X": [100, 200],
                "CURRENT_FIX_Y": [150, 250],
                "CURRENT_FIX_DURATION": [250, 300],
            }
        )
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
        df = pd.DataFrame(
            {
                "participant_id": ["p1", "p1"],
                "trial_id": ["t1", "t1"],
                "x": [100, 200],
                "y": [150, 250],
                "timestamp": [0, 1],
            }
        )
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


class TestCompositeTrialId:
    """Composite (multi-column) trial mapping: build unique_trial_id on the fly."""

    WORD_SCHEMA = {
        "participant": "participant_id",
        "trial": ["participant_id", "para", "rep"],
        "text_id": "para",
        "word_id": "word_id",
        "text": None,
        "line": None,
        "x": "x",
        "y": "y",
        "width": "width",
        "height": "height",
        "left": None,
        "right": None,
        "top": None,
        "bottom": None,
    }

    FIX_SCHEMA = {
        "participant": "participant_id",
        "trial": ["participant_id", "para", "rep"],
        "text_id": "para",
        "fixation_id": None,
        "timestamp": None,
        "duration": "dur",
        "x": "x",
        "y": "y",
        "word_id": None,
        "pass_index": None,
        "saccade_type": None,
        "saccade_amplitude": None,
        "eye": None,
        "noise_flag": None,
    }

    @staticmethod
    def _words(**extra):
        return pd.DataFrame(
            {
                "participant_id": ["p1", "p1", "p2", "p2"],
                "para": ["A", "A", "A", "B"],
                "rep": [False, True, False, False],
                "word_id": [1, 1, 1, 1],
                "x": [0, 0, 0, 0],
                "y": [0, 0, 0, 0],
                "width": [10, 10, 10, 10],
                "height": [10, 10, 10, 10],
                **extra,
            }
        )

    def test_trial_mapping_columns(self):
        assert trial_mapping_columns("trial_id") == ["trial_id"]
        assert trial_mapping_columns(["a", "b"]) == ["a", "b"]
        assert trial_mapping_columns(("a",)) == ["a"]

    def test_trial_id_series_joins_with_underscore(self):
        df = pd.DataFrame({"p": ["p1"], "t": [3], "r": [True]})
        assert trial_id_series(df, ["p", "t", "r"]).tolist() == ["p1_3_True"]
        assert trial_id_series(df, "t").tolist() == ["3"]

    def test_words_composite_builds_unique_trial_id(self):
        result = normalize_words(self._words(), self.WORD_SCHEMA)
        assert result["trial_id"].tolist() == [
            "p1_A_False",
            "p1_A_True",
            "p2_A_False",
            "p2_B_False",
        ]
        assert (result["unique_trial_id"] == result["trial_id"]).all()

    def test_composite_wins_over_raw_unique_trial_id_column(self):
        # An explicit multi-column choice is authoritative even when the data
        # already carries a unique_trial_id column.
        words = self._words(unique_trial_id=["u1", "u1", "u2", "u3"])
        result = normalize_words(words, self.WORD_SCHEMA)
        assert result["trial_id"].tolist() == [
            "p1_A_False",
            "p1_A_True",
            "p2_A_False",
            "p2_B_False",
        ]

    def test_single_element_list_matches_plain_string_mapping(self):
        words = self._words()
        as_list = normalize_words(words, {**self.WORD_SCHEMA, "trial": ["para"]})
        as_str = normalize_words(words, {**self.WORD_SCHEMA, "trial": "para"})
        assert as_list["trial_id"].tolist() == as_str["trial_id"].tolist()

    def test_fixations_align_with_words_on_same_composite(self):
        words = normalize_words(self._words(), self.WORD_SCHEMA)
        fixations = pd.DataFrame(
            {
                "participant_id": ["p1", "p1", "p2"],
                "para": ["A", "A", "B"],
                "rep": [False, True, False],
                "x": [1, 2, 3],
                "y": [1, 2, 3],
                "dur": [100, 120, 90],
            }
        )
        result = normalize_fixations(fixations, self.FIX_SCHEMA)
        assert (result["unique_trial_id"] == result["trial_id"]).all()
        assert set(result["trial_id"]) <= set(words["trial_id"])

    def test_raw_gaze_composite(self):
        raw = pd.DataFrame(
            {
                "participant_id": ["p1", "p1"],
                "para": ["A", "A"],
                "rep": [False, False],
                "x": [1.0, 2.0],
                "y": [3.0, 4.0],
            }
        )
        schema = {
            "participant": "participant_id",
            "trial": ["participant_id", "para", "rep"],
            "text": None,
            "x": "x",
            "y": "y",
            "timestamp": None,
        }
        result = normalize_raw_gaze(raw, schema)
        assert result["trial_id"].tolist() == ["p1_A_False", "p1_A_False"]
        assert (result["unique_trial_id"] == result["trial_id"]).all()

    def test_onestop_composite_reproduces_unique_trial_id_partition(self):
        # OneStop's unique trial id is participant + paragraph + repeated
        # reading; composing those three columns must partition the bundled
        # OneStop sample exactly like the precomputed unique_trial_id.
        from scanpath_studio.data import load_sample_data

        words, _ = load_sample_data()
        composite_cols = [
            "participant_id",
            "unique_paragraph_id",
            "repeated_reading_trial",
        ]
        stripped = words.drop(columns=["unique_trial_id"])
        schema = propose_word_schema(stripped)
        schema["trial"] = composite_cols
        result = normalize_words(stripped, schema)

        pairs = pd.DataFrame(
            {
                "composite": result["trial_id"].to_numpy(),
                "original": words["unique_trial_id"].astype(str).to_numpy(),
            }
        ).drop_duplicates()
        # Bijection: every composite id maps to exactly one original unique
        # trial id and vice versa.
        assert pairs["composite"].nunique() == pairs["original"].nunique() == len(pairs)


class TestFilterData:
    """Tests for filter_data function."""

    def test_filter_data_by_participants(
        self, normalized_words_df, normalized_fixations_df
    ):
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
        words_df = pd.DataFrame(
            {
                "participant_id": ["p1"],
                "trial_id": ["t1"],
            }
        )
        words_filtered, fixations_filtered = filter_data(
            words_df, normalized_fixations_df, filters
        )
        assert all(fixations_filtered["pass_index"] == 1)

    def test_filter_data_exclude_noise(self, normalized_fixations_df):
        normalized_fixations_df["noise_flag"] = [False, False, True]
        filters = {"participants": ["p1"], "trials": ["t1"], "include_noise": False}
        words_df = pd.DataFrame(
            {
                "participant_id": ["p1"],
                "trial_id": ["t1"],
            }
        )
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
        width, height = compute_canvas_size(
            normalized_words_df, normalized_fixations_df
        )
        assert isinstance(width, int)
        assert isinstance(height, int)
        assert width >= 100
        assert height >= 100


class TestComputeWordMetrics:
    """Tests for compute_word_metrics function."""

    def test_compute_word_metrics_basic(
        self, normalized_words_df, normalized_fixations_df
    ):
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

    def test_default_filters_with_pass_index(
        self, normalized_words_df, normalized_fixations_df
    ):
        normalized_fixations_df["pass_index"] = [1, 1, 2]
        filters = default_filters(normalized_words_df, normalized_fixations_df)
        assert "pass_indices" in filters

    def test_default_filters_with_saccade_type(
        self, normalized_words_df, normalized_fixations_df
    ):
        normalized_fixations_df["saccade_type"] = ["RIGHT", "LEFT", "RIGHT"]
        filters = default_filters(normalized_words_df, normalized_fixations_df)
        assert "saccade_types" in filters
