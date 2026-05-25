from __future__ import annotations

import importlib.resources as resources
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from .constants import DEFAULT_FIGURE_SIZE, PACKAGE_NAME


def pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return the first matching column name from a candidate list."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


# Candidate column names checked during auto-inference. Centralised so the
# proposal step and the override UI share the same defaults.
PARTICIPANT_CANDIDATES = [
    "participant_id",
    "subject_id",
    "participant",
    "recording_session_label",
]
TRIAL_CANDIDATES = [
    "unique_trial_id",
    "trial_id",
    "unique_paragraph_id",
    "paragraph_id",
    "trial",
    "trial_index",
    "TRIAL_INDEX",
]
PARAGRAPH_CANDIDATES = ["unique_paragraph_id", "paragraph_id", "PARAGRAPH_ID"]
TEXT_CANDIDATES = [
    "text",
    "IA_LABEL",
    "ia_label",
    "label",
    "word",
    "WORD",
    "content",
    "CONTENT",
    "token",
    "TOKEN",
]
WORD_ID_CANDIDATES = ["word_id", "IA_ID", "ia_id", "ia_index"]
LINE_CANDIDATES = ["line_idx", "line", "line_index", "IA_LINE_ID"]

WORD_X_CANDIDATES = ["x", "X", "left"]
WORD_Y_CANDIDATES = ["y", "Y", "top"]
WORD_WIDTH_CANDIDATES = ["width", "WIDTH"]
WORD_HEIGHT_CANDIDATES = ["height", "HEIGHT"]
WORD_LEFT_CANDIDATES = ["IA_LEFT", "ia_left", "left"]
WORD_RIGHT_CANDIDATES = ["IA_RIGHT", "ia_right", "right"]
WORD_TOP_CANDIDATES = ["IA_TOP", "ia_top", "top"]
WORD_BOTTOM_CANDIDATES = ["IA_BOTTOM", "ia_bottom", "bottom"]

FIX_X_CANDIDATES = ["x", "X", "CURRENT_FIX_X", "FPOGX"]
FIX_Y_CANDIDATES = ["y", "Y", "CURRENT_FIX_Y", "FPOGY"]
FIX_DURATION_CANDIDATES = ["duration_ms", "CURRENT_FIX_DURATION", "CURRENT_FIX_LEN"]
FIX_TIMESTAMP_CANDIDATES = [
    "timestamp_ms",
    "CURRENT_FIX_START",
    "CURRENT_FIX_START_TIME",
    "CURRENT_FIX_TIME",
    "CURRENT_FIX_ONSET",
]
FIX_FIXATION_ID_CANDIDATES = ["fixation_id", "CURRENT_FIX_INDEX", "CURRENT_FIX_NUM"]
FIX_WORD_ID_CANDIDATES = [
    "word_id",
    "IA_ID",
    "ia_id",
    "CURRENT_FIX_INTEREST_AREA_ID",
    "CURRENT_FIX_INTEREST_AREA_INDEX",
]
FIX_PASS_INDEX_CANDIDATES = ["pass_index", "reread", "PASS_INDEX"]
FIX_SACCADE_TYPE_CANDIDATES = ["saccade_type", "SACCADE_TYPE", "NEXT_SAC_DIRECTION"]
FIX_SACCADE_AMPLITUDE_CANDIDATES = [
    "saccade_amplitude",
    "NEXT_SAC_AMPLITUDE",
    "PREVIOUS_SAC_AMPLITUDE",
]
FIX_EYE_CANDIDATES = ["eye", "EYE_USED", "eye_used", "EYE_TRACKED"]
FIX_NOISE_CANDIDATES = ["noise_flag", "CURRENT_FIX_VALIDITY", "CURRENT_FIX_VALID"]

RAW_GAZE_X_CANDIDATES = ["x", "X", "FPOGX", "gaze_x", "GAZE_X"]
RAW_GAZE_Y_CANDIDATES = ["y", "Y", "FPOGY", "gaze_y", "GAZE_Y"]
RAW_GAZE_TIMESTAMP_CANDIDATES = [
    "timestamp",
    "time",
    "ms",
    "timestamp_ms",
    "time_ms",
]


def propose_word_schema(words: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Return a candidate column mapping for words/IA data without erroring."""
    return dict(
        participant=pick_column(words, PARTICIPANT_CANDIDATES),
        trial=pick_column(words, TRIAL_CANDIDATES),
        paragraph=pick_column(words, PARAGRAPH_CANDIDATES),
        word_id=pick_column(words, WORD_ID_CANDIDATES),
        text=pick_column(words, TEXT_CANDIDATES),
        line=pick_column(words, LINE_CANDIDATES),
        x=pick_column(words, WORD_X_CANDIDATES),
        y=pick_column(words, WORD_Y_CANDIDATES),
        width=pick_column(words, WORD_WIDTH_CANDIDATES),
        height=pick_column(words, WORD_HEIGHT_CANDIDATES),
        left=pick_column(words, WORD_LEFT_CANDIDATES),
        right=pick_column(words, WORD_RIGHT_CANDIDATES),
        top=pick_column(words, WORD_TOP_CANDIDATES),
        bottom=pick_column(words, WORD_BOTTOM_CANDIDATES),
    )


def propose_fix_schema(fixations: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Return a candidate column mapping for fixations data without erroring."""
    return dict(
        participant=pick_column(fixations, PARTICIPANT_CANDIDATES),
        trial=pick_column(fixations, TRIAL_CANDIDATES),
        paragraph=pick_column(fixations, PARAGRAPH_CANDIDATES),
        fixation_id=pick_column(fixations, FIX_FIXATION_ID_CANDIDATES),
        timestamp=pick_column(fixations, FIX_TIMESTAMP_CANDIDATES),
        duration=pick_column(fixations, FIX_DURATION_CANDIDATES),
        x=pick_column(fixations, FIX_X_CANDIDATES),
        y=pick_column(fixations, FIX_Y_CANDIDATES),
        word_id=pick_column(fixations, FIX_WORD_ID_CANDIDATES),
        pass_index=pick_column(fixations, FIX_PASS_INDEX_CANDIDATES),
        saccade_type=pick_column(fixations, FIX_SACCADE_TYPE_CANDIDATES),
        saccade_amplitude=pick_column(fixations, FIX_SACCADE_AMPLITUDE_CANDIDATES),
        eye=pick_column(fixations, FIX_EYE_CANDIDATES),
        noise_flag=pick_column(fixations, FIX_NOISE_CANDIDATES),
    )


def propose_raw_gaze_schema(raw_gaze: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Return a candidate column mapping for raw gaze data without erroring."""
    return dict(
        participant=pick_column(raw_gaze, PARTICIPANT_CANDIDATES),
        trial=pick_column(raw_gaze, TRIAL_CANDIDATES),
        text=pick_column(raw_gaze, TEXT_CANDIDATES),
        x=pick_column(raw_gaze, RAW_GAZE_X_CANDIDATES),
        y=pick_column(raw_gaze, RAW_GAZE_Y_CANDIDATES),
        timestamp=pick_column(raw_gaze, RAW_GAZE_TIMESTAMP_CANDIDATES),
    )


def validate_word_schema(schema: Dict[str, Optional[str]]) -> list:
    """Return a list of human-readable problems with a words/IA schema."""
    problems = []
    for key, label in [
        ("participant", "Participant ID"),
        ("trial", "Trial ID"),
        ("word_id", "Word/IA ID"),
    ]:
        if not schema.get(key):
            problems.append(f"missing {label}")
    has_xywh = all(schema.get(k) for k in ["x", "y", "width", "height"])
    has_box = all(schema.get(k) for k in ["left", "right", "top", "bottom"])
    if not has_xywh and not has_box:
        problems.append(
            "need either (x, y, width, height) or (left, right, top, bottom)"
        )
    return problems


def validate_fix_schema(schema: Dict[str, Optional[str]]) -> list:
    """Return a list of human-readable problems with a fixations schema."""
    problems = []
    for key, label in [
        ("participant", "Participant ID"),
        ("trial", "Trial ID"),
        ("duration", "Duration"),
        ("x", "X"),
        ("y", "Y"),
    ]:
        if not schema.get(key):
            problems.append(f"missing {label}")
    return problems


def validate_raw_gaze_schema(schema: Dict[str, Optional[str]]) -> list:
    """Return a list of human-readable problems with a raw gaze schema."""
    problems = []
    for key, label in [
        ("participant", "Participant ID"),
        ("trial", "Trial ID"),
        ("x", "X"),
        ("y", "Y"),
    ]:
        if not schema.get(key):
            problems.append(f"missing {label}")
    return problems


def read_table(file_like_or_path) -> pd.DataFrame:
    """Read a tabular file by extension: csv, parquet, or feather."""
    name = getattr(file_like_or_path, "name", str(file_like_or_path)).lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(file_like_or_path)
    if name.endswith(".feather"):
        return pd.read_feather(file_like_or_path)
    return pd.read_csv(file_like_or_path)


def _load_bundled(name: str) -> pd.DataFrame:
    """Load a single bundled sample, preferring Parquet over CSV."""
    data_root = resources.files(PACKAGE_NAME).joinpath("sample_data")
    for ext in (".parquet", ".csv"):
        resource = data_root / f"{name}{ext}"
        try:
            with resources.as_file(resource) as path:
                if not path.is_file():
                    continue
                return read_table(path)
        except FileNotFoundError:
            continue
    return pd.DataFrame()


@st.cache_data
def load_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load bundled demo IA and fixation tables (prefer Parquet)."""
    words = _load_bundled("ia")
    fixations = _load_bundled("fixations")
    if words.empty or fixations.empty:
        st.error(
            "Bundled sample data not found. Expected ia.{parquet,csv} and "
            "fixations.{parquet,csv} under the installed package's sample_data "
            "directory."
        )
        return pd.DataFrame(), pd.DataFrame()
    return words, fixations


@st.cache_data
def load_sample_raw_gaze() -> pd.DataFrame:
    """Load bundled raw gaze sample (millisecond-level x,y)."""
    return _load_bundled("raw_gaze")


def infer_raw_gaze_schema(raw_gaze: pd.DataFrame) -> Optional[Dict[str, str]]:
    """Infer schema for raw millisecond-level gaze data."""
    schema = propose_raw_gaze_schema(raw_gaze)
    problems = validate_raw_gaze_schema(schema)
    if problems:
        st.error(f"Missing required raw gaze fields: {', '.join(problems)}")
        return None
    return schema


def normalize_raw_gaze(raw_gaze: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
    """Normalize raw gaze data to canonical column names."""
    df = pd.DataFrame()
    df["participant_id"] = raw_gaze[schema["participant"]].astype(str)
    trial_col = (
        "unique_trial_id" if "unique_trial_id" in raw_gaze.columns else schema["trial"]
    )
    df["trial_id"] = raw_gaze[trial_col].astype(str)
    if "unique_trial_id" in raw_gaze.columns:
        df["unique_trial_id"] = raw_gaze["unique_trial_id"].astype(str)
    if schema.get("text"):
        df["text"] = raw_gaze[schema["text"]].astype(str)
    else:
        df["text"] = ""
    df["x"] = pd.to_numeric(raw_gaze[schema["x"]], errors="coerce")
    df["y"] = pd.to_numeric(raw_gaze[schema["y"]], errors="coerce")
    if schema.get("timestamp"):
        df["timestamp_ms"] = pd.to_numeric(
            raw_gaze[schema["timestamp"]], errors="coerce"
        )
    else:
        # Each row represents one millisecond, so use row index within trial as timestamp
        df["timestamp_ms"] = df.groupby(["participant_id", "trial_id"]).cumcount()
    return df


def infer_word_schema(words: pd.DataFrame) -> Optional[Dict[str, str]]:
    schema = propose_word_schema(words)
    problems = validate_word_schema(schema)
    if problems:
        st.error(f"Words/IA schema problems: {'; '.join(problems)}")
        return None
    return schema


def infer_fix_schema(fixations: pd.DataFrame) -> Optional[Dict[str, str]]:
    schema = propose_fix_schema(fixations)
    problems = validate_fix_schema(schema)
    if problems:
        st.error(f"Fixations schema problems: {'; '.join(problems)}")
        return None
    return schema


def normalize_words(words: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame()
    df["participant_id"] = words[schema["participant"]].astype(str)
    trial_col = (
        "unique_trial_id" if "unique_trial_id" in words.columns else schema["trial"]
    )
    df["trial_id"] = words[trial_col].astype(str)
    if "unique_trial_id" in words.columns:
        df["unique_trial_id"] = words["unique_trial_id"].astype(str)
    if "unique_paragraph_id" in words.columns:
        df["unique_paragraph_id"] = words["unique_paragraph_id"].astype(str)
        df["paragraph_id"] = df["unique_paragraph_id"]
    elif schema.get("paragraph"):
        df["paragraph_id"] = words[schema["paragraph"]].astype(str)
    else:
        df["paragraph_id"] = df["trial_id"]
    df["word_id"] = pd.to_numeric(words[schema["word_id"]], errors="coerce")
    if schema.get("text"):
        df["text"] = words[schema["text"]].astype(str)
    else:
        df["text"] = df["word_id"].apply(lambda v: f"w{int(v)}" if pd.notna(v) else "")
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    if schema.get("line"):
        df["line_idx"] = pd.to_numeric(words[schema["line"]], errors="coerce")
    else:
        df["line_idx"] = 1

    if all(schema.get(k) for k in ["x", "y", "width", "height"]):
        df["x"] = pd.to_numeric(words[schema["x"]], errors="coerce")
        df["y"] = pd.to_numeric(words[schema["y"]], errors="coerce")
        df["width"] = pd.to_numeric(words[schema["width"]], errors="coerce")
        df["height"] = pd.to_numeric(words[schema["height"]], errors="coerce")
    else:
        left = pd.to_numeric(words[schema["left"]], errors="coerce")
        right = pd.to_numeric(words[schema["right"]], errors="coerce")
        top = pd.to_numeric(words[schema["top"]], errors="coerce")
        bottom = pd.to_numeric(words[schema["bottom"]], errors="coerce")
        df["x"] = left
        df["y"] = top
        df["width"] = right - left
        df["height"] = bottom - top

    extra_meta = [
        "TRIAL_INDEX",
        "trial_index",
        "article_batch",
        "article_id",
        "difficulty_level",
        "article_title",
        "question",
        "selected_answer",
        "is_correct",
        "repeated_reading_trial",
    ]
    for col in extra_meta:
        if col in words.columns:
            if col == "repeated_reading_trial":
                df[col] = words[col].fillna(False).astype(bool)
            else:
                df[col] = words[col]

    metric_map = {
        "IA_FIRST_FIXATION_DURATION": ("first_fixation_ms", "numeric"),
        "IA_DWELL_TIME": ("total_fixation_duration_ms", "numeric"),
        "IA_FIRST_RUN_DWELL_TIME": ("first_pass_gaze_duration_ms", "numeric"),
        "IA_SECOND_RUN_DWELL_TIME": ("higher_pass_fixation_duration_ms", "numeric"),
        "IA_LAST_RUN_DWELL_TIME": ("last_run_dwell_time_ms", "numeric"),
        "IA_FIXATION_COUNT": ("n_fixations", "numeric"),
        "IA_SKIP": ("skip_flag", "boolean"),
        "IA_REGRESSION_IN_COUNT": ("regression_in_count", "numeric"),
        "IA_REGRESSION_OUT_COUNT": ("regression_out_count", "numeric"),
        "IA_REGRESSION_IN": ("regression_in_flag", "boolean"),
        "IA_REGRESSION_OUT": ("regression_out_flag", "boolean"),
        "IA_REGRESSION_PATH_DURATION": ("regression_path_duration_ms", "numeric"),
        "TRIAL_DWELL_TIME": ("trial_dwell_time_ms", "numeric"),
        "TRIAL_FIXATION_COUNT": ("trial_fixation_count", "numeric"),
        "TRIAL_IA_COUNT": ("trial_ia_count", "numeric"),
        "word_length": ("word_length", "numeric"),
        "word_length_no_punctuation": ("word_length_no_punctuation", "numeric"),
        "gpt2_surprisal": ("gpt2_surprisal", "numeric"),
        "wordfreq_frequency": ("wordfreq_frequency", "numeric"),
        "subtlex_frequency": ("subtlex_frequency", "numeric"),
        "universal_pos": ("universal_pos", "string"),
        "ptb_pos": ("ptb_pos", "string"),
        "Reduced_POS": ("reduced_pos", "string"),
        "dependency_relation": ("dependency_relation", "string"),
        "morphological_features": ("morphological_features", "string"),
        "entity_type": ("entity_type", "string"),
        "head_word_index": ("head_word_index", "numeric"),
        "distance_to_head": ("distance_to_head", "numeric"),
        "left_dependents_count": ("left_dependents_count", "numeric"),
        "right_dependents_count": ("right_dependents_count", "numeric"),
    }
    for source, (dest, kind) in metric_map.items():
        if source not in words.columns:
            continue
        if kind == "numeric":
            df[dest] = pd.to_numeric(words[source], errors="coerce")
        elif kind == "string":
            df[dest] = words[source].astype(str)
        else:
            df[dest] = words[source].fillna(False).astype(bool)

    return df


def normalize_fixations(
    fixations: pd.DataFrame, schema: Dict[str, str]
) -> pd.DataFrame:
    df = pd.DataFrame()
    df["participant_id"] = fixations[schema["participant"]].astype(str)
    trial_col = (
        "unique_trial_id" if "unique_trial_id" in fixations.columns else schema["trial"]
    )
    df["trial_id"] = fixations[trial_col].astype(str)
    if "unique_trial_id" in fixations.columns:
        df["unique_trial_id"] = fixations["unique_trial_id"].astype(str)
    paragraph_col = (
        "unique_paragraph_id"
        if "unique_paragraph_id" in fixations.columns
        else schema.get("paragraph")
    )
    if paragraph_col:
        df["paragraph_id"] = fixations[paragraph_col].astype(str)
    else:
        df["paragraph_id"] = df["trial_id"]
    if "unique_paragraph_id" in fixations.columns:
        df["unique_paragraph_id"] = fixations["unique_paragraph_id"].astype(str)
    df["x"] = pd.to_numeric(fixations[schema["x"]], errors="coerce")
    df["y"] = pd.to_numeric(fixations[schema["y"]], errors="coerce")
    df["duration_ms"] = pd.to_numeric(
        fixations[schema["duration"]], errors="coerce"
    ).fillna(0)

    if schema.get("timestamp"):
        df["timestamp_ms"] = pd.to_numeric(
            fixations[schema["timestamp"]], errors="coerce"
        ).fillna(0)
    else:
        df["timestamp_ms"] = df.groupby(["participant_id", "trial_id"]).cumcount()

    if schema.get("fixation_id"):
        df["fixation_id"] = fixations[schema["fixation_id"]]
    else:
        df["fixation_id"] = df.groupby(["participant_id", "trial_id"]).cumcount().add(1)

    if schema.get("word_id"):
        df["word_id"] = pd.to_numeric(fixations[schema["word_id"]], errors="coerce")
    else:
        df["word_id"] = np.nan
    if schema.get("pass_index"):
        df["pass_index"] = pd.to_numeric(
            fixations[schema["pass_index"]], errors="coerce"
        )
    else:
        df["pass_index"] = 1
    if schema.get("saccade_type"):
        df["saccade_type"] = fixations[schema["saccade_type"]].astype(str)
    else:
        df["saccade_type"] = "unknown"
    if schema.get("saccade_amplitude"):
        df["saccade_amplitude"] = pd.to_numeric(
            fixations[schema["saccade_amplitude"]], errors="coerce"
        )
    if schema.get("eye"):
        df["eye"] = fixations[schema["eye"]].astype(str)
    else:
        df["eye"] = "Both"
    if schema.get("noise_flag"):
        raw_flag = fixations[schema["noise_flag"]]
        if pd.api.types.is_bool_dtype(raw_flag):
            df["noise_flag"] = raw_flag.fillna(False)
        elif pd.api.types.is_numeric_dtype(raw_flag):
            df["noise_flag"] = raw_flag.fillna(0).astype(bool)
        else:
            ok_tokens = {"ok", "good", "valid", "true", "1"}
            df["noise_flag"] = ~raw_flag.astype(str).str.strip().str.lower().isin(
                ok_tokens
            )
    else:
        df["noise_flag"] = False

    meta_cols = [
        "TRIAL_INDEX",
        "trial_index",
        "article_batch",
        "article_id",
        "difficulty_level",
        "article_title",
        "question",
        "selected_answer",
        "is_correct",
    ]
    for col in meta_cols:
        if col in fixations.columns:
            df[col] = fixations[col]
    if "repeated_reading_trial" in fixations.columns:
        df["repeated_reading_trial"] = (
            fixations["repeated_reading_trial"].fillna(False).astype(bool)
        )

    df["order_in_trial"] = (
        df.sort_values(["timestamp_ms", "duration_ms"])
        .groupby(["participant_id", "trial_id"])
        .cumcount()
        + 1
    )
    return df


def filter_data(
    words: pd.DataFrame,
    fixations: pd.DataFrame,
    filters: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    participants = filters.get("participants") or list(words["participant_id"].unique())
    trials = filters.get("trials") or list(words["trial_id"].unique())
    word_mask = words["participant_id"].isin(participants) & words["trial_id"].isin(
        trials
    )
    words_filtered = words[word_mask]

    fix_mask = fixations["participant_id"].isin(participants) & fixations[
        "trial_id"
    ].isin(trials)
    if "pass_index" in fixations.columns:
        pass_indices = filters.get("pass_indices")
        if pass_indices:
            fix_mask &= fixations["pass_index"].isin(pass_indices)
    if "saccade_type" in fixations.columns:
        saccade_types = filters.get("saccade_types")
        if saccade_types:
            fix_mask &= fixations["saccade_type"].isin(saccade_types)
    if "eye" in fixations.columns:
        eyes = filters.get("eyes")
        if eyes:
            fix_mask &= fixations["eye"].isin(eyes)
    include_noise = filters.get("include_noise", True)
    if not include_noise and "noise_flag" in fixations.columns:
        fix_mask &= ~fixations["noise_flag"].fillna(False)
    fixations_filtered = fixations[fix_mask]
    return words_filtered, fixations_filtered


def filter_raw_gaze(
    raw_gaze: pd.DataFrame,
    participants: list,
    trials: list,
) -> pd.DataFrame:
    """Filter raw gaze data by participants and trials."""
    if raw_gaze.empty:
        return raw_gaze
    mask = raw_gaze["participant_id"].isin(participants) & raw_gaze["trial_id"].isin(
        trials
    )
    return raw_gaze[mask]


def compute_canvas_size(
    words: pd.DataFrame, fixations: pd.DataFrame
) -> Tuple[int, int]:
    """Estimate canvas size from word boxes and fixation extents.

    Returns the smallest power-of-100 dimensions that comfortably enclose the
    rightmost/bottommost data point. Falls back to DEFAULT_FIGURE_SIZE when
    nothing is available.
    """
    default_w, default_h = DEFAULT_FIGURE_SIZE
    x_candidates: list[float] = []
    y_candidates: list[float] = []
    if words is not None and not words.empty and "x" in words.columns:
        x_candidates.append(float((words["x"] + words.get("width", 0)).max()))
        y_candidates.append(float((words["y"] + words.get("height", 0)).max()))
    if fixations is not None and not fixations.empty and "x" in fixations.columns:
        x_candidates.append(float(fixations["x"].max()))
        y_candidates.append(float(fixations["y"].max()))
    if not x_candidates or not y_candidates:
        return max(int(default_w), 100), max(int(default_h), 100)
    width = int(np.ceil(max(x_candidates) / 100.0) * 100)
    height = int(np.ceil(max(y_candidates) / 100.0) * 100)
    return max(width, 100), max(height, 100)


@st.cache_data(show_spinner=False)
def compute_word_metrics(words: pd.DataFrame, fixations: pd.DataFrame) -> pd.DataFrame:
    """Return per-word reading measures.

    If the words table already carries pre-aggregated measures (EyeLink IA
    export), those values are preserved. Anything missing is computed from
    fixations + bounding boxes via `measures.compute_per_word_measures`.

    Cached: this function is invoked across the app on each rerun. Streamlit's
    hash is by DataFrame identity/content, so identical inputs reuse the result.
    """
    from .measures import compute_per_word_measures

    if words.empty:
        return words.copy()

    enriched = compute_per_word_measures(fixations, words)

    metric_fields = [
        "first_fixation_ms",
        "first_pass_gaze_duration_ms",
        "regression_path_duration_ms",
        "total_fixation_duration_ms",
        "higher_pass_fixation_duration_ms",
        "last_run_dwell_time_ms",
        "n_fixations",
        "skip_flag",
        "regression_in_count",
        "regression_out_count",
        "regression_in_flag",
        "regression_out_flag",
        "trial_dwell_time_ms",
        "trial_fixation_count",
        "trial_ia_count",
        "word_length",
        "word_length_no_punctuation",
        "gaze_duration_ms",
        "first_fix_x",
        "first_fix_y",
        "gpt2_surprisal",
        "wordfreq_frequency",
        "subtlex_frequency",
        "universal_pos",
        "ptb_pos",
        "reduced_pos",
        "dependency_relation",
        "morphological_features",
        "entity_type",
        "head_word_index",
        "distance_to_head",
        "left_dependents_count",
        "right_dependents_count",
    ]
    base_fields = [
        "participant_id",
        "trial_id",
        "paragraph_id",
        "word_id",
        "text",
        "line_idx",
    ]
    present_fields = [
        col for col in base_fields + metric_fields if col in enriched.columns
    ]
    metrics = enriched[present_fields].copy()

    numeric_fields = [
        "first_fixation_ms",
        "first_pass_gaze_duration_ms",
        "regression_path_duration_ms",
        "total_fixation_duration_ms",
        "higher_pass_fixation_duration_ms",
        "last_run_dwell_time_ms",
        "trial_dwell_time_ms",
        "trial_fixation_count",
        "trial_ia_count",
        "regression_in_count",
        "regression_out_count",
        "word_length",
        "word_length_no_punctuation",
        "gaze_duration_ms",
        "first_fix_x",
        "first_fix_y",
        "gpt2_surprisal",
        "wordfreq_frequency",
        "subtlex_frequency",
        "head_word_index",
        "distance_to_head",
        "left_dependents_count",
        "right_dependents_count",
    ]
    for col in numeric_fields:
        if col in metrics.columns:
            metrics[col] = pd.to_numeric(metrics[col], errors="coerce")
    if "n_fixations" in metrics.columns:
        metrics["n_fixations"] = (
            pd.to_numeric(metrics["n_fixations"], errors="coerce")
            .fillna(0)
            .astype("Int64")
        )
    for col in ["skip_flag", "regression_in_flag", "regression_out_flag"]:
        if col in metrics.columns:
            metrics[col] = metrics[col].fillna(False).astype(bool)
    return metrics


def default_filters(words: pd.DataFrame, fixations: pd.DataFrame) -> Dict:
    filters = dict(
        participants=sorted(words["participant_id"].unique()),
        trials=sorted(words["trial_id"].unique()),
    )
    if "pass_index" in fixations.columns:
        filters["pass_indices"] = sorted(fixations["pass_index"].dropna().unique())
    if "saccade_type" in fixations.columns:
        filters["saccade_types"] = sorted(
            fixations["saccade_type"].dropna().astype(str).unique()
        )
    if "eye" in fixations.columns:
        filters["eyes"] = sorted(fixations["eye"].dropna().astype(str).unique())
    filters["include_noise"] = False if "noise_flag" in fixations.columns else True
    return filters
