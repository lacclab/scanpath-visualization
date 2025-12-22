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


@st.cache_data
def load_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the bundled demo data (csv only) so users can try the app instantly."""
    data_root = resources.files(PACKAGE_NAME).joinpath("sample_data")
    words_resource = data_root / "ia.csv"
    fixations_resource = data_root / "fixations.csv"

    try:
        with resources.as_file(words_resource) as words_path, resources.as_file(fixations_resource) as fixations_path:
            words = pd.read_csv(words_path)
            fixations = pd.read_csv(fixations_path)
    except FileNotFoundError:
        st.error(
            "Bundled sample data not found. Expected ia.csv and fixations.csv "
            "under the installed package's sample_data directory."
        )
        return pd.DataFrame(), pd.DataFrame()

    return words, fixations


def infer_word_schema(words: pd.DataFrame) -> Optional[Dict[str, str]]:
    participant = pick_column(words, ["participant_id", "subject_id", "participant", "recording_session_label"])
    trial = pick_column(
        words,
        ["unique_trial_id", "trial_id", "unique_paragraph_id", "paragraph_id", "trial", "trial_index", "TRIAL_INDEX"],
    )
    paragraph = pick_column(words, ["unique_paragraph_id", "paragraph_id", "PARAGRAPH_ID"])
    word_id = pick_column(words, ["word_id", "IA_ID", "ia_id", "ia_index"])
    text = pick_column(
        words,
        [
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
        ],
    )
    line = pick_column(words, ["line_idx", "line", "line_index", "IA_LINE_ID"])

    x = pick_column(words, ["x", "X", "left"])
    y = pick_column(words, ["y", "Y", "top"])
    width = pick_column(words, ["width", "WIDTH"])
    height = pick_column(words, ["height", "HEIGHT"])
    left = pick_column(words, ["IA_LEFT", "ia_left", "left"])
    right = pick_column(words, ["IA_RIGHT", "ia_right", "right"])
    top = pick_column(words, ["IA_TOP", "ia_top", "top"])
    bottom = pick_column(words, ["IA_BOTTOM", "ia_bottom", "bottom"])

    missing_core = [name for name, val in [("participant", participant), ("trial", trial), ("word_id", word_id)] if val is None]
    if missing_core:
        st.error(f"Missing required word fields in uploaded data: {', '.join(missing_core)}")
        return None

    has_xywh = all([x, y, width, height])
    has_box = all([left, right, top, bottom])
    if not has_xywh and not has_box:
        st.error("Words/IA data needs either (x, y, width, height) or (IA_LEFT, IA_RIGHT, IA_TOP, IA_BOTTOM).")
        return None

    return dict(
        participant=participant,
        trial=trial,
        paragraph=paragraph,
        word_id=word_id,
        text=text,
        line=line,
        x=x,
        y=y,
        width=width,
        height=height,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
    )


def infer_fix_schema(fixations: pd.DataFrame) -> Optional[Dict[str, str]]:
    participant = pick_column(fixations, ["participant_id", "subject_id", "participant", "recording_session_label"])
    trial = pick_column(
        fixations,
        ["unique_trial_id", "trial_id", "unique_paragraph_id", "paragraph_id", "trial", "trial_index", "TRIAL_INDEX"],
    )
    paragraph = pick_column(fixations, ["unique_paragraph_id", "paragraph_id", "PARAGRAPH_ID"])
    fixation_id = pick_column(fixations, ["fixation_id", "CURRENT_FIX_INDEX", "CURRENT_FIX_NUM"])
    timestamp = pick_column(
        fixations,
        ["timestamp_ms", "CURRENT_FIX_START", "CURRENT_FIX_START_TIME", "CURRENT_FIX_TIME", "CURRENT_FIX_ONSET"],
    )
    duration = pick_column(fixations, ["duration_ms", "CURRENT_FIX_DURATION", "CURRENT_FIX_LEN"])
    x = pick_column(fixations, ["x", "X", "CURRENT_FIX_X", "FPOGX"])
    y = pick_column(fixations, ["y", "Y", "CURRENT_FIX_Y", "FPOGY"])
    word_id = pick_column(
        fixations,
        ["word_id", "IA_ID", "ia_id", "CURRENT_FIX_INTEREST_AREA_ID", "CURRENT_FIX_INTEREST_AREA_INDEX"],
    )
    pass_index = pick_column(fixations, ["pass_index", "reread", "PASS_INDEX"])
    saccade_type = pick_column(fixations, ["saccade_type", "SACCADE_TYPE", "NEXT_SAC_DIRECTION"])
    eye = pick_column(fixations, ["eye", "EYE_USED", "eye_used", "EYE_TRACKED"])
    noise_flag = pick_column(fixations, ["noise_flag", "CURRENT_FIX_VALIDITY", "CURRENT_FIX_VALID"])

    missing_core = [name for name, val in [("participant", participant), ("trial", trial), ("duration", duration), ("x", x), ("y", y)] if val is None]
    if missing_core:
        st.error(f"Missing required fixation fields in uploaded data: {', '.join(missing_core)}")
        return None

    return dict(
        participant=participant,
        trial=trial,
        paragraph=paragraph,
        fixation_id=fixation_id,
        timestamp=timestamp,
        duration=duration,
        x=x,
        y=y,
        word_id=word_id,
        pass_index=pass_index,
        saccade_type=saccade_type,
        eye=eye,
        noise_flag=noise_flag,
    )


def normalize_words(words: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame()
    df["participant_id"] = words[schema["participant"]].astype(str)
    trial_col = "unique_trial_id" if "unique_trial_id" in words.columns else schema["trial"]
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
        "TRIAL_DWELL_TIME": ("trial_dwell_time_ms", "numeric"),
        "TRIAL_FIXATION_COUNT": ("trial_fixation_count", "numeric"),
        "TRIAL_IA_COUNT": ("trial_ia_count", "numeric"),
        "word_length": ("word_length", "numeric"),
        "word_length_no_punctuation": ("word_length_no_punctuation", "numeric"),
    }
    for source, (dest, kind) in metric_map.items():
        if source not in words.columns:
            continue
        if kind == "numeric":
            df[dest] = pd.to_numeric(words[source], errors="coerce")
        else:
            df[dest] = words[source].fillna(False).astype(bool)

    return df


def normalize_fixations(fixations: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame()
    df["participant_id"] = fixations[schema["participant"]].astype(str)
    trial_col = "unique_trial_id" if "unique_trial_id" in fixations.columns else schema["trial"]
    df["trial_id"] = fixations[trial_col].astype(str)
    if "unique_trial_id" in fixations.columns:
        df["unique_trial_id"] = fixations["unique_trial_id"].astype(str)
    paragraph_col = "unique_paragraph_id" if "unique_paragraph_id" in fixations.columns else schema.get("paragraph")
    if paragraph_col:
        df["paragraph_id"] = fixations[paragraph_col].astype(str)
    else:
        df["paragraph_id"] = df["trial_id"]
    if "unique_paragraph_id" in fixations.columns:
        df["unique_paragraph_id"] = fixations["unique_paragraph_id"].astype(str)
    df["x"] = pd.to_numeric(fixations[schema["x"]], errors="coerce")
    df["y"] = pd.to_numeric(fixations[schema["y"]], errors="coerce")
    df["duration_ms"] = pd.to_numeric(fixations[schema["duration"]], errors="coerce").fillna(0)

    if schema.get("timestamp"):
        df["timestamp_ms"] = pd.to_numeric(fixations[schema["timestamp"]], errors="coerce").fillna(0)
    else:
        df["timestamp_ms"] = df.groupby(["participant_id", "trial_id"]).cumcount()

    if schema.get("fixation_id"):
        df["fixation_id"] = fixations[schema["fixation_id"]]
    else:
        df["fixation_id"] = (
            df.groupby(["participant_id", "trial_id"])
            .cumcount()
            .add(1)
        )

    if schema.get("word_id"):
        df["word_id"] = pd.to_numeric(fixations[schema["word_id"]], errors="coerce")
    else:
        df["word_id"] = np.nan
    if schema.get("pass_index"):
        df["pass_index"] = pd.to_numeric(fixations[schema["pass_index"]], errors="coerce")
    else:
        df["pass_index"] = 1
    if schema.get("saccade_type"):
        df["saccade_type"] = fixations[schema["saccade_type"]].astype(str)
    else:
        df["saccade_type"] = "unknown"
    if schema.get("eye"):
        df["eye"] = fixations[schema["eye"]].astype(str)
    else:
        df["eye"] = "Both"
    if schema.get("noise_flag"):
        df["noise_flag"] = fixations[schema["noise_flag"]]
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
        df["repeated_reading_trial"] = fixations["repeated_reading_trial"].fillna(False).astype(bool)

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
    word_mask = words["participant_id"].isin(participants) & words["trial_id"].isin(trials)
    words_filtered = words[word_mask]

    fix_mask = fixations["participant_id"].isin(participants) & fixations["trial_id"].isin(trials)
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


def compute_canvas_size(words: pd.DataFrame, fixations: pd.DataFrame) -> Tuple[int, int]:
    """Return the default canvas size; users can override in the UI."""
    default_w, default_h = DEFAULT_FIGURE_SIZE
    return max(int(default_w), 100), max(int(default_h), 100)


def compute_word_metrics(words: pd.DataFrame, fixations: pd.DataFrame) -> pd.DataFrame:
    _ = fixations  # metrics come directly from IA/words data; keep signature stable
    metric_fields = [
        "first_fixation_ms",
        "total_fixation_duration_ms",
        "first_pass_gaze_duration_ms",
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
    ]
    base_fields = ["participant_id", "trial_id", "paragraph_id", "word_id", "text", "line_idx"]
    present_fields = [col for col in base_fields + metric_fields if col in words.columns]
    metrics = words[present_fields].copy()

    numeric_fields = [
        "first_fixation_ms",
        "total_fixation_duration_ms",
        "first_pass_gaze_duration_ms",
        "higher_pass_fixation_duration_ms",
        "last_run_dwell_time_ms",
        "trial_dwell_time_ms",
        "trial_fixation_count",
        "trial_ia_count",
        "regression_in_count",
        "regression_out_count",
        "word_length",
        "word_length_no_punctuation",
    ]
    for col in numeric_fields:
        if col in metrics.columns:
            metrics[col] = pd.to_numeric(metrics[col], errors="coerce")
    if "n_fixations" in metrics.columns:
        metrics["n_fixations"] = pd.to_numeric(metrics["n_fixations"], errors="coerce").fillna(0).astype("Int64")
    if "skip_flag" in metrics.columns:
        metrics["skip_flag"] = metrics["skip_flag"].fillna(False).astype(bool)
    if "regression_in_flag" in metrics.columns:
        metrics["regression_in_flag"] = metrics["regression_in_flag"].fillna(False).astype(bool)
    if "regression_out_flag" in metrics.columns:
        metrics["regression_out_flag"] = metrics["regression_out_flag"].fillna(False).astype(bool)

    if "first_pass_gaze_duration_ms" in metrics.columns and "gaze_duration_ms" not in metrics.columns:
        metrics["gaze_duration_ms"] = metrics["first_pass_gaze_duration_ms"]
    return metrics


def default_filters(words: pd.DataFrame, fixations: pd.DataFrame) -> Dict:
    filters = dict(
        participants=sorted(words["participant_id"].unique()),
        trials=sorted(words["trial_id"].unique()),
    )
    if "pass_index" in fixations.columns:
        filters["pass_indices"] = sorted(fixations["pass_index"].dropna().unique())
    if "saccade_type" in fixations.columns:
        filters["saccade_types"] = sorted(fixations["saccade_type"].dropna().astype(str).unique())
    if "eye" in fixations.columns:
        filters["eyes"] = sorted(fixations["eye"].dropna().astype(str).unique())
    filters["include_noise"] = False if "noise_flag" in fixations.columns else True
    return filters
