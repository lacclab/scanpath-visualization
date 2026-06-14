from __future__ import annotations

import glob
import importlib.resources as resources
import io
import os
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st

from .constants import DEFAULT_FIGURE_SIZE, PACKAGE_NAME


def frame_fingerprint(df: Optional[pd.DataFrame]) -> tuple:
    """Cheap, content-sensitive identity for a DataFrame.

    Used as an *explicit* ``@st.cache_data`` key for functions that take an
    underscore-prefixed (un-hashed) frame argument — so Streamlit never re-hashes
    a multi-million-row frame on every rerun just to look up the cache. We sample
    shape + column names + a hash of the first/last rows, which uniquely
    identifies real datasets while staying O(1) in the row count.

    Two genuinely different frames could in principle collide, but the head/tail
    hash plus the exact row count makes that astronomically unlikely for real
    eye-tracking tables.

    ``hash_pandas_object`` raises on columns of unhashable objects (lists/arrays —
    e.g. parquet-preserved span-index fields). We stringify and retry rather than
    drop the content signal entirely: zeroing the hash would collapse every frame
    of the same shape + columns to one fingerprint, serving stale cached results
    when switching between two such frames. Only a last-resort failure falls back
    to ``(n, columns, 0, 0)``.
    """
    if df is None or getattr(df, "empty", True):
        return (0, ())
    cols = tuple(map(str, df.columns))
    n = int(len(df))

    def _hash(sample: pd.DataFrame) -> int:
        try:
            return int(pd.util.hash_pandas_object(sample, index=True).sum())
        except TypeError:
            # Unhashable cell objects — stringify so content still drives the key.
            return int(pd.util.hash_pandas_object(sample.astype(str), index=True).sum())

    try:
        head = _hash(df.head(64))
        tail = _hash(df.tail(64))
    except Exception:
        head = tail = 0
    return (n, cols, head, tail)


# ---------------------------------------------------------------------------
# Server-side OneStop data source.
#
# When the env var `ONESTOP_DATA_DIR` points at a OneStop lacclab export
# folder (containing `ia_Paragraph.csv.zip` and `fixations_Paragraph.csv.zip`),
# `load_onestop_server_bundle()` returns them as the (words, fixations) tuple
# the rest of the pipeline expects. The schema is identical to the bundled
# sample (the sample is a 3-pid subset of OneStop), so no extra normalisation
# is required.
#
# Drives the "OneStop server bundle" data source option in app.py, used by
# an external review-app deep-link integration (single pid+trial into this UI).
# ---------------------------------------------------------------------------

ONESTOP_DATA_DIR_ENV = "ONESTOP_DATA_DIR"


def onestop_data_dir() -> Optional[Path]:
    """Resolved value of `$ONESTOP_DATA_DIR`, or `None` if unset/blank."""
    raw = os.environ.get(ONESTOP_DATA_DIR_ENV, "").strip()
    return Path(raw) if raw else None


def onestop_full_bundle_exists() -> bool:
    """True when the full OneStop CSV.zip exports are present (not just per-pid
    shards).

    When the whole corpus is available the app loads it once and filters in-app
    (so switching participant is instant); a shards-only setup must instead load
    one participant's shard at a time (it can't materialize the ~60 GB corpus)."""
    base = onestop_data_dir()
    if base is None:
        return False
    return (base / "ia_Paragraph.csv.zip").exists() and (
        base / "fixations_Paragraph.csv.zip"
    ).exists()


def _onestop_shard_paths(base: Path, pid: str) -> Tuple[Path, Path]:
    """Resolved per-participant shard paths under `<base>/by_pid/`."""
    pid = pid.strip().lower()
    return (
        base / "by_pid" / "ia" / f"{pid}.parquet",
        base / "by_pid" / "fixations" / f"{pid}.parquet",
    )


def onestop_data_provenance(participant: Optional[str] = None) -> dict:
    """Where the currently-loaded OneStop data came from, for the Raw Data tab.

    Parses `ONESTOP_DATA_DIR` (typically `…/onestop_<cohort>/reports/<source>/<date>/full/`)
    to surface cohort, export source (lacclab / public / osf), and date in the
    UI so reviewers can verify they're looking at the right export. Also
    reports the per-pid shard's mtime when a participant is set — that's the
    timestamp of the actual data the page is currently rendering.

    Returns an empty dict when `ONESTOP_DATA_DIR` is unset (i.e. the OneStop
    data source isn't in use — caller should suppress the provenance panel).
    """
    base = onestop_data_dir()
    if base is None:
        return {}

    info: dict = {"data_dir": str(base)}

    # Best-effort parse of the canonical path layout.
    parts = base.resolve().parts
    try:
        # Look for "reports" anchor and grab source/date after it.
        i = parts.index("reports")
        info["source"] = parts[i + 1]  # lacclab / public / osf
        info["date"] = parts[i + 2]  # YYYYMMDD
    except (ValueError, IndexError):
        pass
    for p in parts:
        if p.startswith("onestop_"):
            info["cohort"] = p.removeprefix("onestop_")  # L1 / L2
            break

    # Reports the per-pid shard's mtime when a participant is set — that's the
    # timestamp of the bytes the page is rendering right now.
    if participant:
        ia_shard, fix_shard = _onestop_shard_paths(base, participant)
        info["loaded_from"] = "per-pid shard"
        info["ia_shard"] = str(ia_shard)
        info["fix_shard"] = str(fix_shard)
        if ia_shard.is_file():
            info["ia_shard_mtime"] = ia_shard.stat().st_mtime
        if fix_shard.is_file():
            info["fix_shard_mtime"] = fix_shard.stat().st_mtime
    else:
        ia_csv = base / "ia_Paragraph.csv.zip"
        fix_csv = base / "fixations_Paragraph.csv.zip"
        info["loaded_from"] = "full CSV.zip export"
        if ia_csv.is_file():
            info["ia_shard"] = str(ia_csv)
            info["ia_shard_mtime"] = ia_csv.stat().st_mtime
        if fix_csv.is_file():
            info["fix_shard"] = str(fix_csv)
            info["fix_shard_mtime"] = fix_csv.stat().st_mtime
    return info


@st.cache_data(show_spinner="Loading OneStop lacclab export…")
def load_onestop_server_bundle(
    participant: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load OneStop lacclab IA + fixation reports from `$ONESTOP_DATA_DIR`.

    Fast path — when `participant` is given and per-pid shards exist under
    `<ONESTOP_DATA_DIR>/by_pid/{ia,fixations}/<pid>.parquet`, load just that
    one participant (sub-second). Shards are generated by
    `python -m scanpath_studio.onestop_shard --data-dir <ONESTOP_DATA_DIR>`.

    Slow path — fall back to loading the full CSV.zip exports (~3 min, ~60 GB
    RAM for the L2 cohort). Used when no participant is specified, or when
    a deep link points at a pid whose shard hasn't been generated yet.
    """
    base = onestop_data_dir()
    if base is None:
        return pd.DataFrame(), pd.DataFrame()

    # Fast path: per-pid shards.
    if participant:
        ia_shard, fix_shard = _onestop_shard_paths(base, participant)
        ia_present = ia_shard.exists()
        fix_present = fix_shard.exists()
        if ia_present and fix_present:
            return pd.read_parquet(ia_shard), pd.read_parquet(fix_shard)
        # NEVER fall through to the 15 GB load when a participant is named —
        # the deep link is for one pid only, so loading the whole cohort just
        # to discover the pid still has no data is pure waste. Surface a clear
        # error and stop. Common cause: pid was excluded from the IA report
        # (no exported reading data), or shards haven't been generated yet.
        missing = [
            p.name
            for p, ok in [(ia_shard, ia_present), (fix_shard, fix_present)]
            if not ok
        ]
        st.error(
            f"No scanpath data for participant {participant!r}. "
            f"Missing shards: {', '.join(missing)}. "
            f"If the pid was added since the last shard run, regenerate with: "
            f"`python -m scanpath_studio.onestop_shard --data-dir <ONESTOP_DATA_DIR>`. "
            f"Pids with no IA report (e.g. metadata-status excluded) cannot be visualized."
        )
        st.stop()

    # Slow path: full CSV.zip load.
    ia_path = base / "ia_Paragraph.csv.zip"
    fix_path = base / "fixations_Paragraph.csv.zip"
    if not ia_path.exists() or not fix_path.exists():
        st.error(
            f"OneStop data not found under {base}. Expected ia_Paragraph.csv.zip + "
            f"fixations_Paragraph.csv.zip."
        )
        return pd.DataFrame(), pd.DataFrame()
    words = pd.read_csv(ia_path, low_memory=False)
    fixations = pd.read_csv(fix_path, low_memory=False)
    return words, fixations


def _norm_col(name) -> str:
    """Fold a column name to its case- and separator-insensitive key.

    Lowercases and drops every non-alphanumeric char, so ``IA_LEFT``,
    ``ia_left``, ``Ia-Left`` and ``ia left`` all collapse to ``ialeft`` —
    letting auto-detection match real-world column names that differ only in
    capitalization or word separators."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return the first matching column name from a candidate list.

    Matching is case- and separator-insensitive (see ``_norm_col``). Candidate
    order is still priority order — the first candidate with any match wins (so
    EyeLink names keep beating Gazepoint), and among equally-normalized columns
    the leftmost one wins."""
    lookup: Dict[str, str] = {}
    for col in df.columns:
        lookup.setdefault(_norm_col(col), col)
    for name in candidates:
        hit = lookup.get(_norm_col(name))
        if hit is not None:
            return hit
    return None


def trial_mapping_columns(trial_mapping) -> list:
    """Column list behind a trial mapping — a plain column name or a list of
    names (the column-mapping UI returns a list when the user composes a
    unique trial ID from several columns)."""
    if isinstance(trial_mapping, str):
        return [trial_mapping]
    return list(trial_mapping)


def trial_id_series(source: pd.DataFrame, trial_mapping) -> pd.Series:
    """Trial-id values for a single-column or composite (multi-column) mapping.

    A multi-column mapping builds a unique trial ID on the fly by joining the
    columns' string values with ``_`` — for datasets that ship no precomputed
    unique-trial column (e.g. OneStop-style participant + paragraph +
    repeated-reading)."""
    cols = trial_mapping_columns(trial_mapping)
    if len(cols) == 1:
        return source[cols[0]].astype(str)
    return source[cols].astype(str).agg("_".join, axis=1)


def _preserve_composite_columns(
    df: pd.DataFrame, source: pd.DataFrame, trial_mapping
) -> pd.DataFrame:
    """Carry a composite mapping's source columns into the normalized frame
    under their original names.

    A multi-column trial mapping gets joined into a single opaque ``trial_id``
    (e.g. ``2_1_1_Ele_l37_1129_False``). Keeping the individual component
    columns lets the trial picker offer one cascading selector per part —
    mirroring the Text / Participant modes (see
    ``utils._select_trial_composite_mode``). No-op for single-column mappings.
    Rows are 1:1 with ``source`` here (no filtering in the composite path), so a
    positional copy stays aligned."""
    cols = trial_mapping_columns(trial_mapping)
    if len(cols) < 2:
        return df
    for col in cols:
        if col not in df.columns and col in source.columns:
            df[col] = source[col].to_numpy()
    return df


# Candidate column names checked during auto-inference. Centralised so the
# proposal step and the override UI share the same defaults. Matching is case-
# and separator-insensitive (see ``pick_column``), so these list only *distinct*
# conventions — no ALL_CAPS / snake_case twins of the same name needed.
PARTICIPANT_CANDIDATES = [
    "participant_id",
    "subject_id",
    "participant",
    "recording_session_label",
    "reader_id",
]
TRIAL_CANDIDATES = [
    "unique_trial_id",
    "trial_id",
    "unique_paragraph_id",
    "paragraph_id",
    "text_id",
    "trial",
    "trial_index",
]
# Source column names that identify which *text* (passage) a row belongs to.
# Output canonical column is `text_id` (was `paragraph_id`); the source names stay
# as the real-world conventions so auto-detection keeps working.
TEXT_ID_CANDIDATES = [
    "unique_paragraph_id",
    "paragraph_id",
    "unique_text_id",
    "text_id",
]
TEXT_CANDIDATES = [
    "text",
    "IA_LABEL",
    "label",
    "word",
    "content",
    "token",
]
WORD_ID_CANDIDATES = ["word_id", "IA_ID", "ia_index", "word_index", "aoi"]
LINE_CANDIDATES = ["line_idx", "line", "line_index", "IA_LINE_ID"]

WORD_X_CANDIDATES = ["x", "left"]
WORD_Y_CANDIDATES = ["y", "top"]
WORD_WIDTH_CANDIDATES = ["width"]
WORD_HEIGHT_CANDIDATES = ["height"]
WORD_LEFT_CANDIDATES = ["IA_LEFT", "left", "start_x"]
WORD_RIGHT_CANDIDATES = ["IA_RIGHT", "right", "end_x"]
WORD_TOP_CANDIDATES = ["IA_TOP", "top", "start_y"]
WORD_BOTTOM_CANDIDATES = ["IA_BOTTOM", "bottom", "end_y"]

FIX_X_CANDIDATES = ["x", "CURRENT_FIX_X", "FPOGX"]
FIX_Y_CANDIDATES = ["y", "CURRENT_FIX_Y", "FPOGY"]
FIX_DURATION_CANDIDATES = [
    "duration_ms",
    "CURRENT_FIX_DURATION",
    "CURRENT_FIX_LEN",
    "duration",
    "fixation_duration",
]
FIX_TIMESTAMP_CANDIDATES = [
    "timestamp_ms",
    "CURRENT_FIX_START",
    "CURRENT_FIX_START_TIME",
    "CURRENT_FIX_TIME",
    "CURRENT_FIX_ONSET",
]
FIX_FIXATION_ID_CANDIDATES = [
    "fixation_id",
    "CURRENT_FIX_INDEX",
    "CURRENT_FIX_NUM",
    "fixation_index",
]
FIX_WORD_ID_CANDIDATES = [
    "word_id",
    "IA_ID",
    "CURRENT_FIX_INTEREST_AREA_ID",
    "CURRENT_FIX_INTEREST_AREA_INDEX",
    "word_index_in_text",
    "word_index",
]
FIX_PASS_INDEX_CANDIDATES = ["pass_index", "reread"]
FIX_SACCADE_TYPE_CANDIDATES = ["saccade_type", "NEXT_SAC_DIRECTION"]
FIX_SACCADE_AMPLITUDE_CANDIDATES = [
    "saccade_amplitude",
    "NEXT_SAC_AMPLITUDE",
    "PREVIOUS_SAC_AMPLITUDE",
]
FIX_EYE_CANDIDATES = ["eye", "EYE_USED", "EYE_TRACKED"]
FIX_NOISE_CANDIDATES = ["noise_flag", "CURRENT_FIX_VALIDITY", "CURRENT_FIX_VALID"]

RAW_GAZE_X_CANDIDATES = ["x", "FPOGX", "gaze_x"]
RAW_GAZE_Y_CANDIDATES = ["y", "FPOGY", "gaze_y"]
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
        text_id=pick_column(words, TEXT_ID_CANDIDATES),
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
        text_id=pick_column(fixations, TEXT_ID_CANDIDATES),
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
    """Return a list of human-readable problems with a words/IA schema.

    Participant ID is optional: word/AoI tables without one are treated as
    stimulus-level (one row per word per *text*, not per reading) and are
    broadcast across the participants found in the fixations — see
    ``broadcast_stimulus_words``."""
    problems = []
    for key, label in [
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
    """Return a list of human-readable problems with a fixations schema.

    X/Y coordinates are optional when a Word/IA ID is mapped: AOI-sequence
    datasets (fixations recorded as "which word", not "which pixel") get
    coordinates from the matching word-box centers — see
    ``fill_fixation_xy_from_words``."""
    problems = []
    # Participant is optional — a dataset without it is treated as a single
    # anonymous reader (see SYNTHETIC_PARTICIPANT).
    for key, label in [
        ("trial", "Trial ID"),
        ("duration", "Duration"),
    ]:
        if not schema.get(key):
            problems.append(f"missing {label}")
    has_xy = schema.get("x") and schema.get("y")
    if not has_xy and not schema.get("word_id"):
        problems.append(
            "need either (X, Y) coordinates or a Word/IA ID "
            "(AOI-only fixations are placed at word-box centers)"
        )
    return problems


def validate_raw_gaze_schema(schema: Dict[str, Optional[str]]) -> list:
    """Return a list of human-readable problems with a raw gaze schema."""
    problems = []
    # Participant optional — single anonymous reader when absent.
    for key, label in [
        ("trial", "Trial ID"),
        ("x", "X"),
        ("y", "Y"),
    ]:
        if not schema.get(key):
            problems.append(f"missing {label}")
    return problems


# Column added when concatenating several files (multi-file upload, glob, or a
# multi-member zip): the source file's stem. Lets datasets that key metadata in
# the *filename* (one file per participant and/or per text, e.g. PoTeC's
# `reader0_b0_scanpath.tsv`) recover it after concatenation — map it as (part
# of) the Trial/Participant ID.
SOURCE_FILE_COLUMN = "source_file"

TablesInput = Union[str, os.PathLike, object, List]


def _read_by_extension(buf, name: str) -> pd.DataFrame:
    """Dispatch a buffer/path to a pandas reader by its (lowercased) name."""
    if name.endswith(".parquet"):
        return pd.read_parquet(buf)
    if name.endswith(".feather"):
        return pd.read_feather(buf)
    if name.endswith((".tsv", ".tab")):
        return pd.read_csv(buf, sep="\t")
    return pd.read_csv(buf)


def _tag_and_concat(
    frames: List[pd.DataFrame], labels: List[str], source_column: Optional[str]
) -> pd.DataFrame:
    """Concatenate frames into one. With more than one frame, tag each with its
    source label in ``source_column`` (unless that frame already carries the
    column, or ``source_column`` is None) so rows stay traceable to their
    origin. Columns are aligned by name; fields absent from a frame become NaN
    for its rows."""
    if len(frames) == 1:
        return frames[0]
    if source_column:
        for df, label in zip(frames, labels):
            if source_column not in df.columns:
                df[source_column] = label
    return pd.concat(frames, ignore_index=True, sort=False)


def _read_zipped_table(file_like_or_path) -> pd.DataFrame:
    """Read table(s) from a ``.zip`` archive (e.g. ``data.csv.zip``).

    Each member is dispatched on its own extension, so a zip may wrap any
    supported format. A multi-member archive is concatenated just like a
    multi-file upload — every member's rows tagged with its stem in
    ``source_file``. pandas infers compression only from string paths, not from
    uploaded file-like objects, so we open the archive ourselves. Raises
    ``ValueError`` if the archive holds no data file (macOS ``__MACOSX``/dotfile
    cruft is ignored)."""
    with zipfile.ZipFile(file_like_or_path) as zf:
        members = [
            m
            for m in zf.namelist()
            if not m.endswith("/") and not Path(m).name.startswith((".", "__"))
        ]
        if not members:
            raise ValueError("the zip archive contains no readable table files")
        frames, labels = [], []
        for member in members:
            with zf.open(member) as inner:
                buf = io.BytesIO(inner.read())
            frames.append(_read_by_extension(buf, member.lower()))
            labels.append(Path(member).stem)
    return _tag_and_concat(frames, labels, SOURCE_FILE_COLUMN)


def read_table(file_like_or_path) -> pd.DataFrame:
    """Read a tabular file by extension: csv, tsv, parquet, feather, or a
    ``.zip`` wrapping one or more of those (e.g. ``data.csv.zip``). A
    multi-member zip is concatenated like a multi-file upload."""
    name = getattr(file_like_or_path, "name", str(file_like_or_path)).lower()
    if name.endswith(".zip"):
        return _read_zipped_table(file_like_or_path)
    return _read_by_extension(file_like_or_path, name)


def expand_table_inputs(inputs: TablesInput) -> list:
    """Flatten a path / glob pattern / file-like / list-of-those into a list.

    Glob patterns are expanded in sorted order so multi-file datasets (one
    file per participant or per stimulus) can be referenced with a single
    pattern like ``scanpaths/*.tsv``. Raises ``FileNotFoundError`` for a
    pattern that matches nothing — silently loading zero files would read as
    success."""
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    expanded: list = []
    for item in inputs:
        if isinstance(item, (str, os.PathLike)) and glob.has_magic(str(item)):
            matches = sorted(glob.glob(str(item), recursive=True))
            if not matches:
                raise FileNotFoundError(f"No files match pattern: {item}")
            expanded.extend(matches)
        else:
            expanded.append(item)
    return expanded


def read_tables(
    inputs: TablesInput, source_column: Optional[str] = SOURCE_FILE_COLUMN
) -> pd.DataFrame:
    """Read one or many tabular files and concatenate them into one frame.

    ``inputs`` may be a single path or file-like object, a glob pattern, or a
    list mixing those (a ``.zip`` member counts as a file too). When more than
    one file is read, each part gets a ``source_file`` column holding the file's
    stem (unless the data already has that column, or ``source_column=None``) so
    rows stay traceable to their origin file. Columns are aligned by name across
    files; fields absent from a file become NaN for its rows."""
    items = expand_table_inputs(inputs)
    frames, labels = [], []
    for item in items:
        frames.append(read_table(item))
        labels.append(Path(getattr(item, "name", str(item))).stem)
    return _tag_and_concat(frames, labels, source_column)


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
    df = pd.DataFrame(index=raw_gaze.index)
    if schema.get("participant"):
        df["participant_id"] = raw_gaze[schema["participant"]].astype(str)
    else:
        df["participant_id"] = SYNTHETIC_PARTICIPANT
    trial_cols = trial_mapping_columns(schema["trial"])
    if len(trial_cols) > 1:
        # User-composed unique trial ID — see normalize_words.
        df["trial_id"] = trial_id_series(raw_gaze, trial_cols)
        df["unique_trial_id"] = df["trial_id"]
    else:
        trial_col = (
            "unique_trial_id"
            if "unique_trial_id" in raw_gaze.columns
            else trial_cols[0]
        )
        df["trial_id"] = raw_gaze[trial_col].astype(str)
        if "unique_trial_id" in raw_gaze.columns:
            df["unique_trial_id"] = raw_gaze["unique_trial_id"].astype(str)
    # Raw gaze has no text/passage concept; mirror trial_id so a raw-gaze-only
    # dataset still works with the trial picker (utils.build_combo_options needs
    # a text_id column).
    df["text_id"] = df["trial_id"]
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
    df = _preserve_composite_columns(df, raw_gaze, schema["trial"])
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


# Placeholder participant id for stimulus-level word/AoI tables (no participant
# column — one row per word per text, shared by every reading). The marker
# column flags the frame so broadcast_stimulus_words() knows to expand it.
STIMULUS_PARTICIPANT = ""
STIMULUS_WORDS_FLAG = "_stimulus_words"

# Synthetic participant id used when a dataset has no participant column at all
# (a single anonymous reader). Distinct from STIMULUS_PARTICIPANT ("") so it
# never collides with the stimulus-word broadcast machinery — participant_id is
# always present downstream (combos/filters/annotations/export/measures groupby),
# and the UI hides the participant selector when there's only this one value.
SYNTHETIC_PARTICIPANT = "(all)"


def broadcast_stimulus_words(
    words: pd.DataFrame, fixations: pd.DataFrame
) -> pd.DataFrame:
    """Expand stimulus-level words across the participants who read each trial.

    Datasets like PoTeC ship word/AoI tables per *text* (no participant
    column) while fixations are per participant × text. After normalization,
    such words carry the ``_stimulus_words`` flag; this replicates each
    trial's word rows once per participant that has fixations for that
    ``trial_id``, so downstream (participant, trial) filtering works
    unchanged. Words for trials nobody read are dropped. No-op for ordinary
    per-participant word tables, or when there are no fixations to broadcast
    against (the stimulus rows then keep their placeholder participant)."""
    if STIMULUS_WORDS_FLAG not in words.columns:
        return words
    words = words.drop(columns=[STIMULUS_WORDS_FLAG])
    if words.empty or fixations.empty:
        # No fixations to broadcast across (e.g. a words-only dataset): there's a
        # single anonymous reader, so give the placeholder a real synthetic id.
        if not words.empty:
            words = words.copy()
            words["participant_id"] = SYNTHETIC_PARTICIPANT
        return words
    pairs = fixations[["participant_id", "trial_id"]].drop_duplicates()
    pairs["participant_id"] = pairs["participant_id"].astype(str)
    pairs["trial_id"] = pairs["trial_id"].astype(str)
    return words.drop(columns=["participant_id"]).merge(
        pairs, on="trial_id", how="inner"
    )


def fill_fixation_xy_from_words(
    fixations: pd.DataFrame, words: pd.DataFrame
) -> pd.DataFrame:
    """Fill missing fixation coordinates from the fixated word's box center.

    AOI-sequence datasets record *which* word/character each fixation landed
    on but not the pixel position. When normalized fixations have NaN x/y and
    a ``word_id``, place them at the center of the matching word box (keyed by
    participant_id + trial_id + word_id). Fixations whose word_id matches no
    box keep NaN coordinates. Rows that already have coordinates are left
    untouched."""
    if fixations.empty or words.empty:
        return fixations
    missing = fixations["x"].isna() | fixations["y"].isna()
    if not missing.any() or "word_id" not in fixations.columns:
        return fixations
    centers = words[["participant_id", "trial_id", "word_id"]].copy()
    centers["_word_cx"] = words["x"] + words["width"] / 2.0
    centers["_word_cy"] = words["y"] + words["height"] / 2.0
    centers = centers.drop_duplicates(["participant_id", "trial_id", "word_id"])
    merged = fixations[["participant_id", "trial_id", "word_id"]].merge(
        centers, on=["participant_id", "trial_id", "word_id"], how="left"
    )
    fixations = fixations.copy()
    fill = missing.to_numpy()
    fixations.loc[fill, "x"] = merged["_word_cx"].to_numpy()[fill]
    fixations.loc[fill, "y"] = merged["_word_cy"].to_numpy()[fill]
    return fixations


def _reconcile_participant_asymmetry(
    words: pd.DataFrame, fixations: pd.DataFrame
) -> pd.DataFrame:
    """Re-key word boxes to the synthetic participant when the fixations have no
    participant but the words do.

    With participant now optional per table, a fixations table can be
    participant-less (every row stamped ``SYNTHETIC_PARTICIPANT``) while the words
    table still carries real participant ids. The trial picker keys off the
    fixations, so it offers ``(all)`` — but the boxes are keyed by the real ids
    and ``extract_trial`` then finds none, rendering fixations with no text. Stamp
    the words with the synthetic id (dropping the now-duplicate per-reader boxes)
    so they line up. No-op unless the fixations are entirely synthetic and the
    words are not — the stimulus-words broadcast already covers the reverse."""
    if words.empty or fixations.empty or "participant_id" not in words.columns:
        return words
    if set(fixations["participant_id"].unique()) != {SYNTHETIC_PARTICIPANT}:
        return words
    word_parts = set(words["participant_id"].unique())
    if not word_parts or word_parts == {SYNTHETIC_PARTICIPANT}:
        return words
    words = words.copy()
    words["participant_id"] = SYNTHETIC_PARTICIPANT
    subset = [
        c for c in ("participant_id", "trial_id", "word_id") if c in words.columns
    ]
    if subset:
        words = words.drop_duplicates(subset=subset)
    return words


def harmonize_frames(
    words: pd.DataFrame, fixations: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-frame fixups applied right after normalization.

    Broadcast stimulus-level words across participants, reconcile a
    participant-less fixations table with participant-bearing words, then fill
    missing fixation coordinates from word-box centers. Call whenever both frames
    are available (the API and the app both route through this)."""
    words = broadcast_stimulus_words(words, fixations)
    words = _reconcile_participant_asymmetry(words, fixations)
    fixations = fill_fixation_xy_from_words(fixations, words)
    return words, fixations


def _disambiguate_repeated_readings(
    df: pd.DataFrame,
    source: pd.DataFrame,
    participant_col: str,
    trial_col: str,
) -> pd.DataFrame:
    """Suffix `trial_id` with `_r2`, `_r3` … when a participant read the same
    paragraph more than once.

    OneStop L2's per-pid parquet shards don't carry a `unique_trial_id` column,
    so the schema-inference fallback uses `unique_paragraph_id` — but that's
    the same string for both readings of a repeated-reading trial. Without
    this fix, the two readings' fixations collapse into one scanpath (and into
    one row of the trial picker), which is what the cached PNG thumbnails
    (which filter on TRIAL_INDEX) correctly avoid. We rank by TRIAL_INDEX so
    the chronologically-first reading keeps its original id; later readings
    get `_r2`, `_r3`, … appended.
    """
    if "unique_trial_id" in source.columns:
        return df
    idx_col = next(
        (c for c in ("TRIAL_INDEX", "trial_index") if c in source.columns), None
    )
    if idx_col is None:
        return df
    rank = (
        source.groupby([participant_col, trial_col])[idx_col]
        .rank(method="dense")
        .astype(int)
        .to_numpy()
    )
    df["trial_id"] = [
        tid if r == 1 else f"{tid}_r{r}"
        for tid, r in zip(df["trial_id"].to_numpy(), rank)
    ]
    return df


# ---------------------------------------------------------------------------
# Optional-field registry. Drives (a) which known optional source columns are
# carried into the normalized frame and (b) the setup wizard's opt-out checklist.
# Each entry: (source, dest, kind, category) where `kind` ∈
# {numeric, string, boolean, passthrough} and `category` ∈
# {measure, linguistic, meta} groups the fields in the UI. Matched by exact
# source name (same as the legacy keep-lists this replaced).
# ---------------------------------------------------------------------------
WORD_OPTIONAL_FIELDS = [
    ("IA_FIRST_FIXATION_DURATION", "first_fixation_ms", "numeric", "measure"),
    ("IA_DWELL_TIME", "total_fixation_duration_ms", "numeric", "measure"),
    ("IA_FIRST_RUN_DWELL_TIME", "first_pass_gaze_duration_ms", "numeric", "measure"),
    (
        "IA_SECOND_RUN_DWELL_TIME",
        "higher_pass_fixation_duration_ms",
        "numeric",
        "measure",
    ),
    ("IA_LAST_RUN_DWELL_TIME", "last_run_dwell_time_ms", "numeric", "measure"),
    ("IA_FIXATION_COUNT", "n_fixations", "numeric", "measure"),
    ("IA_SKIP", "skip_flag", "boolean", "measure"),
    ("IA_REGRESSION_IN_COUNT", "regression_in_count", "numeric", "measure"),
    ("IA_REGRESSION_OUT_COUNT", "regression_out_count", "numeric", "measure"),
    ("IA_REGRESSION_IN", "regression_in_flag", "boolean", "measure"),
    ("IA_REGRESSION_OUT", "regression_out_flag", "boolean", "measure"),
    (
        "IA_REGRESSION_PATH_DURATION",
        "regression_path_duration_ms",
        "numeric",
        "measure",
    ),
    ("TRIAL_DWELL_TIME", "trial_dwell_time_ms", "numeric", "measure"),
    ("TRIAL_FIXATION_COUNT", "trial_fixation_count", "numeric", "measure"),
    ("TRIAL_IA_COUNT", "trial_ia_count", "numeric", "measure"),
    ("word_length", "word_length", "numeric", "measure"),
    ("word_length_no_punctuation", "word_length_no_punctuation", "numeric", "measure"),
    ("gpt2_surprisal", "gpt2_surprisal", "numeric", "linguistic"),
    ("wordfreq_frequency", "wordfreq_frequency", "numeric", "linguistic"),
    ("subtlex_frequency", "subtlex_frequency", "numeric", "linguistic"),
    ("universal_pos", "universal_pos", "string", "linguistic"),
    ("ptb_pos", "ptb_pos", "string", "linguistic"),
    ("Reduced_POS", "reduced_pos", "string", "linguistic"),
    ("dependency_relation", "dependency_relation", "string", "linguistic"),
    ("morphological_features", "morphological_features", "string", "linguistic"),
    ("entity_type", "entity_type", "string", "linguistic"),
    ("head_word_index", "head_word_index", "numeric", "linguistic"),
    ("distance_to_head", "distance_to_head", "numeric", "linguistic"),
    ("left_dependents_count", "left_dependents_count", "numeric", "linguistic"),
    ("right_dependents_count", "right_dependents_count", "numeric", "linguistic"),
    (SOURCE_FILE_COLUMN, SOURCE_FILE_COLUMN, "passthrough", "meta"),
    ("TRIAL_INDEX", "TRIAL_INDEX", "passthrough", "meta"),
    ("trial_index", "trial_index", "passthrough", "meta"),
    ("article_batch", "article_batch", "passthrough", "meta"),
    ("article_id", "article_id", "passthrough", "meta"),
    ("difficulty_level", "difficulty_level", "passthrough", "meta"),
    ("article_title", "article_title", "passthrough", "meta"),
    ("question", "question", "passthrough", "meta"),
    ("question_preview", "question_preview", "boolean", "meta"),
    ("selected_answer", "selected_answer", "passthrough", "meta"),
    ("is_correct", "is_correct", "passthrough", "meta"),
    ("repeated_reading_trial", "repeated_reading_trial", "boolean", "meta"),
    ("critical_span_indices", "critical_span_indices", "passthrough", "meta"),
    ("distractor_span_indices", "distractor_span_indices", "passthrough", "meta"),
    ("aspan_ind_start", "aspan_ind_start", "passthrough", "meta"),
    ("aspan_ind_end", "aspan_ind_end", "passthrough", "meta"),
    ("dspan_ind_start", "dspan_ind_start", "passthrough", "meta"),
    ("dspan_ind_end", "dspan_ind_end", "passthrough", "meta"),
    ("is_in_aspan", "is_in_aspan", "boolean", "meta"),
    ("is_in_dspan", "is_in_dspan", "boolean", "meta"),
]

FIX_OPTIONAL_FIELDS = [
    (SOURCE_FILE_COLUMN, SOURCE_FILE_COLUMN, "passthrough", "meta"),
    ("TRIAL_INDEX", "TRIAL_INDEX", "passthrough", "meta"),
    ("trial_index", "trial_index", "passthrough", "meta"),
    ("article_batch", "article_batch", "passthrough", "meta"),
    ("article_id", "article_id", "passthrough", "meta"),
    ("difficulty_level", "difficulty_level", "passthrough", "meta"),
    ("article_title", "article_title", "passthrough", "meta"),
    ("question", "question", "passthrough", "meta"),
    ("selected_answer", "selected_answer", "passthrough", "meta"),
    ("is_correct", "is_correct", "passthrough", "meta"),
    ("repeated_reading_trial", "repeated_reading_trial", "boolean", "meta"),
    ("question_preview", "question_preview", "boolean", "meta"),
]


def _schema_source_columns(schema: Dict) -> set:
    """Set of raw source column names a normalization schema references."""
    cols: set = set()
    for value in schema.values():
        if not value:
            continue
        if isinstance(value, list):
            cols.update(value)
        else:
            cols.add(value)
    return cols


def _apply_optional_fields(
    df: pd.DataFrame, source: pd.DataFrame, registry: list, keep: Optional[set]
) -> set:
    """Carry registry-listed optional source columns into ``df`` (renamed +
    dtype-coerced). ``keep`` is ``None`` (carry every detected field — the
    backward-compatible default) or a set of *source* column names to limit to.
    Returns the set of source columns actually emitted."""
    emitted: set = set()
    for src, dest, kind, _category in registry:
        if src not in source.columns:
            continue
        if keep is not None and src not in keep:
            continue
        emitted.add(src)
        col = source[src]
        if kind == "numeric":
            df[dest] = pd.to_numeric(col, errors="coerce")
        elif kind == "string":
            df[dest] = col.astype(str)
        elif kind == "boolean":
            df[dest] = col.fillna(False).astype(bool)
        else:
            df[dest] = col
    return emitted


def _carry_extra_columns(
    df: pd.DataFrame, source: pd.DataFrame, keep: Optional[set], skip: set
) -> None:
    """Carry user-chosen extra ``keep`` source columns through verbatim, skipping
    those already emitted (canonical / registry) or in ``skip``."""
    if not keep:
        return
    for col in keep:
        if col in source.columns and col not in skip and col not in df.columns:
            df[col] = source[col].to_numpy()


def categorize_columns(raw: pd.DataFrame, schema: Dict, registry: list) -> Dict:
    """Split a raw frame's columns into {mapped, detected_optional, unclaimed}.

    ``mapped`` = source columns the schema references; ``detected_optional`` =
    registry entries present in the frame (each ``{source, dest, category}``);
    ``unclaimed`` = everything else (offered as filter fields / extra keeps)."""
    mapped = {c for c in _schema_source_columns(schema) if c in raw.columns}
    detected = [
        {"source": src, "dest": dest, "category": category}
        for src, dest, _kind, category in registry
        if src in raw.columns
    ]
    detected_sources = {d["source"] for d in detected}
    unclaimed = [
        c for c in raw.columns if c not in mapped and c not in detected_sources
    ]
    return {"mapped": mapped, "detected_optional": detected, "unclaimed": unclaimed}


def compute_keep_columns(
    schema: Dict,
    *,
    optional_sources: Optional[Iterable[str]] = None,
    filter_fields: Optional[Iterable[str]] = None,
    keep_columns: Optional[Iterable[str]] = None,
) -> set:
    """Source columns to retain before normalization (everything else is dropped
    for speed). Union of: schema-mapped sources, always-kept structural columns,
    chosen optional fields, chosen filter fields, and extra keep columns."""
    keep = set(_schema_source_columns(schema))
    # Structural columns consulted directly by normalize_* (not via schema).
    for col in (
        SOURCE_FILE_COLUMN,
        "unique_trial_id",
        "unique_paragraph_id",
        "TRIAL_INDEX",
        "trial_index",
    ):
        keep.add(col)
    for group in (optional_sources, filter_fields, keep_columns):
        if group:
            keep.update(group)
    return keep


def normalize_words(
    words: pd.DataFrame, schema: Dict[str, str], *, keep_columns: Optional[set] = None
) -> pd.DataFrame:
    # The explicit index makes scalar assignments (e.g. the stimulus-level
    # participant placeholder) fill every row even when assigned first.
    df = pd.DataFrame(index=words.index)
    if schema.get("participant"):
        df["participant_id"] = words[schema["participant"]].astype(str)
    else:
        # Stimulus-level word/AoI table (one row per word per text, shared by
        # all participants) — broadcast_stimulus_words() expands it across the
        # participants found in the fixations.
        df["participant_id"] = STIMULUS_PARTICIPANT
        df[STIMULUS_WORDS_FLAG] = True
    trial_cols = trial_mapping_columns(schema["trial"])
    if len(trial_cols) > 1:
        # User-composed unique trial ID: authoritative, so it wins over a raw
        # `unique_trial_id` column and needs no repeated-reading suffixing.
        df["trial_id"] = trial_id_series(words, trial_cols)
        df["unique_trial_id"] = df["trial_id"]
    else:
        trial_col = (
            "unique_trial_id" if "unique_trial_id" in words.columns else trial_cols[0]
        )
        df["trial_id"] = words[trial_col].astype(str)
        if schema.get("participant"):
            df = _disambiguate_repeated_readings(
                df, words, schema["participant"], trial_col
            )
        if "unique_trial_id" in words.columns:
            df["unique_trial_id"] = words["unique_trial_id"].astype(str)
    if "unique_paragraph_id" in words.columns:
        df["unique_text_id"] = words["unique_paragraph_id"].astype(str)
        df["text_id"] = df["unique_text_id"]
    elif schema.get("text_id"):
        df["text_id"] = words[schema["text_id"]].astype(str)
    else:
        df["text_id"] = df["trial_id"]
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

    emitted = _apply_optional_fields(df, words, WORD_OPTIONAL_FIELDS, keep_columns)
    if keep_columns is not None:
        _carry_extra_columns(
            df, words, keep_columns, _schema_source_columns(schema) | emitted
        )

    df = _preserve_composite_columns(df, words, schema["trial"])
    return df


def normalize_fixations(
    fixations: pd.DataFrame,
    schema: Dict[str, str],
    *,
    keep_columns: Optional[set] = None,
) -> pd.DataFrame:
    # Explicit index so a constant participant placeholder fills every row.
    df = pd.DataFrame(index=fixations.index)
    if schema.get("participant"):
        df["participant_id"] = fixations[schema["participant"]].astype(str)
    else:
        # No participant column → a single anonymous reader.
        df["participant_id"] = SYNTHETIC_PARTICIPANT
    trial_cols = trial_mapping_columns(schema["trial"])
    if len(trial_cols) > 1:
        # User-composed unique trial ID — see normalize_words.
        df["trial_id"] = trial_id_series(fixations, trial_cols)
        df["unique_trial_id"] = df["trial_id"]
    else:
        trial_col = (
            "unique_trial_id"
            if "unique_trial_id" in fixations.columns
            else trial_cols[0]
        )
        df["trial_id"] = fixations[trial_col].astype(str)
        if schema.get("participant"):
            df = _disambiguate_repeated_readings(
                df, fixations, schema["participant"], trial_col
            )
        if "unique_trial_id" in fixations.columns:
            df["unique_trial_id"] = fixations["unique_trial_id"].astype(str)
    text_id_col = (
        "unique_paragraph_id"
        if "unique_paragraph_id" in fixations.columns
        else schema.get("text_id")
    )
    if text_id_col:
        df["text_id"] = fixations[text_id_col].astype(str)
    else:
        df["text_id"] = df["trial_id"]
    if "unique_paragraph_id" in fixations.columns:
        df["unique_text_id"] = fixations["unique_paragraph_id"].astype(str)
    # X/Y may be unmapped for AOI-sequence datasets (no pixel coordinates) —
    # left NaN here and filled from word-box centers by harmonize_frames().
    for coord in ("x", "y"):
        if schema.get(coord):
            df[coord] = pd.to_numeric(fixations[schema[coord]], errors="coerce")
        else:
            df[coord] = np.nan
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

    emitted = _apply_optional_fields(df, fixations, FIX_OPTIONAL_FIELDS, keep_columns)
    if keep_columns is not None:
        _carry_extra_columns(
            df, fixations, keep_columns, _schema_source_columns(schema) | emitted
        )

    df = _preserve_composite_columns(df, fixations, schema["trial"])

    df["order_in_trial"] = (
        df.sort_values(["timestamp_ms", "duration_ms"])
        .groupby(["participant_id", "trial_id"])
        .cumcount()
        + 1
    )
    return df


# Canonical columns produced by normalize_words / normalize_fixations. Used to
# build typed empty frames when a dataset ships only one of the two reports,
# so every downstream consumer can keep selecting columns unconditionally.
WORDS_CANONICAL_COLUMNS: Dict[str, str] = {
    "participant_id": "object",
    "trial_id": "object",
    "text_id": "object",
    "word_id": "float64",
    "text": "object",
    "line_idx": "float64",
    "x": "float64",
    "y": "float64",
    "width": "float64",
    "height": "float64",
}
FIX_CANONICAL_COLUMNS: Dict[str, str] = {
    "participant_id": "object",
    "trial_id": "object",
    "text_id": "object",
    "x": "float64",
    "y": "float64",
    "duration_ms": "float64",
    "timestamp_ms": "float64",
    "fixation_id": "float64",
    "word_id": "float64",
    "pass_index": "float64",
    "saccade_type": "object",
    "eye": "object",
    "noise_flag": "bool",
    "order_in_trial": "int64",
}


def empty_words_frame() -> pd.DataFrame:
    """An empty words frame with the canonical post-normalization columns."""
    return pd.DataFrame(
        {col: pd.Series(dtype=dt) for col, dt in WORDS_CANONICAL_COLUMNS.items()}
    )


def empty_fixations_frame() -> pd.DataFrame:
    """An empty fixations frame with the canonical post-normalization columns."""
    return pd.DataFrame(
        {col: pd.Series(dtype=dt) for col, dt in FIX_CANONICAL_COLUMNS.items()}
    )


def _union_column_values(
    words: pd.DataFrame, fixations: pd.DataFrame, column: str
) -> list:
    """Sorted union of a column's values across both frames (either may be
    empty — single-report datasets have words or fixations, not both)."""
    values: set = set()
    for df in (words, fixations):
        if df is not None and not df.empty and column in df.columns:
            values.update(df[column].unique())
    return sorted(values)


def filter_data(
    words: pd.DataFrame,
    fixations: pd.DataFrame,
    filters: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # When the participant/trial selection covers the whole frame (the default —
    # any narrowing already happened upstream in filter_trials), skip the two
    # O(n) membership masks entirely; only the optional fixation-level filters
    # below apply. ``default_filters`` sets the cover-all flags.
    cover_all = bool(
        filters.get("_participants_cover_all") and filters.get("_trials_cover_all")
    )
    if cover_all:
        # participant/trial cover the whole frame and default_filters set the
        # pass/saccade/eye filters to their full value sets (no-ops), so only the
        # noise filter can actually narrow the fixations. Return the frames
        # untouched (no full-frame mask, no copy) when there's nothing to drop —
        # the common large-upload case where there's no noise data.
        include_noise = filters.get("include_noise", True)
        if (
            not include_noise
            and "noise_flag" in fixations.columns
            and bool(fixations["noise_flag"].fillna(False).to_numpy().any())
        ):
            return words, fixations[~fixations["noise_flag"].fillna(False)]
        return words, fixations

    participants = filters.get("participants") or _union_column_values(
        words, fixations, "participant_id"
    )
    trials = filters.get("trials") or _union_column_values(words, fixations, "trial_id")
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


def filter_trials(
    words: pd.DataFrame,
    fixations: pd.DataFrame,
    participants: Optional[list] = None,
    metadata: Optional[Dict[str, set]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Narrow words + fixations by participant and categorical trial metadata.

    ``metadata`` maps a column name to the set of allowed values. Only columns
    present on a frame are applied, so a condition like ``question_preview``
    (Hunting/Gathering) narrows both words and fixations — the column is copied
    onto both during normalization. A falsy selection means "no constraint".
    """
    w, f = words, fixations
    if participants:
        # participant_id is already string after normalization (as the metadata
        # filters below also assume), so skip a full-column .astype(str) recast.
        keep = set(map(str, participants))
        w = w[w["participant_id"].isin(keep)]
        f = f[f["participant_id"].isin(keep)]
    for col, allowed in (metadata or {}).items():
        if not allowed:
            continue
        allowed = set(allowed)
        if col in w.columns:
            w = w[w[col].isin(allowed)]
        if col in f.columns:
            f = f[f[col].isin(allowed)]
    return w, f


def filter_to_keys(
    words: pd.DataFrame,
    fixations: pd.DataFrame,
    keys: set,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only rows whose (participant_id, trial_id) is in ``keys``.

    ``keys`` is a set of ``(str, str)`` tuples. Used to apply annotation-based
    filtering (favorites / tags). Vectorized via a MultiIndex membership test so
    it stays fast on large fixation tables."""

    def _restrict(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        idx = pd.MultiIndex.from_arrays(
            [df["participant_id"].astype(str), df["trial_id"].astype(str)]
        )
        return df[idx.isin(keys)]

    return _restrict(words), _restrict(fixations)


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
    # NaN maxima happen when fixations ship without coordinates (AOI-sequence
    # data) and no word boxes were available to fill them in.
    x_candidates = [v for v in x_candidates if np.isfinite(v)]
    y_candidates = [v for v in y_candidates if np.isfinite(v)]
    if not x_candidates or not y_candidates:
        return max(int(default_w), 100), max(int(default_h), 100)
    width = int(np.ceil(max(x_candidates) / 100.0) * 100)
    height = int(np.ceil(max(y_candidates) / 100.0) * 100)
    return max(width, 100), max(height, 100)


# Primary EyeLink IA measures. When a words frame already carries all of these
# (a pre-aggregated export, e.g. OneStop), the fixation-based recompute is a
# fallback whose output is discarded by the "existing values win" merge — so we
# skip it entirely. See compute_per_word_measures for the precedence rule.
_PREAGGREGATED_METRIC_COLUMNS = [
    "first_fixation_ms",
    "first_pass_gaze_duration_ms",
    "total_fixation_duration_ms",
    "n_fixations",
]


def compute_word_metrics(words: pd.DataFrame, fixations: pd.DataFrame) -> pd.DataFrame:
    """Return per-word reading measures.

    If the words table already carries pre-aggregated measures (EyeLink IA
    export), those values are preserved. Anything missing is computed from
    fixations + bounding boxes via `measures.compute_per_word_measures`.

    Cached on a cheap content *fingerprint* of the inputs (see
    ``frame_fingerprint``) rather than a full DataFrame hash, so a rerun that
    doesn't change the data reuses the result without re-hashing millions of
    rows. The frames themselves are passed un-hashed (underscore args).
    """
    return _compute_word_metrics_cached(
        words,
        fixations,
        cache_key=(frame_fingerprint(words), frame_fingerprint(fixations)),
    )


@st.cache_data(show_spinner="Computing reading measures…")
def _compute_word_metrics_cached(
    _words: pd.DataFrame, _fixations: pd.DataFrame, cache_key
) -> pd.DataFrame:
    from .measures import compute_per_word_measures

    if _words.empty:
        return _words.copy()

    # Pre-aggregated reading measures win over computed ones, so when the words
    # frame already carries the EyeLink IA measures we skip the O(fixations)
    # assignment + per-word temporal walk entirely (minutes on the full corpus).
    if all(col in _words.columns for col in _PREAGGREGATED_METRIC_COLUMNS):
        enriched = _words
    else:
        enriched = compute_per_word_measures(_fixations, _words)

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
        "text_id",
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
    """Default ("everything selected") filter dict for the current frames.

    Cached on a cheap content fingerprint so the full-column ``unique()`` scans
    don't re-run on every rerun when the data hasn't changed.
    """
    return _default_filters_cached(
        words,
        fixations,
        cache_key=(frame_fingerprint(words), frame_fingerprint(fixations)),
    )


@st.cache_data(show_spinner=False)
def _default_filters_cached(
    _words: pd.DataFrame, _fixations: pd.DataFrame, cache_key
) -> Dict:
    filters = dict(
        participants=_union_column_values(_words, _fixations, "participant_id"),
        trials=_union_column_values(_words, _fixations, "trial_id"),
        # The participant/trial lists above are the *full* unique set of the
        # (already trial-filtered) frame, so filter_data's membership masks are
        # no-ops — flag that so it can skip the two O(n) scans.
        _participants_cover_all=True,
        _trials_cover_all=True,
    )
    if "pass_index" in _fixations.columns:
        filters["pass_indices"] = sorted(_fixations["pass_index"].dropna().unique())
    if "saccade_type" in _fixations.columns:
        filters["saccade_types"] = sorted(
            _fixations["saccade_type"].dropna().astype(str).unique()
        )
    if "eye" in _fixations.columns:
        filters["eyes"] = sorted(_fixations["eye"].dropna().astype(str).unique())
    filters["include_noise"] = False if "noise_flag" in _fixations.columns else True
    return filters
