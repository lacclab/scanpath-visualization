"""Compute reading measures from fixations and word bounding boxes.

Definitions follow standard reading-research conventions (Rayner 1998;
Inhoff & Radach 1998). All measures are per (participant, trial, word). When a
column already exists on the words dataframe (e.g. pre-aggregated EyeLink IA
metrics) it is preserved; we only compute values that are not present.

Canonical output columns added to words:
- first_fixation_ms       — FFD: duration of the first fixation on this word
- first_pass_gaze_ms      — FPRT / gaze duration
- regression_path_ms      — RPD / go-past time
- total_fixation_ms       — TFD / dwell
- n_fixations             — fixation count
- skip_flag               — True if no first-pass fixation
- regression_in_flag      — True if any later fixation returned here
- regression_out_flag     — True if a fixation here was followed by a regression
- first_fix_x, first_fix_y — landing position of the first-pass first fixation

The fixations dataframe is also enriched with:
- word_id            — assigned via bbox containment + nearest-word fallback
- saccade_amplitude  — pixel distance from the previous fixation in the trial
- progression        — 1 if the next fixation moves to a later word, -1 if earlier, 0 otherwise
- is_regression      — True if this fixation lands on a word earlier than the
                       running maximum word reached in the trial
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

# Default line-misregistration tolerance (px): a fixation that falls outside
# every word box snaps to the nearest word centre within this radius before it
# is left unassigned. Shared by the grouped assigner here and the single-frame
# helper used for model scanpaths in :mod:`scanpath_studio.similarity`.
LINE_MISREGISTRATION_PX = 50.0

# A recorded ``timestamp_ms`` series is trusted as real reading time only when
# its span covers at least this fraction of the summed fixation durations.
# Fixations don't overlap, so a genuine recording spans at least its total
# dwell; the 0,1,2,… row index ``data.normalize_fixations`` synthesises when the
# source has no timestamps collapses to a few ms and must NOT be read as
# milliseconds. Shared by the similarity time-curve and the animation clock.
REAL_TIMESTAMP_DWELL_FRAC = 0.5


def _within_box(
    fix_x: pd.Series, fix_y: pd.Series, word_row: pd.Series, pad: float = 0.0
) -> pd.Series:
    return (
        (fix_x >= word_row["x"] - pad)
        & (fix_x <= word_row["x"] + word_row["width"] + pad)
        & (fix_y >= word_row["y"] - pad)
        & (fix_y <= word_row["y"] + word_row["height"] + pad)
    )


def _assign_word_ids_single(
    fix_chunk: pd.DataFrame,
    word_chunk: pd.DataFrame,
    nearest_within_px: float = LINE_MISREGISTRATION_PX,
) -> np.ndarray:
    """Vectorized fixation→word_id assignment for a single trial's frames.

    Tests every fixation in ``fix_chunk`` against every word box in
    ``word_chunk`` (no participant/trial grouping — the caller is responsible
    for slicing to one trial, or for passing frames whose ids deliberately
    don't match, as the model scanpaths do). Fixations outside every box snap
    to the nearest word centre within ``nearest_within_px`` (line-
    misregistration tolerance), else NaN.

    Returns a float array aligned to ``fix_chunk`` rows (NaN = out of text).
    """
    wx0 = pd.to_numeric(word_chunk["x"], errors="coerce").to_numpy(dtype=float)
    wy0 = pd.to_numeric(word_chunk["y"], errors="coerce").to_numpy(dtype=float)
    wx1 = wx0 + pd.to_numeric(word_chunk["width"], errors="coerce").to_numpy(
        dtype=float
    )
    wy1 = wy0 + pd.to_numeric(word_chunk["height"], errors="coerce").to_numpy(
        dtype=float
    )
    wids = word_chunk["word_id"].to_numpy()
    wcx = (wx0 + wx1) / 2.0
    wcy = (wy0 + wy1) / 2.0

    fx = pd.to_numeric(fix_chunk["x"], errors="coerce").to_numpy(dtype=float)
    fy = pd.to_numeric(fix_chunk["y"], errors="coerce").to_numpy(dtype=float)

    in_box = (
        (fx[:, None] >= wx0[None, :])
        & (fx[:, None] <= wx1[None, :])
        & (fy[:, None] >= wy0[None, :])
        & (fy[:, None] <= wy1[None, :])
    )
    word_idx = np.where(in_box.any(axis=1), in_box.argmax(axis=1), -1)

    # Fallback: nearest word center within nearest_within_px.
    unassigned = word_idx == -1
    if unassigned.any() and nearest_within_px > 0:
        dists = np.sqrt(
            (fx[unassigned, None] - wcx[None, :]) ** 2
            + (fy[unassigned, None] - wcy[None, :]) ** 2
        )
        nearest = dists.argmin(axis=1)
        within = dists[np.arange(len(nearest)), nearest] <= nearest_within_px
        word_idx[unassigned] = np.where(within, nearest, -1)

    return np.where(word_idx >= 0, wids[np.clip(word_idx, 0, None)], np.nan)


def assign_fixations_to_words(
    fixations: pd.DataFrame,
    words: pd.DataFrame,
    *,
    overwrite: bool = False,
    nearest_within_px: float = LINE_MISREGISTRATION_PX,
) -> pd.DataFrame:
    """Assign each fixation to a word via bounding-box containment.

    If a fixation does not fall inside any word box, assign it to the nearest
    word center within `nearest_within_px` pixels (a common practice for line
    misregistration). Beyond that radius, the fixation gets word_id=NaN.

    If `overwrite=False` and the fixations already carry word_id values, those
    are kept; only NaN rows get re-assigned.
    """
    if fixations.empty or words.empty:
        return fixations

    out = fixations.copy()
    if "word_id" not in out.columns or overwrite:
        out["word_id"] = np.nan

    need_idx = out["word_id"].isna()
    if not need_idx.any():
        return out

    # Per (participant, trial), do a fast vectorized box-test against that
    # trial's words.
    groups = out[need_idx].groupby(["participant_id", "trial_id"], sort=False)
    word_groups = words.groupby(["participant_id", "trial_id"], sort=False)

    assignments = pd.Series(np.nan, index=out.index[need_idx], dtype=float)
    for (pid, tid), fix_chunk in groups:
        try:
            wchunk = word_groups.get_group((pid, tid))
        except KeyError:
            continue
        if wchunk.empty:
            continue
        assignments.loc[fix_chunk.index] = _assign_word_ids_single(
            fix_chunk, wchunk, nearest_within_px
        )

    out.loc[need_idx, "word_id"] = assignments
    return out


def enrich_fixations(fixations: pd.DataFrame, words: pd.DataFrame) -> pd.DataFrame:
    """Add saccade_amplitude, progression, and is_regression to fixations."""
    if fixations.empty:
        return fixations
    out = fixations.copy()
    out = out.sort_values(["participant_id", "trial_id", "timestamp_ms"])

    g = out.groupby(["participant_id", "trial_id"], sort=False)
    dx = g["x"].diff()
    dy = g["y"].diff()
    if "saccade_amplitude" not in out.columns:
        out["saccade_amplitude"] = np.sqrt(dx * dx + dy * dy)
    else:
        out["saccade_amplitude"] = out["saccade_amplitude"].where(
            out["saccade_amplitude"].notna(), np.sqrt(dx * dx + dy * dy)
        )

    next_word = g["word_id"].shift(-1)
    out["progression"] = np.sign(next_word - out["word_id"]).fillna(0).astype(int)

    running_max = g["word_id"].cummax()
    out["is_regression"] = (out["word_id"] < running_max).fillna(False).astype(bool)
    return out


def rebased_fixation_onsets(ordered_fixations: pd.DataFrame) -> np.ndarray:
    """Fixation onset times (ms), rebased so the first fixation is t=0.

    ``ordered_fixations`` must already be in reading order (sorted by
    ``timestamp_ms``); the returned array is aligned to its rows. Uses the
    recorded ``timestamp_ms`` when they look like real times — their span is at
    least ``REAL_TIMESTAMP_DWELL_FRAC`` of the summed durations — otherwise lays
    fixations back-to-back by their durations, so a synthesised 0,1,2,… index
    doesn't crush the time axis. Shared by the similarity time-curve
    (:func:`scanpath_studio.similarity._rebased_onsets`) and the animation clock
    (:func:`scanpath_studio.plots._scanpath_anim_specs`).
    """
    if ordered_fixations.empty:
        return np.array([], dtype=float)
    if "duration_ms" in ordered_fixations.columns:
        dur = (
            pd.to_numeric(ordered_fixations["duration_ms"], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=float)
        )
    else:
        dur = np.zeros(len(ordered_fixations))
    contiguous = np.concatenate(([0.0], np.cumsum(dur)[:-1])) if len(dur) else dur
    if "timestamp_ms" in ordered_fixations.columns:
        ts = pd.to_numeric(ordered_fixations["timestamp_ms"], errors="coerce").to_numpy(
            dtype=float
        )
        total_dwell = float(dur.sum()) if len(dur) else 0.0
        if (
            len(ts)
            and not np.isnan(ts).any()
            and (ts[-1] - ts[0]) >= REAL_TIMESTAMP_DWELL_FRAC * total_dwell
        ):
            return ts - ts[0]
    return contiguous


# ---------------------------------------------------------------------------
# Geometry helpers: line clustering, in-text test, fixation -> line.
#
# These power the "highlight out-of-text fixations" and "color fixations by
# line" plot options. They are deliberately pure (no Streamlit, no plotting)
# so they can be unit-tested against a known synthetic layout.
# ---------------------------------------------------------------------------


def _box_bounds(
    words: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x0, y0, x1, y1) arrays for a words frame's bounding boxes."""
    x0 = words["x"].to_numpy(dtype=float)
    y0 = words["y"].to_numpy(dtype=float)
    x1 = x0 + words["width"].to_numpy(dtype=float)
    y1 = y0 + words["height"].to_numpy(dtype=float)
    return x0, y0, x1, y1


def _in_any_box(fix_chunk: pd.DataFrame, word_chunk: pd.DataFrame) -> np.ndarray:
    """Boolean array (aligned to fix_chunk order): is each fixation inside any box?"""
    x0, y0, x1, y1 = _box_bounds(word_chunk)
    fx = pd.to_numeric(fix_chunk["x"], errors="coerce").to_numpy(dtype=float)
    fy = pd.to_numeric(fix_chunk["y"], errors="coerce").to_numpy(dtype=float)
    inside = (
        (fx[:, None] >= x0[None, :])
        & (fx[:, None] <= x1[None, :])
        & (fy[:, None] >= y0[None, :])
        & (fy[:, None] <= y1[None, :])
    )
    return inside.any(axis=1)


def cluster_word_lines(words: pd.DataFrame, tol_frac: float = 0.5) -> pd.Series:
    """Assign each word a 0-based line id by clustering on vertical position.

    OneStop IA exports rarely carry a real per-word line number (the
    normalized ``line_idx`` is often a constant), so we infer visual lines from
    word-box geometry: words are sorted by vertical center and a new line
    starts whenever the center jumps by more than ``tol_frac`` of the median
    word height. Lines are numbered top-to-bottom starting at 0.

    Returns an int Series aligned to ``words.index`` (empty when ``words`` is
    empty). Mirrors the line clustering used by ``plots.build_critical_span_overlay``.
    """
    if words.empty:
        return pd.Series([], dtype="int64", index=words.index)
    heights = pd.to_numeric(words["height"], errors="coerce")
    typical_h = float(heights.median()) if heights.notna().any() else 1.0
    typical_h = typical_h if typical_h > 0 else 1.0
    y_center = pd.to_numeric(words["y"], errors="coerce") + heights.fillna(0) / 2.0
    order = y_center.sort_values(kind="stable")
    line_of_sorted = (
        (order.diff().fillna(0) > typical_h * tol_frac).cumsum().astype(int)
    )
    return line_of_sorted.reindex(words.index)


def _groupwise(fixations: pd.DataFrame, words: pd.DataFrame, fn) -> pd.Series:
    """Apply ``fn(fix_chunk, word_chunk)`` per (participant, trial), aligning
    the per-chunk result back onto a Series indexed like ``fixations``.

    Falls back to a single group when the id columns are absent (e.g. the
    figure builders pass already-sliced single-trial frames)."""
    keys = ["participant_id", "trial_id"]
    # object dtype so a chunk fn may return bools (in-text) or floats (line ids)
    # without triggering a dtype-incompatibility cast; callers coerce the result.
    out = pd.Series(np.nan, index=fixations.index, dtype=object)
    has_groups = set(keys).issubset(fixations.columns) and set(keys).issubset(
        words.columns
    )
    if not has_groups:
        out.loc[:] = fn(fixations, words)
        return out
    word_groups = words.groupby(keys, sort=False)
    for key, fix_chunk in fixations.groupby(keys, sort=False):
        try:
            word_chunk = word_groups.get_group(key)
        except KeyError:
            continue
        if word_chunk.empty:
            continue
        out.loc[fix_chunk.index] = fn(fix_chunk, word_chunk)
    return out


def fixation_in_text_mask(fixations: pd.DataFrame, words: pd.DataFrame) -> pd.Series:
    """Boolean Series: True where a fixation falls inside any word box.

    "Out-of-text" fixations are simply ``~fixation_in_text_mask(...)``. Works
    on multi-trial frames (grouped by participant/trial) or on a single
    already-sliced trial. Fixations with non-finite coordinates count as
    out-of-text (mask = False)."""
    if fixations.empty:
        return pd.Series([], dtype=bool, index=fixations.index)
    if words.empty:
        return pd.Series(False, index=fixations.index)
    res = _groupwise(fixations, words, _in_any_box)
    return res.where(res.notna(), other=False).astype(bool)


def assign_fixation_lines(fixations: pd.DataFrame, words: pd.DataFrame) -> pd.Series:
    """Assign each fixation the 0-based line id of the nearest text line.

    Lines are derived from word geometry via :func:`cluster_word_lines`; each
    fixation is mapped to the line whose mean vertical center is closest to the
    fixation's y. Returns a float Series (NaN where unmappable) aligned to
    ``fixations.index`` so it can be used as a categorical color field."""
    if fixations.empty:
        return pd.Series([], dtype="float64", index=fixations.index)
    if words.empty:
        return pd.Series(np.nan, index=fixations.index, dtype="float64")

    def _nearest_line(fix_chunk: pd.DataFrame, word_chunk: pd.DataFrame) -> np.ndarray:
        lines = cluster_word_lines(word_chunk)
        y_center = (
            pd.to_numeric(word_chunk["y"], errors="coerce")
            + pd.to_numeric(word_chunk["height"], errors="coerce").fillna(0) / 2.0
        )
        centers = y_center.groupby(lines).mean()
        line_ids = centers.index.to_numpy(dtype=float)
        line_cy = centers.to_numpy(dtype=float)
        fy = pd.to_numeric(fix_chunk["y"], errors="coerce").to_numpy(dtype=float)
        dist = np.abs(fy[:, None] - line_cy[None, :])
        nearest = np.where(np.isnan(fy), -1, dist.argmin(axis=1))
        result = np.where(nearest >= 0, line_ids[np.clip(nearest, 0, None)], np.nan)
        return result

    return pd.to_numeric(_groupwise(fixations, words, _nearest_line), errors="coerce")


def compute_per_word_measures(
    fixations: pd.DataFrame, words: pd.DataFrame
) -> pd.DataFrame:
    """Compute canonical reading measures per word.

    Returns a copy of `words` with computed columns added. Existing values on
    `words` (e.g. EyeLink IA metrics) take precedence over computed ones.
    """
    if words.empty:
        return words.copy()

    enriched = (
        enrich_fixations(assign_fixations_to_words(fixations, words), words)
        if not fixations.empty
        else fixations
    )

    out = words.copy()
    key_cols = ["participant_id", "trial_id", "word_id"]

    # Initialize defaults
    computed = (
        words[key_cols]
        .drop_duplicates()
        .assign(
            _first_fixation_ms=np.nan,
            _first_pass_gaze_ms=np.nan,
            _regression_path_ms=np.nan,
            _total_fixation_ms=0.0,
            _n_fixations=0,
            _skip_flag=True,
            _regression_in_flag=False,
            _regression_out_flag=False,
            _first_fix_x=np.nan,
            _first_fix_y=np.nan,
        )
    )

    if not fixations.empty and "word_id" in enriched.columns:
        per_word_rows = []
        # Group fixations by trial to walk them in temporal order.
        for (pid, tid), fix_chunk in enriched.dropna(subset=["word_id"]).groupby(
            ["participant_id", "trial_id"], sort=False
        ):
            fix_chunk = fix_chunk.sort_values("timestamp_ms")
            # Total / n / first-fixation are per-word aggregations
            grp = fix_chunk.groupby("word_id")
            tot = grp["duration_ms"].sum()
            n = grp.size()
            ffd = grp["duration_ms"].first()
            ffx = grp["x"].first()
            ffy = grp["y"].first()

            # First-pass gaze: walk the trial in order, accumulate runs.
            first_pass_gaze: dict[float, float] = {}
            regression_path: dict[float, float] = {}
            regression_in: set = set()
            regression_out: set = set()

            running_max = -np.inf
            current_run_word: float | None = None
            current_run_duration: float = 0.0
            # For regression-path: from first entry into a word until first
            # fixation past it, sum all durations.
            first_entry_seen: set = set()
            rp_open_for: dict[float, float] = {}

            prev_word: float | None = None
            for row in fix_chunk.itertuples():
                w = float(row.word_id)
                dur = float(row.duration_ms)

                # First-pass gaze duration: continuous run on this word
                # starting from first entry, ending the first time we leave.
                if w not in first_pass_gaze:
                    if current_run_word == w:
                        current_run_duration += dur
                    else:
                        if current_run_word is not None:
                            first_pass_gaze.setdefault(
                                current_run_word, current_run_duration
                            )
                        current_run_word = w
                        current_run_duration = dur
                else:
                    # Already past first pass; reset run tracker.
                    if (
                        current_run_word is not None
                        and current_run_word not in first_pass_gaze
                    ):
                        first_pass_gaze.setdefault(
                            current_run_word, current_run_duration
                        )
                    current_run_word = None
                    current_run_duration = 0.0

                # Regression-path: from the first entry into a word, sum
                # durations until the next fixation lands on a strictly later
                # word.
                if w not in first_entry_seen:
                    first_entry_seen.add(w)
                    rp_open_for[w] = dur
                else:
                    for k in list(rp_open_for.keys()):
                        if w <= k:
                            # Still within or back-tracking; keep accumulating.
                            rp_open_for[k] += dur
                # Close any open RP windows for words we've now moved past.
                for k in list(rp_open_for.keys()):
                    if w > k and k != w:
                        regression_path.setdefault(k, rp_open_for.pop(k))

                # Regression-in: stepping back to an earlier word counts the
                # destination as receiving an in-regression.
                if prev_word is not None and w < prev_word:
                    regression_in.add(w)
                    regression_out.add(prev_word)

                running_max = max(running_max, w)
                prev_word = w

            # Flush remaining first-pass run
            if current_run_word is not None and current_run_word not in first_pass_gaze:
                first_pass_gaze[current_run_word] = current_run_duration
            # Flush remaining regression-path windows (reader never moved past)
            for k, v in rp_open_for.items():
                regression_path.setdefault(k, v)

            for w in tot.index:
                per_word_rows.append(
                    dict(
                        participant_id=pid,
                        trial_id=tid,
                        word_id=w,
                        _first_fixation_ms=float(ffd.loc[w]),
                        _first_pass_gaze_ms=float(first_pass_gaze.get(w, np.nan)),
                        _regression_path_ms=float(
                            regression_path.get(w, np.nan)
                            if w in first_entry_seen
                            else np.nan
                        ),
                        _total_fixation_ms=float(tot.loc[w]),
                        _n_fixations=int(n.loc[w]),
                        _skip_flag=bool(np.isnan(first_pass_gaze.get(w, np.nan))),
                        _regression_in_flag=w in regression_in,
                        _regression_out_flag=w in regression_out,
                        _first_fix_x=float(ffx.loc[w]),
                        _first_fix_y=float(ffy.loc[w]),
                    )
                )

        if per_word_rows:
            new_df = pd.DataFrame(per_word_rows)
            updated = computed.merge(
                new_df, on=key_cols, how="left", suffixes=("_default", "")
            )
            for col in new_df.columns:
                if col in key_cols:
                    continue
                default_col = f"{col}_default"
                if default_col in updated.columns:
                    updated[col] = updated[col].where(
                        updated[col].notna(), updated[default_col]
                    )
                    updated = updated.drop(columns=default_col)
            computed = updated

    out = out.merge(computed, on=key_cols, how="left")

    # Map computed -> canonical name, keeping any existing value.
    rename_map = {
        "_first_fixation_ms": "first_fixation_ms",
        "_first_pass_gaze_ms": "first_pass_gaze_duration_ms",
        "_regression_path_ms": "regression_path_duration_ms",
        "_total_fixation_ms": "total_fixation_duration_ms",
        "_n_fixations": "n_fixations",
        "_skip_flag": "skip_flag",
        "_regression_in_flag": "regression_in_flag",
        "_regression_out_flag": "regression_out_flag",
        "_first_fix_x": "first_fix_x",
        "_first_fix_y": "first_fix_y",
    }
    for src, dst in rename_map.items():
        if src not in out.columns:
            continue
        if dst in out.columns:
            out[dst] = out[dst].where(out[dst].notna(), out[src])
        else:
            out[dst] = out[src]
        out = out.drop(columns=src)

    # Canonical aliases used elsewhere in the app
    if (
        "first_pass_gaze_duration_ms" in out.columns
        and "gaze_duration_ms" not in out.columns
    ):
        out["gaze_duration_ms"] = out["first_pass_gaze_duration_ms"]

    # Ensure dtypes
    for col in ["n_fixations"]:
        if col in out.columns:
            out[col] = (
                pd.to_numeric(out[col], errors="coerce").fillna(0).astype("Int64")
            )
    for col in ["skip_flag", "regression_in_flag", "regression_out_flag"]:
        if col in out.columns:
            out[col] = (
                out[col].astype(object).where(out[col].notna(), False).astype(bool)
            )

    return out


def compute_trial_measures(
    fixations: pd.DataFrame, words: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: returns (enriched_fixations, words_with_measures)."""
    enriched_fix = (
        enrich_fixations(assign_fixations_to_words(fixations, words), words)
        if not fixations.empty
        else fixations
    )
    enriched_words = compute_per_word_measures(fixations, words)
    return enriched_fix, enriched_words
