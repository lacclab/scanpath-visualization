"""Utility functions for trial selection, statistics, and labelling."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import streamlit as st

from .data import frame_fingerprint


# -----------------------------------------------------------------------------
# Trial combo building
# -----------------------------------------------------------------------------


def build_combo_options(
    fixations: pd.DataFrame,
) -> Tuple[pd.DataFrame, list[str], Dict[str, Tuple[str, str]]]:
    """Build participant/trial/text combinations for selection UI.

    Returns:
        Tuple of (combos DataFrame, label list, label-to-combo mapping).

    Cached on a cheap fingerprint of the frame + the composite-trial columns, so
    the full-frame ``drop_duplicates`` + label build don't re-run on every rerun
    (e.g. selecting a different trial). The session-state read happens here, in
    the un-cached wrapper, and is threaded into the cached core as an argument.
    """
    composite_cols = tuple(st.session_state.get("_composite_trial_columns") or [])
    return _build_combo_options_cached(
        fixations,
        composite_cols,
        cache_key=(frame_fingerprint(fixations), composite_cols),
    )


@st.cache_data(show_spinner="Building trial list…")
def _build_combo_options_cached(
    _fixations: pd.DataFrame,
    composite_cols: Tuple[str, ...],
    cache_key,
) -> Tuple[pd.DataFrame, list[str], Dict[str, Tuple[str, str]]]:
    fixations = _fixations
    trial_col = (
        "unique_trial_id" if "unique_trial_id" in fixations.columns else "trial_id"
    )
    # The text/passage column is optional. Normalized frames carry a text_id (it
    # falls back to trial_id when no text is mapped), but a frame may arrive with
    # only the source name (e.g. unique_paragraph_id — the pre-rename text id, and
    # which can also be a composite-trial component). Detect it via the same
    # priority list normalization uses, and *copy* it to text_id rather than
    # renaming so a shared composite component column survives for the picker.
    text_col = next(
        (
            c
            for c in ("unique_text_id", "text_id", "unique_paragraph_id", "paragraph_id")
            if c in fixations.columns
        ),
        None,
    )
    combo_cols = ["participant_id", trial_col]
    if text_col is not None and text_col not in combo_cols:
        combo_cols.append(text_col)
    for col in ["unique_trial_id", "unique_text_id", "TRIAL_INDEX", "trial_index"]:
        if col in fixations.columns and col not in combo_cols:
            combo_cols.append(col)
    # Carry the composite trial id's component columns through, so the trial
    # picker can offer one cascading selector per part (see select_trial).
    for col in composite_cols:
        if col in fixations.columns and col not in combo_cols:
            combo_cols.append(col)

    combos = fixations[combo_cols].drop_duplicates().rename(columns={trial_col: "trial_id"})
    if "text_id" not in combos.columns:
        combos["text_id"] = (
            combos[text_col] if text_col is not None else combos["trial_id"]
        )
    if trial_col == "unique_trial_id" and "unique_trial_id" not in combos.columns:
        combos["unique_trial_id"] = combos["trial_id"]
    if text_col == "unique_text_id" and "unique_text_id" not in combos.columns:
        combos["unique_text_id"] = combos["text_id"]
    sort_cols = ["participant_id"]
    if "TRIAL_INDEX" in combos.columns:
        sort_cols.append("TRIAL_INDEX")
    elif "trial_index" in combos.columns:
        sort_cols.append("trial_index")
    sort_cols.append("trial_id")
    combos = combos.sort_values(sort_cols)

    combo_labels = [
        f"{row.participant_id} / {row.trial_id} · {row.text_id}"
        for row in combos.itertuples()
    ]
    label_to_combo = dict(
        zip(
            combo_labels,
            combos[["participant_id", "trial_id"]].itertuples(index=False, name=None),
        )
    )
    return combos, combo_labels, label_to_combo


@st.cache_data(show_spinner=False)
def _trial_positions(_frame: pd.DataFrame, cache_key) -> Dict[Tuple[str, str], object]:
    """Map ``(participant_id, trial_id)`` → positional row indices.

    Built once per frame (cached on its fingerprint) so extracting a single
    trial is an O(trial) ``iloc`` rather than an O(corpus) boolean mask on every
    rerun — and shared across the tabs, which all slice the same filtered frames.
    """
    if _frame is None or _frame.empty:
        return {}
    grouped = _frame.groupby(["participant_id", "trial_id"], sort=False).indices
    # Normalise keys to (str, str) so lookups match the picker's string values.
    return {(str(p), str(t)): idx for (p, t), idx in grouped.items()}


def extract_trial(frame: pd.DataFrame, participant_id, trial_id) -> pd.DataFrame:
    """Rows of one (participant, trial), sliced via the cached position index.

    Equivalent to ``frame[(frame.participant_id == p) & (frame.trial_id == t)]``
    but O(trial) instead of O(corpus) once the index is built — the per-rerun win
    on large datasets, where every tab extracts the selected trial."""
    if frame is None or getattr(frame, "empty", True):
        return frame
    positions = _trial_positions(frame, cache_key=frame_fingerprint(frame))
    pos = positions.get((str(participant_id), str(trial_id)))
    if pos is None or len(pos) == 0:
        return frame.iloc[0:0]
    return frame.iloc[pos]


# -----------------------------------------------------------------------------
# Trial selection UI
# -----------------------------------------------------------------------------


def _select_trial_none_mode(
    combos: pd.DataFrame, trial_field: str, text_field: str, key_prefix: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Handle trial selection when mode is 'None' (direct trial selection)."""
    available_trials = combos.drop_duplicates(subset=[trial_field])
    trial_options = sorted(available_trials[trial_field].dropna().astype(str).unique())
    if not trial_options:
        st.warning("No trials available after filtering.")
        st.stop()

    # Session state for navigation
    state_key = f"{key_prefix}_trial_index" if key_prefix else "trial_index"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0

    current_idx = st.session_state[state_key]
    current_idx = max(0, min(current_idx, len(trial_options) - 1))
    st.session_state[state_key] = current_idx

    # Navigation buttons — compact (no `width="stretch"`) so they don't wrap
    # to 3 lines inside the narrow 30% side panel. `vertical_alignment="center"`
    # drops the arrows to line up with the labelled selectbox beside them
    # (otherwise they sit flush to the top, leaving empty space below).
    nav_col1, nav_col2, select_col = st.columns([1, 1, 4], vertical_alignment="center")
    with nav_col1:
        if st.button(
            "◀",
            key=f"{key_prefix}_prev_btn" if key_prefix else "prev_btn",
            disabled=current_idx <= 0,
            help="Previous trial",
        ):
            st.session_state[state_key] = current_idx - 1
            st.rerun()
    with nav_col2:
        if st.button(
            "▶",
            key=f"{key_prefix}_next_btn" if key_prefix else "next_btn",
            disabled=current_idx >= len(trial_options) - 1,
            help="Next trial",
        ):
            st.session_state[state_key] = current_idx + 1
            st.rerun()
    with select_col:
        selected_trial_label = st.selectbox(
            "Unique trial id",
            options=trial_options,
            index=current_idx,
            key=f"{key_prefix}_trial_id" if key_prefix else None,
        )
        if selected_trial_label:
            new_idx = trial_options.index(selected_trial_label)
            if new_idx != current_idx:
                st.session_state[state_key] = new_idx

    if not selected_trial_label:
        return None, None, None

    chosen = available_trials[
        available_trials[trial_field].astype(str) == selected_trial_label
    ].iloc[0]
    selected_text = str(chosen[text_field]) if text_field in chosen.index else None
    return chosen["participant_id"], chosen["trial_id"], selected_text


def _select_trial_text_mode(
    combos: pd.DataFrame, text_field: str, key_prefix: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Handle trial selection when mode is 'Text' (select by text first)."""
    text_options = sorted(combos[text_field].dropna().astype(str).unique())
    if not text_options:
        st.warning("No texts available after filtering.")
        st.stop()

    selected_text = st.selectbox(
        "Text",
        options=text_options,
        key=f"{key_prefix}_text_id" if key_prefix else None,
    )
    if not selected_text:
        st.warning("No text selected after filtering.")
        st.stop()

    text_combos = combos[combos[text_field].astype(str) == str(selected_text)]
    participant_options = sorted(text_combos["participant_id"].dropna().unique())
    if not participant_options:
        st.warning("No participants available for this text.")
        st.stop()

    selected_participant = st.selectbox(
        "Participant",
        options=participant_options,
        key=f"{key_prefix}_participant_text" if key_prefix else None,
    )

    # Handle multiple readings
    candidate_trials = (
        text_combos[text_combos["participant_id"] == selected_participant]
        .drop_duplicates(subset=["trial_id"])
        .sort_values("trial_id")
    )
    if candidate_trials.empty:
        return None, None, selected_text

    if len(candidate_trials) > 1:
        trial_options = candidate_trials["trial_id"].tolist()
        selected_trial = st.selectbox(
            "Reading (multiple trials available)",
            options=trial_options,
            key=f"{key_prefix}_reading_text" if key_prefix else None,
            help="This participant read this text multiple times.",
        )
    else:
        selected_trial = candidate_trials.iloc[0]["trial_id"]

    return selected_participant, selected_trial, selected_text


def _select_trial_participant_mode(
    combos: pd.DataFrame,
    text_field: str,
    trial_index_field: Optional[str],
    key_prefix: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Handle trial selection when mode is 'Participant' (select participant first)."""
    participants = sorted(combos["participant_id"].dropna().unique())
    if not participants:
        st.warning("No participants available after filtering.")
        st.stop()

    selected_participant = st.selectbox(
        "Participant",
        options=participants,
        key=f"{key_prefix}_participant" if key_prefix else None,
    )

    participant_trials = combos[combos["participant_id"] == selected_participant]
    if participant_trials.empty:
        st.warning("No trials available for this participant.")
        return None, None, None

    # Within the chosen participant, let the user pick which trial by trial
    # index, by text, or by trial id — only offering the methods the data
    # supports (trial index needs a TRIAL_INDEX column). "Trial index" is the
    # default so deep links (?trial=N) and the prior behavior still land.
    methods: list[str] = []
    if trial_index_field and participant_trials[trial_index_field].notna().any():
        methods.append("Trial index")
    methods.append("Text")
    methods.append("Trial ID")
    # A filter can drop "Trial index" out of `methods` (a participant with no
    # trial-index values) while session_state still holds it from a prior run —
    # clear the stale pick so st.radio falls back to the first option instead of
    # raising (same guard as the select_slider below / _select_trial_composite_mode).
    sub_mode_key = f"{key_prefix}_participant_by" if key_prefix else None
    if (
        sub_mode_key
        and sub_mode_key in st.session_state
        and st.session_state[sub_mode_key] not in methods
    ):
        del st.session_state[sub_mode_key]
    if len(methods) > 1:
        st.markdown("##### Pick trial by")
        sub_mode = st.radio(
            "Pick trial by",
            options=methods,
            horizontal=True,
            key=sub_mode_key,
            label_visibility="collapsed",
        )
    else:
        sub_mode = methods[0]

    if sub_mode == "Trial index":
        slider_field = trial_index_field
        slider_label = "Trial index"
        slider_options = sorted(
            participant_trials[trial_index_field].dropna().unique().tolist()
        )
    elif sub_mode == "Text":
        slider_field = text_field
        slider_label = "Text"
        slider_options = sorted(
            participant_trials[text_field].dropna().astype(str).unique().tolist()
        )
    else:  # Trial ID
        slider_field = "trial_id"
        slider_label = "Trial ID"
        slider_options = sorted(
            participant_trials["trial_id"].dropna().astype(str).unique().tolist()
        )

    if not slider_options:
        return None, None, None

    state_key = f"{key_prefix}_slider" if key_prefix else "slider"
    # Switching sub-method changes the option universe; drop a now-invalid
    # stored value so st.select_slider falls back to the first option instead of
    # raising (mirrors the guard in _select_trial_composite_mode).
    if (
        state_key in st.session_state
        and st.session_state[state_key] not in slider_options
    ):
        del st.session_state[state_key]

    if len(slider_options) == 1:
        # st.select_slider breaks in the browser with a single option
        # (RangeError: min (0) is equal/bigger than max (0)) — common when the
        # data holds a single trial or the filters narrow a participant down
        # to one. Show the lone value as plain text instead.
        slider_value = slider_options[0]
        st.session_state[state_key] = slider_value
        st.caption(f"{slider_label}: **{slider_value}** (only one available)")
        return _resolve_participant_trial(
            participant_trials,
            selected_participant,
            slider_field,
            slider_value,
            text_field,
            key_prefix,
        )

    # Prev/Next buttons flank the slider so reviewers can step through trials
    # without dragging. Mutations live in an `on_click` callback (the slider's
    # session-state key can't be mutated after the widget instantiates).

    def _step_slider(direction: int) -> None:
        opts = slider_options
        current = st.session_state.get(state_key)
        try:
            idx = opts.index(current) if current is not None else 0
        except ValueError:
            idx = 0
        new_idx = max(0, min(idx + direction, len(opts) - 1))
        st.session_state[state_key] = opts[new_idx]

    current_val = st.session_state.get(state_key, slider_options[0])
    try:
        current_pos = slider_options.index(current_val)
    except ValueError:
        current_pos = 0

    nav_prev, slider_col, nav_next = st.columns([1, 8, 1], vertical_alignment="center")
    with nav_prev:
        st.button(
            "◀",
            key=f"{key_prefix}_slider_prev" if key_prefix else "slider_prev",
            disabled=current_pos <= 0,
            help=f"Previous {slider_label.lower()}",
            on_click=_step_slider,
            args=(-1,),
        )
    with nav_next:
        st.button(
            "▶",
            key=f"{key_prefix}_slider_next" if key_prefix else "slider_next",
            disabled=current_pos >= len(slider_options) - 1,
            help=f"Next {slider_label.lower()}",
            on_click=_step_slider,
            args=(+1,),
        )
    with slider_col:
        slider_value = st.select_slider(
            slider_label,
            options=slider_options,
            key=state_key,
        )
    if slider_value is None:
        return None, None, None

    return _resolve_participant_trial(
        participant_trials,
        selected_participant,
        slider_field,
        slider_value,
        text_field,
        key_prefix,
    )


def _resolve_participant_trial(
    participant_trials: pd.DataFrame,
    selected_participant: str,
    slider_field: str,
    slider_value,
    text_field: str,
    key_prefix: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Map a Participant-mode slider value to a concrete (participant, trial, text),
    offering a 'Reading' selectbox when the value matches several trials."""
    # Text + trial-id sliders carry string values, so match as strings; the
    # trial-index slider carries the raw (numeric) value and matches exactly.
    if slider_field in (text_field, "trial_id"):
        trial_candidates = participant_trials[
            participant_trials[slider_field].astype(str) == str(slider_value)
        ]
        selected_text = (
            str(slider_value)
            if slider_field == text_field
            else (
                str(trial_candidates.iloc[0][text_field])
                if not trial_candidates.empty and text_field in trial_candidates.columns
                else None
            )
        )
    else:
        trial_candidates = participant_trials[
            participant_trials[slider_field] == slider_value
        ]
        selected_text = (
            str(trial_candidates.iloc[0][text_field])
            if not trial_candidates.empty and text_field in trial_candidates.columns
            else None
        )

    trial_candidates = trial_candidates.drop_duplicates(
        subset=["trial_id"]
    ).sort_values("trial_id")
    if trial_candidates.empty:
        return None, None, selected_text

    if len(trial_candidates) > 1:
        trial_options = trial_candidates["trial_id"].tolist()
        selected_trial = st.selectbox(
            "Reading (multiple trials available)",
            options=trial_options,
            key=f"{key_prefix}_reading_participant" if key_prefix else None,
            help="This participant read this text multiple times.",
        )
    else:
        selected_trial = trial_candidates.iloc[0]["trial_id"]

    return selected_participant, selected_trial, selected_text


# Friendly labels for the cascading selectors in composite-trial mode. Unknown
# component columns fall back to their raw name (most informative for arbitrary
# uploads); the canonical ids reuse the Participant / Text wording so the UI
# reads the same as the dedicated modes. Composite components are preserved under
# their *source* names (data._preserve_composite_columns), so the pre-rename
# paragraph names map to "Text" too — the canonical text_id was paragraph_id.
_COMPONENT_LABELS = {
    "participant_id": "Participant",
    "unique_text_id": "Text",
    "text_id": "Text",
    "unique_paragraph_id": "Text",
    "paragraph_id": "Text",
}


def _component_label(col: str) -> str:
    return _COMPONENT_LABELS.get(col, col)


def _composite_columns_for(combos: pd.DataFrame) -> list[str]:
    """Composite trial-id component columns that are actually present in
    ``combos`` — empty unless the trial id was built from several columns
    (set in ``app.prepare_data`` / preserved by ``data._preserve_composite_columns``)."""
    cols = st.session_state.get("_composite_trial_columns") or []
    return [c for c in cols if c in combos.columns]


def _select_trial_composite_mode(
    combos: pd.DataFrame,
    component_cols: list[str],
    text_field: str,
    key_prefix: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Trial selection when the trial id was composed from several columns.

    Renders one cascading selector per component (each narrowed by the previous
    picks), mirroring the Text / Participant modes instead of a single opaque
    ``a_b_c`` dropdown."""
    st.caption("Composite trial id — pick each part to narrow to a trial.")
    filtered = combos
    for col in component_cols:
        options = sorted(filtered[col].dropna().astype(str).unique())
        if not options:
            st.warning("No trials available after filtering.")
            return None, None, None
        state_key = (
            f"{key_prefix}_composite_{col}" if key_prefix else f"composite_{col}"
        )
        # A change to an earlier selector can drop the stored value out of this
        # selector's (now narrower) option set — clear it so st.selectbox falls
        # back to the first valid option instead of raising.
        if state_key in st.session_state and st.session_state[state_key] not in options:
            del st.session_state[state_key]
        chosen = st.selectbox(_component_label(col), options=options, key=state_key)
        filtered = filtered[filtered[col].astype(str) == str(chosen)]
        if filtered.empty:
            st.warning("No trial matches the selected combination.")
            return None, None, None

    candidates = filtered.drop_duplicates(subset=["trial_id"]).sort_values("trial_id")
    if candidates.empty:
        return None, None, None
    if len(candidates) > 1:
        # The components didn't fully determine a single trial — offer the
        # remaining ones, like the other modes' "Reading" selector.
        trial_options = candidates["trial_id"].astype(str).tolist()
        selected_trial = st.selectbox(
            "Reading (multiple trials available)",
            options=trial_options,
            key=f"{key_prefix}_composite_reading"
            if key_prefix
            else "composite_reading",
            help="More than one trial shares these values.",
        )
        row = candidates[
            candidates["trial_id"].astype(str) == str(selected_trial)
        ].iloc[0]
    else:
        row = candidates.iloc[0]
        selected_trial = row["trial_id"]

    text = str(row[text_field]) if text_field in row.index else None
    return row["participant_id"], selected_trial, text


def select_trial(
    combos: pd.DataFrame, key_prefix: str = ""
) -> Tuple[Optional[str], Optional[str], str, Optional[str]]:
    """Select a trial using a three-mode UI (Trial/Text/Participant).

    In ``Trial`` mode a composite trial id (built from several mapped columns)
    is broken into one cascading selector per component; otherwise a single
    unique-trial dropdown is shown.

    Returns:
        Tuple of (participant_id, trial_id, selection_mode, selected_text).
    """
    if combos.empty:
        st.warning("No trials available after filtering.")
        st.stop()

    trial_field = (
        "unique_trial_id" if "unique_trial_id" in combos.columns else "trial_id"
    )
    text_field = "unique_text_id" if "unique_text_id" in combos.columns else "text_id"
    trial_index_field = next(
        (c for c in ["TRIAL_INDEX", "trial_index"] if c in combos.columns), None
    )

    # Only offer the Text / Participant modes when those dimensions actually vary
    # — a single anonymous participant or a single text makes them no-ops, so the
    # picker collapses to a plain trial dropdown.
    modes = ["Trial"]
    if text_field in combos.columns and combos[text_field].nunique(dropna=True) > 1:
        modes.append("Text")
    if (
        "participant_id" in combos.columns
        and combos["participant_id"].nunique(dropna=True) > 1
    ):
        modes.append("Participant")

    mode_key = f"{key_prefix}_select_trial_mode" if key_prefix else None
    # Drop a stale stored mode that's no longer offered so the radio doesn't raise.
    if mode_key and st.session_state.get(mode_key) not in modes:
        st.session_state.pop(mode_key, None)
    if len(modes) > 1:
        st.markdown("#### Select trials by")
        selection_mode = st.radio(
            label="Select trials by",
            options=modes,
            horizontal=True,
            key=mode_key,
            label_visibility="collapsed",
        )
    else:
        selection_mode = "Trial"

    if selection_mode == "Trial":
        composite_cols = _composite_columns_for(combos)
        if len(composite_cols) >= 2:
            participant, trial, text = _select_trial_composite_mode(
                combos, composite_cols, text_field, key_prefix
            )
        else:
            participant, trial, text = _select_trial_none_mode(
                combos, trial_field, text_field, key_prefix
            )
    elif selection_mode == "Text":
        participant, trial, text = _select_trial_text_mode(
            combos, text_field, key_prefix
        )
    else:
        participant, trial, text = _select_trial_participant_mode(
            combos, text_field, trial_index_field, key_prefix
        )

    return participant, trial, selection_mode, text


# -----------------------------------------------------------------------------
# Statistics and metadata
# -----------------------------------------------------------------------------


def compute_trial_stats(
    trial_words: pd.DataFrame, trial_fixations: pd.DataFrame
) -> Dict[str, float]:
    """Compute summary statistics for a single trial."""
    total_time = None
    if "trial_dwell_time_ms" in trial_words.columns:
        dwell_values = (
            pd.to_numeric(trial_words["trial_dwell_time_ms"], errors="coerce")
            .dropna()
            .unique()
        )
        if len(dwell_values):
            total_time = float(dwell_values[0])
    if total_time is None:
        total_time = (
            float(trial_fixations["duration_ms"].sum())
            if not trial_fixations.empty
            else 0.0
        )
    return dict(
        total_reading_time_ms=total_time,
        total_reading_time_s=total_time / 1000.0,
        word_count=int(len(trial_words)),
        fixation_count=int(len(trial_fixations)),
    )


def gather_trial_metadata(
    trial_words: pd.DataFrame, trial_fixations: pd.DataFrame, fields: Iterable[str]
) -> pd.DataFrame:
    """Gather metadata for selected fields from words and fixations."""
    rows = []
    for field in fields:
        if field in trial_words.columns:
            series = pd.Series(trial_words[field])
        elif field in trial_fixations.columns:
            series = pd.Series(trial_fixations[field])
        else:
            continue

        cleaned = series.dropna()
        if cleaned.empty:
            value = "—"
        else:
            unique_values = cleaned.unique()
            if len(unique_values) == 1:
                value = unique_values[0]
            else:
                numeric_series = pd.to_numeric(cleaned, errors="coerce")
                numeric_values = numeric_series.dropna()
                is_numeric = (
                    not pd.api.types.is_bool_dtype(cleaned)
                    and (
                        pd.api.types.is_numeric_dtype(cleaned)
                        or len(numeric_values) == len(cleaned)
                    )
                    and not numeric_values.empty
                )
                if is_numeric:
                    value = f"mean={numeric_values.mean():.2f}, std={numeric_values.std():.2f}"
                else:
                    modes = cleaned.mode(dropna=True)
                    mode_value = modes.iloc[0] if not modes.empty else "—"
                    value = f"{mode_value} (mode, {len(unique_values)} unique)"
        rows.append({"Field": field, "Value": value})

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Value"] = df["Value"].astype(str)
    return df


def safe_summary(series: pd.Series) -> dict:
    """Compute summary statistics for a series, handling empty data."""
    if series.empty:
        nan_val = float("nan")
        return dict(mean=nan_val, std=nan_val, min=nan_val, max=nan_val, median=nan_val)
    return dict(
        mean=float(series.mean()),
        std=float(series.std(ddof=0)),
        min=float(series.min()),
        max=float(series.max()),
        median=float(series.median()),
    )


# -----------------------------------------------------------------------------
# Comparison helpers
# -----------------------------------------------------------------------------


def friendly_trial_label(
    participant_id: str,
    trial_id: str,
    text_id: Optional[str],
    existing_labels: set[str],
    prefix: str = "",
) -> str:
    """Create a short, de-duplicated label for comparison dropdowns/legends."""
    trial_str = str(trial_id) if trial_id is not None else ""
    text_str = str(text_id) if text_id is not None else ""
    text_str = text_str.strip()
    trial_contains_text = text_str and text_str.lower() in trial_str.lower()

    if text_str:
        base = f"{text_str} · {participant_id}"
        if not trial_contains_text:
            base = f"{base} (trial {trial_str})" if trial_str else base
        elif trial_str != text_str:
            # Surface any trial_id suffix beyond the text id (e.g. a
            # repeat-reading "_r2" tag added during normalization). Without
            # this the primary and compare titles look identical when a
            # participant re-read the same text.
            extra = trial_str
            if extra.lower().startswith(text_str.lower()):
                extra = extra[len(text_str) :].lstrip("_- ")
            if extra:
                base = f"{text_str} ({extra}) · {participant_id}"
    else:
        base = f"{trial_str} · {participant_id}" if trial_str else participant_id

    label = f"{prefix}{base}"
    if label in existing_labels:
        label = f"{prefix}{base} [{trial_str or 'trial'}]"
    existing_labels.add(label)
    return label


def build_comparison_options(
    combos: pd.DataFrame,
    selection_mode: str,
    primary_participant: str,
    primary_trial: str,
    primary_text: Optional[str],
) -> list[Tuple[str, str, str]]:
    """Build prioritized list of comparison trial options.

    Returns list of (participant_id, trial_id, label) tuples, prioritized by:
    - Same text trials first (marked with ★)
    - Other trials after
    """
    text_field = "unique_text_id" if "unique_text_id" in combos.columns else "text_id"

    options: list[Tuple[str, str, str]] = []
    added = set()
    used_labels: set[str] = set()

    def add_options(df: pd.DataFrame, prefix: str = ""):
        for row in df.itertuples():
            key = (row.participant_id, row.trial_id)
            if key not in added and key != (primary_participant, primary_trial):
                text_id = getattr(row, text_field, "")
                label = friendly_trial_label(
                    row.participant_id,
                    row.trial_id,
                    text_id,
                    used_labels,
                    prefix=prefix,
                )
                options.append((row.participant_id, row.trial_id, label))
                added.add(key)

    if primary_text:
        # Same text first
        same_text_all = combos[
            (combos[text_field].astype(str) == str(primary_text))
        ].drop_duplicates(subset=["participant_id", "trial_id"])
        add_options(same_text_all, "★ ")

        # Then other texts
        other_texts = combos[
            (combos[text_field].astype(str) != str(primary_text))
        ].drop_duplicates(subset=["participant_id", "trial_id"])
        add_options(other_texts)
    else:
        all_others = combos.drop_duplicates(subset=["participant_id", "trial_id"])
        add_options(all_others)

    return options
