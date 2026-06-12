from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from .annotations import known_tags
from .constants import BACKGROUND_PRESETS, DEFAULT_BACKGROUND_COLOR

NONE_OPTION = "(none)"


# Help text for the (multi-capable) Trial ID mapping, shared by all tables.
_TRIAL_MAPPING_HELP = (
    "Pick the column holding your unique trial ID — or pick SEVERAL columns "
    "to build one on the fly (values joined with '_'), e.g. participant + "
    "paragraph + repeated-reading when no single column identifies a trial. "
    "Use the same columns for every uploaded table so trials line up."
)

WORD_FIELD_SPECS: List[Dict] = [
    {"key": "participant", "label": "Participant ID", "required": True},
    {
        "key": "trial",
        "label": "Trial ID",
        "required": True,
        "multi": True,
        "help": _TRIAL_MAPPING_HELP,
    },
    {"key": "word_id", "label": "Word/IA ID", "required": True},
    {"key": "text", "label": "Word text/label", "required": False},
    {"key": "paragraph", "label": "Paragraph ID", "required": False},
    {"key": "line", "label": "Line index", "required": False},
    {
        "key": "x",
        "label": "Box x (top-left)",
        "required": False,
        "help": "Provide x, y, width, height — or use the four IA_LEFT/RIGHT/TOP/BOTTOM fields below.",
    },
    {"key": "y", "label": "Box y (top-left)", "required": False},
    {"key": "width", "label": "Box width", "required": False},
    {"key": "height", "label": "Box height", "required": False},
    {"key": "left", "label": "Box left (IA_LEFT)", "required": False},
    {"key": "right", "label": "Box right (IA_RIGHT)", "required": False},
    {"key": "top", "label": "Box top (IA_TOP)", "required": False},
    {"key": "bottom", "label": "Box bottom (IA_BOTTOM)", "required": False},
]

FIX_FIELD_SPECS: List[Dict] = [
    {"key": "participant", "label": "Participant ID", "required": True},
    {
        "key": "trial",
        "label": "Trial ID",
        "required": True,
        "multi": True,
        "help": _TRIAL_MAPPING_HELP,
    },
    {"key": "x", "label": "X coordinate", "required": True},
    {"key": "y", "label": "Y coordinate", "required": True},
    {"key": "duration", "label": "Duration (ms)", "required": True},
    {"key": "timestamp", "label": "Timestamp (ms)", "required": False},
    {"key": "fixation_id", "label": "Fixation ID", "required": False},
    {"key": "paragraph", "label": "Paragraph ID", "required": False},
    {"key": "word_id", "label": "Word/IA ID", "required": False},
    {"key": "pass_index", "label": "Pass index", "required": False},
    {"key": "saccade_type", "label": "Saccade type", "required": False},
    {"key": "eye", "label": "Eye", "required": False},
    {"key": "noise_flag", "label": "Noise flag", "required": False},
]

RAW_GAZE_FIELD_SPECS: List[Dict] = [
    {"key": "participant", "label": "Participant ID", "required": True},
    {
        "key": "trial",
        "label": "Trial ID",
        "required": True,
        "multi": True,
        "help": _TRIAL_MAPPING_HELP,
    },
    {"key": "x", "label": "X coordinate", "required": True},
    {"key": "y", "label": "Y coordinate", "required": True},
    {"key": "timestamp", "label": "Timestamp (ms)", "required": False},
    {"key": "text", "label": "Word text/label", "required": False},
]


def column_mapping_ui(
    df: pd.DataFrame,
    table_label: str,
    state_key_prefix: str,
    field_specs: List[Dict],
    proposed: Dict[str, Optional[str]],
    expand_on_problem: bool = True,
    problems: Optional[List[str]] = None,
) -> Dict[str, Optional[str]]:
    """Render a sidebar expander letting users override the inferred column mapping.

    Returns a mapping {field_key: column_name_or_None}. Fields marked
    ``multi: True`` (Trial ID) render as a multiselect: picking several columns
    yields a list, meaning "build this ID on the fly by joining the columns'
    values" (see ``data.trial_id_series``); a single pick stays a plain string.
    """
    options = [NONE_OPTION] + list(df.columns)
    expanded = bool(expand_on_problem and problems)

    with st.sidebar.expander(f"Column mapping — {table_label}", expanded=expanded):
        st.caption(
            "Auto-detected from your CSV. Override any row if your column names differ."
        )
        if problems:
            st.warning(
                "Fix these before the app can use this table: " + "; ".join(problems)
            )
        mapping: Dict[str, Optional[str]] = {}
        for spec in field_specs:
            key = spec["key"]
            default = proposed.get(key)
            label = spec["label"] + (" *" if spec.get("required") else "")
            if spec.get("multi"):
                state_key = f"{state_key_prefix}_{key}"
                proposed_default = [default] if default in df.columns else []
                stored = st.session_state.get(state_key)
                if stored is None:
                    # Seed via session state instead of `default=` so the
                    # stale-column reset below never fights a default arg.
                    st.session_state[state_key] = proposed_default
                else:
                    # A new upload changes the column universe — silently
                    # keeping stale picks would leave the field empty (the
                    # selectboxes self-heal via their index fallback; a
                    # multiselect doesn't). Drop unknown columns and fall
                    # back to the auto-proposal when nothing survives.
                    valid = [c for c in stored if c in df.columns]
                    if len(valid) != len(stored):
                        st.session_state[state_key] = valid or proposed_default
                chosen_cols = st.multiselect(
                    label,
                    options=list(df.columns),
                    key=state_key,
                    help=spec.get("help"),
                )
                if not chosen_cols:
                    mapping[key] = None
                elif len(chosen_cols) == 1:
                    mapping[key] = chosen_cols[0]
                else:
                    mapping[key] = list(chosen_cols)
                continue
            index = options.index(default) if default in options else 0
            chosen = st.selectbox(
                label,
                options=options,
                index=index,
                key=f"{state_key_prefix}_{key}",
                help=spec.get("help"),
            )
            mapping[key] = None if chosen == NONE_OPTION else chosen
    return mapping


def data_dictionary_help_text() -> str:
    return (
        "Data dictionary / expected columns:\n"
        "The app auto-detects column names from csv tables using common conventions.\n"
        "- Words/IA: tries `participant_id`/`subject_id`, `unique_trial_id`/`trial_id`/`unique_paragraph_id`, "
        "`IA_ID`/`word_id`, optional `IA_LABEL`/`text`, paragraph ids, and bounding boxes via either "
        "`(x, y, width, height)` or `(IA_LEFT, IA_RIGHT, IA_TOP, IA_BOTTOM)`.\n"
        "- Fixations: tries `participant_id`/`subject_id`, `unique_trial_id`/`trial_id`/`unique_paragraph_id`, "
        "`CURRENT_FIX_DURATION`, `CURRENT_FIX_X`/`CURRENT_FIX_Y`, and optionally `CURRENT_FIX_START`, "
        "`IA_ID`, `pass_index`/`reread`, `saccade_type`, `eye`, `noise_flag`.\n"
        "- Raw gaze (optional): millisecond-level data with `participant_id`, `trial_id`, `x`, `y`. "
        "Each row represents one timepoint.\n"
        "If your columns are named differently, after uploading expand the "
        "*Column mapping* sections in the sidebar to map each field to your column.\n"
        "No single column uniquely identifies a trial? Map *Trial ID* to several "
        "columns (e.g. participant + paragraph + repeated reading) and the app "
        "builds a combined unique trial ID on the fly.\n"
        "Multi-file datasets: upload several files at once (e.g. one per "
        "participant or text) and they're concatenated, each row tagged by its "
        "`source_file`.\n"
        "Single-report datasets: upload only a words/IA table OR only a "
        "fixations table — the missing layer is skipped. A words-only table "
        "still draws a heatmap from its pre-aggregated reading measures.\n"
        "Stimulus-level word boxes (no participant column) are shared across "
        "every participant who read that trial; fixations with a word/AoI id "
        "but no x/y are placed at the matching word-box centers.\n"
        "Only fields present in your data are used for filters, coloring, and tooltips.\n"
        "Areas of interest (word boxes) are taken from your data, not computed; "
        "fixations are tied to words by bounding-box containment with a small "
        "nearest-word fallback, and fixations outside every box are flagged out-of-text."
    )


def sidebar_controls(
    trial_fixations: pd.DataFrame, base_font_size: int, has_raw_gaze: bool = False
) -> Dict:
    viz = st.sidebar.expander("Visualization controls", expanded=True)
    show_words = viz.checkbox("Bounding boxes", value=True, key="global_show_words")
    show_labels = viz.checkbox("Text", value=True, key="global_show_labels")
    show_fix = viz.checkbox("Fixations", value=True, key="global_show_fix")
    show_order = viz.checkbox("Fixation index", value=True, key="global_show_order")
    show_saccades = viz.checkbox("Saccades", value=True, key="global_show_saccades")
    show_saccade_arrows = viz.checkbox(
        "Saccade direction arrows",
        value=False,
        key="global_show_saccade_arrows",
        help="Draw an arrowhead on each saccade pointing in the gaze direction.",
    )
    show_heatmap = viz.checkbox("Heatmap", value=True, key="global_show_heatmap")
    heatmap_style = viz.radio(
        "Heatmap style",
        options=["Word boxes", "Interpolated"],
        index=0,
        horizontal=True,
        key="global_heatmap_style",
        help=(
            "Word boxes: tint each word box by fixation count / duration. "
            "Interpolated: a smooth Gaussian density over the fixations "
            "themselves, independent of the word boxes (classic eye-movement "
            "heatmap)."
        ),
    )
    show_raw_gaze = viz.checkbox(
        "Raw gaze data",
        value=False,
        help="Display millisecond-level gaze positions as small dots. "
        + ("" if has_raw_gaze else "(No raw gaze data loaded)"),
        disabled=not has_raw_gaze,
        key="global_show_raw_gaze",
    )
    critical_span_style = viz.radio(
        "Text Highlighting",
        options=["Mark text", "Mark border", "None"],
        index=0,
        horizontal=True,
        key="global_critical_span_style",
        help=(
            "Mark text: color the critical-span words in dark pink. "
            "Mark border: draw a thin black outline around the span. "
            "None: don't mark the critical span."
        ),
    )
    color_by_line = viz.checkbox(
        "Color fixations by line",
        value=False,
        key="global_color_by_line",
        help=(
            "Tint each fixation by the text line it lands on (lines inferred "
            "from word positions). Overrides 'Color fixations by'."
        ),
    )
    highlight_out_of_text = viz.checkbox(
        "Mark out-of-text fixations",
        value=False,
        key="global_highlight_out_of_text",
        help="Draw a red ✕ on fixations that fall outside every word box.",
    )

    # Plot background. White by default; some analyses prefer a neutral gray.
    bg_options = list(BACKGROUND_PRESETS.keys()) + ["Custom…"]
    bg_choice = viz.selectbox(
        "Plot background",
        options=bg_options,
        index=0,
        key="global_bg_choice",
        help="Background of the plotting area (and exported figures).",
    )
    if bg_choice == "Custom…":
        background_color = viz.color_picker(
            "Custom background color",
            value=DEFAULT_BACKGROUND_COLOR,
            key="global_bg_custom",
        )
    else:
        background_color = BACKGROUND_PRESETS[bg_choice]

    preferred_color_fields = [
        "duration_ms",
        "pass_index",
        "eye",
        "saccade_type",
        "saccade_amplitude",
        "word_id",
        "timestamp_ms",
        "is_regression",
        "progression",
        "gpt2_surprisal",
        "wordfreq_frequency",
        "subtlex_frequency",
        "universal_pos",
        "ptb_pos",
    ]
    color_fields = [f for f in preferred_color_fields if f in trial_fixations.columns]
    if not color_fields:
        color_fields = ["duration_ms"]
    color_by = viz.selectbox(
        "Color fixations by",
        options=color_fields,
        index=color_fields.index("duration_ms") if "duration_ms" in color_fields else 0,
    )
    heatmap_metric = viz.selectbox(
        "Heatmap metric",
        options=["duration_ms", "counts"],
        help="Heatmap can be raw counts or weighted by fixation duration.",
        index=0,
    )

    numeric_fields = [
        col
        for col in trial_fixations.columns
        if pd.api.types.is_numeric_dtype(trial_fixations[col])
    ]
    if not numeric_fields:
        st.error("No numeric fields found in fixations to map axes.")
        st.stop()
    x_default = "x" if "x" in numeric_fields else numeric_fields[0]
    y_default = (
        "y"
        if "y" in numeric_fields
        else numeric_fields[min(1, len(numeric_fields) - 1)]
    )
    x_field = viz.selectbox(
        "X axis field", options=numeric_fields, index=numeric_fields.index(x_default)
    )
    y_field = viz.selectbox(
        "Y axis field", options=numeric_fields, index=numeric_fields.index(y_default)
    )

    order_font_color = "#111111"
    order_font_size = int(base_font_size)
    size_min, size_max = 8, 24
    show_colorbars = False
    fixation_color_range = None
    heatmap_range = None
    fixation_colorscale = "Blues"
    heatmap_colorscale = "Oranges"
    # Collapsible panel (like "Filter trials"). Its widgets always render, so
    # deep-linked colorscale presets apply even while collapsed; the URL handler
    # sets `global_advanced` to auto-open it (see app._apply_url_preset).
    advanced_open = bool(st.session_state.get("global_advanced", False))
    with st.sidebar.expander("Advanced styling", expanded=advanced_open):
        from .constants import (
            COLORSCALES,
            DEFAULT_FIXATION_COLORSCALE,
            DEFAULT_HEATMAP_COLORSCALE,
        )

        order_font_color = st.color_picker("Order label color", value="#111111")
        order_font_size = st.slider("Order label size", 6, 72, int(base_font_size))
        size_min, size_max = st.slider("Fixation marker size (px)", 4, 40, (8, 24))
        fixation_colorscale = st.selectbox(
            "Fixation colorscale",
            options=COLORSCALES,
            index=COLORSCALES.index(DEFAULT_FIXATION_COLORSCALE),
            help="Color palette for fixation markers when coloring by numeric values.",
            key="global_fixation_colorscale",
        )
        heatmap_colorscale = st.selectbox(
            "Heatmap colorscale",
            options=COLORSCALES,
            index=COLORSCALES.index(DEFAULT_HEATMAP_COLORSCALE),
            help="Color palette for the density heatmap overlay.",
            key="global_heatmap_colorscale",
        )
        show_colorbars = st.checkbox("Color bars", value=False)
        if show_colorbars and pd.api.types.is_numeric_dtype(trial_fixations[color_by]):
            cmin = float(trial_fixations[color_by].min())
            cmax = float(trial_fixations[color_by].max())
            fixation_color_range = st.slider(
                "Fixation color range",
                min_value=cmin,
                max_value=cmax if cmax > cmin else cmin + 1.0,
                value=(cmin, cmax if cmax > cmin else cmin + 1.0),
                step=(cmax - cmin) / 100 if cmax > cmin else 1.0,
            )
        if show_colorbars and show_heatmap:
            heat_data = (
                trial_fixations["duration_ms"]
                if heatmap_metric == "duration_ms"
                else None
            )
            if heat_data is not None and len(heat_data) > 0:
                hmin = float(heat_data.min())
                hmax = float(heat_data.max())
                heatmap_range = st.slider(
                    "Heatmap color range",
                    min_value=hmin,
                    max_value=hmax if hmax > hmin else hmin + 1.0,
                    value=(hmin, hmax if hmax > hmin else hmin + 1.0),
                    step=(hmax - hmin) / 100 if hmax > hmin else 1.0,
                )

    return dict(
        show_words=show_words,
        show_labels=show_labels,
        show_fix=show_fix,
        show_order=show_order,
        show_saccades=show_saccades,
        show_saccade_arrows=show_saccade_arrows,
        show_heatmap=show_heatmap,
        heatmap_style=heatmap_style,
        show_raw_gaze=show_raw_gaze,
        color_by=color_by,
        heatmap_metric=heatmap_metric,
        x_field=x_field,
        y_field=y_field,
        marker_size_range=(size_min, size_max),
        order_font_size=order_font_size,
        order_font_color=order_font_color,
        show_colorbars=show_colorbars,
        fixation_color_range=fixation_color_range,
        heatmap_range=heatmap_range,
        fixation_colorscale=fixation_colorscale,
        heatmap_colorscale=heatmap_colorscale,
        critical_span_style=critical_span_style,
        color_by_line=color_by_line,
        highlight_out_of_text=highlight_out_of_text,
        background_color=background_color,
    )


def _bool_metadata_filter(
    label: str,
    col: str,
    df: pd.DataFrame,
    true_label: str,
    false_label: str,
    key: str,
) -> Optional[set]:
    """Friendly multiselect for a boolean metadata column.

    Returns the set of raw bool values to keep, or None when the column is
    absent, has fewer than two classes, or the user kept everything (no
    narrowing)."""
    if col not in df.columns:
        return None
    present = set(pd.Series(df[col]).dropna().astype(bool).unique())
    label_to_val = {true_label: True, false_label: False}
    options = [lbl for lbl, val in label_to_val.items() if val in present]
    if len(options) < 2:
        return None
    chosen = st.multiselect(label, options=options, default=options, key=key)
    if not chosen or set(chosen) == set(options):
        return None
    return {label_to_val[c] for c in chosen}


def sidebar_trial_filters(words: pd.DataFrame, fixations: pd.DataFrame) -> Dict:
    """Render the 'Filter trials' sidebar panel and return the selections.

    Lets the user narrow the trial pool by participant and by categorical
    condition (Hunting/Gathering, difficulty, first/repeated reading,
    correctness) as well as by annotation state (favorites / tags). Only
    *narrowing* selections are returned; an untouched field is omitted so
    downstream filtering is a no-op for it.
    """
    result: Dict = {
        "participants": None,
        "metadata": {},
        "favorites_only": False,
        "required_tags": [],
        "excluded_tags": [],
    }
    with st.sidebar.expander("Filter trials", expanded=False):
        st.caption("Narrow the trial pool shown in every tab.")

        # Union across both frames — single-report datasets have participants
        # in only one of them.
        parts = sorted(
            set(words["participant_id"].dropna().astype(str))
            | set(fixations["participant_id"].dropna().astype(str))
        )
        if len(parts) > 1:
            chosen = st.multiselect(
                "Participants", options=parts, default=parts, key="filter_participants"
            )
            if chosen and len(chosen) < len(parts):
                result["participants"] = chosen

        # Hunting (question previewed) vs Gathering (ordinary reading).
        regime = _bool_metadata_filter(
            "Reading regime",
            "question_preview",
            words,
            "Hunting",
            "Gathering",
            "filter_regime",
        )
        if regime is not None:
            result["metadata"]["question_preview"] = regime

        if "difficulty_level" in words.columns:
            diffs = sorted(words["difficulty_level"].dropna().astype(str).unique())
            if len(diffs) > 1:
                chosen = st.multiselect(
                    "Difficulty", options=diffs, default=diffs, key="filter_difficulty"
                )
                if chosen and len(chosen) < len(diffs):
                    result["metadata"]["difficulty_level"] = set(chosen)

        repeat = _bool_metadata_filter(
            "Reading number",
            "repeated_reading_trial",
            words,
            "Repeated",
            "First",
            "filter_repeat",
        )
        if repeat is not None:
            result["metadata"]["repeated_reading_trial"] = repeat

        correct = _bool_metadata_filter(
            "Answer",
            "is_correct",
            words,
            "Correct",
            "Incorrect",
            "filter_correct",
        )
        if correct is not None:
            result["metadata"]["is_correct"] = correct

        st.markdown("**By annotation**")
        result["favorites_only"] = st.checkbox(
            "⭐ Favorites only", value=False, key="filter_favorites"
        )
        tags = known_tags()
        if tags:
            result["required_tags"] = st.multiselect(
                "With any of these tags",
                options=tags,
                default=[],
                key="filter_req_tags",
            )
            result["excluded_tags"] = st.multiselect(
                "Excluding tags",
                options=tags,
                default=[],
                key="filter_exc_tags",
                help="e.g. hide everything tagged 'To exclude'.",
            )
    return result
