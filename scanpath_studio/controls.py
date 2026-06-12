from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from .annotations import known_tags
from .constants import (
    BACKGROUND_PRESETS,
    COLORSCALES,
    DEFAULT_BACKGROUND_COLOR,
    DEFAULT_FIXATION_COLORSCALE,
    DEFAULT_HEATMAP_COLORSCALE,
)

NONE_OPTION = "(none)"

# Static defaults for the keyed visualization widgets that the plot-config
# restore (app._restore_plot_config) can set. Seeded into session_state so those
# widgets render WITHOUT a `value=`/`index=` argument — that keeps their key
# programmatically settable without Streamlit's "default value but also set via
# Session State API" warning. Data-dependent defaults (color-by / axis fields /
# sizing / canvas) are seeded locally where they're computed.
_VIZ_WIDGET_DEFAULTS = {
    "global_show_words": True,
    "global_show_labels": True,
    "global_show_fix": True,
    "global_show_order": True,
    "global_show_saccades": True,
    "global_show_saccade_arrows": False,
    "global_show_heatmap": True,
    "global_show_raw_gaze": False,
    "global_heatmap_style": "Word boxes",
    "global_heatmap_metric": "duration_ms",
    "global_show_colorbars": False,
    "global_order_font_color": "#111111",
    "global_fixation_colorscale": DEFAULT_FIXATION_COLORSCALE,
    "global_heatmap_colorscale": DEFAULT_HEATMAP_COLORSCALE,
}


# Help text for the (multi-capable) Trial ID mapping, shared by all tables.
_TRIAL_MAPPING_HELP = (
    "Pick the column holding your unique trial ID — or pick SEVERAL columns "
    "to build one on the fly (values joined with '_'), e.g. participant + "
    "paragraph + repeated-reading when no single column identifies a trial. "
    "Use the same columns for every uploaded table so trials line up."
)

# Word-box geometry is one rectangle in two interchangeable encodings. The
# mapping UI shows a format picker plus four fields instead of all eight at once;
# both encodings normalize to canonical x/y/width/height in
# ``data.normalize_words``, so the returned schema still carries all eight keys.
BOX_FORMAT_EDGES = "Edges"
BOX_FORMAT_ORIGIN = "Origin + size"
_BOX_SUBFIELDS: Dict[str, List[tuple]] = {
    BOX_FORMAT_EDGES: [
        ("left", "Box left"),
        ("right", "Box right"),
        ("top", "Box top"),
        ("bottom", "Box bottom"),
    ],
    BOX_FORMAT_ORIGIN: [
        ("x", "Box x (top-left)"),
        ("y", "Box y (top-left)"),
        ("width", "Box width"),
        ("height", "Box height"),
    ],
}
_ALL_BOX_KEYS = [key for fields in _BOX_SUBFIELDS.values() for key, _ in fields]


def _default_box_format(proposed: Dict[str, Optional[str]]) -> str:
    """Which box encoding to show first, from what auto-detect found.

    Edges if all four edge columns were detected, else origin+size if those four
    were, else edges."""
    if all(proposed.get(k) for k in ("left", "right", "top", "bottom")):
        return BOX_FORMAT_EDGES
    if all(proposed.get(k) for k in ("x", "y", "width", "height")):
        return BOX_FORMAT_ORIGIN
    return BOX_FORMAT_EDGES


WORD_FIELD_SPECS: List[Dict] = [
    {
        "key": "participant",
        "label": "Participant ID",
        "required": False,
        "help": "Optional — omit for stimulus-level word boxes shared across all readers.",
    },
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
        "key": "box",
        "kind": "box",
        "label": "Word box",
        "required": True,
        "help": "Bounding box per word/AOI. Edges = left/right/top/bottom (EyeLink IA_*); origin+size = x/y/width/height.",
    },
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
    {
        "key": "x",
        "label": "X coordinate",
        "required": False,
        "help": "Fixation pixel X. Leave empty for AOI-only data and map "
        "Word/IA ID instead — those fixations are placed at word-box centers.",
    },
    {"key": "y", "label": "Y coordinate", "required": False},
    {"key": "duration", "label": "Duration (ms)", "required": True},
    {"key": "timestamp", "label": "Timestamp (ms)", "required": False},
    {"key": "fixation_id", "label": "Fixation ID", "required": False},
    {"key": "paragraph", "label": "Paragraph ID", "required": False},
    {
        "key": "word_id",
        "label": "Word/IA ID",
        "required": False,
        "help": "Which word/AOI each fixation landed on. Authoritative when "
        "present (overrides geometric assignment), and supplies the location "
        "when X/Y are absent.",
    },
    {"key": "pass_index", "label": "Pass index", "required": False},
    {"key": "saccade_type", "label": "Saccade type", "required": False},
    {
        "key": "saccade_amplitude",
        "label": "Saccade amplitude",
        "required": False,
        "help": "Pixel distance from the previous fixation; computed from X/Y "
        "when not provided.",
    },
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
    A field marked ``kind: "box"`` (the word box) renders a coordinate-format
    radio plus the four sub-fields for that format, and expands into all eight
    box keys (the four inactive ones set to None) so the returned schema keeps
    its fixed shape.
    """
    options = [NONE_OPTION] + list(df.columns)
    expanded = bool(expand_on_problem and problems)

    def _selectbox(field_key: str, field_label: str, help_text=None) -> Optional[str]:
        default = proposed.get(field_key)
        index = options.index(default) if default in options else 0
        chosen = st.selectbox(
            field_label,
            options=options,
            index=index,
            key=f"{state_key_prefix}_{field_key}",
            help=help_text,
        )
        return None if chosen == NONE_OPTION else chosen

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
            if spec.get("kind") == "box":
                fmt_key = f"{state_key_prefix}_box_format"
                if fmt_key not in st.session_state:
                    # Seed via session state (no `index=`) so it survives reruns
                    # and never fights a default arg — same pattern as the
                    # multiselect below.
                    st.session_state[fmt_key] = _default_box_format(proposed)
                star = " \\*" if spec.get("required") else ""
                st.markdown(f"**{spec['label']}**{star}")
                if spec.get("help"):
                    st.caption(spec["help"])
                fmt = st.radio(
                    "Coordinate format",
                    options=list(_BOX_SUBFIELDS),
                    key=fmt_key,
                    horizontal=True,
                    label_visibility="collapsed",
                )
                # Always emit all eight box keys; only the active format's four
                # get a column, the rest stay None.
                mapping.update({box_key: None for box_key in _ALL_BOX_KEYS})
                for sub_key, sub_label in _BOX_SUBFIELDS[fmt]:
                    mapping[sub_key] = _selectbox(sub_key, sub_label)
                continue
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
            mapping[key] = _selectbox(key, label, spec.get("help"))
    return mapping


def data_dictionary_help_text() -> str:
    return (
        "Data dictionary / expected columns:\n"
        "The app auto-detects column names from csv tables using common conventions.\n"
        "- Words/IA: tries `participant_id`/`subject_id`, `unique_trial_id`/`trial_id`/`unique_paragraph_id`, "
        "`IA_ID`/`word_id`, optional `IA_LABEL`/`text`, paragraph ids, and bounding boxes via either "
        "edges `(IA_LEFT, IA_RIGHT, IA_TOP, IA_BOTTOM)` or origin+size `(x, y, width, height)` — "
        "pick which in the *Word box* selector.\n"
        "- Fixations: tries `participant_id`/`subject_id`, `unique_trial_id`/`trial_id`/`unique_paragraph_id`, "
        "`CURRENT_FIX_DURATION`, `CURRENT_FIX_X`/`CURRENT_FIX_Y`, and optionally `CURRENT_FIX_START`, "
        "`IA_ID`, `pass_index`/`reread`, `saccade_type`, `saccade_amplitude`, "
        "`eye`, `noise_flag`. X/Y are optional when `IA_ID` is mapped "
        "(AOI-only fixations are placed at word-box centers).\n"
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


# Field-option helpers — shared by the sidebar selectors and the plot-config
# restore path (`app._restore_plot_config`) so both agree on what's valid for
# the current data.
def color_field_options(trial_fixations: pd.DataFrame) -> List[str]:
    """Columns offered in the 'Color fixations by' selector — a preferred order
    intersected with what's present, falling back to ``['duration_ms']``."""
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
    fields = [f for f in preferred_color_fields if f in trial_fixations.columns]
    return fields or ["duration_ms"]


def numeric_field_options(trial_fixations: pd.DataFrame) -> List[str]:
    """Numeric columns offered as X/Y axis fields."""
    return [
        col
        for col in trial_fixations.columns
        if pd.api.types.is_numeric_dtype(trial_fixations[col])
    ]


def _drop_stale(state_key: str, options: list) -> None:
    """Clear a persisted selectbox value that isn't valid for the current
    ``options`` (e.g. after switching datasets, or restoring a config built on
    different data) so ``st.selectbox`` falls back to its ``index=`` default
    instead of raising. Mirrors the guard in ``utils._select_trial_composite_mode``."""
    if state_key in st.session_state and st.session_state[state_key] not in options:
        del st.session_state[state_key]


def _clamp_range(state_key: str, lo: float, hi: float) -> None:
    """Clamp a persisted ``(min, max)`` range-slider value into ``[lo, hi]`` so a
    restored value built on different data can't fall outside the slider bounds
    and raise. Drops anything that isn't a 2-tuple."""
    val = st.session_state.get(state_key)
    if not (isinstance(val, (list, tuple)) and len(val) == 2):
        st.session_state.pop(state_key, None)
        return
    try:
        a, b = float(val[0]), float(val[1])
    except (TypeError, ValueError):
        del st.session_state[state_key]
        return
    a, b = max(lo, min(a, hi)), max(lo, min(b, hi))
    st.session_state[state_key] = (min(a, b), max(a, b))


def sidebar_controls(
    trial_fixations: pd.DataFrame, base_font_size: int, has_raw_gaze: bool = False
) -> Dict:
    # Seed defaults for the widgets that the plot-config restore can set
    # (app._restore_plot_config). These render WITHOUT a `value=`/`index=`
    # argument so their key can be assigned programmatically without tripping
    # Streamlit's "default value but also set via Session State API" warning;
    # the default lives here instead. Widgets the restore never touches keep
    # their inline `value=`. `setdefault` leaves any URL-preset / restored value
    # in place.
    for _key, _default in _VIZ_WIDGET_DEFAULTS.items():
        st.session_state.setdefault(_key, _default)

    # Keyed wrapper → stable `.st-key-…` selector for the spotlight tour.
    viz = st.sidebar.container(key="tour_grp_viz_controls").expander(
        "Visualization controls", expanded=True
    )
    show_words = viz.checkbox("Bounding boxes", key="global_show_words")
    show_labels = viz.checkbox("Text", key="global_show_labels")
    show_fix = viz.checkbox("Fixations", key="global_show_fix")
    show_order = viz.checkbox("Fixation index", key="global_show_order")
    show_saccades = viz.checkbox("Saccades", key="global_show_saccades")
    show_saccade_arrows = viz.checkbox(
        "Saccade direction arrows",
        key="global_show_saccade_arrows",
        help="Draw an arrowhead on each saccade pointing in the gaze direction.",
    )
    show_heatmap = viz.checkbox("Heatmap", key="global_show_heatmap")
    heatmap_style = viz.radio(
        "Heatmap style",
        options=["Word boxes", "Interpolated"],
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

    color_fields = color_field_options(trial_fixations)
    _drop_stale("global_color_by", color_fields)
    st.session_state.setdefault(
        "global_color_by",
        "duration_ms" if "duration_ms" in color_fields else color_fields[0],
    )
    color_by = viz.selectbox(
        "Color fixations by",
        options=color_fields,
        key="global_color_by",
    )
    heatmap_metric = viz.selectbox(
        "Heatmap metric",
        options=["duration_ms", "counts"],
        help="Heatmap can be raw counts or weighted by fixation duration.",
        key="global_heatmap_metric",
    )

    numeric_fields = numeric_field_options(trial_fixations)
    if not numeric_fields:
        st.error("No numeric fields found in fixations to map axes.")
        st.stop()
    x_default = "x" if "x" in numeric_fields else numeric_fields[0]
    y_default = (
        "y"
        if "y" in numeric_fields
        else numeric_fields[min(1, len(numeric_fields) - 1)]
    )
    _drop_stale("global_x_field", numeric_fields)
    st.session_state.setdefault("global_x_field", x_default)
    x_field = viz.selectbox(
        "X axis field",
        options=numeric_fields,
        key="global_x_field",
    )
    _drop_stale("global_y_field", numeric_fields)
    st.session_state.setdefault("global_y_field", y_default)
    y_field = viz.selectbox(
        "Y axis field",
        options=numeric_fields,
        key="global_y_field",
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
    st.session_state.setdefault("global_order_font_size", int(base_font_size))
    st.session_state.setdefault("global_marker_size_range", (8, 24))
    with st.sidebar.expander("Advanced styling", expanded=advanced_open):
        order_font_color = st.color_picker(
            "Order label color", key="global_order_font_color"
        )
        order_font_size = st.slider(
            "Order label size", 6, 72, key="global_order_font_size"
        )
        size_min, size_max = st.slider(
            "Fixation marker size (px)", 4, 40, key="global_marker_size_range"
        )
        fixation_colorscale = st.selectbox(
            "Fixation colorscale",
            options=COLORSCALES,
            help="Color palette for fixation markers when coloring by numeric values.",
            key="global_fixation_colorscale",
        )
        heatmap_colorscale = st.selectbox(
            "Heatmap colorscale",
            options=COLORSCALES,
            help="Color palette for the density heatmap overlay.",
            key="global_heatmap_colorscale",
        )
        show_colorbars = st.checkbox("Color bars", key="global_show_colorbars")
        if show_colorbars and pd.api.types.is_numeric_dtype(trial_fixations[color_by]):
            cmin = float(trial_fixations[color_by].min())
            cmax = float(trial_fixations[color_by].max())
            cmax_eff = cmax if cmax > cmin else cmin + 1.0
            _clamp_range("global_fixation_color_range", cmin, cmax_eff)
            st.session_state.setdefault("global_fixation_color_range", (cmin, cmax_eff))
            fixation_color_range = st.slider(
                "Fixation color range",
                min_value=cmin,
                max_value=cmax_eff,
                step=(cmax - cmin) / 100 if cmax > cmin else 1.0,
                key="global_fixation_color_range",
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
                hmax_eff = hmax if hmax > hmin else hmin + 1.0
                _clamp_range("global_heatmap_color_range", hmin, hmax_eff)
                st.session_state.setdefault(
                    "global_heatmap_color_range", (hmin, hmax_eff)
                )
                heatmap_range = st.slider(
                    "Heatmap color range",
                    min_value=hmin,
                    max_value=hmax_eff,
                    step=(hmax - hmin) / 100 if hmax > hmin else 1.0,
                    key="global_heatmap_color_range",
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
    # Keyed wrapper → stable `.st-key-…` selector for the spotlight tour.
    with st.sidebar.container(key="tour_grp_filter_trials").expander(
        "Filter trials", expanded=False
    ):
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
