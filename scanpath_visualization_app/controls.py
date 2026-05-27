from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

NONE_OPTION = "(none)"


WORD_FIELD_SPECS: List[Dict] = [
    {"key": "participant", "label": "Participant ID", "required": True},
    {"key": "trial", "label": "Trial ID", "required": True},
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
    {"key": "trial", "label": "Trial ID", "required": True},
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
    {"key": "trial", "label": "Trial ID", "required": True},
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

    Returns a mapping {field_key: column_name_or_None} based on user selections.
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
        "Only fields present in your data are used for filters, coloring, and tooltips."
    )


def sidebar_controls(
    trial_fixations: pd.DataFrame, base_font_size: int, has_raw_gaze: bool = False
) -> Dict:
    st.sidebar.header("Visualization controls")
    show_words = st.sidebar.checkbox("Show word boxes", value=True,
                                     key="global_show_words")
    show_labels = st.sidebar.checkbox("Show word labels", value=True,
                                      key="global_show_labels")
    show_fix = st.sidebar.checkbox("Show fixations", value=True,
                                   key="global_show_fix")
    show_order = st.sidebar.checkbox("Number fixation order", value=True,
                                     key="global_show_order")
    show_saccades = st.sidebar.checkbox("Show saccades", value=True,
                                        key="global_show_saccades")
    show_heatmap = st.sidebar.checkbox("Add density heatmap", value=True,
                                       key="global_show_heatmap")
    show_raw_gaze = st.sidebar.checkbox(
        "Show raw gaze data",
        value=False,
        help="Display millisecond-level gaze positions as small dots. "
        + ("" if has_raw_gaze else "(No raw gaze data loaded)"),
        disabled=not has_raw_gaze,
        key="global_show_raw_gaze",
    )

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
    color_by = st.sidebar.selectbox(
        "Color fixations by",
        options=color_fields,
        index=color_fields.index("duration_ms") if "duration_ms" in color_fields else 0,
    )
    heatmap_metric = st.sidebar.selectbox(
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
    x_field = st.sidebar.selectbox(
        "X axis field", options=numeric_fields, index=numeric_fields.index(x_default)
    )
    y_field = st.sidebar.selectbox(
        "Y axis field", options=numeric_fields, index=numeric_fields.index(y_default)
    )

    st.sidebar.subheader("Advanced styling")
    advanced = st.sidebar.checkbox("Advanced styling", value=False,
                                   key="global_advanced")
    order_font_color = "#111111"
    order_font_size = int(base_font_size)
    size_min, size_max = 8, 24
    show_colorbars = False
    fixation_color_range = None
    heatmap_range = None
    fixation_colorscale = "Blues"
    heatmap_colorscale = "Oranges"
    if advanced:
        from .constants import (
            COLORSCALES,
            DEFAULT_FIXATION_COLORSCALE,
            DEFAULT_HEATMAP_COLORSCALE,
        )

        order_font_color = st.sidebar.color_picker("Order label color", value="#111111")
        order_font_size = st.sidebar.slider(
            "Order label size", 6, 72, int(base_font_size)
        )
        size_min, size_max = st.sidebar.slider(
            "Fixation marker size (px)", 4, 40, (8, 24)
        )
        fixation_colorscale = st.sidebar.selectbox(
            "Fixation colorscale",
            options=COLORSCALES,
            index=COLORSCALES.index(DEFAULT_FIXATION_COLORSCALE),
            help="Color palette for fixation markers when coloring by numeric values.",
            key="global_fixation_colorscale",
        )
        heatmap_colorscale = st.sidebar.selectbox(
            "Heatmap colorscale",
            options=COLORSCALES,
            index=COLORSCALES.index(DEFAULT_HEATMAP_COLORSCALE),
            help="Color palette for the density heatmap overlay.",
            key="global_heatmap_colorscale",
        )
        show_colorbars = st.sidebar.checkbox("Show color bars", value=False)
        if show_colorbars and pd.api.types.is_numeric_dtype(trial_fixations[color_by]):
            cmin = float(trial_fixations[color_by].min())
            cmax = float(trial_fixations[color_by].max())
            fixation_color_range = st.sidebar.slider(
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
                heatmap_range = st.sidebar.slider(
                    "Heatmap color range",
                    min_value=hmin,
                    max_value=hmax if hmax > hmin else hmin + 1.0,
                    value=(hmin, hmax if hmax > hmin else hmin + 1.0),
                    step=(hmax - hmin) / 100 if hmax > hmin else 1.0,
                )
    else:
        show_colorbars = False

    return dict(
        show_words=show_words,
        show_labels=show_labels,
        show_fix=show_fix,
        show_order=show_order,
        show_saccades=show_saccades,
        show_heatmap=show_heatmap,
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
    )
