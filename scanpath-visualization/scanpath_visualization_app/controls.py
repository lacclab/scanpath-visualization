from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st


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
        "Only fields present in your data are used for filters, coloring, and tooltips."
    )


def render_dictionary() -> None:
    with st.expander("Data dictionary / expected columns"):
        st.markdown(data_dictionary_help_text())


def sidebar_controls(trial_fixations: pd.DataFrame, base_font_size: int, has_raw_gaze: bool = False) -> Dict:
    st.sidebar.header("Visualization controls")
    show_words = st.sidebar.checkbox("Show word boxes", value=True)
    show_labels = st.sidebar.checkbox("Show word labels", value=True)
    show_fix = st.sidebar.checkbox("Show fixations", value=True)
    show_order = st.sidebar.checkbox("Number fixation order", value=True)
    show_saccades = st.sidebar.checkbox("Show saccades", value=True)
    show_heatmap = st.sidebar.checkbox("Add density heatmap", value=True)
    show_raw_gaze = st.sidebar.checkbox(
        "Show raw gaze data",
        value=False,
        help="Display millisecond-level gaze positions as small dots.",
        disabled=not has_raw_gaze,
    ) if has_raw_gaze else False

    color_fields = [
        field
        for field in ["duration_ms", "pass_index", "eye", "saccade_type", "word_id", "timestamp_ms"]
        if field in trial_fixations.columns
    ]
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

    numeric_fields = [col for col in trial_fixations.columns if pd.api.types.is_numeric_dtype(trial_fixations[col])]
    if not numeric_fields:
        st.error("No numeric fields found in fixations to map axes.")
        st.stop()
    x_default = "x" if "x" in numeric_fields else numeric_fields[0]
    y_default = "y" if "y" in numeric_fields else numeric_fields[min(1, len(numeric_fields) - 1)]
    x_field = st.sidebar.selectbox("X axis field", options=numeric_fields, index=numeric_fields.index(x_default))
    y_field = st.sidebar.selectbox("Y axis field", options=numeric_fields, index=numeric_fields.index(y_default))

    st.sidebar.subheader("Advanced styling")
    advanced = st.sidebar.checkbox("Advanced styling", value=False)
    order_font_color = "#111111"
    order_font_size = int(base_font_size)
    size_min, size_max = 8, 24
    show_colorbars = False
    fixation_color_range = None
    heatmap_range = None
    fixation_colorscale = "Blues"
    heatmap_colorscale = "Oranges"
    if advanced:
        from .constants import COLORSCALES, DEFAULT_FIXATION_COLORSCALE, DEFAULT_HEATMAP_COLORSCALE
        order_font_color = st.sidebar.color_picker("Order label color", value="#111111")
        order_font_size = st.sidebar.slider("Order label size", 6, 72, int(base_font_size))
        size_min, size_max = st.sidebar.slider("Fixation marker size (px)", 4, 40, (8, 24))
        fixation_colorscale = st.sidebar.selectbox(
            "Fixation colorscale",
            options=COLORSCALES,
            index=COLORSCALES.index(DEFAULT_FIXATION_COLORSCALE),
            help="Color palette for fixation markers when coloring by numeric values.",
        )
        heatmap_colorscale = st.sidebar.selectbox(
            "Heatmap colorscale",
            options=COLORSCALES,
            index=COLORSCALES.index(DEFAULT_HEATMAP_COLORSCALE),
            help="Color palette for the density heatmap overlay.",
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
            heat_data = trial_fixations["duration_ms"] if heatmap_metric == "duration_ms" else None
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
