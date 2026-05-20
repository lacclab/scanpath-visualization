"""Tab rendering functions for the Scanpath Visualization app."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from scanpath_visualization_app.plots import (
    make_comparison_figure,
    make_scanpath_animation,
    make_scanpath_figure,
)
from scanpath_visualization_app.utils import (
    build_comparison_options,
    compute_trial_stats,
    export_filtered_trials,
    friendly_trial_label,
    gather_trial_metadata,
    safe_summary,
    select_trial,
)


# -----------------------------------------------------------------------------
# Single Trial Tab
# -----------------------------------------------------------------------------


def _build_figure_settings(viz_settings: dict, effective_show_raw_gaze: bool) -> dict:
    """Convert viz_settings to figure-compatible settings dict."""
    return dict(
        show_words=viz_settings["show_words"],
        show_word_labels=viz_settings["show_labels"],
        show_fixations=viz_settings["show_fix"],
        show_order=viz_settings["show_order"],
        show_saccades=viz_settings["show_saccades"],
        show_heatmap=viz_settings["show_heatmap"],
        show_raw_gaze=effective_show_raw_gaze,
        color_by=viz_settings["color_by"],
        heatmap_metric=(
            viz_settings["heatmap_metric"]
            if viz_settings["heatmap_metric"] != "counts"
            else None
        ),
        marker_size_range=viz_settings["marker_size_range"],
        order_font_size=viz_settings["order_font_size"],
        order_font_color=viz_settings["order_font_color"],
        show_colorbars=viz_settings["show_colorbars"],
        fixation_color_range=viz_settings["fixation_color_range"],
        heatmap_range=viz_settings["heatmap_range"],
        fixation_colorscale=viz_settings["fixation_colorscale"],
        heatmap_colorscale=viz_settings["heatmap_colorscale"],
    )


def _render_comparison_controls(
    combos: pd.DataFrame,
    selection_mode: str,
    selected_participant: str,
    selected_trial: str,
    selected_text: Optional[str],
) -> tuple[Optional[str], Optional[str], str]:
    """Render comparison toggle and trial selector, return (participant, trial, layout)."""
    compare_enabled = st.checkbox(
        "Compare with another trial",
        value=False,
        key="single_compare_toggle",
        help="Overlay another trial's scanpath or view them side by side.",
    )

    if not compare_enabled:
        return None, None, "overlay"

    comparison_options = build_comparison_options(
        combos, selection_mode, selected_participant, selected_trial, selected_text
    )

    if not comparison_options:
        st.info("No other trials available for comparison.")
        return None, None, "overlay"

    option_labels = [opt[2] for opt in comparison_options]
    label_to_trial = {opt[2]: (opt[0], opt[1]) for opt in comparison_options}

    col_trial, col_layout = st.columns([2, 1])
    with col_trial:
        selected_compare_label = st.selectbox(
            "Compare with trial",
            options=option_labels,
            key="single_compare_trial",
            help="★ indicates same text as primary trial",
        )
    with col_layout:
        layout_label = st.radio(
            "View",
            options=["Overlay", "Side by side"],
            index=0,
            key="single_compare_layout",
            horizontal=True,
        )
    layout = "side_by_side" if layout_label == "Side by side" else "overlay"

    if selected_compare_label:
        participant, trial = label_to_trial[selected_compare_label]
        return participant, trial, layout
    return None, None, layout


def _render_trial_stats(trial_words: pd.DataFrame, trial_fixations: pd.DataFrame):
    """Render trial statistics metrics."""
    stats = compute_trial_stats(trial_words, trial_fixations)
    stat_cols = st.columns(3)
    stat_cols[0].metric("Total reading time (s)", f"{stats['total_reading_time_s']:.1f}")
    stat_cols[1].metric("Number of words", f"{stats['word_count']:,}")
    stat_cols[2].metric("Number of fixations", f"{stats['fixation_count']:,}")


def _render_metadata_selector(
    words_filtered: pd.DataFrame, fixations_filtered: pd.DataFrame,
    trial_words: pd.DataFrame, trial_fixations: pd.DataFrame
):
    """Render metadata field selector and display table."""
    metadata_candidates = []
    for col in list(words_filtered.columns) + list(fixations_filtered.columns):
        if col not in metadata_candidates:
            metadata_candidates.append(col)

    default_metadata = [
        field
        for field in [
            "difficulty_level",
            "repeated_reading_trial",
            "selected_answer",
            "is_correct",
        ]
        if field in metadata_candidates
    ]
    selected_metadata = st.multiselect(
        "Trial metadata fields",
        options=metadata_candidates,
        default=default_metadata or metadata_candidates,
    )
    if selected_metadata:
        metadata_df = gather_trial_metadata(trial_words, trial_fixations, selected_metadata)
        if not metadata_df.empty:
            st.dataframe(metadata_df, hide_index=True, width="stretch")


def _render_plot_config_expander(
    selected_participant: str,
    selected_trial: str,
    canvas_width: int,
    canvas_height: int,
    x_field: str,
    y_field: str,
    figure_settings: dict,
    viz_settings: dict,
    base_font_size: int,
    trial_raw_gaze: pd.DataFrame,
):
    """Render the plot configuration expander."""
    with st.expander("Plot configuration"):
        plot_config = {
            "selection": {
                "participant_id": selected_participant,
                "trial_id": selected_trial,
            },
            "canvas_px": {"width": int(canvas_width), "height": int(canvas_height)},
            "axes": {"x_field": x_field, "y_field": y_field},
            "layers": {
                "words": figure_settings["show_words"],
                "word_labels": figure_settings["show_word_labels"],
                "fixations": figure_settings["show_fixations"],
                "order_labels": figure_settings["show_order"],
                "saccades": figure_settings["show_saccades"],
                "heatmap": figure_settings["show_heatmap"],
                "raw_gaze": figure_settings["show_raw_gaze"],
            },
            "coloring": {
                "color_by": figure_settings["color_by"],
                "heatmap_metric": viz_settings["heatmap_metric"],
                "show_colorbars": figure_settings["show_colorbars"],
                "fixation_range": (
                    list(figure_settings["fixation_color_range"])
                    if figure_settings["fixation_color_range"]
                    else None
                ),
                "heatmap_range": (
                    list(figure_settings["heatmap_range"])
                    if figure_settings["heatmap_range"]
                    else None
                ),
                "fixation_colorscale": figure_settings["fixation_colorscale"],
                "heatmap_colorscale": figure_settings["heatmap_colorscale"],
            },
            "sizing": {
                "marker_size_range": [int(s) for s in figure_settings["marker_size_range"]],
                "order_font_size": int(figure_settings["order_font_size"]),
                "order_font_color": figure_settings["order_font_color"],
                "base_font_size": int(base_font_size),
            },
            "raw_gaze": {
                "available": not trial_raw_gaze.empty,
                "points": len(trial_raw_gaze) if not trial_raw_gaze.empty else 0,
            },
        }
        st.json(plot_config)


def render_single_trial_tab(
    words_filtered: pd.DataFrame,
    fixations_filtered: pd.DataFrame,
    combos: pd.DataFrame,
    *,
    canvas_width: int,
    canvas_height: int,
    base_font_size: int,
    font_family: str,
    viz_settings: dict,
    raw_gaze: Optional[pd.DataFrame] = None,
) -> None:
    """Render the single trial visualization tab."""
    selected_participant, selected_trial, selection_mode, selected_text = select_trial(
        combos, key_prefix="single"
    )
    if not (selected_participant and selected_trial):
        return

    # Filter data for selected trial
    trial_words = words_filtered[
        (words_filtered["participant_id"] == selected_participant)
        & (words_filtered["trial_id"] == selected_trial)
    ]
    trial_fixations = fixations_filtered[
        (fixations_filtered["participant_id"] == selected_participant)
        & (fixations_filtered["trial_id"] == selected_trial)
    ]

    # Handle raw gaze data
    trial_raw_gaze = pd.DataFrame()
    if raw_gaze is not None and not raw_gaze.empty:
        trial_raw_gaze = raw_gaze[
            (raw_gaze["participant_id"] == selected_participant)
            & (raw_gaze["trial_id"] == selected_trial)
        ]

    st.markdown(f"Showing **{selected_trial}** ")

    # Check raw gaze availability
    trial_has_raw_gaze = not trial_raw_gaze.empty
    global_raw_toggle = bool(viz_settings.get("show_raw_gaze"))
    if global_raw_toggle and not trial_has_raw_gaze:
        st.toast("Raw gaze not available for this trial.", icon="⚠️")
    effective_show_raw_gaze = bool(global_raw_toggle and trial_has_raw_gaze)

    # Build settings
    figure_settings = _build_figure_settings(viz_settings, effective_show_raw_gaze)
    figure_settings["raw_gaze"] = trial_raw_gaze if trial_has_raw_gaze else None
    x_field = viz_settings["x_field"]
    y_field = viz_settings["y_field"]

    # Comparison controls
    compare_participant, compare_trial, compare_layout = _render_comparison_controls(
        combos, selection_mode, selected_participant, selected_trial, selected_text
    )

    # Render figure
    if compare_participant is not None and compare_trial is not None:
        _render_comparison_figure(
            combos, words_filtered, fixations_filtered,
            selected_participant, selected_trial, selected_text,
            compare_participant, compare_trial,
            canvas_width, canvas_height, font_family, base_font_size, viz_settings,
            layout=compare_layout,
        )
    else:
        fig = make_scanpath_figure(
            trial_words,
            trial_fixations,
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            base_font_size=int(base_font_size),
            font_family=font_family,
            x_field=x_field,
            y_field=y_field,
            **figure_settings,
        )
        st.plotly_chart(fig, width="content", config={"responsive": False})

    # Stats, metadata, config
    _render_trial_stats(trial_words, trial_fixations)
    _render_metadata_selector(words_filtered, fixations_filtered, trial_words, trial_fixations)
    _render_plot_config_expander(
        selected_participant, selected_trial, canvas_width, canvas_height,
        x_field, y_field, figure_settings, viz_settings, base_font_size, trial_raw_gaze
    )

    # Export button
    if st.button("Export all filtered trials as PNG (zip)"):
        export_filtered_trials(
            combos,
            words_filtered,
            fixations_filtered,
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            base_font_size=int(base_font_size),
            font_family=font_family,
            x_field=x_field,
            y_field=y_field,
            settings=figure_settings,
        )


def _render_comparison_figure(
    combos: pd.DataFrame,
    words_filtered: pd.DataFrame,
    fixations_filtered: pd.DataFrame,
    selected_participant: str,
    selected_trial: str,
    selected_text: Optional[str],
    compare_participant: str,
    compare_trial: str,
    canvas_width: int,
    canvas_height: int,
    font_family: str,
    base_font_size: int,
    viz_settings: dict,
    layout: str = "overlay",
):
    """Render comparison figure for two trials."""
    paragraph_field = (
        "unique_paragraph_id"
        if "unique_paragraph_id" in combos.columns
        else "paragraph_id"
    )

    def _lookup_text_id(participant_id: str, trial_id: str) -> Optional[str]:
        match = combos[
            (combos["participant_id"] == participant_id)
            & (combos["trial_id"] == trial_id)
        ]
        if match.empty or paragraph_field not in match.columns:
            return None
        return str(match.iloc[0][paragraph_field])

    label_pool: set[str] = set()
    primary_text_id = selected_text or _lookup_text_id(selected_participant, selected_trial)
    compare_text_id = _lookup_text_id(compare_participant, compare_trial)
    primary_label = friendly_trial_label(
        selected_participant, selected_trial, primary_text_id, label_pool
    )
    compare_label = friendly_trial_label(
        compare_participant, compare_trial, compare_text_id, label_pool
    )

    fig_compare = make_comparison_figure(
        words_filtered,
        fixations_filtered,
        (selected_participant, selected_trial),
        (compare_participant, compare_trial),
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        font_family=font_family,
        base_font_size=int(base_font_size),
        show_words=viz_settings["show_words"],
        show_word_labels=viz_settings["show_labels"],
        trial_labels=(primary_label, compare_label),
        layout=layout,
    )
    st.plotly_chart(fig_compare, width="content", config={"responsive": False})


# -----------------------------------------------------------------------------
# Animation Tab
# -----------------------------------------------------------------------------


def render_animation_tab(
    words_filtered: pd.DataFrame,
    fixations_filtered: pd.DataFrame,
    combos: pd.DataFrame,
    *,
    canvas_width: int,
    canvas_height: int,
    base_font_size: int,
    font_family: str,
    viz_settings: dict,
) -> None:
    """Render the animated scanpath tab."""
    selected_participant, selected_trial, _mode, _text = select_trial(
        combos, key_prefix="anim"
    )
    if not (selected_participant and selected_trial):
        return

    trial_words = words_filtered[
        (words_filtered["participant_id"] == selected_participant)
        & (words_filtered["trial_id"] == selected_trial)
    ]
    trial_fixations = fixations_filtered[
        (fixations_filtered["participant_id"] == selected_participant)
        & (fixations_filtered["trial_id"] == selected_trial)
    ]

    st.markdown(f"Showing animated scanpath for **{selected_trial}**")

    # Playback speed
    speed_options = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0]
    speed_labels = ["×0.25", "×0.5", "×1", "×1.5", "×2", "×4", "×8"]
    playback_speed = st.select_slider(
        "Playback speed",
        options=speed_options,
        value=1.0,
        format_func=lambda x: speed_labels[speed_options.index(x)],
        help="Controls playback speed relative to actual fixation durations.",
        key="anim_playback_speed",
    )

    if trial_fixations.empty:
        st.warning("No fixations available for this trial.")
        return

    n_fixations = len(trial_fixations)
    total_duration_ms = trial_fixations["duration_ms"].sum()
    playback_duration_s = total_duration_ms / playback_speed / 1000
    st.info(
        f"**{n_fixations} fixations** · Total duration: {total_duration_ms / 1000:.1f}s "
        f"· Playback time at ×{playback_speed}: {playback_duration_s:.1f}s"
    )

    fig = make_scanpath_animation(
        trial_words,
        trial_fixations,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        base_font_size=int(base_font_size),
        font_family=font_family,
        playback_speed=playback_speed,
        show_words=viz_settings["show_words"],
        show_word_labels=viz_settings["show_labels"],
        show_saccades=viz_settings["show_saccades"],
        show_order=viz_settings["show_order"],
        marker_size_range=viz_settings["marker_size_range"],
        order_font_size=viz_settings["order_font_size"],
        order_font_color=viz_settings["order_font_color"],
    )
    st.plotly_chart(fig, width="content", config={"responsive": False})

    st.caption(
        "**Controls:** Use ▶ Play to auto-advance through fixations, ⏸ Pause to stop, "
        "or drag the slider to jump to any fixation. Orange highlight shows the current fixation."
    )


# -----------------------------------------------------------------------------
# Data Tables Tabs
# -----------------------------------------------------------------------------


def _render_paginated_dataframe(
    df: pd.DataFrame, page_size: int, key: str, caption: str
) -> None:
    """Render a dataframe with pagination if needed."""
    total_rows = len(df)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)

    if total_rows > page_size:
        st.info(f"Showing {total_rows:,} rows with pagination ({page_size:,} per page).")
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            key=key,
            help=f"Total pages: {total_pages:,}",
        )
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)
        display_df = df.iloc[start_idx:end_idx]
        st.caption(f"Showing rows {start_idx + 1:,} – {end_idx:,} of {total_rows:,}")
    else:
        display_df = df

    st.dataframe(display_df, hide_index=True, width="stretch")
    st.caption(caption)


def render_metrics_tab(
    words_filtered: pd.DataFrame, fixations_filtered: pd.DataFrame
) -> None:
    """Render word-level metrics tab."""
    from scanpath_visualization_app.data import compute_word_metrics

    st.subheader("Word-level data")
    metrics = compute_word_metrics(words_filtered, fixations_filtered)
    _render_paginated_dataframe(
        metrics, 1000, "metrics_page",
        "Word-level data with computed reading metrics where available."
    )


def render_fixations_tab(fixations_filtered: pd.DataFrame) -> None:
    """Render fixation-level data tab."""
    st.subheader("Fixation-level data")
    _render_paginated_dataframe(
        fixations_filtered, 1000, "fixations_page",
        "All fixation records after applying filters; includes ids, timing, and optional flags."
    )


def render_raw_gaze_tab(raw_gaze_filtered: pd.DataFrame) -> None:
    """Render raw gaze data tab."""
    st.subheader("Raw gaze data")
    if raw_gaze_filtered.empty:
        st.info("No raw gaze data available after filtering.")
        return
    _render_paginated_dataframe(
        raw_gaze_filtered, 1000, "raw_gaze_page",
        "Millisecond-level gaze samples after applying filters."
    )


def render_raw_data_tab(
    words_filtered: pd.DataFrame,
    fixations_filtered: pd.DataFrame,
    raw_gaze_filtered: pd.DataFrame,
) -> None:
    """Render the raw data tab with sub-tabs."""
    word_tab, fixation_tab, raw_gaze_tab = st.tabs(
        ["Word-level", "Fixation-level", "Raw gaze"]
    )
    with word_tab:
        render_metrics_tab(words_filtered, fixations_filtered)
    with fixation_tab:
        render_fixations_tab(fixations_filtered)
    with raw_gaze_tab:
        render_raw_gaze_tab(raw_gaze_filtered)


# -----------------------------------------------------------------------------
# Statistics Tab
# -----------------------------------------------------------------------------


def render_data_statistics_tab(
    words_filtered: pd.DataFrame,
    fixations_filtered: pd.DataFrame,
    raw_gaze_filtered: pd.DataFrame,
) -> None:
    """Render dataset statistics tab."""
    st.subheader("Dataset statistics")

    # Count unique entities
    participant_ids = set(words_filtered["participant_id"].unique()) | set(
        fixations_filtered["participant_id"].unique()
    )
    trial_ids = set(words_filtered["trial_id"].unique()) | set(
        fixations_filtered["trial_id"].unique()
    )
    paragraph_col = (
        "unique_paragraph_id"
        if "unique_paragraph_id" in words_filtered.columns
        else "paragraph_id"
    )
    text_ids = (
        set(words_filtered[paragraph_col].unique())
        if paragraph_col in words_filtered.columns
        else set()
    )

    # Top-level metrics
    top_cols = st.columns(6)
    top_cols[0].metric("Participants", f"{len(participant_ids):,}")
    top_cols[1].metric("Texts", f"{len(text_ids):,}")
    top_cols[2].metric("Trials", f"{len(trial_ids):,}")
    top_cols[3].metric("Fixations", f"{len(fixations_filtered):,}")
    top_cols[4].metric("Words", f"{len(words_filtered):,}")
    top_cols[5].metric(
        "Gaze points",
        f"{len(raw_gaze_filtered):,}" if not raw_gaze_filtered.empty else "0",
        help="Counts raw gaze samples if provided.",
    )

    st.divider()

    # Detailed statistics
    trial_source = fixations_filtered if not fixations_filtered.empty else words_filtered
    trials_per_participant = (
        trial_source.groupby("participant_id")["trial_id"].nunique()
        if not trial_source.empty
        else pd.Series(dtype=float)
    )
    fixations_per_trial = (
        fixations_filtered.groupby(["participant_id", "trial_id"]).size()
        if not fixations_filtered.empty
        else pd.Series(dtype=float)
    )
    words_per_trial = (
        words_filtered.groupby(["participant_id", "trial_id"]).size()
        if not words_filtered.empty
        else pd.Series(dtype=float)
    )

    stats_df = pd.DataFrame([
        {"Metric": "Trials per participant", **safe_summary(trials_per_participant)},
        {"Metric": "Fixations per trial", **safe_summary(fixations_per_trial)},
        {"Metric": "Words per trial", **safe_summary(words_per_trial)},
    ])
    stats_df = stats_df.rename(columns={
        "mean": "Mean", "std": "Std", "min": "Min", "median": "Median", "max": "Max"
    })

    st.dataframe(
        stats_df,
        hide_index=True,
        width="stretch",
        column_config={
            col: st.column_config.NumberColumn(format="%.2f")
            for col in ["Mean", "Std", "Min", "Median", "Max"]
        },
    )
    st.caption(
        "Statistics computed after filtering; missing values indicate empty source data."
    )
