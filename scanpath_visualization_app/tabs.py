"""Tab rendering functions for the Scanpath Visualization app."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from scanpath_visualization_app.data import compute_word_metrics

from scanpath_visualization_app.plots import (
    make_comparison_figure,
    make_fixation_duration_histogram,
    make_scanpath_animation,
    make_scanpath_figure,
    make_word_measure_bar_figure,
)
from scanpath_visualization_app.export import (
    ExportProgress,
    bulk_export,
    render_export_options,
)
from scanpath_visualization_app.utils import (
    build_comparison_options,
    compute_trial_stats,
    friendly_trial_label,
    gather_trial_metadata,
    safe_summary,
    select_trial,
)


# -----------------------------------------------------------------------------
# Single Trial Tab
# -----------------------------------------------------------------------------


def _safe_filename(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(text))


_MIME_FOR_FORMAT = {
    "PNG": "image/png",
    "SVG": "image/svg+xml",
    "PDF": "application/pdf",
}


def _render_save_plot_button(
    fig,
    *,
    canvas_width: int,
    canvas_height: int,
    slug: str,
    key_prefix: str,
) -> None:
    """Render a Generate-then-download flow for the currently displayed figure.

    Pulls width/height from the figure's own layout (so stacked / multi-panel
    figures save at their on-screen size). Only invokes the expensive
    `fig.to_image` call when the user clicks Generate.
    """
    if fig is None:
        return
    file_stem = f"scanpath_{_safe_filename(slug)}"
    cols = st.columns([1, 1, 2])
    with cols[0]:
        fmt = st.radio(
            "Save format",
            options=["PNG", "SVG", "PDF"],
            index=0,
            horizontal=True,
            key=f"{key_prefix}_save_format",
        )
    with cols[1]:
        generate = st.button(
            f"Render {fmt}",
            key=f"{key_prefix}_save_generate",
            help="Generates the file. The download button below appears once it's ready.",
        )

    if not generate:
        return

    fig_width = int(fig.layout.width or canvas_width)
    fig_height = int(fig.layout.height or canvas_height)
    try:
        data = fig.to_image(
            format=fmt.lower(),
            width=fig_width,
            height=fig_height,
            scale=2 if fmt == "PNG" else 1,
        )
    except Exception as exc:
        st.warning(f"Could not render {fmt} for download: {exc}")
        return
    st.download_button(
        f"Save plot ({fmt})",
        data=data,
        file_name=f"{file_stem}.{fmt.lower()}",
        mime=_MIME_FOR_FORMAT[fmt],
        key=f"{key_prefix}_save_button",
    )


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
        critical_span_style=viz_settings.get("critical_span_style", "Mark text"),
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
            options=["Overlay", "Side by side", "Stacked"],
            index=0,
            key="single_compare_layout",
            horizontal=True,
            help="Stacked = trials shown one above the other.",
        )
    layout_map = {
        "Overlay": "overlay",
        "Side by side": "side_by_side",
        "Stacked": "stacked",
    }
    layout = layout_map.get(layout_label, "overlay")

    if selected_compare_label:
        participant, trial = label_to_trial[selected_compare_label]
        return participant, trial, layout
    return None, None, layout


_CRITICAL_SPAN_BG = "#FCE7F3"  # light pink — critical-span words
_DISTRACTOR_SPAN_BG = "#E5E7EB"  # light grey — distractor-span words


def _ordered_words(trial_words: pd.DataFrame) -> pd.DataFrame:
    """Return trial_words sorted into reading order."""
    for col in ("word_id", "IA_ID"):
        if col in trial_words.columns:
            return trial_words.sort_values(col)
    return trial_words


def _render_paragraph_with_spans(trial_words: pd.DataFrame) -> None:
    """Render paragraph text with critical-span (green) and distractor (orange)
    word backgrounds. Falls back to plain text when span columns are absent."""
    if "text" not in trial_words.columns or trial_words.empty:
        return
    ordered = _ordered_words(trial_words)
    has_critical = "is_in_aspan" in ordered.columns
    has_distractor = "is_in_dspan" in ordered.columns
    if not (has_critical or has_distractor):
        st.write(" ".join(ordered["text"].astype(str).tolist()))
        return
    import html as _html

    parts: list[str] = []
    for row in ordered.itertuples():
        word = _html.escape(str(getattr(row, "text", "")))
        bg = ""
        if has_critical and bool(getattr(row, "is_in_aspan", False)):
            bg = _CRITICAL_SPAN_BG
        elif has_distractor and bool(getattr(row, "is_in_dspan", False)):
            bg = _DISTRACTOR_SPAN_BG
        if bg:
            parts.append(
                f'<span style="background-color:{bg};padding:0 2px;border-radius:2px;">{word}</span>'
            )
        else:
            parts.append(word)
    html_body = " ".join(parts)
    st.markdown(
        f'<div style="line-height:1.6;">{html_body}</div>',
        unsafe_allow_html=True,
    )


def _span_text(trial_words: pd.DataFrame, mask_col: str) -> str:
    """Return the joined text of words where mask_col is True."""
    if mask_col not in trial_words.columns or "text" not in trial_words.columns:
        return ""
    ordered = _ordered_words(trial_words)
    mask = ordered[mask_col].fillna(False).astype(bool)
    return " ".join(ordered.loc[mask, "text"].astype(str).tolist())


def _render_trial_header(
    participant: str,
    trial_id: str,
    trial_words: pd.DataFrame,
    prefix: str = "Trial:",
) -> None:
    """Render the one-line participant + trial id + text id header.

    The paragraph text / question / spans live in
    `_render_paragraph_panel` so they can sit under the figure (single tab)
    while the header stays in the side panel.
    """
    parts = [f"**{prefix}** `{trial_id}`", f"participant `{participant}`"]
    text_id = None
    for col in ("unique_paragraph_id", "paragraph_id"):
        if col in trial_words.columns and not trial_words.empty:
            value = trial_words[col].iloc[0]
            if pd.notna(value):
                text_id = value
                break
    if text_id is not None:
        parts.append(f"Text: `{text_id}`")
    st.markdown(" · ".join(parts))


def _render_paragraph_panel(
    trial_words: pd.DataFrame, *, expanded: bool = True
) -> None:
    """Render the paragraph expander with highlighted spans, question, and
    critical-/distractor-span text. Skips silently when no word text is
    available."""
    if "text" not in trial_words.columns or trial_words.empty:
        return
    with st.expander("Paragraph text", expanded=expanded):
        _render_paragraph_with_spans(trial_words)
        question_val = None
        if "question" in trial_words.columns:
            qs = trial_words["question"].dropna()
            if not qs.empty:
                question_val = str(qs.iloc[0])
        if question_val:
            st.markdown(f"**Question:** {question_val}")
        critical_text = _span_text(trial_words, "is_in_aspan")
        if critical_text:
            st.markdown(
                f'<span style="background-color:{_CRITICAL_SPAN_BG};'
                f'padding:0 4px;border-radius:2px;">'
                f"<b>Critical span:</b></span> {critical_text}",
                unsafe_allow_html=True,
            )
        distractor_text = _span_text(trial_words, "is_in_dspan")
        if distractor_text:
            st.markdown(
                f'<span style="background-color:{_DISTRACTOR_SPAN_BG};'
                f'padding:0 4px;border-radius:2px;">'
                f"<b>Distractor span:</b></span> {distractor_text}",
                unsafe_allow_html=True,
            )


def _render_trial_stats(trial_words: pd.DataFrame, trial_fixations: pd.DataFrame):
    """Render trial statistics metrics."""
    stats = compute_trial_stats(trial_words, trial_fixations)
    stat_cols = st.columns(3)
    stat_cols[0].metric(
        "Total reading time (s)", f"{stats['total_reading_time_s']:.1f}"
    )
    stat_cols[1].metric("Number of words", f"{stats['word_count']:,}")
    stat_cols[2].metric("Number of fixations", f"{stats['fixation_count']:,}")


# Row-wise palette for the trial-metadata table. Picked light so the text
# stays readable. `is_correct` keeps the row background neutral but tints
# the value text green/red — adding a background there clashed with the
# ✓ / ✗ color.
_DIFFICULTY_COLORS = {"ele": "#d4edda", "adv": "#fff3cd"}
_REPEAT_COLOR = "#cfe2ff"  # light blue — clearly distinct from the Ele/Adv tints
_PREVIEW_COLOR = "#DBEAFE"  # light blue — Hunting/preview trials


def _style_metadata_row(row: pd.Series) -> list[str]:
    field = str(row.get("Field", ""))
    value = str(row.get("Value", ""))
    bg = ""
    if field == "difficulty_level":
        for prefix, color in _DIFFICULTY_COLORS.items():
            if value.lower().startswith(prefix):
                bg = f"background-color: {color};"
                break
    elif field == "repeated_reading_trial" and value.lower().startswith("true"):
        bg = f"background-color: {_REPEAT_COLOR};"
    elif field == "question_preview" and value.lower().startswith("true"):
        bg = f"background-color: {_PREVIEW_COLOR};"

    field_style = bg
    value_style = bg
    if field == "is_correct":
        if value.startswith("✓"):
            value_style = "color: #198754; font-weight: 600;"
        elif value.startswith("✗"):
            value_style = "color: #dc3545; font-weight: 600;"
    return [field_style, value_style]


def _render_metadata_selector(
    words_filtered: pd.DataFrame,
    fixations_filtered: pd.DataFrame,
    trial_words: pd.DataFrame,
    trial_fixations: pd.DataFrame,
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
            "question_preview",
            "selected_answer",
            "is_correct",
        ]
        if field in metadata_candidates
    ]
    # Reserve the table slot first; the multiselect renders below it so the
    # filter chips sit under the data they configure. Streamlit fills the
    # container later in the same run, preserving the visual order.
    table_slot = st.container()
    selected_metadata = st.multiselect(
        "Trial metadata fields",
        options=metadata_candidates,
        default=default_metadata or metadata_candidates,
    )
    if selected_metadata:
        metadata_df = gather_trial_metadata(
            trial_words, trial_fixations, selected_metadata
        )
        if not metadata_df.empty:
            # Prefix is_correct with ✓ / ✗ so reviewers can scan the table
            # without parsing the True/False text.
            mask = metadata_df["Field"] == "is_correct"
            if mask.any():
                val = str(metadata_df.loc[mask, "Value"].iloc[0])
                if val.lower().startswith("true"):
                    metadata_df.loc[mask, "Value"] = f"✓ {val}"
                elif val.lower().startswith("false"):
                    metadata_df.loc[mask, "Value"] = f"✗ {val}"
            styled = metadata_df.style.apply(_style_metadata_row, axis=1)
            with table_slot:
                st.dataframe(styled, hide_index=True, width="stretch")


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
                "marker_size_range": [
                    int(s) for s in figure_settings["marker_size_range"]
                ],
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


def _ordered_trial_ids(combos: pd.DataFrame) -> list[str]:
    """Stable trial_id ordering used by the under-image Prev/Next buttons.

    Mirrors `_select_trial_none_mode`'s sort so the under-image nav lands on
    the same trial as the side-panel Prev/Next when both are present.
    """
    trial_col = "unique_trial_id" if "unique_trial_id" in combos.columns else "trial_id"
    return sorted(combos[trial_col].dropna().astype(str).unique().tolist())


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
    """Render the single trial visualization tab.

    Layout: 30 / 70 split — every per-trial control (selectors, paragraph
    text, compare toggle, save button, stats, metadata, plot config) sits in
    the left side panel, the plot dominates the right column. Bulk export
    sits below full-width because its progress bar / artifact dropdowns need
    the room.
    """
    col_side, col_main = st.columns([3, 7], gap="medium")

    with col_side:
        # Stats render at the very top of the side panel so reviewers see the
        # trial's totals before scrolling into the picker / metadata. The slot
        # is reserved here and filled later once `trial_words` / `trial_fixations`
        # are available (they depend on the picker selection).
        stats_slot = st.container()
        selected_participant, selected_trial, selection_mode, selected_text = (
            select_trial(combos, key_prefix="single")
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
    trial_raw_gaze = pd.DataFrame()
    if raw_gaze is not None and not raw_gaze.empty:
        trial_raw_gaze = raw_gaze[
            (raw_gaze["participant_id"] == selected_participant)
            & (raw_gaze["trial_id"] == selected_trial)
        ]

    trial_has_raw_gaze = not trial_raw_gaze.empty
    global_raw_toggle = bool(viz_settings.get("show_raw_gaze"))
    effective_show_raw_gaze = bool(global_raw_toggle and trial_has_raw_gaze)
    figure_settings = _build_figure_settings(viz_settings, effective_show_raw_gaze)
    figure_settings["raw_gaze"] = trial_raw_gaze if trial_has_raw_gaze else None
    x_field = viz_settings["x_field"]
    y_field = viz_settings["y_field"]

    with col_side:
        _render_trial_header(selected_participant, selected_trial, trial_words)
        if global_raw_toggle and not trial_has_raw_gaze:
            st.warning("Raw gaze not available for this trial.", icon="⚠️")
        compare_participant, compare_trial, compare_layout = (
            _render_comparison_controls(
                combos,
                selection_mode,
                selected_participant,
                selected_trial,
                selected_text,
            )
        )

    with col_main:
        if compare_participant is not None and compare_trial is not None:
            displayed_fig = _render_comparison_figure(
                combos,
                words_filtered,
                fixations_filtered,
                selected_participant,
                selected_trial,
                selected_text,
                compare_participant,
                compare_trial,
                canvas_width,
                canvas_height,
                font_family,
                base_font_size,
                viz_settings,
                layout=compare_layout,
            )
            save_slug = (
                f"{selected_participant}__{selected_trial}__vs__"
                f"{compare_participant}__{compare_trial}"
            )
        else:
            displayed_fig = make_scanpath_figure(
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
            st.plotly_chart(
                displayed_fig, width="stretch", config={"responsive": False}
            )
            save_slug = f"{selected_participant}__{selected_trial}"

        # Save format / Render-PNG button lives directly under the plot so
        # the export action is right where the eye is when reviewers want it.
        _render_save_plot_button(
            displayed_fig,
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            slug=save_slug,
            key_prefix="single",
        )

        # Paragraph text (with question + critical/distractor span overlays)
        # sits below the figure so reviewers can read the text while
        # comparing it to the scanpath right above.
        _render_paragraph_panel(trial_words, expanded=True)

    # Publish the single-tab's selection so the Animation tab can mirror it.
    # Read by `render_animation_tab` (one-way sync; anim can still be moved
    # independently — see `_sync_anim_from_single`).
    st.session_state["shared_selected_pid"] = selected_participant
    st.session_state["shared_selected_trial_id"] = selected_trial

    # Fill the reserved stats slot now that the trial data is in hand.
    with stats_slot:
        _render_trial_stats(trial_words, trial_fixations)

    with col_side:
        _render_metadata_selector(
            words_filtered, fixations_filtered, trial_words, trial_fixations
        )
        _render_plot_config_expander(
            selected_participant,
            selected_trial,
            canvas_width,
            canvas_height,
            x_field,
            y_field,
            figure_settings,
            viz_settings,
            base_font_size,
            trial_raw_gaze,
        )

    _render_bulk_export(
        combos,
        words_filtered,
        fixations_filtered,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        base_font_size=int(base_font_size),
        font_family=font_family,
        x_field=x_field,
        y_field=y_field,
        figure_settings=figure_settings,
    )


def _render_bulk_export(
    combos: pd.DataFrame,
    words_filtered: pd.DataFrame,
    fixations_filtered: pd.DataFrame,
    *,
    canvas_width: int,
    canvas_height: int,
    base_font_size: int,
    font_family: str,
    x_field: str,
    y_field: str,
    figure_settings: dict,
) -> None:
    """Render configurable bulk-export UI (artifact picker + run + download)."""
    st.divider()
    st.markdown("### Bulk export")
    st.caption(
        f"Up to **{len(combos)}** currently filtered trials are available. "
        "Choose a scope and which artifacts to bundle below."
    )
    options = render_export_options(st, combos, key_prefix="single_export")
    run_col, info_col = st.columns([1, 3])
    with run_col:
        run = st.button(
            "Build export",
            type="primary",
            disabled=(
                combos.empty
                or not any(
                    [
                        options.include_png,
                        options.include_svg,
                        options.include_pdf,
                        options.include_plot_config,
                        options.include_fixations,
                        options.include_measures,
                        options.include_mega_table,
                    ]
                )
            ),
        )
    progress_bar = info_col.progress(0.0, text="Idle")
    if run:

        def on_progress(p: ExportProgress) -> None:
            frac = p.finished_trials / p.total_trials if p.total_trials else 1.0
            progress_bar.progress(
                min(max(frac, 0.0), 1.0),
                text=(
                    f"Exporting trial {p.finished_trials}/{p.total_trials} "
                    f"— {p.bytes_written / 1024:.0f} KB so far"
                ),
            )

        zip_bytes, progress = bulk_export(
            combos,
            words_filtered,
            fixations_filtered,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            base_font_size=base_font_size,
            font_family=font_family,
            x_field=x_field,
            y_field=y_field,
            settings=figure_settings,
            options=options,
            progress_callback=on_progress,
        )
        progress_bar.progress(1.0, text="Ready")
        if progress.errors:
            with st.expander("Export warnings"):
                for err in progress.errors:
                    st.write(err)
        st.download_button(
            "Download zip",
            data=zip_bytes,
            file_name=f"scanpath_export_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S}.zip",
            mime="application/zip",
            type="primary",
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
    primary_text_id = selected_text or _lookup_text_id(
        selected_participant, selected_trial
    )
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
    st.plotly_chart(fig_compare, width="stretch", config={"responsive": False})
    return fig_compare


# -----------------------------------------------------------------------------
# Animation Tab
# -----------------------------------------------------------------------------


def _sync_anim_from_single(combos: pd.DataFrame) -> None:
    """Pre-set anim tab's selector state to mirror the single tab's selection.

    Fires only when the single tab's selection changed since the last run
    (tracked via `_prev_shared_trial`). This means user changes inside the
    anim tab are preserved across reruns — anim only "snaps" back to single
    when the single tab is the active driver. One-way sync by design.
    """
    shared_pid = st.session_state.get("shared_selected_pid")
    shared_trial = st.session_state.get("shared_selected_trial_id")
    if not (shared_pid and shared_trial):
        return
    current = (shared_pid, shared_trial)
    if st.session_state.get("_prev_shared_trial") == current:
        return
    trial_options = _ordered_trial_ids(combos)
    try:
        idx = trial_options.index(str(shared_trial))
    except ValueError:
        return
    st.session_state["anim_select_trial_mode"] = "None"
    st.session_state["anim_trial_index"] = idx
    st.session_state["_prev_shared_trial"] = current


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
    """Render the animated scanpath tab.

    Layout mirrors the Interactive Plot tab: 30 / 70 split, selectors / info /
    export controls in the left side panel and the animation plot on the
    right.
    """
    # Sync trial selection from the single tab BEFORE the picker widgets
    # render, so they pick up the seeded state as their initial value.
    _sync_anim_from_single(combos)

    col_side, col_main = st.columns([3, 7], gap="medium")

    with col_side:
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

    # Playback speed — rendered on the right (next to the animation plot)
    # because that's where the eye actually goes when adjusting playback.
    # Frame durations are floor-clamped at 50 ms (see `make_scanpath_animation`),
    # so higher speeds aren't more expensive to render than lower ones; they
    # just cap there for the shortest fixations.
    speed_options = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 8.0]
    speed_labels = ["×0.25", "×0.5", "×1", "×1.5", "×2", "×2.5", "×3", "×4", "×8"]
    with col_main:
        playback_speed = st.select_slider(
            "Playback speed",
            options=speed_options,
            value=3.0,
            format_func=lambda x: speed_labels[speed_options.index(x)],
            help="Controls playback speed relative to actual fixation durations.",
            key="anim_playback_speed",
        )

    with col_side:
        _render_trial_header(
            selected_participant,
            selected_trial,
            trial_words,
            prefix="Animated scanpath:",
        )
        _render_paragraph_panel(trial_words, expanded=False)

        if trial_fixations.empty:
            st.warning("No fixations available for this trial.")
        else:
            n_fixations = len(trial_fixations)
            total_duration_ms = trial_fixations["duration_ms"].sum()
            playback_duration_s = total_duration_ms / playback_speed / 1000
            st.info(
                f"**{n_fixations} fixations** · Total duration: "
                f"{total_duration_ms / 1000:.1f}s · Playback time at "
                f"×{playback_speed}: {playback_duration_s:.1f}s"
            )

    if trial_fixations.empty:
        return

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
    with col_main:
        st.plotly_chart(fig, width="stretch", config={"responsive": False})

    with col_side:
        st.caption(
            "**Controls:** Use ▶ Play to auto-advance through fixations, "
            "⏸ Pause to stop, or drag the slider to jump to any fixation. "
            "Orange highlight shows the current fixation."
        )
        html_bytes = fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")
        st.download_button(
            "Export animation (HTML)",
            data=html_bytes,
            file_name=f"animation_{_safe_filename(selected_participant)}__{_safe_filename(selected_trial)}.html",
            mime="text/html",
            key="anim_export_html",
            help="Self-contained HTML you can open in any browser; keeps play/slider interactivity.",
        )


# -----------------------------------------------------------------------------
# Data Tables Tabs
# -----------------------------------------------------------------------------


def _render_paginated_dataframe(
    df: pd.DataFrame,
    page_size: int,
    key: str,
    caption: str,
    download_name: Optional[str] = None,
) -> None:
    """Render a dataframe with pagination + download buttons (CSV + Parquet)."""
    total_rows = len(df)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)

    if total_rows > page_size:
        st.info(
            f"Showing {total_rows:,} rows with pagination ({page_size:,} per page)."
        )
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

    if download_name and not df.empty:
        col_csv, col_parquet, _ = st.columns([1, 1, 4])
        with col_csv:
            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"{download_name}.csv",
                mime="text/csv",
                key=f"{key}_csv_download",
            )
        with col_parquet:
            import io as _io

            buf = _io.BytesIO()
            try:
                df.to_parquet(buf, index=False)
                st.download_button(
                    "Download Parquet",
                    data=buf.getvalue(),
                    file_name=f"{download_name}.parquet",
                    mime="application/octet-stream",
                    key=f"{key}_parquet_download",
                )
            except Exception:
                pass


def render_metrics_tab(
    words_filtered: pd.DataFrame, fixations_filtered: pd.DataFrame
) -> None:
    """Render word-level metrics tab."""
    st.subheader("Word-level data")
    metrics = compute_word_metrics(words_filtered, fixations_filtered)
    _render_paginated_dataframe(
        metrics,
        1000,
        "metrics_page",
        "Word-level data with computed reading metrics where available.",
        download_name="word_measures",
    )


def render_fixations_tab(fixations_filtered: pd.DataFrame) -> None:
    """Render fixation-level data tab."""
    st.subheader("Fixation-level data")
    _render_paginated_dataframe(
        fixations_filtered,
        1000,
        "fixations_page",
        "All fixation records after applying filters; includes ids, timing, and optional flags.",
        download_name="fixations",
    )


def render_raw_gaze_tab(raw_gaze_filtered: pd.DataFrame) -> None:
    """Render raw gaze data tab."""
    st.subheader("Raw gaze data")
    if raw_gaze_filtered.empty:
        st.info("No raw gaze data available after filtering.")
        return
    _render_paginated_dataframe(
        raw_gaze_filtered,
        1000,
        "raw_gaze_page",
        "Millisecond-level gaze samples after applying filters.",
        download_name="raw_gaze",
    )


def _render_data_provenance() -> None:
    """Show a 'source / cohort / date / file mtime' banner above the Raw Data
    sub-tabs so reviewers can verify which OneStop export they're looking at.

    Only renders when ONESTOP_DATA_DIR is set (i.e. OneStop server bundle is
    the active data source). For uploads / bundled demo, falls through silently.
    """
    from datetime import datetime
    from .data import onestop_data_provenance

    # Honour the deep-link participant so the per-pid shard's mtime is shown.
    pid = st.session_state.get("single_participant")
    info = onestop_data_provenance(participant=pid)
    if not info:
        return

    def _fmt_mtime(ts):
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "—"

    cols = st.columns(4)
    cols[0].metric("Source", info.get("source", "—"))
    cols[1].metric("Cohort", info.get("cohort", "—"))
    cols[2].metric("Date", info.get("date", "—"))
    cols[3].metric("File mtime", _fmt_mtime(info.get("ia_shard_mtime")))

    with st.expander("Data provenance — full paths"):
        st.caption(f"`ONESTOP_DATA_DIR = {info.get('data_dir', '?')}`")
        st.caption(
            f"loaded from: **{info.get('loaded_from', '?')}**"
            + (f"  ·  participant `{pid}`" if pid else "")
        )
        if "ia_shard" in info:
            st.caption(
                f"IA file: `{info['ia_shard']}`  "
                f"·  mtime `{_fmt_mtime(info.get('ia_shard_mtime'))}`"
            )
        if "fix_shard" in info:
            st.caption(
                f"Fixations file: `{info['fix_shard']}`  "
                f"·  mtime `{_fmt_mtime(info.get('fix_shard_mtime'))}`"
            )


def render_raw_data_tab(
    words_filtered: pd.DataFrame,
    fixations_filtered: pd.DataFrame,
    raw_gaze_filtered: pd.DataFrame,
) -> None:
    """Render the raw data tab with sub-tabs."""
    _render_data_provenance()
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


_MEASURE_OPTIONS = [
    ("first_fixation_ms", "First-fixation duration (FFD)"),
    ("first_pass_gaze_duration_ms", "Gaze duration / FPRT"),
    ("regression_path_duration_ms", "Regression-path duration (go-past)"),
    ("total_fixation_duration_ms", "Total fixation duration / dwell"),
    ("n_fixations", "Fixation count"),
    ("skip_flag", "Skip flag"),
    ("regression_in_flag", "Regression in"),
    ("regression_out_flag", "Regression out"),
    ("gpt2_surprisal", "GPT-2 surprisal"),
    ("wordfreq_frequency", "Word frequency (wordfreq)"),
    ("subtlex_frequency", "Word frequency (SUBTLEX)"),
]


def render_data_statistics_tab(
    words_filtered: pd.DataFrame,
    fixations_filtered: pd.DataFrame,
    raw_gaze_filtered: pd.DataFrame,
    combos: pd.DataFrame,
    *,
    canvas_width: int,
    base_font_size: int,
    font_family: str,
) -> None:
    """Render dataset statistics tab with reading-research summaries."""
    st.subheader("Dataset statistics")

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

    rr_cols = st.columns(4)
    if not fixations_filtered.empty and "duration_ms" in fixations_filtered.columns:
        mean_fix_dur = float(fixations_filtered["duration_ms"].mean())
        rr_cols[0].metric("Mean fixation dur (ms)", f"{mean_fix_dur:.0f}")
    else:
        rr_cols[0].metric("Mean fixation dur (ms)", "—")
    if (
        not fixations_filtered.empty
        and "saccade_amplitude" in fixations_filtered.columns
    ):
        mean_sac = float(fixations_filtered["saccade_amplitude"].dropna().mean() or 0)
        rr_cols[1].metric("Mean saccade amp (px)", f"{mean_sac:.0f}")
    else:
        rr_cols[1].metric("Mean saccade amp (px)", "—")
    if "is_regression" in fixations_filtered.columns and not fixations_filtered.empty:
        reg_rate = float(fixations_filtered["is_regression"].mean()) * 100
        rr_cols[2].metric("Regression rate", f"{reg_rate:.1f} %")
    elif "regression_out_flag" in words_filtered.columns and not words_filtered.empty:
        reg_rate = float(words_filtered["regression_out_flag"].mean()) * 100
        rr_cols[2].metric("Words w/ regression-out", f"{reg_rate:.1f} %")
    else:
        rr_cols[2].metric("Regression rate", "—")
    if (
        not fixations_filtered.empty
        and "duration_ms" in fixations_filtered.columns
        and not words_filtered.empty
    ):
        total_ms = fixations_filtered.groupby(["participant_id", "trial_id"])[
            "duration_ms"
        ].sum()
        n_words = words_filtered.groupby(["participant_id", "trial_id"]).size()
        per_trial = (
            n_words.reindex(total_ms.index).fillna(0) * 60_000
        ) / total_ms.replace(0, pd.NA)
        wpm = float(per_trial.dropna().mean()) if per_trial.dropna().size else 0.0
        rr_cols[3].metric("Reading speed (wpm)", f"{wpm:.0f}")
    else:
        rr_cols[3].metric("Reading speed (wpm)", "—")

    st.divider()

    trial_source = (
        fixations_filtered if not fixations_filtered.empty else words_filtered
    )
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

    stats_df = pd.DataFrame(
        [
            {
                "Metric": "Trials per participant",
                **safe_summary(trials_per_participant),
            },
            {"Metric": "Fixations per trial", **safe_summary(fixations_per_trial)},
            {"Metric": "Words per trial", **safe_summary(words_per_trial)},
        ]
    )
    stats_df = stats_df.rename(
        columns={
            "mean": "Mean",
            "std": "Std",
            "min": "Min",
            "median": "Median",
            "max": "Max",
        }
    )

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

    st.divider()
    st.subheader("Fixation duration distribution")
    if fixations_filtered.empty:
        st.info("No fixations available for distribution plot.")
    else:
        overlay_measures = (
            compute_word_metrics(words_filtered, fixations_filtered)
            if not words_filtered.empty
            else None
        )
        hist = make_fixation_duration_histogram(
            fixations_filtered,
            canvas_width=int(canvas_width),
            base_font_size=int(base_font_size),
            font_family=font_family,
            overlay_words=overlay_measures,
        )
        st.plotly_chart(hist, width="content", config={"responsive": False})

    st.divider()
    st.subheader("Per-word measure")
    if combos.empty:
        st.info("No trials available — adjust the filters to pick a trial here.")
        return
    selected_participant, selected_trial, _mode, _text = select_trial(
        combos, key_prefix="stats_measures"
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
    if trial_words.empty or trial_fixations.empty:
        st.info("Select a trial with both words and fixations to see measures.")
        return

    measures_df = compute_word_metrics(trial_words, trial_fixations)
    available_measures = [
        (key, label) for key, label in _MEASURE_OPTIONS if key in measures_df.columns
    ]
    if not available_measures:
        st.info("No per-word measures available for this trial.")
        return

    key_to_label = dict(available_measures)
    selected_measure = st.selectbox(
        "Measure",
        options=list(key_to_label.keys()),
        format_func=lambda k: key_to_label[k],
        key="word_measure_choice",
    )
    bar_fig = make_word_measure_bar_figure(
        measures_df,
        measure=selected_measure,
        canvas_width=int(canvas_width),
        base_font_size=int(base_font_size),
        font_family=font_family,
    )
    st.plotly_chart(bar_fig, width="content", config={"responsive": False})
    st.caption(
        "Per-word measures computed per (participant, trial, word). When the input "
        "table carries EyeLink IA exports (e.g. IA_FIRST_FIXATION_DURATION), "
        "those values are preserved; missing values are computed from "
        "fixations + word bounding boxes."
    )
