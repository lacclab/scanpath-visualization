from __future__ import annotations

import io
import zipfile
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    from .constants import FONT_FAMILY
    from .controls import data_dictionary_help_text, sidebar_controls
    from .data import (
        compute_canvas_size,
        compute_word_metrics,
        default_filters,
        filter_data,
        infer_fix_schema,
        infer_word_schema,
        load_sample_data,
        normalize_fixations,
        normalize_words,
    )
    from .plots import make_comparison_figure, make_scanpath_figure
except ImportError:
    # Allow running via `streamlit run scanpath_visualization_app/app.py`
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scanpath_visualization_app.constants import FONT_FAMILY
    from scanpath_visualization_app.controls import data_dictionary_help_text, sidebar_controls
    from scanpath_visualization_app.data import (
        compute_canvas_size,
        compute_word_metrics,
        default_filters,
        filter_data,
        infer_fix_schema,
        infer_word_schema,
        load_sample_data,
        normalize_fixations,
        normalize_words,
    )
    from scanpath_visualization_app.plots import make_comparison_figure, make_scanpath_figure


st.set_page_config(
    page_title="Scanpath Visualization",
    page_icon="👀",
    layout="wide",
)


def load_words_and_fixations(data_choice: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return uploaded csvs or fall back to bundled demo data."""
    if data_choice == "Upload csv tables":
        uploaded_words = st.sidebar.file_uploader("Words/IA csv", type=["csv"])
        uploaded_fixations = st.sidebar.file_uploader("Fixations csv", type=["csv"])
        if uploaded_words and uploaded_fixations:
            return pd.read_csv(uploaded_words), pd.read_csv(uploaded_fixations)
        st.sidebar.info("Upload both files or switch to demo data.")
        return load_sample_data()
    return load_sample_data()


def prepare_data(words_df: pd.DataFrame, fixations_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Infer schemas and normalize incoming dataframes."""
    word_schema = infer_word_schema(words_df)
    fix_schema = infer_fix_schema(fixations_df)
    if not word_schema or not fix_schema:
        st.stop()
    return normalize_words(words_df, word_schema), normalize_fixations(fixations_df, fix_schema)


def build_combo_options(fixations: pd.DataFrame) -> Tuple[pd.DataFrame, list[str], Dict[str, Tuple[str, str]]]:
    paragraph_col = "unique_paragraph_id" if "unique_paragraph_id" in fixations.columns else "paragraph_id"
    trial_col = "unique_trial_id" if "unique_trial_id" in fixations.columns else "trial_id"
    combo_cols = ["participant_id", trial_col, paragraph_col]
    for col in ["unique_trial_id", "unique_paragraph_id", "TRIAL_INDEX", "trial_index"]:
        if col in fixations.columns and col not in combo_cols:
            combo_cols.append(col)

    combos = (
        fixations[combo_cols]
        .drop_duplicates()
        .rename(columns={trial_col: "trial_id", paragraph_col: "paragraph_id"})
    )
    if trial_col == "unique_trial_id" and "unique_trial_id" not in combos.columns:
        combos["unique_trial_id"] = combos["trial_id"]
    if paragraph_col == "unique_paragraph_id" and "unique_paragraph_id" not in combos.columns:
        combos["unique_paragraph_id"] = combos["paragraph_id"]
    sort_cols = ["participant_id"]
    if "TRIAL_INDEX" in combos.columns:
        sort_cols.append("TRIAL_INDEX")
    elif "trial_index" in combos.columns:
        sort_cols.append("trial_index")
    sort_cols.append("trial_id")
    combos = combos.sort_values(sort_cols)

    combo_labels = [f"{row.participant_id} / {row.trial_id} · {row.paragraph_id}" for row in combos.itertuples()]
    label_to_combo = dict(
        zip(
            combo_labels,
            combos[["participant_id", "trial_id"]].itertuples(index=False, name=None),
        )
    )
    return combos, combo_labels, label_to_combo


def clamp_canvas_size(words: pd.DataFrame, fixations: pd.DataFrame) -> Tuple[int, int]:
    default_canvas_w, default_canvas_h = compute_canvas_size(words, fixations)
    return (
        min(max(default_canvas_w, 100), 10000),
        min(max(default_canvas_h, 100), 10000),
    )


def select_trial(combos: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    if combos.empty:
        st.warning("No trials available after filtering.")
        st.stop()

    selection_mode = st.radio(
        label="Select trials by",
        options=["None", "Text", "Participant"],
        horizontal=True,
    )
    selected_participant = None
    selected_trial = None

    trial_field = "unique_trial_id" if "unique_trial_id" in combos.columns else "trial_id"
    paragraph_field = "unique_paragraph_id" if "unique_paragraph_id" in combos.columns else "paragraph_id"
    trial_index_field = None
    for candidate in ["TRIAL_INDEX", "trial_index"]:
        if candidate in combos.columns:
            trial_index_field = candidate
            break

    if selection_mode == "None":
        available_trials = combos.drop_duplicates(subset=[trial_field])
        trial_options = sorted(available_trials[trial_field].dropna().astype(str).unique())
        if not trial_options:
            st.warning("No trials available after filtering.")
            st.stop()
        selected_trial_label = st.selectbox("Unique trial id", options=trial_options)
        if selected_trial_label:
            chosen = available_trials[available_trials[trial_field].astype(str) == selected_trial_label].iloc[0]
            selected_participant = chosen["participant_id"]
            selected_trial = chosen["trial_id"]
    elif selection_mode == "Text":
        paragraph_options = sorted(combos[paragraph_field].dropna().astype(str).unique())
        if not paragraph_options:
            st.warning("No text ids available after filtering.")
            st.stop()
        selected_paragraph = st.selectbox("Text id", options=paragraph_options)
        if not selected_paragraph:
            st.warning("No text selected after filtering.")
            st.stop()
        paragraph_combos = combos[combos[paragraph_field].astype(str) == str(selected_paragraph)]
        participant_options = sorted(paragraph_combos["participant_id"].dropna().unique())
        if not participant_options:
            st.warning("No participants available for this text.")
            st.stop()
        selected_participant = st.selectbox("Participant", options=participant_options)
        candidate_trials = (
            paragraph_combos[paragraph_combos["participant_id"] == selected_participant]
            .drop_duplicates(subset=["trial_id"])
            .sort_values("trial_id")
        )
        if not candidate_trials.empty:
            selected_trial = candidate_trials.iloc[0]["trial_id"]
    else:
        participants = sorted(combos["participant_id"].dropna().unique())
        if not participants:
            st.warning("No participants available after filtering.")
            st.stop()
        selected_participant = st.selectbox("Participant", options=participants)
        participant_trials = combos[combos["participant_id"] == selected_participant]
        if participant_trials.empty:
            st.warning("No trials available for this participant.")
            return None, None

        use_trial_index = trial_index_field and participant_trials[trial_index_field].notna().any()
        if use_trial_index:
            slider_options = sorted(participant_trials[trial_index_field].dropna().unique().tolist())
            slider_label = "Trial index"
            slider_field = trial_index_field
        else:
            slider_options = sorted(participant_trials[paragraph_field].dropna().astype(str).unique().tolist())
            slider_label = "Text id"
            slider_field = paragraph_field

        if not slider_options:
            st.warning("No trials available for this selection.")
            return None, None

        slider_value = st.select_slider(slider_label, options=slider_options)
        if slider_value is None:
            return None, None

        if slider_field == paragraph_field:
            trial_candidates = participant_trials[participant_trials[slider_field].astype(str) == str(slider_value)]
        else:
            trial_candidates = participant_trials[participant_trials[slider_field] == slider_value]

        trial_candidates = trial_candidates.drop_duplicates(subset=["trial_id"]).sort_values("trial_id")
        if not trial_candidates.empty:
            selected_trial = trial_candidates.iloc[0]["trial_id"]

    return selected_participant, selected_trial


def compute_trial_stats(trial_words: pd.DataFrame, trial_fixations: pd.DataFrame) -> Dict[str, float]:
    total_time = None
    if "trial_dwell_time_ms" in trial_words.columns:
        dwell_values = pd.to_numeric(trial_words["trial_dwell_time_ms"], errors="coerce").dropna().unique()
        if len(dwell_values):
            total_time = float(dwell_values[0])
    if total_time is None:
        total_time = float(trial_fixations["duration_ms"].sum()) if not trial_fixations.empty else 0.0
    return dict(
        total_reading_time_ms=total_time,
        total_reading_time_s=total_time / 1000.0,
        word_count=int(len(trial_words)),
        fixation_count=int(len(trial_fixations)),
    )


def gather_trial_metadata(
    trial_words: pd.DataFrame, trial_fixations: pd.DataFrame, fields: Iterable[str]
) -> pd.DataFrame:
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
                    and (pd.api.types.is_numeric_dtype(cleaned) or len(numeric_values) == len(cleaned))
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


def export_filtered_trials(
    combos: pd.DataFrame,
    words: pd.DataFrame,
    fixations: pd.DataFrame,
    *,
    canvas_width: int,
    canvas_height: int,
    base_font_size: int,
    font_family: str,
    x_field: str,
    y_field: str,
    settings: Dict,
) -> None:
    if combos.empty:
        st.warning("No trials to export.")
        return

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for combo in combos.itertuples(index=False):
            combo_words = words[
                (words["participant_id"] == combo.participant_id)
                & (words["trial_id"] == combo.trial_id)
            ]
            combo_fix = fixations[
                (fixations["participant_id"] == combo.participant_id)
                & (fixations["trial_id"] == combo.trial_id)
            ]
            combo_fig = make_scanpath_figure(
                combo_words,
                combo_fix,
                canvas_width=int(canvas_width),
                canvas_height=int(canvas_height),
                base_font_size=int(base_font_size),
                font_family=font_family,
                x_field=x_field,
                y_field=y_field,
                show_words=settings["show_words"],
                show_word_labels=settings["show_word_labels"],
                show_fixations=settings["show_fixations"],
                show_order=settings["show_order"],
                show_saccades=settings["show_saccades"],
                show_heatmap=settings["show_heatmap"],
                color_by=settings["color_by"],
                heatmap_metric=settings["heatmap_metric"],
                marker_size_range=settings["marker_size_range"],
                order_font_size=settings["order_font_size"],
                order_font_color=settings["order_font_color"],
                show_colorbars=settings["show_colorbars"],
                fixation_color_range=settings["fixation_color_range"],
                heatmap_range=settings["heatmap_range"],
            )
            img_bytes = combo_fig.to_image(
                format="png",
                width=int(canvas_width),
                height=int(canvas_height),
            )
            filename = f"{combo.participant_id}_{combo.trial_id}.png"
            zf.writestr(filename, img_bytes)
    buf.seek(0)
    st.download_button(
        "Download zip",
        data=buf.getvalue(),
        file_name="scanpaths.zip",
        mime="application/zip",
    )


def render_single_trial_tab(
    words_filtered: pd.DataFrame,
    fixations_filtered: pd.DataFrame,
    combos: pd.DataFrame,
    *,
    canvas_width: int,
    canvas_height: int,
    base_font_size: int,
    font_family: str,
) -> None:
    selected_participant, selected_trial = select_trial(combos)
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

    st.markdown(
        f"Showing **{selected_trial}** "
    )

    stats = compute_trial_stats(trial_words, trial_fixations)
    stat_cols = st.columns(3)
    stat_cols[0].metric("Total reading time (s)", f"{stats['total_reading_time_s']:.1f}")
    stat_cols[1].metric("Number of words", f"{stats['word_count']:,}")
    stat_cols[2].metric("Number of fixations", f"{stats['fixation_count']:,}")

    metadata_candidates: list[str] = []
    for col in list(words_filtered.columns) + list(fixations_filtered.columns):
        if col not in metadata_candidates:
            metadata_candidates.append(col)
    available_metadata = metadata_candidates
    default_metadata = [
        field
        for field in [
            "difficulty_level",
            "repeated_reading_trial",
            "selected_answer",
            "is_correct",
        ]
        if field in available_metadata
    ]
    selected_metadata = st.multiselect(
        "Trial metadata fields",
        options=available_metadata,
        default=default_metadata or available_metadata,
    )
    if selected_metadata:
        metadata_df = gather_trial_metadata(trial_words, trial_fixations, selected_metadata)
        if not metadata_df.empty:
            st.dataframe(metadata_df, hide_index=True, width='stretch')

    viz_settings = sidebar_controls(trial_fixations, base_font_size)
    figure_settings = dict(
        show_words=viz_settings["show_words"],
        show_word_labels=viz_settings["show_labels"],
        show_fixations=viz_settings["show_fix"],
        show_order=viz_settings["show_order"],
        show_saccades=viz_settings["show_saccades"],
        show_heatmap=viz_settings["show_heatmap"],
        color_by=viz_settings["color_by"],
        heatmap_metric=viz_settings["heatmap_metric"] if viz_settings["heatmap_metric"] != "counts" else None,
        marker_size_range=viz_settings["marker_size_range"],
        order_font_size=viz_settings["order_font_size"],
        order_font_color=viz_settings["order_font_color"],
        show_colorbars=viz_settings["show_colorbars"],
        fixation_color_range=viz_settings["fixation_color_range"],
        heatmap_range=viz_settings["heatmap_range"],
    )
    x_field = viz_settings["x_field"]
    y_field = viz_settings["y_field"]

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
    st.plotly_chart(fig, width='content', config={"responsive": False})

    with st.expander("Plot configuration"):
        plot_config = {
            "selection": {"participant_id": selected_participant, "trial_id": selected_trial},
            "canvas_px": {"width": int(canvas_width), "height": int(canvas_height)},
            "axes": {"x_field": x_field, "y_field": y_field},
            "layers": {
                "words": figure_settings["show_words"],
                "word_labels": figure_settings["show_word_labels"],
                "fixations": figure_settings["show_fixations"],
                "order_labels": figure_settings["show_order"],
                "saccades": figure_settings["show_saccades"],
                "heatmap": figure_settings["show_heatmap"],
            },
            "coloring": {
                "color_by": figure_settings["color_by"],
                "heatmap_metric": viz_settings["heatmap_metric"],
                "show_colorbars": figure_settings["show_colorbars"],
                "fixation_range": list(figure_settings["fixation_color_range"]) if figure_settings["fixation_color_range"] else None,
                "heatmap_range": list(figure_settings["heatmap_range"]) if figure_settings["heatmap_range"] else None,
            },
            "sizing": {
                "marker_size_range": [int(s) for s in figure_settings["marker_size_range"]],
                "order_font_size": int(figure_settings["order_font_size"]),
                "order_font_color": figure_settings["order_font_color"],
                "base_font_size": int(base_font_size),
            },
        }
        st.json(plot_config)

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


def render_comparison_tab(
    words_filtered: pd.DataFrame,
    fixations_filtered: pd.DataFrame,
    combo_labels: Iterable[str],
    label_to_combo: Dict[str, Tuple[str, str]],
    *,
    canvas_width: int,
    canvas_height: int,
    font_family: str,
    base_font_size: int,
) -> None:
    st.subheader("Overlay two trials")
    labels = list(combo_labels)
    if len(labels) < 2:
        st.info("Select at least two trials in the filters to compare.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        label_a = st.selectbox("Primary trial", options=labels, index=0)
    with col_b:
        label_b = st.selectbox("Trial to overlay", options=labels, index=1)

    trial_a = label_to_combo[label_a]
    trial_b = label_to_combo[label_b]
    fig_compare = make_comparison_figure(
        words_filtered,
        fixations_filtered,
        trial_a,
        trial_b,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        font_family=font_family,
        base_font_size=int(base_font_size),
    )
    st.plotly_chart(fig_compare, width='content', config={"responsive": False})


def render_metrics_tab(words_filtered: pd.DataFrame, fixations_filtered: pd.DataFrame) -> None:
    st.subheader("Word-level data")
    metrics = compute_word_metrics(words_filtered, fixations_filtered)
    columns_to_show = [
        "participant_id",
        "trial_id",
        "paragraph_id",
        "word_id",
        "text",
        "line_idx",
        "first_fixation_ms",
        "first_pass_gaze_duration_ms",
        "gaze_duration_ms",
        "total_fixation_duration_ms",
        "higher_pass_fixation_duration_ms",
        "n_fixations",
        "skip_flag",
        "regression_out_count",
        "regression_in_count",
    ]
    display_cols = [col for col in columns_to_show if col in metrics.columns]
    st.dataframe(
        metrics[display_cols],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Metrics pulled from IA data: first fixation (FF), first-pass gaze duration (fpGD/GD), total fixation duration (FD), higher-pass dwell time, fixation counts, skips, and regressions."
    )
    st.download_button(
        label="Download metrics as CSV",
        data=metrics.to_csv(index=False),
        file_name="word_metrics.csv",
        mime="text/csv",
    )


def render_fixations_tab(fixations_filtered: pd.DataFrame) -> None:
    st.subheader("Fixation-level data")
    table_cols = [
        col
        for col in [
            "participant_id",
            "trial_id",
            "unique_trial_id",
            "paragraph_id",
            "unique_paragraph_id",
            "fixation_id",
            "order_in_trial",
            "timestamp_ms",
            "duration_ms",
            "word_id",
            "pass_index",
            "saccade_type",
            "eye",
            "noise_flag",
        ]
        if col in fixations_filtered.columns
    ]
    st.dataframe(
        fixations_filtered[table_cols],
        hide_index=True,
        width="stretch",
    )
    st.caption("All fixation records after applying filters; includes ids, timing, and optional flags.")
    st.download_button(
        label="Download filtered fixations as CSV",
        data=fixations_filtered[table_cols].to_csv(index=False),
        file_name="filtered_fixations.csv",
        mime="text/csv",
    )


def render_raw_data_tab(words_filtered: pd.DataFrame, fixations_filtered: pd.DataFrame) -> None:
    word_tab, fixation_tab = st.tabs(["Word-level", "Fixation-level"])
    with word_tab:
        render_metrics_tab(words_filtered, fixations_filtered)
    with fixation_tab:
        render_fixations_tab(fixations_filtered)


def main() -> None:
    st.title("Scanpath Visualization")
    st.caption(
        "Visualize eye-tracking-while-reading scanpaths!"
    )

    st.sidebar.header("Experimental Setup")
    data_choice = st.sidebar.radio(
        "Data source",
        ["Use bundled demo", "Upload csv tables"],
        index=0,
        help=data_dictionary_help_text(),
    )
    words_df, fixations_df = load_words_and_fixations(data_choice)
    words_df, fixations_df = prepare_data(words_df, fixations_df)

    filters = default_filters(words_df, fixations_df)
    words_filtered, fixations_filtered = filter_data(
        words_df,
        fixations_df,
        filters,
    )

    if words_filtered.empty or fixations_filtered.empty:
        st.warning("No data after filtering. Try selecting more participants or trials.")
        return

    combos, combo_labels, label_to_combo = build_combo_options(fixations_filtered)
    canvas_width, canvas_height = clamp_canvas_size(words_filtered, fixations_filtered)
    canvas_width = st.sidebar.number_input(
        "Monitor width (px)",
        min_value=100,
        max_value=10000,
        value=canvas_width,
        step=10,
        help="Use the real monitor width in pixels to keep coordinates true to scale.",
    )
    canvas_height = st.sidebar.number_input(
        "Monitor height (px)",
        min_value=100,
        max_value=10000,
        value=canvas_height,
        step=10,
        help="Use the real monitor height in pixels to keep coordinates true to scale.",
    )
    base_font_size = st.sidebar.number_input(
        "Figure font size (px)",
        min_value=6,
        max_value=72,
        value=12,
        step=1,
        help="Controls label text size; uses a Lucida Sans monospace family.",
    )
    st.sidebar.divider()
    font_family = FONT_FAMILY

    tab_single, tab_compare, tab_raw = st.tabs(
        ["Interactive Plot", "Comparison", "Raw Data"]
    )

    with tab_single:
        render_single_trial_tab(
            words_filtered,
            fixations_filtered,
            combos,
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            base_font_size=int(base_font_size),
            font_family=font_family,
        )

    with tab_compare:
        render_comparison_tab(
            words_filtered,
            fixations_filtered,
            combo_labels,
            label_to_combo,
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            font_family=font_family,
            base_font_size=int(base_font_size),
        )

    with tab_raw:
        render_raw_data_tab(words_filtered, fixations_filtered)


if __name__ == "__main__":
    main()
