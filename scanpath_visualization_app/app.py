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
        filter_raw_gaze,
        infer_fix_schema,
        infer_raw_gaze_schema,
        infer_word_schema,
        load_sample_data,
        load_sample_raw_gaze,
        normalize_fixations,
        normalize_raw_gaze,
        normalize_words,
    )
    from .plots import (
        make_comparison_figure,
        make_scanpath_animation,
        make_scanpath_figure,
    )
except ImportError:
    # Allow running via `streamlit run scanpath_visualization_app/app.py`
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scanpath_visualization_app.constants import FONT_FAMILY
    from scanpath_visualization_app.controls import (
        data_dictionary_help_text,
        sidebar_controls,
    )
    from scanpath_visualization_app.data import (
        compute_canvas_size,
        compute_word_metrics,
        default_filters,
        filter_data,
        filter_raw_gaze,
        infer_fix_schema,
        infer_raw_gaze_schema,
        infer_word_schema,
        load_sample_data,
        load_sample_raw_gaze,
        normalize_fixations,
        normalize_raw_gaze,
        normalize_words,
    )
    from scanpath_visualization_app.plots import (
        make_comparison_figure,
        make_scanpath_animation,
        make_scanpath_figure,
    )


st.set_page_config(
    page_title="Scanpath Visualization",
    page_icon="👀",
    layout="wide",
)
st.markdown(
    """
    <style>
    section.main > div.block-container {padding-top: 0.5rem; padding-bottom: 0.5rem;}
    /* Remove all whitespace around plotly charts */
    div[data-testid="stPlotlyChart"] {margin: 0 !important; padding: 0 !important; line-height: 0 !important;}
    div[data-testid="stPlotlyChart"] > div {margin: 0 !important; padding: 0 !important;}
    div[data-testid="stPlotlyChart"] iframe {display: block !important; margin: 0 !important; padding: 0 !important;}
    .stPlotlyChart {margin: 0 !important; padding: 0 !important;}
    /* Target parent containers */
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stPlotlyChart"]) {padding: 0 !important; margin: 0 !important; gap: 0 !important;}
    div[data-testid="element-container"]:has(> div[data-testid="stPlotlyChart"]) {margin: 0 !important; padding: 0 !important;}
    /* Reduce gap in vertical blocks globally */
    div[data-testid="stVerticalBlock"] {gap: 0rem !important;}
    div[data-testid="stVerticalBlock"] > div {margin-bottom: 0.25rem !important;}
    /* Target the js-plotly-plot container */
    .js-plotly-plot, .plot-container, .plotly {margin: 0 !important; padding: 0 !important;}
    .main-svg {display: block !important;}
    /* Remove extra spacing from streamlit elements near charts */
    div[data-testid="stMarkdown"] + div[data-testid="element-container"]:has(div[data-testid="stPlotlyChart"]) {margin-top: 0 !important;}
    div[data-testid="element-container"]:has(div[data-testid="stPlotlyChart"]) + div[data-testid="stExpander"] {margin-top: 0.5rem !important;}
    /* Reduce spacing around dataframes */
    div[data-testid="stDataFrame"] {margin-bottom: 0 !important;}
    div[data-testid="element-container"]:has(div[data-testid="stDataFrame"]) {margin-bottom: 0.25rem !important;}
    /* Reduce multiselect spacing */
    div[data-testid="stMultiSelect"] {margin-bottom: 0.25rem !important;}
    /* Disable fade in/out animations on element updates */
    div[data-testid="stPlotlyChart"], div[data-testid="element-container"], .stMarkdown, .element-container {
        animation: none !important;
        transition: none !important;
    }
    div[data-testid="stPlotlyChart"] * {
        animation: none !important;
        transition: none !important;
    }
    /* Disable Streamlit's stale element fade effect */
    [data-stale="true"] {
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
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


def prepare_data(
    words_df: pd.DataFrame, fixations_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Infer schemas and normalize incoming dataframes."""
    word_schema = infer_word_schema(words_df)
    fix_schema = infer_fix_schema(fixations_df)
    if not word_schema or not fix_schema:
        st.stop()
    return normalize_words(words_df, word_schema), normalize_fixations(
        fixations_df, fix_schema
    )


def build_combo_options(
    fixations: pd.DataFrame,
) -> Tuple[pd.DataFrame, list[str], Dict[str, Tuple[str, str]]]:
    paragraph_col = (
        "unique_paragraph_id"
        if "unique_paragraph_id" in fixations.columns
        else "paragraph_id"
    )
    trial_col = (
        "unique_trial_id" if "unique_trial_id" in fixations.columns else "trial_id"
    )
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
    if (
        paragraph_col == "unique_paragraph_id"
        and "unique_paragraph_id" not in combos.columns
    ):
        combos["unique_paragraph_id"] = combos["paragraph_id"]
    sort_cols = ["participant_id"]
    if "TRIAL_INDEX" in combos.columns:
        sort_cols.append("TRIAL_INDEX")
    elif "trial_index" in combos.columns:
        sort_cols.append("trial_index")
    sort_cols.append("trial_id")
    combos = combos.sort_values(sort_cols)

    combo_labels = [
        f"{row.participant_id} / {row.trial_id} · {row.paragraph_id}"
        for row in combos.itertuples()
    ]
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


def select_trial(
    combos: pd.DataFrame, key_prefix: str = ""
) -> Tuple[Optional[str], Optional[str], str, Optional[str]]:
    """Select a trial from the available combinations.

    Returns:
        Tuple of (participant_id, trial_id, selection_mode, selected_text)
        - selection_mode: "None", "Text", or "Participant"
        - selected_text: the text/paragraph id when in Text or Participant mode
    """
    if combos.empty:
        st.warning("No trials available after filtering.")
        st.stop()

    selection_mode = st.radio(
        label="Select trials by",
        options=["None", "Text", "Participant"],
        horizontal=True,
        key=f"{key_prefix}_select_trial_mode" if key_prefix else None,
    )
    selected_participant = None
    selected_trial = None
    selected_text = None

    trial_field = (
        "unique_trial_id" if "unique_trial_id" in combos.columns else "trial_id"
    )
    paragraph_field = (
        "unique_paragraph_id"
        if "unique_paragraph_id" in combos.columns
        else "paragraph_id"
    )
    trial_index_field = None
    for candidate in ["TRIAL_INDEX", "trial_index"]:
        if candidate in combos.columns:
            trial_index_field = candidate
            break

    if selection_mode == "None":
        available_trials = combos.drop_duplicates(subset=[trial_field])
        trial_options = sorted(
            available_trials[trial_field].dropna().astype(str).unique()
        )
        if not trial_options:
            st.warning("No trials available after filtering.")
            st.stop()

        # Initialize session state for trial index if needed
        state_key = f"{key_prefix}_trial_index" if key_prefix else "trial_index"
        if state_key not in st.session_state:
            st.session_state[state_key] = 0

        # Ensure index is within bounds
        current_idx = st.session_state[state_key]
        if current_idx >= len(trial_options):
            current_idx = len(trial_options) - 1
            st.session_state[state_key] = current_idx
        if current_idx < 0:
            current_idx = 0
            st.session_state[state_key] = current_idx

        # Navigation buttons and selectbox
        nav_col1, nav_col2, select_col = st.columns([1, 1, 4])
        with nav_col1:
            if st.button(
                "← Prev",
                key=f"{key_prefix}_prev_btn" if key_prefix else "prev_btn",
                disabled=current_idx <= 0,
                width='stretch',
            ):
                st.session_state[state_key] = current_idx - 1
                st.rerun()
        with nav_col2:
            if st.button(
                "Next →",
                key=f"{key_prefix}_next_btn" if key_prefix else "next_btn",
                disabled=current_idx >= len(trial_options) - 1,
                width='stretch',
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
            # Update session state if user manually selects a different trial
            if selected_trial_label:
                new_idx = trial_options.index(selected_trial_label)
                if new_idx != current_idx:
                    st.session_state[state_key] = new_idx

        if selected_trial_label:
            chosen = available_trials[
                available_trials[trial_field].astype(str) == selected_trial_label
            ].iloc[0]
            selected_participant = chosen["participant_id"]
            selected_trial = chosen["trial_id"]
            selected_text = (
                str(chosen[paragraph_field])
                if paragraph_field in chosen.index
                else None
            )
    elif selection_mode == "Text":
        paragraph_options = sorted(
            combos[paragraph_field].dropna().astype(str).unique()
        )
        if not paragraph_options:
            st.warning("No text ids available after filtering.")
            st.stop()
        selected_paragraph = st.selectbox(
            "Text id",
            options=paragraph_options,
            key=f"{key_prefix}_text_id" if key_prefix else None,
        )
        if not selected_paragraph:
            st.warning("No text selected after filtering.")
            st.stop()
        selected_text = selected_paragraph
        paragraph_combos = combos[
            combos[paragraph_field].astype(str) == str(selected_paragraph)
        ]
        participant_options = sorted(
            paragraph_combos["participant_id"].dropna().unique()
        )
        if not participant_options:
            st.warning("No participants available for this text.")
            st.stop()
        selected_participant = st.selectbox(
            "Participant",
            options=participant_options,
            key=f"{key_prefix}_participant_text" if key_prefix else None,
        )
        # Handle multiple readings: allow selection if participant read this text multiple times
        candidate_trials = (
            paragraph_combos[paragraph_combos["participant_id"] == selected_participant]
            .drop_duplicates(subset=["trial_id"])
            .sort_values("trial_id")
        )
        if candidate_trials.empty:
            st.warning("No trials available for this selection.")
            return None, None, selection_mode, selected_text
        if len(candidate_trials) > 1:
            # Multiple readings - let user pick
            trial_options = candidate_trials["trial_id"].tolist()
            selected_trial = st.selectbox(
                "Reading (multiple trials available)",
                options=trial_options,
                key=f"{key_prefix}_reading_text" if key_prefix else None,
                help="This participant read this text multiple times.",
            )
        else:
            selected_trial = candidate_trials.iloc[0]["trial_id"]
    else:
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
            return None, None, selection_mode, None

        use_trial_index = (
            trial_index_field and participant_trials[trial_index_field].notna().any()
        )
        if use_trial_index:
            slider_options = sorted(
                participant_trials[trial_index_field].dropna().unique().tolist()
            )
            slider_label = "Trial index"
            slider_field = trial_index_field
        else:
            slider_options = sorted(
                participant_trials[paragraph_field]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            slider_label = "Text id"
            slider_field = paragraph_field

        if not slider_options:
            st.warning("No trials available for this selection.")
            return None, None, selection_mode, None

        slider_value = st.select_slider(
            slider_label,
            options=slider_options,
            key=f"{key_prefix}_slider" if key_prefix else None,
        )
        if slider_value is None:
            return None, None, selection_mode, None

        if slider_field == paragraph_field:
            trial_candidates = participant_trials[
                participant_trials[slider_field].astype(str) == str(slider_value)
            ]
            selected_text = str(slider_value)
        else:
            trial_candidates = participant_trials[
                participant_trials[slider_field] == slider_value
            ]
            # Get the text from the trial candidate
            if (
                not trial_candidates.empty
                and paragraph_field in trial_candidates.columns
            ):
                selected_text = str(trial_candidates.iloc[0][paragraph_field])

        trial_candidates = trial_candidates.drop_duplicates(
            subset=["trial_id"]
        ).sort_values("trial_id")
        if trial_candidates.empty:
            st.warning("No trials available for this selection.")
            return None, None, selection_mode, selected_text
        if len(trial_candidates) > 1:
            # Multiple readings - let user pick
            trial_options = trial_candidates["trial_id"].tolist()
            selected_trial = st.selectbox(
                "Reading (multiple trials available)",
                options=trial_options,
                key=f"{key_prefix}_reading_participant" if key_prefix else None,
                help="This participant read this text multiple times.",
            )
        else:
            selected_trial = trial_candidates.iloc[0]["trial_id"]

    return selected_participant, selected_trial, selection_mode, selected_text


def compute_trial_stats(
    trial_words: pd.DataFrame, trial_fixations: pd.DataFrame
) -> Dict[str, float]:
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

    total_trials = len(combos)
    progress = st.progress(0, text="Preparing exports...")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for idx, combo in enumerate(combos.itertuples(index=False), start=1):
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
                fixation_colorscale=settings.get("fixation_colorscale", "Blues"),
                heatmap_colorscale=settings.get("heatmap_colorscale", "Oranges"),
            )
            img_bytes = combo_fig.to_image(
                format="png",
                width=int(canvas_width),
                height=int(canvas_height),
            )
            filename = f"{combo.participant_id}_{combo.trial_id}.png"
            zf.writestr(filename, img_bytes)
            progress.progress(
                int(idx / total_trials * 100),
                text=f"Exporting trial {idx} of {total_trials}...",
            )

    progress.progress(100, text="Export ready! Click below to download.")
    buf.seek(0)
    st.download_button(
        "Download zip",
        data=buf.getvalue(),
        file_name="scanpaths.zip",
        mime="application/zip",
    )


def _friendly_trial_label(
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
        base = f"{participant_id} · {text_str}"
        # Only append trial id when it isn't already encoded in the text label
        if not trial_contains_text:
            base = f"{base} (trial {trial_str})" if trial_str else base
    else:
        base = f"{participant_id} · {trial_str}" if trial_str else participant_id

    label = f"{prefix}{base}"
    if label in existing_labels:
        label = f"{prefix}{base} [{trial_str or 'trial'}]"
    existing_labels.add(label)
    return label


def _build_comparison_options(
    combos: pd.DataFrame,
    selection_mode: str,
    primary_participant: str,
    primary_trial: str,
    primary_text: Optional[str],
) -> list[Tuple[str, str, str]]:
    """Build prioritized list of comparison trial options.

    Returns list of (participant_id, trial_id, label) tuples, prioritized by:
    - If selection_mode is "Text": same participant first (different readings)
    - If selection_mode is "Participant": same text first (other readers)
    - Otherwise: all other trials
    """
    paragraph_field = (
        "unique_paragraph_id"
        if "unique_paragraph_id" in combos.columns
        else "paragraph_id"
    )

    options: list[Tuple[str, str, str]] = []
    added = set()
    used_labels: set[str] = set()

    # Helper to add options from a filtered dataframe
    def add_options(df: pd.DataFrame, prefix: str = ""):
        for row in df.itertuples():
            key = (row.participant_id, row.trial_id)
            if key not in added and key != (primary_participant, primary_trial):
                text_id = getattr(row, paragraph_field, "")
                label = _friendly_trial_label(
                    row.participant_id,
                    row.trial_id,
                    text_id,
                    used_labels,
                    prefix=prefix,
                )
                options.append((row.participant_id, row.trial_id, label))
                added.add(key)

    if selection_mode == "Text" and primary_text:
        # Prioritize same participant (different readings of same text)
        same_participant_same_text = combos[
            (combos["participant_id"] == primary_participant)
            & (combos[paragraph_field].astype(str) == str(primary_text))
        ].drop_duplicates(subset=["trial_id"])
        add_options(same_participant_same_text, "★ ")

        # Then other participants reading same text
        other_participants_same_text = combos[
            (combos["participant_id"] != primary_participant)
            & (combos[paragraph_field].astype(str) == str(primary_text))
        ].drop_duplicates(subset=["participant_id", "trial_id"])
        add_options(other_participants_same_text)

        # Then all other trials
        all_others = combos.drop_duplicates(subset=["participant_id", "trial_id"])
        add_options(all_others)

    elif selection_mode == "Participant" and primary_text:
        # Prioritize same text (other readers)
        other_participants_same_text = combos[
            (combos["participant_id"] != primary_participant)
            & (combos[paragraph_field].astype(str) == str(primary_text))
        ].drop_duplicates(subset=["participant_id", "trial_id"])
        add_options(other_participants_same_text, "★ ")

        # Then same participant different texts
        same_participant_other_texts = combos[
            (combos["participant_id"] == primary_participant)
            & (combos[paragraph_field].astype(str) != str(primary_text))
        ].drop_duplicates(subset=["trial_id"])
        add_options(same_participant_other_texts)

        # Then all other trials
        all_others = combos.drop_duplicates(subset=["participant_id", "trial_id"])
        add_options(all_others)
    else:
        # No special prioritization
        all_others = combos.drop_duplicates(subset=["participant_id", "trial_id"])
        add_options(all_others)

    return options


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
    selected_participant, selected_trial, selection_mode, selected_text = select_trial(
        combos, key_prefix="single"
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

    # Filter raw gaze data for this trial
    trial_raw_gaze = pd.DataFrame()
    if raw_gaze is not None and not raw_gaze.empty:
        trial_raw_gaze = raw_gaze[
            (raw_gaze["participant_id"] == selected_participant)
            & (raw_gaze["trial_id"] == selected_trial)
        ]

    st.markdown(f"Showing **{selected_trial}** ")

    figure_settings = dict(
        show_words=viz_settings["show_words"],
        show_word_labels=viz_settings["show_labels"],
        show_fixations=viz_settings["show_fix"],
        show_order=viz_settings["show_order"],
        show_saccades=viz_settings["show_saccades"],
        show_heatmap=viz_settings["show_heatmap"],
        show_raw_gaze=viz_settings["show_raw_gaze"],
        color_by=viz_settings["color_by"],
        heatmap_metric=viz_settings["heatmap_metric"]
        if viz_settings["heatmap_metric"] != "counts"
        else None,
        marker_size_range=viz_settings["marker_size_range"],
        order_font_size=viz_settings["order_font_size"],
        order_font_color=viz_settings["order_font_color"],
        show_colorbars=viz_settings["show_colorbars"],
        fixation_color_range=viz_settings["fixation_color_range"],
        heatmap_range=viz_settings["heatmap_range"],
        fixation_colorscale=viz_settings["fixation_colorscale"],
        heatmap_colorscale=viz_settings["heatmap_colorscale"],
        raw_gaze=trial_raw_gaze if not trial_raw_gaze.empty else None,
    )
    x_field = viz_settings["x_field"]
    y_field = viz_settings["y_field"]

    # Comparison toggle (above the plot)
    compare_enabled = st.checkbox(
        "Compare with another trial",
        value=False,
        key="single_compare_toggle",
        help="Overlay another trial's scanpath for comparison.",
    )

    compare_participant = None
    compare_trial = None
    if compare_enabled:
        comparison_options = _build_comparison_options(
            combos, selection_mode, selected_participant, selected_trial, selected_text
        )

        if not comparison_options:
            st.info("No other trials available for comparison.")
        else:
            # Build selectbox options
            option_labels = [opt[2] for opt in comparison_options]
            label_to_trial = {opt[2]: (opt[0], opt[1]) for opt in comparison_options}

            # Show hint about prioritization
            if selection_mode == "Text":
                hint = "★ indicates same participant (multiple readings)"
            elif selection_mode == "Participant":
                hint = "★ indicates same text (other readers)"
            else:
                hint = None

            selected_compare_label = st.selectbox(
                "Compare with trial",
                options=option_labels,
                key="single_compare_trial",
                help=hint,
            )

            if selected_compare_label:
                compare_participant, compare_trial = label_to_trial[
                    selected_compare_label
                ]

    # Show comparison figure if enabled and a trial is selected, otherwise show single trial figure
    if (
        compare_enabled
        and compare_participant is not None
        and compare_trial is not None
    ):
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
        primary_label = _friendly_trial_label(
            selected_participant,
            selected_trial,
            primary_text_id,
            label_pool,
        )
        compare_label = _friendly_trial_label(
            compare_participant,
            compare_trial,
            compare_text_id,
            label_pool,
        )
        trial_a = (selected_participant, selected_trial)
        trial_b = (compare_participant, compare_trial)

        fig_compare = make_comparison_figure(
            words_filtered,
            fixations_filtered,
            trial_a,
            trial_b,
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            font_family=font_family,
            base_font_size=int(base_font_size),
            show_words=viz_settings["show_words"],
            show_word_labels=viz_settings["show_labels"],
            trial_labels=(primary_label, compare_label),
        )
        st.plotly_chart(fig_compare, width="content", config={"responsive": False})
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

    stats = compute_trial_stats(trial_words, trial_fixations)
    stat_cols = st.columns(3)
    stat_cols[0].metric(
        "Total reading time (s)", f"{stats['total_reading_time_s']:.1f}"
    )
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
        metadata_df = gather_trial_metadata(
            trial_words, trial_fixations, selected_metadata
        )
        if not metadata_df.empty:
            st.dataframe(metadata_df, hide_index=True, width="stretch")

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
                "fixation_range": list(figure_settings["fixation_color_range"])
                if figure_settings["fixation_color_range"]
                else None,
                "heatmap_range": list(figure_settings["heatmap_range"])
                if figure_settings["heatmap_range"]
                else None,
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


def render_metrics_tab(
    words_filtered: pd.DataFrame, fixations_filtered: pd.DataFrame
) -> None:
    st.subheader("Word-level data")
    metrics = compute_word_metrics(words_filtered, fixations_filtered)

    PAGE_SIZE = 1000
    total_rows = len(metrics)
    total_pages = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)

    if total_rows > PAGE_SIZE:
        st.info(
            f"Showing {total_rows:,} rows with pagination ({PAGE_SIZE:,} per page)."
        )
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            key="metrics_page",
            help=f"Total pages: {total_pages:,}",
        )
        start_idx = (page - 1) * PAGE_SIZE
        end_idx = min(start_idx + PAGE_SIZE, total_rows)
        display_df = metrics.iloc[start_idx:end_idx]
        st.caption(f"Showing rows {start_idx + 1:,} – {end_idx:,} of {total_rows:,}")
    else:
        display_df = metrics

    st.dataframe(
        display_df,
        hide_index=True,
        width='stretch',
    )
    st.caption("Word-level data with computed reading metrics where available.")


def render_fixations_tab(fixations_filtered: pd.DataFrame) -> None:
    st.subheader("Fixation-level data")

    PAGE_SIZE = 1000
    total_rows = len(fixations_filtered)
    total_pages = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)

    if total_rows > PAGE_SIZE:
        st.info(
            f"Showing {total_rows:,} rows with pagination ({PAGE_SIZE:,} per page)."
        )
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            key="fixations_page",
            help=f"Total pages: {total_pages:,}",
        )
        start_idx = (page - 1) * PAGE_SIZE
        end_idx = min(start_idx + PAGE_SIZE, total_rows)
        display_df = fixations_filtered.iloc[start_idx:end_idx]
        st.caption(f"Showing rows {start_idx + 1:,} – {end_idx:,} of {total_rows:,}")
    else:
        display_df = fixations_filtered

    st.dataframe(
        display_df,
        hide_index=True,
        width='stretch',
    )
    st.caption(
        "All fixation records after applying filters; includes ids, timing, and optional flags."
    )


def render_raw_gaze_tab(raw_gaze_filtered: pd.DataFrame) -> None:
    st.subheader("Raw gaze data")

    if raw_gaze_filtered.empty:
        st.info("No raw gaze data available after filtering.")
        return

    PAGE_SIZE = 1000
    total_rows = len(raw_gaze_filtered)
    total_pages = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)

    if total_rows > PAGE_SIZE:
        st.info(
            f"Showing {total_rows:,} rows with pagination ({PAGE_SIZE:,} per page)."
        )
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            key="raw_gaze_page",
            help=f"Total pages: {total_pages:,}",
        )
        start_idx = (page - 1) * PAGE_SIZE
        end_idx = min(start_idx + PAGE_SIZE, total_rows)
        display_df = raw_gaze_filtered.iloc[start_idx:end_idx]
        st.caption(f"Showing rows {start_idx + 1:,} – {end_idx:,} of {total_rows:,}")
    else:
        display_df = raw_gaze_filtered

    st.dataframe(display_df, hide_index=True, width="stretch")
    st.caption("Millisecond-level gaze samples after applying filters.")


def render_raw_data_tab(
    words_filtered: pd.DataFrame,
    fixations_filtered: pd.DataFrame,
    raw_gaze_filtered: pd.DataFrame,
) -> None:
    word_tab, fixation_tab, raw_gaze_tab = st.tabs(
        ["Word-level", "Fixation-level", "Raw gaze"]
    )
    with word_tab:
        render_metrics_tab(words_filtered, fixations_filtered)
    with fixation_tab:
        render_fixations_tab(fixations_filtered)
    with raw_gaze_tab:
        render_raw_gaze_tab(raw_gaze_filtered)


def _safe_summary(series: pd.Series) -> dict:
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


def render_data_statistics_tab(
    words_filtered: pd.DataFrame,
    fixations_filtered: pd.DataFrame,
    raw_gaze_filtered: pd.DataFrame,
) -> None:
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
        set(words_filtered[paragraph_col].unique()) if paragraph_col in words_filtered.columns else set()
    )

    num_participants = len(participant_ids)
    num_trials = len(trial_ids)
    num_texts = len(text_ids)
    num_fixations = len(fixations_filtered)
    num_words = len(words_filtered)
    num_gaze_points = len(raw_gaze_filtered) if not raw_gaze_filtered.empty else 0

    top_cols = st.columns(6)
    top_cols[0].metric("Participants", f"{num_participants:,}")
    top_cols[1].metric("Texts", f"{num_texts:,}")
    top_cols[2].metric("Trials", f"{num_trials:,}")
    top_cols[3].metric("Fixations", f"{num_fixations:,}")
    top_cols[4].metric("Words", f"{num_words:,}")
    top_cols[5].metric(
        "Gaze points", f"{num_gaze_points:,}", help="Counts raw gaze samples if provided."
    )

    st.divider()

    # Trials per participant
    trial_source = fixations_filtered if not fixations_filtered.empty else words_filtered
    trials_per_participant = (
        trial_source.groupby("participant_id")["trial_id"].nunique()
        if not trial_source.empty
        else pd.Series(dtype=float)
    )
    trials_summary = _safe_summary(trials_per_participant)

    # Fixations per trial
    fixations_per_trial = (
        fixations_filtered.groupby(["participant_id", "trial_id"]).size()
        if not fixations_filtered.empty
        else pd.Series(dtype=float)
    )
    fixations_summary = _safe_summary(fixations_per_trial)

    # Words per trial
    words_per_trial = (
        words_filtered.groupby(["participant_id", "trial_id"]).size()
        if not words_filtered.empty
        else pd.Series(dtype=float)
    )
    words_summary = _safe_summary(words_per_trial)

    stats_df = pd.DataFrame(
        [
            {
                "Metric": "Trials per participant",
                "Mean": trials_summary["mean"],
                "Std": trials_summary["std"],
                "Min": trials_summary["min"],
                "Median": trials_summary["median"],
                "Max": trials_summary["max"],
            },
            {
                "Metric": "Fixations per trial",
                "Mean": fixations_summary["mean"],
                "Std": fixations_summary["std"],
                "Min": fixations_summary["min"],
                "Median": fixations_summary["median"],
                "Max": fixations_summary["max"],
            },
            {
                "Metric": "Words per trial",
                "Mean": words_summary["mean"],
                "Std": words_summary["std"],
                "Min": words_summary["min"],
                "Median": words_summary["median"],
                "Max": words_summary["max"],
            },
        ]
    )

    st.dataframe(
        stats_df,
        hide_index=True,
        width="stretch",
        column_config={col: st.column_config.NumberColumn(format="%.2f") for col in ["Mean", "Std", "Min", "Median", "Max"]},
    )

    st.caption(
        "Statistics computed after filtering; missing values indicate the source data was empty for that metric."
    )


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
    """Render the animated scanpath tab showing fixations progressing over time."""
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

    # Playback speed control
    speed_options = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0]
    speed_labels = ["×0.25", "×0.5", "×1", "×1.5", "×2", "×4", "×8"]
    playback_speed = st.select_slider(
        "Playback speed",
        options=speed_options,
        value=1.0,
        format_func=lambda x: speed_labels[speed_options.index(x)],
        help="Controls playback speed relative to actual fixation durations. ×1 = real-time, ×2 = twice as fast.",
        key="anim_playback_speed",
    )

    if trial_fixations.empty:
        st.warning("No fixations available for this trial.")
        return

    n_fixations = len(trial_fixations)
    total_duration_ms = trial_fixations["duration_ms"].sum()
    playback_duration_s = total_duration_ms / playback_speed / 1000
    st.info(
        f"**{n_fixations} fixations** · Total duration: {total_duration_ms / 1000:.1f}s · Playback time at ×{playback_speed}: {playback_duration_s:.1f}s"
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
        "or drag the slider to jump to any fixation. Orange highlight shows the current fixation. "
        "Each fixation is displayed proportional to its actual duration."
    )


def main() -> None:
    st.title("Scanpath Visualization")
    st.caption("Visualize eye-tracking-while-reading scanpaths!")

    st.sidebar.header("Experimental Setup")
    data_choice = st.sidebar.radio(
        "Data source",
        ["Use bundled demo", "Upload csv tables"],
        index=0,
        help=data_dictionary_help_text(),
    )
    words_df, fixations_df = load_words_and_fixations(data_choice)
    words_df, fixations_df = prepare_data(words_df, fixations_df)

    # Load and process raw gaze data
    raw_gaze_df = pd.DataFrame()
    if data_choice == "Use bundled demo":
        raw_gaze_df = load_sample_raw_gaze()
        if not raw_gaze_df.empty:
            raw_gaze_schema = infer_raw_gaze_schema(raw_gaze_df)
            if raw_gaze_schema:
                raw_gaze_df = normalize_raw_gaze(raw_gaze_df, raw_gaze_schema)
            else:
                st.sidebar.warning("Could not infer raw gaze schema from sample data")
                raw_gaze_df = pd.DataFrame()
        else:
            st.sidebar.info("No sample raw gaze data available")
    else:
        uploaded_raw_gaze = st.sidebar.file_uploader(
            "Raw gaze csv (optional)",
            type=["csv"],
            help="Optional: millisecond-level gaze data with participant_id, trial_id, x, y columns.",
        )
        if uploaded_raw_gaze:
            raw_gaze_df = pd.read_csv(uploaded_raw_gaze)
            raw_gaze_schema = infer_raw_gaze_schema(raw_gaze_df)
            if raw_gaze_schema:
                raw_gaze_df = normalize_raw_gaze(raw_gaze_df, raw_gaze_schema)
            else:
                raw_gaze_df = pd.DataFrame()

    filters = default_filters(words_df, fixations_df)
    words_filtered, fixations_filtered = filter_data(
        words_df,
        fixations_df,
        filters,
    )

    # Filter raw gaze data
    if not raw_gaze_df.empty:
        raw_gaze_filtered = filter_raw_gaze(
            raw_gaze_df,
            filters.get("participants", []),
            filters.get("trials", []),
        )
        if raw_gaze_filtered.empty and not raw_gaze_df.empty:
            st.sidebar.warning(
                f"Raw gaze data was filtered out. "
                f"Original: {len(raw_gaze_df)} rows, "
                f"After filter: {len(raw_gaze_filtered)} rows"
            )
    else:
        raw_gaze_filtered = pd.DataFrame()

    if words_filtered.empty or fixations_filtered.empty:
        st.warning(
            "No data after filtering. Try selecting more participants or trials."
        )
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
        value=16,
        step=1,
        help="Match the font size used in your experiment to keep bounding boxes aligned.",
    )
    font_family = st.sidebar.text_input(
        "Word label font family",
        value=FONT_FAMILY,
        help="Use the exact font from the experiment (e.g., 'Arial' or a fall-back stack).",
    )
    st.sidebar.divider()

    # Create visualization controls once, shared by all tabs
    # Use filtered fixations to determine available fields
    has_raw_gaze = not raw_gaze_filtered.empty
    viz_settings = sidebar_controls(
        fixations_filtered, base_font_size, has_raw_gaze=has_raw_gaze
    )

    tab_single, tab_animation, tab_raw, tab_stats = st.tabs(
        ["Interactive Plot", "Animated Scanpath", "Raw Data", "Data Statistics"]
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
            viz_settings=viz_settings,
            raw_gaze=raw_gaze_filtered,
        )

    with tab_animation:
        render_animation_tab(
            words_filtered,
            fixations_filtered,
            combos,
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            base_font_size=int(base_font_size),
            font_family=font_family,
            viz_settings=viz_settings,
        )

    with tab_raw:
        render_raw_data_tab(words_filtered, fixations_filtered, raw_gaze_filtered)

    with tab_stats:
        render_data_statistics_tab(
            words_filtered, fixations_filtered, raw_gaze_filtered
        )


if __name__ == "__main__":
    main()
