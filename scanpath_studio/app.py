"""Scanpath Studio Streamlit app.

This is the main entry point for the Streamlit application that visualizes
eye-tracking scanpaths over text.

Architecture:
    - Entry point: main() function configures Streamlit and orchestrates the UI
    - Data flow: CSV upload → schema inference → normalization → filtering → plotting
    - UI structure: Sidebar controls + 4 tabbed views (Interactive, Animation, Raw Data, Stats)

Data Pipeline:
    1. Load raw CSVs (words + fixations + optional raw gaze)
    2. Infer schema via candidate column matching
    3. Normalize to canonical column names
    4. Apply participant/trial/paragraph filters
    5. Build trial combinations for selection
    6. Render visualizations with user-controlled settings

Usage:
    # Development mode (watch for changes):
    $ streamlit run scanpath_studio/app.py

    # Package mode:
    $ python -m scanpath_studio
    # or
    $ scanpath-studio
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
import streamlit as st

# Allow running via `streamlit run scanpath_studio/app.py` by adding the
# repository root to sys.path when executed as a script instead of a package.
if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from scanpath_studio.annotations import (
    filter_keys,
    render_annotations_sidebar,
)
from scanpath_studio.constants import DEFAULT_LINE_SPACING, FONT_FAMILY
from scanpath_studio.controls import (
    FIX_FIELD_SPECS,
    RAW_GAZE_FIELD_SPECS,
    WORD_FIELD_SPECS,
    column_mapping_ui,
    data_dictionary_help_text,
    sidebar_controls,
    sidebar_trial_filters,
)
from scanpath_studio.data import (
    compute_canvas_size,
    default_filters,
    filter_data,
    filter_raw_gaze,
    filter_to_keys,
    filter_trials,
    infer_fix_schema,
    infer_raw_gaze_schema,
    infer_word_schema,
    load_onestop_server_bundle,
    load_sample_data,
    load_sample_raw_gaze,
    normalize_fixations,
    normalize_raw_gaze,
    normalize_words,
    onestop_data_dir,
    propose_fix_schema,
    propose_raw_gaze_schema,
    propose_word_schema,
    read_table,
    trial_mapping_columns,
    validate_fix_schema,
    validate_raw_gaze_schema,
    validate_word_schema,
)
from scanpath_studio.styles import get_app_css
from scanpath_studio.tabs import (
    render_animation_tab,
    render_data_statistics_tab,
    render_multiple_comparison_tab,
    render_raw_data_tab,
    render_single_trial_tab,
)
from scanpath_studio.utils import (  # noqa: F401
    build_combo_options,
    build_comparison_options as _build_comparison_options,
    compute_trial_stats,
    friendly_trial_label as _friendly_trial_label,
    gather_trial_metadata,
)

UPLOAD_CHOICE = "Upload tables"
DEMO_CHOICE = "Use bundled demo"
# A tiny, fully-specified synthetic trial (scanpath_studio.synthetic)
# with known ground-truth reading measures — handy for sanity-checking the viz
# against documented expected values.
SYNTHETIC_CHOICE = "Synthetic test trial"
# Server-side OneStop lacclab bundle. Only offered when $ONESTOP_DATA_DIR is
# set; selected automatically when the page is opened with `?source=onestop`
# in the URL. See data.load_onestop_server_bundle().
ONESTOP_CHOICE = "OneStop server bundle"

# URL query-param → session_state key map for the deep-link API. Used by
# `_apply_url_preset()` to preset widgets when the page is opened from an
# external tool with a deep link.
#
# Selection prefixes — every selectable tab (Interactive Plot, Animated
# Scanpath, …) renders its own `select_trial` with a different `key_prefix`,
# so a URL deep link has to seed all of them or only the first tab lands on
# the requested trial. Keep this list in sync with the `key_prefix=` values
# passed to `select_trial` in tabs.py.
_SELECTION_PREFIXES = ("single", "anim", "multi")
_URL_PRESETS = {
    # viz prefs (`controls.sidebar_controls`)
    "show_order": ("global_show_order", lambda v: v not in {"0", "false", "no"}),
    "hide_fixation_numbers": ("global_show_order", lambda v: v in {"0", "false", "no"}),
    "show_saccades": ("global_show_saccades", lambda v: v not in {"0", "false", "no"}),
    "show_heatmap": ("global_show_heatmap", lambda v: v not in {"0", "false", "no"}),
    "show_words": ("global_show_words", lambda v: v not in {"0", "false", "no"}),
    "show_labels": ("global_show_labels", lambda v: v not in {"0", "false", "no"}),
    "show_fixations": ("global_show_fix", lambda v: v not in {"0", "false", "no"}),
    "heatmap_colorscale": ("global_heatmap_colorscale", str),
    "fixation_colorscale": ("global_fixation_colorscale", str),
}


def _apply_url_preset() -> Optional[str]:
    """Read `st.query_params` and preset Streamlit session state for deep links.

    Returns the URL-requested `source` ("onestop"/"demo"/"upload") or `None`.
    Call this at the very top of `main()` — before any widgets render — so
    session_state values are picked up as the widgets' initial values.

    URL schema (all params optional):
        ?source=onestop          → force "OneStop server bundle" data source
        &participant=p001        → preselect participant (Participant mode)
        &trial=37                → preselect trial_index slider
        &tab=animation           → land on Animated Scanpath tab
        &heatmap_colorscale=Greens
        &hide_fixation_numbers=1
        &show_saccades=1
        &show_heatmap=1
        ...etc — see _URL_PRESETS above

    Bonus side-effect: when any colorscale is set via URL, also forces the
    "Advanced styling" sidebar expander open so the value is visible/editable.

    External tools can deep-link into this app via the URL schema above to
    land on a specific trial with the reviewer's preferred viz settings.
    """
    qp = st.query_params
    if not qp:
        return None

    # Seed selection state for every tab that exposes a `select_trial` widget.
    # `?participant=` + `?trial=` map onto Participant mode with the matching
    # participant / slider value. Without this loop the Animated Scanpath tab
    # (key_prefix="anim") would default to "Trial" mode and land on the
    # alphabetically-first trial instead of the deep-linked one.
    if "participant" in qp or "trial" in qp:
        for prefix in _SELECTION_PREFIXES:
            st.session_state.setdefault(f"{prefix}_select_trial_mode", "Participant")
            if "participant" in qp:
                st.session_state.setdefault(
                    f"{prefix}_participant", str(qp["participant"])
                )
            if "trial" in qp:
                try:
                    st.session_state.setdefault(f"{prefix}_slider", int(qp["trial"]))
                except (ValueError, TypeError):
                    st.warning(f"Ignored bad URL param ?trial={qp['trial']!r}")

    for url_key, (state_key, coerce) in _URL_PRESETS.items():
        if url_key not in qp:
            continue
        raw = qp[url_key]
        try:
            value = coerce(raw)
        except (ValueError, TypeError):
            st.warning(f"Ignored bad URL param ?{url_key}={raw!r}")
            continue
        st.session_state.setdefault(state_key, value)

    # Heatmap / fixation colorscale only render under the Advanced expander —
    # auto-open it so the URL value is exposed in the sidebar.
    if "heatmap_colorscale" in qp or "fixation_colorscale" in qp:
        st.session_state.setdefault("global_advanced", True)

    source = qp.get("source")
    return source.lower() if source else None


def configure_page() -> None:
    """Streamlit page config + custom CSS.

    When loaded from an iframe with `?embed=true`, Streamlit's built-in embed
    mode already hides the header/menu — we additionally collapse the sidebar
    so the iframe is mostly the plot.
    """
    is_embed = (st.query_params.get("embed") or "").lower() in {"true", "1"}
    st.set_page_config(
        page_title="Scanpath Studio",
        page_icon="👀",
        layout="wide",
        initial_sidebar_state="collapsed" if is_embed else "auto",
    )
    st.markdown(get_app_css(), unsafe_allow_html=True)


def _render_about_panel() -> None:
    """Compact header with title + Lab/Code pill links."""
    from scanpath_studio.constants import CITATION

    title_col, links_col = st.columns([5, 2])
    with title_col:
        st.title("Scanpath Studio")
        st.caption("Interactive exploration of eye movements in reading.")
    with links_col:
        st.markdown(
            f"""<div class="header-link-row">
              <a class="header-link lab" href="https://lacclab.github.io/" target="_blank" rel="noopener">🧪 LaCC Lab</a>
              <a class="header-link code" href="{CITATION["url"]}" target="_blank" rel="noopener">💻 Code</a>
            </div>""",
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_words_and_fixations(
    data_choice: str,
    participant: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load word and fixation data from user uploads or bundled demo files.

    Args:
        data_choice: Either "Upload csv tables" / "Use bundled demo" / "OneStop server bundle"
        participant: Lowercased participant_id from the URL deep link. When set
            AND `data_choice == ONESTOP_CHOICE`, the OneStop loader fast-paths
            to just that pid's Parquet shard — sub-second instead of ~3 min.
            Ignored for the other data sources.

    Returns:
        Tuple of (words_df, fixations_df) as raw DataFrames before normalization

    UI Effects:
        - Renders file uploaders in sidebar when data_choice is "Upload csv tables"
        - Shows info message if uploads are incomplete
        - Falls back to sample data if uploads missing
    """
    if data_choice == SYNTHETIC_CHOICE:
        from scanpath_studio.synthetic import load_synthetic_data

        return load_synthetic_data()
    if data_choice == UPLOAD_CHOICE:
        uploaded_words = st.sidebar.file_uploader(
            "Words/IA table", type=["csv", "parquet", "feather"]
        )
        uploaded_fixations = st.sidebar.file_uploader(
            "Fixations table", type=["csv", "parquet", "feather"]
        )
        if uploaded_words and uploaded_fixations:
            return read_table(uploaded_words), read_table(uploaded_fixations)
        st.sidebar.info("Upload both files or switch to demo data.")
        return load_sample_data()
    if data_choice == ONESTOP_CHOICE:
        words, fixations = load_onestop_server_bundle(participant=participant)
        if words.empty or fixations.empty:
            st.sidebar.warning(
                "OneStop bundle unavailable — falling back to demo data."
            )
            return load_sample_data()
        return words, fixations
    return load_sample_data()


def prepare_data(
    words_df: pd.DataFrame,
    fixations_df: pd.DataFrame,
    allow_override: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Infer schemas and normalize incoming dataframes to canonical column names.

    When ``allow_override`` is True, render sidebar expanders that let the user
    pick the exact column names for each field (pre-filled with auto-detection).
    Otherwise fall back to the original infer-then-stop flow used for demo data.
    """
    if allow_override:
        word_proposed = propose_word_schema(words_df)
        word_problems = validate_word_schema(word_proposed)
        word_schema = column_mapping_ui(
            words_df,
            table_label="Words/IA",
            state_key_prefix="col_map_words",
            field_specs=WORD_FIELD_SPECS,
            proposed=word_proposed,
            problems=word_problems,
        )
        word_problems = validate_word_schema(word_schema)
        if word_problems:
            st.sidebar.error("Words/IA: " + "; ".join(word_problems))
            st.stop()

        fix_proposed = propose_fix_schema(fixations_df)
        fix_problems = validate_fix_schema(fix_proposed)
        fix_schema = column_mapping_ui(
            fixations_df,
            table_label="Fixations",
            state_key_prefix="col_map_fix",
            field_specs=FIX_FIELD_SPECS,
            proposed=fix_proposed,
            problems=fix_problems,
        )
        fix_problems = validate_fix_schema(fix_schema)
        if fix_problems:
            st.sidebar.error("Fixations: " + "; ".join(fix_problems))
            st.stop()
    else:
        word_schema = infer_word_schema(words_df)
        fix_schema = infer_fix_schema(fixations_df)
        if not word_schema or not fix_schema:
            st.stop()

    # Remember whether the trial id was composed from several columns, so the
    # trial picker can offer one cascading selector per component (see
    # utils._select_trial_composite_mode). Recomputed every run so it clears
    # when switching to a single-column / non-upload data source.
    trial_cols = trial_mapping_columns(word_schema["trial"])
    st.session_state["_composite_trial_columns"] = (
        trial_cols if len(trial_cols) > 1 else None
    )

    return normalize_words(words_df, word_schema), normalize_fixations(
        fixations_df, fix_schema
    )


def load_raw_gaze_data(data_choice: str) -> pd.DataFrame:
    """Load and normalize optional raw gaze data (millisecond-level eye positions).

    Raw gaze data provides finer temporal resolution than fixation-level data
    and enables overlay visualizations showing continuous gaze paths.

    Args:
        data_choice: Either "Upload csv tables" or "Use bundled demo"

    Returns:
        Normalized raw gaze DataFrame with canonical columns, or empty DataFrame
        if not available or schema inference fails

    Canonical Columns (raw gaze):
        participant_id, trial_id, x, y, timestamp_ms (optional: eye, noise_flag)

    UI Effects:
        - Renders optional file uploader for "Upload csv tables" mode
        - Shows warning if schema inference fails
        - Shows info message if sample data unavailable
    """
    raw_gaze_df = pd.DataFrame()

    if data_choice == SYNTHETIC_CHOICE:
        # The synthetic trial has no raw gaze; skip the uploader entirely.
        return raw_gaze_df

    if data_choice == DEMO_CHOICE:
        raw_gaze_df = load_sample_raw_gaze()
        if not raw_gaze_df.empty:
            raw_gaze_schema = infer_raw_gaze_schema(raw_gaze_df)
            if raw_gaze_schema:
                raw_gaze_df = normalize_raw_gaze(raw_gaze_df, raw_gaze_schema)
            else:
                st.sidebar.warning("Could not infer raw gaze schema from sample data")
                raw_gaze_df = pd.DataFrame()
    else:
        uploaded_raw_gaze = st.sidebar.file_uploader(
            "Raw gaze table (optional)",
            type=["csv", "parquet", "feather"],
            help="Optional: millisecond-level gaze with participant_id, trial_id, x, y.",
        )
        if uploaded_raw_gaze:
            raw_gaze_df = read_table(uploaded_raw_gaze)
            proposed = propose_raw_gaze_schema(raw_gaze_df)
            initial_problems = validate_raw_gaze_schema(proposed)
            raw_gaze_schema = column_mapping_ui(
                raw_gaze_df,
                table_label="Raw gaze",
                state_key_prefix="col_map_raw_gaze",
                field_specs=RAW_GAZE_FIELD_SPECS,
                proposed=proposed,
                problems=initial_problems,
            )
            problems = validate_raw_gaze_schema(raw_gaze_schema)
            if problems:
                st.sidebar.warning("Raw gaze ignored — " + "; ".join(problems))
                raw_gaze_df = pd.DataFrame()
            else:
                raw_gaze_df = normalize_raw_gaze(raw_gaze_df, raw_gaze_schema)

    return raw_gaze_df


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------


def _sidebar_group(title: str) -> None:
    """Render a section title that groups the toggles below it in the sidebar."""
    st.sidebar.markdown(f"### {title}")


def render_sidebar_data_source() -> str:
    """Render the data source selection radio button in sidebar.

    Returns:
        Selected data source: "Use bundled demo" or "Upload csv tables"

    UI Components:
        - Section header: "Experimental Setup"
        - Radio button with two options and help text
        - Help text explains expected CSV column formats
    """
    # Only offer the OneStop bundle when $ONESTOP_DATA_DIR is set on the
    # server. Outside that context the choice would be a dead-end, so we hide it.
    options = [DEMO_CHOICE, SYNTHETIC_CHOICE, UPLOAD_CHOICE]
    if onestop_data_dir() is not None:
        options.insert(0, ONESTOP_CHOICE)
    # Default to OneStop when it's available AND a deep-link forced it via
    # session_state; otherwise the first option in the list.
    default = st.session_state.get("data_source_choice", options[0])
    if default not in options:
        default = options[0]
    source = st.sidebar.expander("Data source", expanded=True)
    return source.radio(
        "Data source",
        options,
        index=options.index(default),
        help=data_dictionary_help_text(),
        key="data_source_choice",
        label_visibility="collapsed",
    )


def render_sidebar_canvas_controls(
    words_filtered: pd.DataFrame,
    fixations_filtered: pd.DataFrame,
    data_choice: Optional[str] = None,
) -> Tuple[int, int, int, str, float, bool]:
    """Render canvas dimension and font controls in sidebar.

    These controls allow users to match the visualization to their experimental
    display setup, ensuring spatial accuracy and proper word box alignment.

    Args:
        words_filtered: Filtered words dataframe (used to compute default dimensions)
        fixations_filtered: Filtered fixations dataframe (used for coordinate ranges)
        data_choice: Currently selected data source. When it's the OneStop server
            bundle or the bundled demo (a OneStop subset), defaults to the
            OneStop monitor resolution (2560x1440, Dell U2715H — OneStopL1 paper
            §Monitor). Otherwise defaults are derived from data extents.

    Returns:
        Tuple of (canvas_width, canvas_height, base_font_size, font_family,
        line_spacing, scale_text_to_boxes). The text-sizing pair keeps the reading
        text true-to-scale: see `plots._word_label_font_px`.
    """
    # OneStop server bundle + bundled demo share the same experimental setup
    # (Dell U2715H, 2560x1440). Data-derived extents undershoot — text only
    # fills part of the screen — so hard-default to the real monitor here.
    if data_choice in (ONESTOP_CHOICE, DEMO_CHOICE):
        default_canvas_w, default_canvas_h = 2560, 1440
    else:
        default_canvas_w, default_canvas_h = compute_canvas_size(
            words_filtered, fixations_filtered
        )
    canvas_width = min(max(default_canvas_w, 100), 10000)
    canvas_height = min(max(default_canvas_h, 100), 10000)

    display = st.sidebar.expander("Display settings", expanded=False)
    canvas_width = display.number_input(
        "Monitor width (px)",
        min_value=100,
        max_value=10000,
        value=canvas_width,
        step=10,
        help="Use the real monitor width in pixels to keep coordinates true to scale.",
    )
    canvas_height = display.number_input(
        "Monitor height (px)",
        min_value=100,
        max_value=10000,
        value=canvas_height,
        step=10,
        help="Use the real monitor height in pixels to keep coordinates true to scale.",
    )
    # Reading text is true-to-scale by default: it auto-sizes to the word boxes
    # (text height = box_height / line_spacing) and scales with the figure, so it
    # always fills the real line slot. Untick to fall back to a fixed font size.
    scale_text_to_boxes = display.checkbox(
        "Scale text to boxes",
        value=True,
        help="Size the reading text from the word boxes (height = box height ÷ "
        "line spacing) so it stays true to the real experiment at any zoom. "
        "Untick to use the fixed 'Figure font size' below instead.",
    )
    line_spacing = display.number_input(
        "Line spacing",
        min_value=1.0,
        max_value=10.0,
        value=float(DEFAULT_LINE_SPACING),
        step=0.5,
        disabled=not scale_text_to_boxes,
        help="Line slots per line of text. OneStop rendered one blank line above "
        "and one below each text line, so the box spans 3 line heights → 3.",
    )
    base_font_size = display.number_input(
        "Figure font size (px)",
        min_value=6,
        max_value=72,
        value=16,
        step=1,
        help="Real (monitor-pixel) font size, scaled true-to-scale with the "
        "figure. Used for the reading text when 'Scale text to boxes' is off or "
        "the data has no word boxes, and always for axis/legend chrome.",
    )
    font_family = display.text_input(
        "Text font",
        value=FONT_FAMILY,
        help="Font for the word labels. Use the exact font from your experiment "
        "(e.g. 'Courier New') or a CSS fallback stack.",
    )

    return (
        int(canvas_width),
        int(canvas_height),
        int(base_font_size),
        font_family,
        float(line_spacing),
        bool(scale_text_to_boxes),
    )


# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------


def main() -> None:
    """Main application entry point.

    Orchestrates the full application workflow:
        1. Configure Streamlit page and custom CSS
        2. Render title and caption
        3. Load and normalize data (words, fixations, optional raw gaze)
        4. Apply user-selected filters (participants, trials, paragraphs)
        5. Render sidebar controls (canvas, fonts, visualization settings)
        6. Render tabbed UI (Interactive Plot, Animation, Raw Data, Statistics)

    Data Flow:
        CSV upload → schema inference → normalization → filtering →
        trial combination building → visualization rendering

    UI Structure:
        Sidebar: Data source, filters, canvas settings, viz controls
        Main area: 4 tabs for different views of the data

    Error Handling:
        - Stops execution if schema inference fails
        - Shows warning if filtering eliminates all data
        - Handles missing raw gaze data gracefully
    """
    configure_page()
    _render_about_panel()

    # Apply deep-link presets BEFORE any widget renders — see _apply_url_preset
    # for the full URL schema. External tools can deep-link into this app with
    # `?source=...&participant=...&trial=...&...` to land on a specific trial
    # with the reviewer's preferred viz settings.
    url_source = _apply_url_preset()
    if url_source == "onestop" and onestop_data_dir() is not None:
        st.session_state.setdefault("data_source_choice", ONESTOP_CHOICE)
    elif url_source == "demo":
        st.session_state.setdefault("data_source_choice", DEMO_CHOICE)
    elif url_source == "upload":
        st.session_state.setdefault("data_source_choice", UPLOAD_CHOICE)

    # Data source selection (sidebar)
    _sidebar_group("📂 Data")
    data_choice = render_sidebar_data_source()

    # Load and prepare core data (words + fixations). Pass the deep-link
    # participant so the OneStop loader can fast-path to a per-pid shard.
    deep_link_pid = st.session_state.get("single_participant")
    words_df, fixations_df = load_words_and_fixations(
        data_choice, participant=deep_link_pid
    )
    words_df, fixations_df = prepare_data(
        words_df, fixations_df, allow_override=(data_choice == UPLOAD_CHOICE)
    )

    # Load optional raw gaze data
    raw_gaze_df = load_raw_gaze_data(data_choice)

    # Trial-level filtering / grouping (sidebar): narrow by participant, by
    # condition (Hunting/Gathering, difficulty, first/repeated reading,
    # correctness), and by annotation state (favorites / tags) before anything
    # downstream sees the data.
    trial_filters = sidebar_trial_filters(words_df, fixations_df)
    words_df, fixations_df = filter_trials(
        words_df,
        fixations_df,
        participants=trial_filters["participants"],
        metadata=trial_filters["metadata"],
    )
    if (
        trial_filters["favorites_only"]
        or trial_filters["required_tags"]
        or trial_filters["excluded_tags"]
    ) and not fixations_df.empty:
        present_keys = {
            (str(p), str(t))
            for p, t in zip(fixations_df["participant_id"], fixations_df["trial_id"])
        }
        kept = set(
            filter_keys(
                list(present_keys),
                favorites_only=trial_filters["favorites_only"],
                required_tags=trial_filters["required_tags"],
                excluded_tags=trial_filters["excluded_tags"],
            )
        )
        words_df, fixations_df = filter_to_keys(words_df, fixations_df, kept)

    # Apply filters (participant/trial/paragraph selection)
    filters = default_filters(words_df, fixations_df)
    words_filtered, fixations_filtered = filter_data(words_df, fixations_df, filters)

    # Filter raw gaze data to match selected participants/trials
    if not raw_gaze_df.empty:
        raw_gaze_filtered = filter_raw_gaze(
            raw_gaze_df,
            filters.get("participants", []),
            filters.get("trials", []),
        )
        if raw_gaze_filtered.empty:
            # Informational, not an error: the loaded raw-gaze samples just
            # don't cover any trial in the current filter (raw gaze typically
            # exists for only a subset of trials). The overlay is optional.
            st.sidebar.caption(
                f"ℹ️ The loaded raw-gaze samples ({len(raw_gaze_df):,} rows) don't "
                "overlap the current trial filter, so the raw-gaze overlay is "
                "unavailable here."
            )
    else:
        raw_gaze_filtered = pd.DataFrame()

    # Check for empty data after filtering
    if words_filtered.empty or fixations_filtered.empty:
        st.warning(
            "No data after filtering. Loosen the **Filter trials** panel "
            "(participants, condition, or annotation filters) in the sidebar."
        )
        return

    # Build trial combinations for selection UI
    combos, _, _ = build_combo_options(fixations_filtered)

    # Canvas and visualization controls (sidebar)
    _sidebar_group("🎨 Visualization")
    (
        canvas_width,
        canvas_height,
        base_font_size,
        font_family,
        line_spacing,
        scale_text_to_boxes,
    ) = render_sidebar_canvas_controls(words_filtered, fixations_filtered, data_choice)

    has_raw_gaze = not raw_gaze_filtered.empty
    viz_settings = sidebar_controls(
        fixations_filtered, base_font_size, has_raw_gaze=has_raw_gaze
    )

    # Sidebar Annotations panel (download/restore JSON + count). The per-trial
    # star/tags/notes editor lives in the Interactive Plot tab.
    _sidebar_group("📝 Annotations")
    render_annotations_sidebar()

    # Tab pre-selection isn't supported by st.tabs (Streamlit limitation), so
    # when the deep link asks for animation we surface a banner pointing to it.
    requested_tab = (st.query_params.get("tab") or "").lower()
    if requested_tab == "animation":
        st.info("🎬 For the animated view, click the **Animated Scanpath** tab below.")

    # Render tabbed interface
    tab_single, tab_animation, tab_multi, tab_raw, tab_stats = st.tabs(
        [
            "Interactive Plot",
            "Animated Scanpath",
            "Multiple Comparison",
            "Raw Data",
            "Data Statistics",
        ]
    )

    with tab_single:
        render_single_trial_tab(
            words_filtered,
            fixations_filtered,
            combos,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            base_font_size=base_font_size,
            font_family=font_family,
            viz_settings=viz_settings,
            raw_gaze=raw_gaze_filtered,
            line_spacing=line_spacing,
            scale_text_to_boxes=scale_text_to_boxes,
        )

    with tab_animation:
        render_animation_tab(
            words_filtered,
            fixations_filtered,
            combos,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            base_font_size=base_font_size,
            font_family=font_family,
            viz_settings=viz_settings,
            line_spacing=line_spacing,
            scale_text_to_boxes=scale_text_to_boxes,
        )

    with tab_multi:
        render_multiple_comparison_tab(
            words_filtered,
            fixations_filtered,
            combos,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            base_font_size=base_font_size,
            font_family=font_family,
            viz_settings=viz_settings,
            line_spacing=line_spacing,
            scale_text_to_boxes=scale_text_to_boxes,
        )

    with tab_raw:
        render_raw_data_tab(words_filtered, fixations_filtered, raw_gaze_filtered)

    with tab_stats:
        render_data_statistics_tab(
            words_filtered,
            fixations_filtered,
            raw_gaze_filtered,
            combos,
            canvas_width=canvas_width,
            base_font_size=base_font_size,
            font_family=font_family,
        )


if __name__ == "__main__":
    main()
