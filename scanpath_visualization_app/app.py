"""Scanpath Visualization Streamlit app.

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
    $ streamlit run scanpath_visualization_app/app.py
    
    # Package mode:
    $ python -m scanpath_visualization_app
    # or
    $ scanpath-visualization
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
import streamlit as st

# Allow running via `streamlit run scanpath_visualization_app/app.py` by adding the
# repository root to sys.path when executed as a script instead of a package.
if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from scanpath_visualization_app.constants import FONT_FAMILY
from scanpath_visualization_app.controls import data_dictionary_help_text, sidebar_controls
from scanpath_visualization_app.data import (
    compute_canvas_size,
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
from scanpath_visualization_app.styles import get_app_css
from scanpath_visualization_app.tabs import (
    render_animation_tab,
    render_data_statistics_tab,
    render_raw_data_tab,
    render_single_trial_tab,
)
from scanpath_visualization_app.utils import build_combo_options

# Re-export utility functions for backward compatibility with tests/imports
from scanpath_visualization_app.utils import (  # noqa: F401
    build_comparison_options as _build_comparison_options,
    compute_trial_stats,
    friendly_trial_label as _friendly_trial_label,
    gather_trial_metadata,
)


def clamp_canvas_size(words: pd.DataFrame, fixations: pd.DataFrame) -> Tuple[int, int]:
    """Clamp canvas dimensions to reasonable bounds (backward compatibility wrapper).
    
    This function ensures canvas dimensions fall within safe rendering limits.
    
    Args:
        words: Normalized words dataframe with spatial coordinates
        fixations: Normalized fixations dataframe with x, y positions
        
    Returns:
        Tuple of (width, height) clamped to range [100, 10000] pixels
        
    Note:
        Kept for backward compatibility with existing test code. New code should
        use compute_canvas_size() directly and apply clamping inline if needed.
    """
    default_canvas_w, default_canvas_h = compute_canvas_size(words, fixations)
    return (
        min(max(default_canvas_w, 100), 10000),
        min(max(default_canvas_h, 100), 10000),
    )


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------


def configure_page() -> None:
    """Set up Streamlit page configuration and custom CSS.
    
    Configures:
        - Page title and favicon
        - Wide layout mode for better use of screen space
        - Custom CSS for consistent styling across components
        
    Note:
        Must be called before any other Streamlit commands.
    """
    st.set_page_config(
        page_title="Scanpath Visualization",
        page_icon="👀",
        layout="wide",
    )
    st.markdown(get_app_css(), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_words_and_fixations(data_choice: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load word and fixation data from user uploads or bundled demo files.
    
    Args:
        data_choice: Either "Upload csv tables" or "Use bundled demo"
        
    Returns:
        Tuple of (words_df, fixations_df) as raw DataFrames before normalization
        
    UI Effects:
        - Renders file uploaders in sidebar when data_choice is "Upload csv tables"
        - Shows info message if uploads are incomplete
        - Falls back to sample data if uploads missing
        
    Note:
        Returned DataFrames have original column names and must be normalized
        via prepare_data() before use in plotting.
    """
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
    """Infer schemas and normalize incoming dataframes to canonical column names.
    
    Args:
        words_df: Raw words dataframe with experiment-specific column names
        fixations_df: Raw fixations dataframe with experiment-specific column names
        
    Returns:
        Tuple of (normalized_words, normalized_fixations) with canonical columns
        
    Canonical Columns:
        Words: participant_id, trial_id, paragraph_id, word_id, text, line_idx,
               x, y, width, height
        Fixations: participant_id, trial_id, paragraph_id, x, y, duration_ms,
                   timestamp_ms (+ optional word_id, pass_index, saccade_type, etc.)
                   
    Raises:
        Streamlit stop signal if schema inference fails (missing required columns)
    """
    word_schema = infer_word_schema(words_df)
    fix_schema = infer_fix_schema(fixations_df)
    if not word_schema or not fix_schema:
        st.stop()
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

    return raw_gaze_df


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------


def render_sidebar_data_source() -> str:
    """Render the data source selection radio button in sidebar.
    
    Returns:
        Selected data source: "Use bundled demo" or "Upload csv tables"
        
    UI Components:
        - Section header: "Experimental Setup"
        - Radio button with two options and help text
        - Help text explains expected CSV column formats
    """
    st.sidebar.header("Experimental Setup")
    return st.sidebar.radio(
        "Data source",
        ["Use bundled demo", "Upload csv tables"],
        index=0,
        help=data_dictionary_help_text(),
    )


def render_sidebar_canvas_controls(
    words_filtered: pd.DataFrame, fixations_filtered: pd.DataFrame
) -> Tuple[int, int, int, str]:
    """Render canvas dimension and font controls in sidebar.
    
    These controls allow users to match the visualization to their experimental
    display setup, ensuring spatial accuracy and proper word box alignment.
    
    Args:
        words_filtered: Filtered words dataframe (used to compute default dimensions)
        fixations_filtered: Filtered fixations dataframe (used for coordinate ranges)
        
    Returns:
        Tuple of (canvas_width, canvas_height, base_font_size, font_family)
        
    UI Components:
        - Monitor width number input (100-10000 px, default from data)
        - Monitor height number input (100-10000 px, default from data)
        - Font size number input (6-72 px, default 16)
        - Font family text input (default from constants.FONT_FAMILY)
        - Divider line at end
        
    Note:
        Default canvas dimensions are computed from actual word/fixation coordinates
        to match the experimental display as closely as possible.
    """
    # Compute smart defaults from data
    default_canvas_w, default_canvas_h = compute_canvas_size(
        words_filtered, fixations_filtered
    )
    canvas_width = min(max(default_canvas_w, 100), 10000)
    canvas_height = min(max(default_canvas_h, 100), 10000)

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

    return int(canvas_width), int(canvas_height), int(base_font_size), font_family


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

    # App header
    st.title("Scanpath Visualization")
    st.caption("Visualize eye-tracking-while-reading scanpaths!")

    # Data source selection (sidebar)
    data_choice = render_sidebar_data_source()

    # Load and prepare core data (words + fixations)
    words_df, fixations_df = load_words_and_fixations(data_choice)
    words_df, fixations_df = prepare_data(words_df, fixations_df)
    
    # Load optional raw gaze data
    raw_gaze_df = load_raw_gaze_data(data_choice)

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
            st.sidebar.warning(
                f"Raw gaze data was filtered out. "
                f"Original: {len(raw_gaze_df)} rows, After filter: 0 rows"
            )
    else:
        raw_gaze_filtered = pd.DataFrame()

    # Check for empty data after filtering
    if words_filtered.empty or fixations_filtered.empty:
        st.warning(
            "No data after filtering. Try selecting more participants or trials."
        )
        return

    # Build trial combinations for selection UI
    combos, _, _ = build_combo_options(fixations_filtered)

    # Canvas and visualization controls (sidebar)
    canvas_width, canvas_height, base_font_size, font_family = (
        render_sidebar_canvas_controls(words_filtered, fixations_filtered)
    )

    has_raw_gaze = not raw_gaze_filtered.empty
    viz_settings = sidebar_controls(
        fixations_filtered, base_font_size, has_raw_gaze=has_raw_gaze
    )

    # Render tabbed interface
    tab_single, tab_animation, tab_raw, tab_stats = st.tabs(
        ["Interactive Plot", "Animated Scanpath", "Raw Data", "Data Statistics"]
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
        )

    with tab_raw:
        render_raw_data_tab(words_filtered, fixations_filtered, raw_gaze_filtered)

    with tab_stats:
        render_data_statistics_tab(
            words_filtered, fixations_filtered, raw_gaze_filtered
        )


if __name__ == "__main__":
    main()
