"""CSS styles for the Scanpath Studio Streamlit app."""

from __future__ import annotations


def get_app_css() -> str:
    """Return custom CSS to reduce whitespace and disable animations."""
    return """
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
    /* "About" popover in the header: pill-shaped trigger + readable body width */
    /* The button is content-sized (width="content") and sits in a narrow column;
       align it to the column's right edge so it lines up with the right edge of
       the page content. The `about_btn` wrapper is a flex column, so this only
       moves the About button — no other popover (e.g. the plot toolbar). */
    .st-key-about_btn { align-items: flex-end; }
    div[data-testid="stPopover"] button { border-radius: 999px; }
    div[data-testid="stPopover"] button p { white-space: nowrap; }
    div[data-testid="stPopoverBody"] {
        min-width: min(28rem, 90vw);
        max-width: min(32rem, 95vw);
    }
    div[data-testid="stPopoverBody"] p { line-height: 1.45; }

    /* === Emphasised loading spinner ============================================
       The cache_data spinners ("Reading uploaded data…", "Normalizing data…", …)
       can run for a while on a large upload, so make them an unmissable pulsing
       banner instead of a small inline spinner. Blue tint + border reads on both
       the light and dark themes; the keyframes are scoped so the app-wide
       "animation: none" rules above don't kill the pulse or the spin. */
    div[data-testid="stSpinner"] {
        display: flex !important;
        align-items: center;
        gap: 0.9rem;
        width: 100%;
        box-sizing: border-box;
        padding: 1.2rem 1.5rem !important;
        margin: 0.7rem 0 !important;
        border-radius: 14px;
        border: 1px solid #185fa5;
        background: linear-gradient(90deg, #1f77b4, #3a8fd0 55%, #5aa9e6);
        box-shadow: 0 8px 24px rgba(31, 119, 180, 0.38);
        animation: sps-spinner-pulse 1.6s ease-in-out infinite !important;
    }
    /* white message text on the filled blue banner */
    div[data-testid="stSpinner"] p,
    div[data-testid="stSpinner"] div {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.2px;
        color: #ffffff !important;
    }
    /* enlarge the spinning Material icon (sized by font-size; it spins natively) */
    div[data-testid="stSpinner"] [data-testid="stIconMaterial"] {
        font-size: 2.5rem !important;
        width: 2.5rem !important;
        height: 2.5rem !important;
        color: #ffffff !important;
    }
    @keyframes sps-spinner-pulse {
        0%   { box-shadow: 0 0 0 0 rgba(90, 169, 230, 0.6), 0 8px 24px rgba(31,119,180,0.38); }
        70%  { box-shadow: 0 0 0 16px rgba(90, 169, 230, 0.0), 0 8px 24px rgba(31,119,180,0.38); }
        100% { box-shadow: 0 0 0 0 rgba(90, 169, 230, 0.0), 0 8px 24px rgba(31,119,180,0.38); }
    }
    </style>
    """
