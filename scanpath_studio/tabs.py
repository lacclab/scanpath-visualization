"""Tab rendering functions for the Scanpath Studio app."""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from scanpath_studio.annotations import render_trial_annotations
from scanpath_studio.constants import DEFAULT_LINE_SPACING
from scanpath_studio.data import compute_word_metrics
from scanpath_studio.model_scanpaths import (
    DEFAULT_N_MODELS,
    generate_model_scanpaths,
)
from scanpath_studio.similarity import (
    METRICS,
    compute_similarity_table,
    nld_by_fixation_index,
    nld_by_time,
)

from scanpath_studio.plots import (
    animation_playback_ms,
    make_comparison_figure,
    make_fixation_duration_histogram,
    make_metric_convergence_figure,
    make_scanpath_animation,
    make_scanpath_figure,
    make_word_measure_bar_figure,
)
from scanpath_studio.export import (
    ExportProgress,
    bulk_export,
    render_export_options,
)
from scanpath_studio.animation_export import (
    AnimationExportError,
    export_animation,
    mime_for,
)
from scanpath_studio.utils import (
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


def _embed_html_iframe(html: str, *, height: int) -> None:
    """Render an HTML string (scripts included) inside a sandboxed iframe.

    ``st.iframe`` (Streamlit >= 1.56) supersedes ``st.components.v1.html``, which
    is deprecated and scheduled for removal. We need an *iframe* — not
    ``st.html`` — because the embedded block runs ``<script>`` (Plotly from the
    CDN plus the fit/scale script), which ``st.html`` strips. Fall back to the
    old API on older Streamlit so the app still runs against the pinned baseline.
    """
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height, scrolling=False)


def _render_true_scale_chart(
    fig, *, key: str, max_height: Optional[int] = None
) -> None:
    """Display a spatial figure true-to-scale, fitted to the column width.

    ``st.plotly_chart`` pins the chart width to the column but keeps the layout
    height, re-laying-out the plot to an unknown scale — which breaks the
    data→pixel sizing the word labels were computed for (text over/under fills
    the boxes). Instead we render the figure at its exact pixel size, then scale
    that whole block *uniformly* with a CSS transform so it fills the column
    width. A uniform transform keeps boxes, fixations and text locked at one true
    scale (unlike a Plotly re-layout, which leaves the font fixed), so the plot
    stays faithful to the experiment at any column width — and never needs
    horizontal scrolling. It is only scaled down to fit (capped at 1×), so on a
    wide monitor it sits at true size with margin rather than being stretched.

    ``max_height`` caps the rendered (scaled) height in px — used for the small
    multiples in the *Multiple Comparison* grid, where the figure should also
    shrink to fit a fixed cell height (whichever of width/height binds), and the
    iframe is sized to that cap so panels don't leave a tall band of whitespace.
    """
    width = int(fig.layout.width or 900)
    height = int(fig.layout.height or 600)
    plot_html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        config={"responsive": False, "displaylogo": False},
        div_id=f"truescale-{key}",
    )
    # Scale factor: shrink to the available width, and (when capped) also to the
    # cell height, never upscaling past 1×.
    if max_height is not None:
        scale_js = f"Math.min(1, avail / W, {int(max_height)} / H)"
        iframe_height = int(max_height) + 12
    else:
        scale_js = "Math.min(1, avail / W)"
        iframe_height = height + 12
    # Wrap the fixed-size plot and scale it to the available (iframe) width.
    # transform-origin top-left keeps it flush-left; the outer box height tracks
    # the scaled height so there's no dead space below.
    html = f"""
    <div id="fit-{key}" style="width:100%;overflow:hidden;">
      <div id="box-{key}" style="width:{width}px;height:{height}px;
           transform-origin:top left;">{plot_html}</div>
    </div>
    <script>
    (function() {{
      var W = {width}, H = {height};
      var outer = document.getElementById("fit-{key}");
      var box = document.getElementById("box-{key}");
      function fit() {{
        var avail = outer.clientWidth || W;
        var s = {scale_js};
        box.style.transform = "scale(" + s + ")";
        outer.style.height = Math.round(H * s) + "px";
      }}
      fit();
      window.addEventListener("resize", fit);
      setTimeout(fit, 150);
    }})();
    </script>
    """
    # Iframe height = full true height (or the cap); the script trims the
    # visible block to the scaled height.
    _embed_html_iframe(html, height=iframe_height)


def _trial_text_id(trial_words: pd.DataFrame) -> Optional[str]:
    """Best-available text identifier for a trial's words (for same-text checks)."""
    for col in ("unique_paragraph_id", "paragraph_id"):
        if col in trial_words.columns and not trial_words.empty:
            value = trial_words[col].iloc[0]
            if pd.notna(value):
                return str(value)
    return None


def _default_second_trial(combos: pd.DataFrame, participant_a, trial_a):
    """Pick a sensible default second scanpath: a DIFFERENT trial on the SAME
    text as A, preferring A's own participant (a re-reading) and otherwise a
    different participant. Returns (participant_b, trial_b, text_str, text_field)
    or None when there's no same-text alternative.
    """
    para_field = (
        "unique_paragraph_id"
        if "unique_paragraph_id" in combos.columns
        else "paragraph_id"
    )
    if para_field not in combos.columns:
        return None
    a_rows = combos[
        (combos["participant_id"] == participant_a)
        & (combos["trial_id"].astype(str) == str(trial_a))
    ]
    if a_rows.empty or pd.isna(a_rows.iloc[0][para_field]):
        return None
    para = a_rows.iloc[0][para_field]
    same_text = combos[combos[para_field] == para].drop_duplicates(
        subset=["participant_id", "trial_id"]
    )
    # A different reading of the same text (exclude exactly A's trial).
    cand = same_text[
        ~(
            (same_text["participant_id"] == participant_a)
            & (same_text["trial_id"].astype(str) == str(trial_a))
        )
    ]
    if cand.empty:
        return None
    same_pid = cand[cand["participant_id"] == participant_a].sort_values("trial_id")
    pick = (
        same_pid
        if not same_pid.empty
        else cand.sort_values(["participant_id", "trial_id"])
    )
    row = pick.iloc[0]
    return row["participant_id"], row["trial_id"], str(para), para_field


def _seed_anim_b_default(combos: pd.DataFrame, participant_a, trial_a) -> None:
    """Seed the second-scanpath selector to `_default_second_trial` the first
    time the overlay is shown for a given text. Re-seeds only when A's text
    changes, so a manual second-scanpath choice isn't clobbered on every rerun.

    Seeds via the selector's "Text" mode (text → participant → reading), which
    is exactly the same-text-different-reader shape we want as the default.
    """
    default = _default_second_trial(combos, participant_a, trial_a)
    seed_key = default[2] if default else f"__none__:{participant_a}:{trial_a}"
    if st.session_state.get("_anim_b_seeded_text") == seed_key:
        return
    st.session_state["_anim_b_seeded_text"] = seed_key
    if default is None:
        return
    pid_b, trial_b, para, para_field = default
    st.session_state["anim_b_select_trial_mode"] = "Text"
    st.session_state["anim_b_text_id"] = para
    st.session_state["anim_b_participant_text"] = pid_b
    readings = (
        combos[
            (combos[para_field].astype(str) == str(para))
            & (combos["participant_id"] == pid_b)
        ]
        .drop_duplicates(subset=["trial_id"])
        .sort_values("trial_id")["trial_id"]
        .tolist()
    )
    if len(readings) > 1:
        st.session_state["anim_b_reading_text"] = trial_b


_MIME_FOR_FORMAT = {
    "PNG": "image/png",
    "SVG": "image/svg+xml",
    "PDF": "application/pdf",
    "HTML": "text/html",
}


def _render_save_plot_button(
    fig,
    *,
    canvas_width: int,
    canvas_height: int,
    slug: str,
    key_prefix: str,
) -> None:
    """Download the currently displayed figure.

    HTML is cheap (no Kaleido/Chrome) so it downloads in a single click. PNG/SVG/
    PDF go through Kaleido, which spins up a headless Chrome — far too slow to run
    on every Streamlit rerun — so those keep a Render step and reveal the download
    only once the image is ready. Width/height come from the figure's own layout
    so stacked / multi-panel figures save at their on-screen size.
    """
    if fig is None:
        return
    file_stem = f"scanpath_{_safe_filename(slug)}"
    cols = st.columns([1, 2])
    with cols[0]:
        fmt = st.radio(
            "Download format",
            # HTML is a browser-free fallback (no Kaleido/Chrome) — useful on
            # Streamlit Cloud where static image export needs a Chromium binary.
            options=["PNG", "SVG", "PDF", "HTML"],
            index=0,
            horizontal=True,
            key=f"{key_prefix}_save_format",
            help="PNG/SVG/PDF need a Chrome/Chromium browser (Kaleido). HTML "
            "is interactive and needs no browser.",
        )

    # HTML: one-click download — no expensive render to defer.
    if fmt == "HTML":
        with cols[1]:
            html_bytes = fig.to_html(include_plotlyjs="cdn", full_html=True).encode(
                "utf-8"
            )
            st.download_button(
                "⬇ Download HTML",
                data=html_bytes,
                file_name=f"{file_stem}.html",
                mime=_MIME_FOR_FORMAT["HTML"],
                key=f"{key_prefix}_save_button_html",
            )
        return

    # PNG/SVG/PDF: render on click (Kaleido/Chrome), then reveal the download.
    with cols[1]:
        generate = st.button(
            f"Render {fmt}",
            key=f"{key_prefix}_save_generate",
            help="Renders the image (needs Chrome/Kaleido); the download button "
            "appears once it's ready.",
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
            # The on-screen figure is sized to fit the column; render raster
            # PNG at 3x so saved figures stay paper-quality (SVG/PDF are
            # vector and unaffected).
            scale=3 if fmt == "PNG" else 1,
        )
    except Exception as exc:
        st.warning(
            f"Could not render {fmt}: {exc}\n\n"
            "Static image export (PNG/SVG/PDF) needs a Chrome/Chromium browser "
            "for Kaleido. On Streamlit Cloud this is installed via `packages.txt`; "
            "if it still fails, choose the **HTML** format above — it needs no browser."
        )
        return
    st.download_button(
        f"⬇ Download {fmt}",
        data=data,
        file_name=f"{file_stem}.{fmt.lower()}",
        mime=_MIME_FOR_FORMAT[fmt],
        key=f"{key_prefix}_save_button",
    )


# Above this many frames, offer to cap the rendered frame count: each frame is a
# headless-Chrome render (~0.1-0.25 s), so a long reading would otherwise spin for
# a minute or more. Capping holds each kept frame proportionally longer, so the
# clip's total duration is unchanged.
_ANIM_FRAME_CAP = 250
# Rough per-frame render cost (warm browser) + one-off browser cold start, for the
# "~Ns to render" estimate. Approximate by design.
_ANIM_RENDER_S_PER_FRAME = 0.18
_ANIM_RENDER_COLD_START_S = 3.0


def _render_animation_export(fig, *, file_stem: str, playback_ms: float) -> None:
    """Export the animated scanpath as interactive HTML or a rasterized GIF/MP4.

    HTML is one click (no browser needed to generate, keeps interactivity). GIF and
    MP4 rasterize every frame through Kaleido (headless Chrome) — too slow to run on
    every rerun — so they follow the same Render-then-download pattern as the static
    image export, with a progress bar and a result cached in session state so the
    download button survives reruns (and a re-render isn't needed unless an option
    changes). The clip reproduces the on-screen Play: every frame held for the same
    average duration, so its runtime equals the quoted playback time.
    """
    n_frames = len(fig.frames or ())
    fmt = st.radio(
        "Export format",
        options=["HTML", "GIF", "MP4"],
        index=0,
        horizontal=True,
        key="anim_export_format",
        help=(
            "HTML keeps the interactive play/slider and needs no browser to "
            "generate. GIF and MP4 are self-playing clips (great for slides or "
            "papers) rendered via Kaleido/Chrome — MP4 is far smaller than GIF."
        ),
    )

    if fmt == "HTML":
        html_bytes = fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")
        st.download_button(
            "⬇ Download HTML",
            data=html_bytes,
            file_name=f"{file_stem}.html",
            mime="text/html",
            key="anim_export_html",
            help="Self-contained HTML you can open in any browser; keeps play/slider interactivity.",
        )
        return

    if n_frames == 0:
        st.info("Nothing to animate for this trial.")
        return

    frame_ms = playback_ms / n_frames if n_frames else 16.0
    clip_s = playback_ms / 1000.0

    scale = st.select_slider(
        "Resolution",
        options=[0.5, 0.75, 1.0, 1.5, 2.0],
        value=1.0,
        format_func=lambda s: f"{s:g}×",
        key="anim_export_scale",
        help="Render scale: 1× matches the on-screen size. Lower is smaller/faster; "
        "higher is crisper/larger.",
    )

    max_frames = None
    if n_frames > _ANIM_FRAME_CAP:
        if st.checkbox(
            f"Limit to {_ANIM_FRAME_CAP} frames for a faster render",
            value=True,
            key="anim_export_limit",
            help="This reading has many fixations. Capping the rendered frames keeps "
            "the export quick; the clip's total duration is unchanged (each kept "
            "frame is held a little longer).",
        ):
            max_frames = _ANIM_FRAME_CAP

    render_frames = min(n_frames, max_frames) if max_frames else n_frames
    est_s = render_frames * _ANIM_RENDER_S_PER_FRAME + _ANIM_RENDER_COLD_START_S
    note = f"{n_frames} frames · clip ≈ {clip_s:.1f}s · ~{est_s:.0f}s to render"
    if render_frames != n_frames:
        note += f" ({render_frames} frames rendered)"
    st.caption(note)
    if fmt == "GIF":
        st.caption(
            "GIF embeds anywhere but is large for long readings — prefer MP4, or "
            "lower the resolution, to shrink the file."
        )

    # Re-render only when an output-affecting option changes; otherwise reuse the
    # cached bytes so the download button persists across reruns.
    sig = (file_stem, fmt, float(scale), max_frames, n_frames, round(frame_ms, 2))
    cache = st.session_state.get("_anim_export_cache")

    if st.button(
        f"Render {fmt}",
        key="anim_export_generate",
        help="Renders each frame via Kaleido (headless Chrome); the download "
        "appears once it's ready.",
    ):
        bar = st.progress(0.0, text=f"Rendering 0/{render_frames} frames…")

        def _on_progress(done: int, total: int) -> None:
            bar.progress(done / total, text=f"Rendering {done}/{total} frames…")

        try:
            data = export_animation(
                fig,
                fmt=fmt.lower(),
                frame_duration_ms=frame_ms,
                scale=float(scale),
                max_frames=max_frames,
                progress_callback=_on_progress,
            )
        except AnimationExportError as exc:
            bar.empty()
            st.warning(
                f"Could not render {fmt}: {exc}\n\n"
                "GIF/MP4 export rasterizes each frame with a Chrome/Chromium browser "
                "(Kaleido). On Streamlit Cloud this is installed via `packages.txt`; "
                "if it still fails, use the **HTML** format above — it needs no browser."
            )
            cache = None
        else:
            bar.empty()
            cache = {"sig": sig, "data": data}
            st.session_state["_anim_export_cache"] = cache

    if cache and cache.get("sig") == sig:
        data = cache["data"]
        size = len(data)
        size_label = (
            f"{size / 1_048_576:.1f} MB"
            if size >= 1_048_576
            else f"{size / 1024:.0f} KB"
        )
        st.download_button(
            f"⬇ Download {fmt} ({size_label})",
            data=data,
            file_name=f"{file_stem}.{fmt.lower()}",
            mime=mime_for(fmt.lower()),
            key="anim_export_download",
        )


def _build_figure_settings(viz_settings: dict, effective_show_raw_gaze: bool) -> dict:
    """Convert viz_settings to figure-compatible settings dict."""
    return dict(
        show_words=viz_settings["show_words"],
        show_word_labels=viz_settings["show_labels"],
        show_fixations=viz_settings["show_fix"],
        show_order=viz_settings["show_order"],
        show_saccades=viz_settings["show_saccades"],
        show_saccade_arrows=viz_settings.get("show_saccade_arrows", False),
        show_heatmap=viz_settings["show_heatmap"],
        heatmap_style=viz_settings.get("heatmap_style", "Word boxes"),
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
        color_by_line=viz_settings.get("color_by_line", False),
        highlight_out_of_text=viz_settings.get("highlight_out_of_text", False),
        background_color=viz_settings.get("background_color"),
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


def _first_str(df: pd.DataFrame, col: str) -> Optional[str]:
    """First non-null value of ``col`` as a string, or None."""
    if col in df.columns:
        vals = df[col].dropna()
        if not vals.empty:
            return str(vals.iloc[0])
    return None


def _first_bool(df: pd.DataFrame, col: str) -> Optional[bool]:
    """First non-null value of ``col`` as a bool, or None when absent/empty."""
    if col in df.columns:
        vals = df[col].dropna()
        if not vals.empty:
            return bool(vals.iloc[0])
    return None


def _span_fixated_note(
    trial_words: pd.DataFrame,
    trial_fixations: Optional[pd.DataFrame],
    mask_col: str,
) -> str:
    """Inline HTML note: was the span fixated, and for how long?

    Counts fixations falling inside any of the span's word boxes — a quick
    "did the reader actually look at the answer?" check tying the scanpath to
    comprehension. Returns "" when fixations or the span column are absent."""
    if (
        trial_fixations is None
        or trial_fixations.empty
        or mask_col not in trial_words.columns
    ):
        return ""
    span_words = trial_words[trial_words[mask_col].fillna(False).astype(bool)]
    if span_words.empty:
        return ""
    from scanpath_studio.measures import fixation_in_text_mask

    mask = fixation_in_text_mask(trial_fixations, span_words)
    n = int(mask.sum())
    if n == 0:
        return ' <span style="color:#dc3545;">— not fixated</span>'
    dwell = float(pd.to_numeric(trial_fixations.loc[mask, "duration_ms"]).sum())
    return f' <span style="color:#198754;">— {n} fixations, {dwell:.0f} ms</span>'


def _render_trial_header(
    participant: str,
    trial_id: str,
    trial_words: pd.DataFrame,
    prefix: str = "Trial:",
) -> None:
    """Render the trial id header with participant + text id stacked below it.

    The paragraph text / question / spans live in
    `_render_paragraph_panel` so they can sit under the figure (single tab)
    while the header stays in the side panel.
    """
    lines = [f"**{prefix}** `{trial_id}`", f"Participant: `{participant}`"]
    text_id = None
    for col in ("unique_paragraph_id", "paragraph_id"):
        if col in trial_words.columns and not trial_words.empty:
            value = trial_words[col].iloc[0]
            if pd.notna(value):
                text_id = value
                break
    if text_id is not None:
        lines.append(f"Text: `{text_id}`")
    # Participant and Text sit on their own lines under the trial id (a markdown
    # hard line break is two trailing spaces + newline).
    st.markdown("  \n".join(lines))


def _render_paragraph_panel(
    trial_words: pd.DataFrame,
    *,
    trial_fixations: Optional[pd.DataFrame] = None,
    expanded: bool = True,
) -> None:
    """Render the paragraph + comprehension-question panel.

    Shows the paragraph with highlighted spans, the reading regime
    (Hunting/Gathering), the question, the participant's selected answer and
    correctness, and the answer (critical) / distractor span texts — each
    annotated with whether it was fixated when ``trial_fixations`` is given.
    Skips silently when no word text is available."""
    if "text" not in trial_words.columns or trial_words.empty:
        return
    with st.expander("Paragraph & question", expanded=expanded):
        _render_paragraph_with_spans(trial_words)

        # Reading regime: in OneStop, question_preview=True means the reader saw
        # the question before the text ("Hunting"); False is ordinary reading
        # ("Gathering").
        preview = _first_bool(trial_words, "question_preview")
        if preview is not None:
            regime = (
                "Hunting (question previewed)"
                if preview
                else "Gathering (ordinary reading)"
            )
            st.caption(f"Reading regime: **{regime}**")

        question_val = _first_str(trial_words, "question")
        if question_val:
            st.markdown(f"**Question:** {question_val}")

        # Participant's answer + correctness. The data carries the chosen option
        # (e.g. 'A'/'B') and a correctness flag, but not the option texts.
        answer_val = _first_str(trial_words, "selected_answer")
        correct = _first_bool(trial_words, "is_correct")
        if answer_val or correct is not None:
            bits = []
            if answer_val:
                bits.append(f"selected **{answer_val}**")
            if correct is not None:
                bits.append("✓ correct" if correct else "✗ incorrect")
            st.markdown("**Answer:** " + " · ".join(bits))

        critical_text = _span_text(trial_words, "is_in_aspan")
        if critical_text:
            note = _span_fixated_note(trial_words, trial_fixations, "is_in_aspan")
            st.markdown(
                f'<span style="background-color:{_CRITICAL_SPAN_BG};'
                f'padding:0 4px;border-radius:2px;">'
                f"<b>Answer (critical) span:</b></span> {critical_text}{note}",
                unsafe_allow_html=True,
            )
        distractor_text = _span_text(trial_words, "is_in_dspan")
        if distractor_text:
            note = _span_fixated_note(trial_words, trial_fixations, "is_in_dspan")
            st.markdown(
                f'<span style="background-color:{_DISTRACTOR_SPAN_BG};'
                f'padding:0 4px;border-radius:2px;">'
                f"<b>Distractor span:</b></span> {distractor_text}{note}",
                unsafe_allow_html=True,
            )


def _in_text_fixation_value(
    trial_words: pd.DataFrame, trial_fixations: pd.DataFrame
) -> Optional[str]:
    """Concise "in / total" count of fixations that landed inside a word box.

    Replaces the old free-text caption ("All 154 fixations landed inside word
    boxes.") with a compact value for the trial summary table. Returns None when
    there's nothing to count."""
    if trial_words.empty or trial_fixations.empty:
        return None
    from scanpath_studio.measures import fixation_in_text_mask

    mask = fixation_in_text_mask(trial_fixations, trial_words)
    n_total = len(mask)
    if n_total == 0:
        return None
    n_in = int(mask.sum())
    return f"{n_in} / {n_total}"


def _summary_rows(
    trial_words: pd.DataFrame, trial_fixations: pd.DataFrame
) -> list[dict]:
    """Field/Value rows summarising a trial — totals + in-text fixation count.

    Folded into the metadata table (formerly the three `st.metric` cards plus
    the out-of-text caption), so a trial's headline numbers live in one place."""
    stats = compute_trial_stats(trial_words, trial_fixations)
    rows = [
        {
            "Field": "Total reading time (s)",
            "Value": f"{stats['total_reading_time_s']:.1f}",
        },
        {"Field": "Number of words", "Value": f"{stats['word_count']:,}"},
        {"Field": "Number of fixations", "Value": f"{stats['fixation_count']:,}"},
    ]
    in_text = _in_text_fixation_value(trial_words, trial_fixations)
    if in_text is not None:
        rows.append({"Field": "Fixations in word boxes", "Value": in_text})
    return rows


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


def _prefix_is_correct(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Prefix the is_correct row's Value with ✓ / ✗ for at-a-glance scanning."""
    mask = metadata_df["Field"] == "is_correct"
    if mask.any():
        val = str(metadata_df.loc[mask, "Value"].iloc[0])
        if val.lower().startswith("true"):
            metadata_df.loc[mask, "Value"] = f"✓ {val}"
        elif val.lower().startswith("false"):
            metadata_df.loc[mask, "Value"] = f"✗ {val}"
    return metadata_df


def _render_metadata_selector(
    words_filtered: pd.DataFrame,
    fixations_filtered: pd.DataFrame,
    trial_words: pd.DataFrame,
    trial_fixations: pd.DataFrame,
    compare: Optional[dict] = None,
):
    """Render the trial summary + metadata table.

    The table always leads with the trial's headline numbers (reading time,
    word / fixation counts, in-text fixation count — see `_summary_rows`), then
    lists the metadata fields chosen in the sidebar picker.

    When ``compare`` is given (keys: ``words``, ``fixations``,
    ``label_primary``, ``label_compare``), the table shows one value column per
    trial so the two compared trials can be read side by side.
    """
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
    # Field picker lives in the sidebar so the chips don't crowd the side panel;
    # the table renders in the panel next to the plot.
    with st.sidebar.expander("Trial metadata fields", expanded=False):
        selected_metadata = st.multiselect(
            "Fields to show",
            options=metadata_candidates,
            default=default_metadata or metadata_candidates,
            key="trial_metadata_fields",
        )

    if compare is None:
        summary = pd.DataFrame(_summary_rows(trial_words, trial_fixations))
        metadata_df = (
            _prefix_is_correct(
                gather_trial_metadata(trial_words, trial_fixations, selected_metadata)
            )
            if selected_metadata
            else pd.DataFrame(columns=["Field", "Value"])
        )
        table = pd.concat([summary, metadata_df], ignore_index=True)
        if not table.empty:
            styled = table.style.apply(_style_metadata_row, axis=1)
            st.dataframe(styled, hide_index=True, width="stretch")
        return

    # Comparison mode: one value column per trial, merged on Field.
    label_a = compare["label_primary"]
    label_b = compare["label_compare"]
    # Summary rows share the same fields in the same order across both trials,
    # so pair them up directly to keep the row order rather than outer-merging.
    summary_b = {
        r["Field"]: r["Value"]
        for r in _summary_rows(compare["words"], compare["fixations"])
    }
    summary = pd.DataFrame(
        [
            {
                "Field": r["Field"],
                label_a: r["Value"],
                label_b: summary_b.get(r["Field"], "—"),
            }
            for r in _summary_rows(trial_words, trial_fixations)
        ]
    )
    if selected_metadata:
        primary = _prefix_is_correct(
            gather_trial_metadata(trial_words, trial_fixations, selected_metadata)
        ).rename(columns={"Value": label_a})
        other = _prefix_is_correct(
            gather_trial_metadata(
                compare["words"], compare["fixations"], selected_metadata
            )
        ).rename(columns={"Value": label_b})
        # Outer merge in case the two trials' sources expose different columns;
        # fill any resulting gap with the same "—" placeholder the summary uses.
        merged = primary.merge(other, on="Field", how="outer").fillna("—")
        order = {field: i for i, field in enumerate(selected_metadata)}
        merged = (
            merged.assign(_o=merged["Field"].map(order))
            .sort_values("_o")
            .drop(columns="_o")
        )
    else:
        merged = pd.DataFrame(columns=["Field", label_a, label_b])
    table = pd.concat([summary, merged], ignore_index=True)
    if not table.empty:
        st.dataframe(table, hide_index=True, width="stretch")


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
    """Render the plot configuration expander in the sidebar.

    Offers the active plot settings as a downloadable JSON sidecar so a reviewer
    can save/share the exact configuration that produced the current figure.
    """
    with st.sidebar.expander("Plot configuration"):
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
                "saccade_arrows": figure_settings.get("show_saccade_arrows", False),
                "heatmap": figure_settings["show_heatmap"],
                "raw_gaze": figure_settings["show_raw_gaze"],
            },
            "coloring": {
                "color_by": figure_settings["color_by"],
                "heatmap_metric": viz_settings["heatmap_metric"],
                "heatmap_style": figure_settings.get("heatmap_style", "Word boxes"),
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
        safe_pid = str(selected_participant).replace("/", "-")
        safe_trial = str(selected_trial).replace("/", "-")
        st.caption(
            "The exact settings behind the current figure — download to save or "
            "share a reproducible configuration."
        )
        st.download_button(
            "⬇ Download plot config (JSON)",
            data=json.dumps(plot_config, indent=2),
            file_name=f"plot_config_{safe_pid}_{safe_trial}.json",
            mime="application/json",
            key="plot_config_download",
            use_container_width=True,
        )


def _ordered_trial_ids(combos: pd.DataFrame) -> list[str]:
    """Stable trial_id ordering used by the under-image Prev/Next buttons.

    Mirrors `_select_trial_none_mode`'s sort so the under-image nav lands on
    the same trial as the side-panel Prev/Next when both are present.
    """
    trial_col = "unique_trial_id" if "unique_trial_id" in combos.columns else "trial_id"
    return sorted(combos[trial_col].dropna().astype(str).unique().tolist())


def _build_compare_meta(
    words_filtered: pd.DataFrame,
    fixations_filtered: pd.DataFrame,
    selected_participant: str,
    selected_trial: str,
    compare_participant: Optional[str],
    compare_trial: Optional[str],
) -> Optional[dict]:
    """Build the second trial's words/fixations + column labels for the
    side-by-side metadata table, or None when no comparison is active."""
    if compare_participant is None or compare_trial is None:
        return None
    compare_words = words_filtered[
        (words_filtered["participant_id"] == compare_participant)
        & (words_filtered["trial_id"] == compare_trial)
    ]
    compare_fix = fixations_filtered[
        (fixations_filtered["participant_id"] == compare_participant)
        & (fixations_filtered["trial_id"] == compare_trial)
    ]
    # Short, distinct column headers: participant ids when comparing different
    # participants (the common same-text case), else the trial ids — the long
    # ids otherwise overflow the narrow panel.
    if str(selected_participant) != str(compare_participant):
        label_primary = str(selected_participant)
        label_compare = str(compare_participant)
    else:
        label_primary = str(selected_trial)
        label_compare = str(compare_trial)
    return {
        "words": compare_words,
        "fixations": compare_fix,
        "label_primary": label_primary,
        "label_compare": label_compare,
    }


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
    line_spacing: float = DEFAULT_LINE_SPACING,
    scale_text_to_boxes: bool = True,
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
    # Carried into both the live figure and the bulk export so exported plots are
    # sized identically to what's on screen (true-to-scale reading text).
    figure_settings["line_spacing"] = line_spacing
    figure_settings["scale_text_to_boxes"] = scale_text_to_boxes
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
        # Trial summary + metadata table (totals, in-text fixation count, then
        # the sidebar-selected metadata fields). Sits high in the side panel,
        # right under the selectors, so the trial's headline numbers are visible
        # without scrolling.
        compare_meta = _build_compare_meta(
            words_filtered,
            fixations_filtered,
            selected_participant,
            selected_trial,
            compare_participant,
            compare_trial,
        )
        _render_metadata_selector(
            words_filtered,
            fixations_filtered,
            trial_words,
            trial_fixations,
            compare=compare_meta,
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
                line_spacing=line_spacing,
                scale_text_to_boxes=scale_text_to_boxes,
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
            _render_true_scale_chart(displayed_fig, key="single")
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
        _render_paragraph_panel(
            trial_words, trial_fixations=trial_fixations, expanded=True
        )

    # Publish the single-tab's selection so the Animation tab can mirror it.
    # Read by `render_animation_tab` (one-way sync; anim can still be moved
    # independently — see `_sync_anim_from_single`).
    st.session_state["shared_selected_pid"] = selected_participant
    st.session_state["shared_selected_trial_id"] = selected_trial

    with col_side:
        render_trial_annotations(selected_participant, selected_trial)

    # Plot configuration renders into the sidebar (see _render_plot_config_expander).
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
    line_spacing: float = DEFAULT_LINE_SPACING,
    scale_text_to_boxes: bool = True,
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
        background_color=viz_settings.get("background_color"),
        line_spacing=line_spacing,
        scale_text_to_boxes=scale_text_to_boxes,
    )
    _render_true_scale_chart(fig_compare, key="compare")
    return fig_compare


# -----------------------------------------------------------------------------
# Animation Tab
# -----------------------------------------------------------------------------


def _sync_anim_from_single(combos: pd.DataFrame) -> None:
    """Pre-set anim tab's selector state to mirror the single tab's selection.

    Fires only when the single tab's selection changed since the last run
    (tracked via `_prev_shared_trial`). One-way sync by design — user
    changes inside the anim tab are preserved across reruns until the
    single tab moves again.

    Respects whichever selection mode (Trial/Text/Participant) the anim tab
    is currently in and writes to the matching state key, so the user's
    mode choice isn't reset every time the single tab advances.
    """
    shared_pid = st.session_state.get("shared_selected_pid")
    shared_trial = st.session_state.get("shared_selected_trial_id")
    if not (shared_pid and shared_trial):
        return
    current = (shared_pid, shared_trial)
    if st.session_state.get("_prev_shared_trial") == current:
        return

    # Find the row in `combos` so we can read trial_index / paragraph_id —
    # needed by the slider (Participant mode) and the text selectbox (Text
    # mode), which key off a different field than `trial_id`.
    match = combos[
        (combos["participant_id"] == shared_pid)
        & (combos["trial_id"].astype(str) == str(shared_trial))
    ]
    if match.empty:
        return
    row = match.iloc[0]

    mode = st.session_state.get("anim_select_trial_mode", "Trial")

    if mode == "Trial":
        trial_options = _ordered_trial_ids(combos)
        try:
            idx = trial_options.index(str(shared_trial))
        except ValueError:
            return
        st.session_state["anim_trial_index"] = idx
    elif mode == "Participant":
        # Slider value is TRIAL_INDEX (int) when present, else paragraph_id
        # (str) — matches `_select_trial_participant_mode`'s preference.
        st.session_state["anim_participant"] = shared_pid
        slider_value = None
        for field in ("TRIAL_INDEX", "trial_index"):
            if field in row.index and pd.notna(row[field]):
                slider_value = row[field]
                break
        if slider_value is None:
            for field in ("unique_paragraph_id", "paragraph_id"):
                if field in row.index and pd.notna(row[field]):
                    slider_value = str(row[field])
                    break
        if slider_value is not None:
            st.session_state["anim_slider"] = slider_value
    elif mode == "Text":
        for field in ("unique_paragraph_id", "paragraph_id"):
            if field in row.index and pd.notna(row[field]):
                st.session_state["anim_text_id"] = str(row[field])
                break
        st.session_state["anim_participant_text"] = shared_pid

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
    line_spacing: float = DEFAULT_LINE_SPACING,
    scale_text_to_boxes: bool = True,
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

    # Optional second scanpath, co-animated on the same real-time clock by
    # `make_scanpath_animation`. Its selector is keyed under "anim_b" so it stays
    # independent of the single-tab → anim sync (which targets "anim"); when the
    # overlay is first enabled (or A's text changes) we seed it to another
    # reading of the SAME text, preferring the same participant — see
    # `_seed_anim_b_default`.
    with col_side:
        compare = st.checkbox(
            "Overlay a second scanpath",
            value=False,
            key="anim_compare",
            help=(
                "Animate a second reading on the same timeline. Works best for "
                "two readings of the same text (e.g. a re-reading, or another "
                "participant)."
            ),
        )
        sel_b_participant = sel_b_trial = None
        if compare:
            _seed_anim_b_default(combos, selected_participant, selected_trial)
            sel_b_participant, sel_b_trial, _mode_b, _text_b = select_trial(
                combos, key_prefix="anim_b"
            )

    trial_words_b = None
    trial_fixations_b = None
    if compare and sel_b_participant and sel_b_trial:
        trial_words_b = words_filtered[
            (words_filtered["participant_id"] == sel_b_participant)
            & (words_filtered["trial_id"] == sel_b_trial)
        ]
        trial_fixations_b = fixations_filtered[
            (fixations_filtered["participant_id"] == sel_b_participant)
            & (fixations_filtered["trial_id"] == sel_b_trial)
        ]

    dual = (
        compare
        and trial_fixations_b is not None
        and not trial_fixations_b.empty
        and not trial_fixations.empty
    )

    # Playback speed — rendered on the right (next to the animation plot)
    # because that's where the eye actually goes when adjusting playback.
    # Frame durations are floor-clamped at ~16 ms (see `make_scanpath_animation`),
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
        _render_paragraph_panel(
            trial_words, trial_fixations=trial_fixations, expanded=False
        )

        if trial_fixations.empty:
            st.warning("No fixations available for this trial.")
        else:
            # Quote the REAL animation runtime (see `animation_playback_ms`) so
            # the stated playback time matches what the user observes; "reading
            # time" is the recorded span (incl. saccade/blink gaps), not the sum
            # of fixation durations.
            reading_span_ms, playback_ms = animation_playback_ms(
                [trial_fixations] + ([trial_fixations_b] if dual else []),
                playback_speed,
            )
            if dual:
                span_a = animation_playback_ms([trial_fixations], 1.0)[0]
                span_b = animation_playback_ms([trial_fixations_b], 1.0)[0]
                st.info(
                    f"**A: {len(trial_fixations)} fixations** "
                    f"({span_a / 1000:.1f}s) · **B: {len(trial_fixations_b)} "
                    f"fixations** ({span_b / 1000:.1f}s)"
                )
                shorter = "A" if span_a <= span_b else "B"
                st.caption(
                    f"Playback at ×{playback_speed}: "
                    f"**{playback_ms / 1000:.1f}s** — both co-animate on one "
                    f"clock; the shorter reading ({shorter}) finishes first and "
                    f"holds while the longer one continues."
                )
                if (sel_b_participant, sel_b_trial) == (
                    selected_participant,
                    selected_trial,
                ):
                    st.caption("⚠️ The second scanpath is the same trial as the first.")
                else:
                    text_a = _trial_text_id(trial_words)
                    text_b = _trial_text_id(trial_words_b)
                    if text_a is not None and text_b is not None and text_a != text_b:
                        st.warning(
                            "The two scanpaths are **different texts**, so the "
                            "shared word boxes don't line up with the second "
                            "reading — the spatial overlay isn't meaningful. This "
                            "view is intended for two readings of the same "
                            "paragraph."
                        )
            else:
                st.info(
                    f"**{len(trial_fixations)} fixations** · Reading time: "
                    f"{reading_span_ms / 1000:.1f}s · Playback at "
                    f"×{playback_speed}: {playback_ms / 1000:.1f}s"
                )
                if compare and sel_b_participant and sel_b_trial:
                    st.warning(
                        "The selected second scanpath has no fixations after "
                        "filtering — showing only the first scanpath."
                    )

    if trial_fixations.empty:
        return

    # One builder for both cases: passing fixations_b/words_b switches it to the
    # dual co-animation; without them it's the classic single replay.
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
        background_color=viz_settings.get("background_color"),
        fixations_b=trial_fixations_b if dual else None,
        words_b=trial_words_b if dual else None,
        label_a=f"{selected_participant} · {selected_trial}" if dual else "Scanpath A",
        label_b=f"{sel_b_participant} · {sel_b_trial}" if dual else "Scanpath B",
        line_spacing=line_spacing,
        scale_text_to_boxes=scale_text_to_boxes,
    )
    with col_main:
        _render_true_scale_chart(fig, key="animation")

    with col_side:
        if dual:
            st.caption(
                "**Controls:** ▶ Play co-animates both scanpaths on one shared "
                "real-time clock (recorded fixation timings ÷ speed), so the "
                "shorter reading finishes first and waits while the longer one "
                "continues. Drag the slider to scrub by elapsed reading time. "
                "Each orange highlight, ringed in its scanpath's colour, marks "
                "that reader's current fixation."
            )
            file_stem = (
                f"animation_{_safe_filename(selected_participant)}__"
                f"{_safe_filename(selected_trial)}__vs__"
                f"{_safe_filename(sel_b_participant)}__"
                f"{_safe_filename(sel_b_trial)}"
            )
        else:
            st.caption(
                "**Controls:** ▶ Play replays the scan in real reading time ÷ "
                "speed, ⏸ Pause stops, and the slider scrubs by elapsed reading "
                "time. Orange highlight shows the current fixation."
            )
            file_stem = (
                f"animation_{_safe_filename(selected_participant)}__"
                f"{_safe_filename(selected_trial)}"
            )
        _render_animation_export(fig, file_stem=file_stem, playback_ms=playback_ms)


# -----------------------------------------------------------------------------
# Multiple Comparison Tab
# -----------------------------------------------------------------------------


def _best_model_indices(table: pd.DataFrame) -> dict:
    """Row index of the best-scoring model per *real* metric column.

    Min when the metric is lower-is-better, max otherwise. Placeholder metrics
    (``fn is None``) and all-NaN columns are skipped, so they get no entry.
    """
    best_index: dict[str, object] = {}
    for metric in METRICS:
        if metric.fn is None or metric.label not in table.columns:
            continue
        col = pd.to_numeric(table[metric.label], errors="coerce")
        if col.notna().any():
            best_index[metric.label] = (
                col.idxmin() if metric.lower_is_better else col.idxmax()
            )
    return best_index


def _metric_arrow(metric) -> str:
    """Direction arrow for a metric header: ↓ when lower is more similar, ↑ when
    higher is."""
    return "↓" if metric.lower_is_better else "↑"


def _style_similarity_table(table: pd.DataFrame):
    """Format the similarity table and highlight the best model per metric.

    Each metric header gets a direction arrow (↓ lower-is-better / ↑
    higher-is-better) so it's clear which way means *more similar*. Placeholder
    metrics (all-NaN columns) render as "—"; for each real metric the
    best-scoring model's cell is tinted green.
    """
    # Header relabelling: "NLD" -> "NLD ↓", etc. Keep the original-label
    # best-index, then translate its keys to the arrowed headers for highlighting.
    header_map = {
        m.label: f"{m.label} {_metric_arrow(m)}"
        for m in METRICS
        if m.label in table.columns
    }
    best_index = _best_model_indices(table)
    best_by_header = {
        header_map[label]: idx
        for label, idx in best_index.items()
        if label in header_map
    }

    display = table.copy()
    for label in header_map:
        display[label] = display[label].map(
            lambda v: f"{v:.3f}" if pd.notna(v) else "—"
        )
    display = display.rename(columns=header_map)

    def _highlight(row: pd.Series) -> list[str]:
        styles = []
        for col in display.columns:
            style = ""
            if col in best_by_header and row.name == best_by_header[col]:
                style = "background-color: #d4edda; font-weight: 600;"
            styles.append(style)
        return styles

    return display.style.apply(_highlight, axis=1)


def render_multiple_comparison_tab(
    words_filtered: pd.DataFrame,
    fixations_filtered: pd.DataFrame,
    combos: pd.DataFrame,
    *,
    canvas_width: int,
    canvas_height: int,
    base_font_size: int,
    font_family: str,
    viz_settings: dict,
    line_spacing: float = DEFAULT_LINE_SPACING,
    scale_text_to_boxes: bool = True,
) -> None:
    """Render the Multiple Comparison tab.

    Shows the real scanpath for the selected trial on top, then a configurable
    grid of model-generated scanpaths over the SAME text, and a similarity
    table scoring each model against the real reading (NLD plus placeholder
    metrics). The model scanpaths are synthetic placeholders until real model
    outputs are connected — see :mod:`scanpath_studio.model_scanpaths`.
    """
    st.caption(
        "Compare a real scanpath with several **model-generated** scanpaths over "
        "the same text, and score how close each model is to the real reading. "
        "⚠️ The model scanpaths are **synthetic placeholders** (reproducible, "
        "reading-like random paths) until real model outputs are connected."
    )

    col_side, col_main = st.columns([3, 7], gap="medium")

    with col_side:
        selected_participant, selected_trial, _mode, _text = select_trial(
            combos, key_prefix="multi"
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
        with col_main:
            st.warning("The selected trial has no words/fixations after filtering.")
        return

    with col_side:
        # The number of models is inferred from the model-scanpath data. Until
        # real model outputs are connected we generate a fixed set of example
        # scanpaths behind the scenes — so there's no user control for it.
        n_models = DEFAULT_N_MODELS
        n_cols = st.slider(
            "Grid columns",
            min_value=1,
            max_value=4,
            value=3,
            key="multi_n_cols",
            help="Columns in the model grid (rows fill automatically).",
        )
        if st.button(
            "🎲 Regenerate",
            key="multi_regen",
            help="Re-draw the synthetic model scanpaths with a fresh random seed.",
        ):
            st.session_state["multi_nonce"] = (
                int(st.session_state.get("multi_nonce", 0)) + 1
            )
        nonce = int(st.session_state.get("multi_nonce", 0))

        paragraph_id = _trial_text_id(trial_words)
        models = generate_model_scanpaths(
            trial_words,
            n_models=n_models,
            reference_trial_id=selected_trial,
            paragraph_id=paragraph_id,
            nonce=nonce,
        )

        # Fixation-index window: restrict which fixations are drawn (and scored
        # in the snapshot table) for the real scanpath and every model.
        max_fix = max([len(trial_fixations)] + [len(m) for m in models.values()])
        if max_fix >= 2:
            # The slider value persists across trial / model-count changes, which
            # shift max_fix. Seed/clamp it via session_state *only* (no value=
            # arg), so the stored value is always inside [1, max_fix] — passing
            # both a value= and a session-state value raises a Streamlit warning,
            # and an out-of-range stored value would otherwise raise outright.
            stored = st.session_state.get("multi_fix_range")
            if isinstance(stored, (tuple, list)) and len(stored) == 2:
                lo = max(1, min(int(stored[0]), int(max_fix)))
                hi = max(lo, min(int(stored[1]), int(max_fix)))
                st.session_state["multi_fix_range"] = (lo, hi)
            else:
                st.session_state["multi_fix_range"] = (1, int(max_fix))
            fix_start, fix_end = st.slider(
                "Fixation index range",
                min_value=1,
                max_value=int(max_fix),
                key="multi_fix_range",
                help="Plot (and score, in the table) only fixations whose index "
                "falls in this range — applied to the real scanpath and every "
                "model. The convergence plots below always span the full reading.",
            )
        else:
            fix_start, fix_end = 1, int(max_fix)

        _render_trial_header(
            selected_participant,
            selected_trial,
            trial_words,
            prefix="Real scanpath:",
        )
        _render_paragraph_panel(
            trial_words, trial_fixations=trial_fixations, expanded=False
        )

    def _slice_range(fix: pd.DataFrame) -> pd.DataFrame:
        """Keep only fixations whose 1-based order index is in [start, end]."""
        if "order_in_trial" in fix.columns and not fix.empty:
            return fix[
                (fix["order_in_trial"] >= fix_start)
                & (fix["order_in_trial"] <= fix_end)
            ]
        return fix

    # Reuse the user's viz toggles, but force a clean comparison view (no
    # heatmap / raw gaze). The small model panels additionally drop word labels
    # and order numbers, which are illegible at grid scale.
    #
    # Normalize to a clean, comparable spatial view regardless of the sidebar
    # choices. The grid is an inherently spatial scanpath view, and the synthetic
    # model frames only carry the canonical fixation columns with *synthetic*
    # participant ids — so any option that (a) reads a real-only column or (b)
    # routes through a (participant_id, trial_id) groupby against the real
    # trial_words would either KeyError or silently mis-render the model panels:
    #   - x/y axes + color_by: a real-only column (saccade_amplitude, surprisal…)
    #     KeyErrors / falls back to a flat colour → pin axes to x/y, colour by
    #     duration_ms (present in every frame);
    #   - color_by_line / highlight_out_of_text: call measures helpers that group
    #     by (participant_id, trial_id); the model frames' "Model N" id never
    #     matches trial_words, so every model fixation would land off-line /
    #     out-of-text → pin both off.
    base_settings = _build_figure_settings(viz_settings, False)
    base_settings["raw_gaze"] = None
    base_settings["line_spacing"] = line_spacing
    base_settings["scale_text_to_boxes"] = scale_text_to_boxes
    real_settings = {
        **base_settings,
        "show_heatmap": False,
        "show_raw_gaze": False,
        "color_by": "duration_ms",
        "color_by_line": False,
        "highlight_out_of_text": False,
    }
    panel_settings = {**real_settings, "show_word_labels": False, "show_order": False}

    def _make_fig(fix: pd.DataFrame, settings: dict):
        return make_scanpath_figure(
            trial_words,
            fix,
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            base_font_size=int(base_font_size),
            font_family=font_family,
            x_field="x",
            y_field="y",
            **settings,
        )

    # Sliced (windowed) frames feed the spatial figures and the snapshot table;
    # the convergence plots below use the FULL scanpaths.
    sliced_real = _slice_range(trial_fixations)
    sliced_models = {name: _slice_range(m) for name, m in models.items()}
    windowed = fix_start > 1 or fix_end < max_fix

    with col_main:
        st.markdown("#### Real scanpath")
        if windowed:
            st.caption(
                f"Showing fixations **{fix_start}–{fix_end}** of up to {max_fix}."
            )
        real_fig = _make_fig(sliced_real, real_settings)
        _render_true_scale_chart(real_fig, key="multi_real")

        # Compute the similarity table once up front: the per-model NLD annotates
        # each grid panel below, and the full table is shown beneath the grid.
        table = compute_similarity_table(sliced_real, sliced_models, trial_words)
        nld_by_model = (
            dict(zip(table["Model"], table["NLD"])) if "NLD" in table.columns else {}
        )

        st.markdown("#### Model-generated scanpaths")
        # Estimate a uniform cell height from the figure aspect + column count
        # so panels line up and don't leave a tall whitespace band below each.
        fig_w = float(real_fig.layout.width or 900)
        fig_h = float(real_fig.layout.height or 600)
        aspect = fig_h / fig_w if fig_w else 0.5
        assumed_col_px = max(200, int(780 / max(1, n_cols)))
        cell_h = max(150, int(assumed_col_px * aspect) + 16)

        names = list(models.keys())
        for start in range(0, len(names), n_cols):
            row_names = names[start : start + n_cols]
            grid_cols = st.columns(n_cols)
            for cell, name in zip(grid_cols, row_names):
                with cell:
                    fix = sliced_models[name]
                    nld = nld_by_model.get(name)
                    if nld is not None and pd.notna(nld):
                        st.caption(f"**{name}** · NLD {nld:.2f} · {len(fix)} fix")
                    else:
                        st.caption(f"**{name}** · {len(fix)} fix")
                    _render_true_scale_chart(
                        _make_fig(fix, panel_settings),
                        key=f"multi_model_{name.replace(' ', '_')}",
                        max_height=cell_h,
                    )

        st.markdown("#### Similarity to the real scanpath")
        st.dataframe(_style_similarity_table(table), hide_index=True, width="stretch")

        # Cumulative metric convergence over the full scanpaths. Memoized so
        # dragging the fixation slider — which does NOT change these curves —
        # doesn't recompute the per-prefix NLDs. The key includes a fingerprint
        # of the real fixation content (not just the participant/trial id
        # strings), so a different dataset that happens to reuse the same ids
        # (e.g. two uploads both labelled participant "1") can't serve stale
        # curves.
        st.markdown("#### Metric convergence")
        st.caption(
            "NLD between the real scanpath and each model, computed cumulatively "
            "over the first *k* fixations (left) and the first *t* seconds of "
            "reading (right). Lower = more similar; computed on the full reading "
            "regardless of the fixation-range slider."
        )
        if trial_fixations.empty:
            fix_fingerprint: tuple = (0,)
        else:
            fix_fingerprint = (
                len(trial_fixations),
                len(trial_words),
                round(float(pd.to_numeric(trial_fixations["x"]).sum()), 3),
                round(float(pd.to_numeric(trial_fixations["y"]).sum()), 3),
                round(float(pd.to_numeric(trial_fixations["duration_ms"]).sum()), 3),
            )
        conv_key = (
            str(selected_participant),
            str(selected_trial),
            int(nonce),
            int(n_models),
            fix_fingerprint,
        )
        if st.session_state.get("_multi_conv_key") != conv_key:
            st.session_state["_multi_conv_key"] = conv_key
            st.session_state["_multi_conv_fix"] = {
                name: nld_by_fixation_index(trial_fixations, m, trial_words)
                for name, m in models.items()
            }
            st.session_state["_multi_conv_time"] = {
                name: nld_by_time(trial_fixations, m, trial_words)
                for name, m in models.items()
            }
        fix_curves = st.session_state["_multi_conv_fix"]
        time_curves = st.session_state["_multi_conv_time"]

        conv_cols = st.columns(2)
        with conv_cols[0]:
            fig_idx = make_metric_convergence_figure(
                fix_curves,
                x_title="Cumulative fixation index",
                y_title="NLD",
                title="NLD vs fixation index",
                canvas_width=520,
                base_font_size=int(base_font_size),
                font_family=font_family,
                highlight_x_range=(fix_start, fix_end) if windowed else None,
            )
            st.plotly_chart(fig_idx, width="stretch", config={"responsive": True})
        with conv_cols[1]:
            fig_time = make_metric_convergence_figure(
                time_curves,
                x_title="Elapsed reading time (s)",
                y_title="NLD",
                title="NLD vs time",
                canvas_width=520,
                base_font_size=int(base_font_size),
                font_family=font_family,
            )
            st.plotly_chart(fig_time, width="stretch", config={"responsive": True})


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
