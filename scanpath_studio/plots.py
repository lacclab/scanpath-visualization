"""Plotly figure builders for scanpath visualization."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .constants import (
    CANVAS_PAD_FRACTION,
    CANVAS_PAD_MIN_PX,
    COMPARISON_PALETTE,
    CURRENT_FIX_COLOR,
    CURRENT_FIX_OUTLINE,
    DEFAULT_FIXATION_COLORSCALE,
    DEFAULT_HEATMAP_COLORSCALE,
    DEFAULT_LINE_SPACING,
    DEFAULT_MARKER_SIZE_RANGE,
    FIX_MARKER_OUTLINE,
    FONT_FAMILY,
    OUT_OF_TEXT_COLOR,
    SACCADE_COLOR,
    WORD_BOX_COLOR,
    WORD_LABEL_COLOR,
)

COLORBAR_LEN_FRACTION = 0.33


def _compute_axis_ranges(
    canvas_width: int,
    canvas_height: int,
    *frames_with_xy: Tuple[Optional[pd.DataFrame], str, str],
    word_frames: Iterable[pd.DataFrame] = (),
) -> Tuple[
    list, list, Optional[float], Optional[float], Optional[float], Optional[float]
]:
    """Compute padded x/y ranges from any number of (frame, x_col, y_col) tuples.

    word_frames contribute box-extent bounds: x, x+width and y, y+height.
    Falls back to (0..canvas_width, canvas_height..0) when there's no data.
    Returns: x_range, y_range (y inverted), and the unpadded mins/maxs.
    """
    x_candidates: list = []
    y_candidates: list = []

    for df, x_col, y_col in frames_with_xy:
        if df is None or df.empty:
            continue
        if x_col in df.columns:
            x_candidates.extend([df[x_col].min(), df[x_col].max()])
        if y_col in df.columns:
            y_candidates.extend([df[y_col].min(), df[y_col].max()])

    for df in word_frames:
        if df is None or df.empty:
            continue
        x_candidates.extend([df["x"].min(), (df["x"] + df["width"]).max()])
        y_candidates.extend([df["y"].min(), (df["y"] + df["height"]).max()])

    x_range = [0, canvas_width]
    y_range = [canvas_height, 0]
    if not x_candidates or not y_candidates:
        return x_range, y_range, None, None, None, None

    x_min = float(np.nanmin(x_candidates))
    x_max = float(np.nanmax(x_candidates))
    y_min = float(np.nanmin(y_candidates))
    y_max = float(np.nanmax(y_candidates))

    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    pad_x = max(CANVAS_PAD_MIN_PX, CANVAS_PAD_FRACTION * x_span)
    pad_y = max(CANVAS_PAD_MIN_PX, CANVAS_PAD_FRACTION * y_span)
    x_range = [x_min - pad_x, x_max + pad_x]
    y_range = [y_max + pad_y, y_min - pad_y]
    return x_range, y_range, x_min, x_max, y_min, y_max


# Cap the *fixed* render size so the true-to-scale plot (rendered at exactly
# these pixels via tabs._render_true_scale_chart) fits a typical research display
# without horizontal scrolling. Aspect ratio is preserved when shrinking — both
# dims scale together, so boxes/text/fixations keep one true scale. A wider
# monitor just leaves margin (the plot is "narrower than the column", never
# stretched); a narrower window scrolls rather than distorting.
_DISPLAY_MAX_HEIGHT = 650
_DISPLAY_MAX_WIDTH = 900


def _fit_display_size(
    canvas_width: int,
    canvas_height: int,
    x_range: list,
    y_range: list,
    spatial_axes: bool,
) -> Tuple[int, int]:
    """Return (width, height) for `fig.update_layout` so the plot fits onscreen.

    With `scaleanchor="x", scaleratio=1` the plot domain shrinks to the data
    aspect ratio, leaving large blank vertical strips when the figure box is
    the full monitor height. We match the figure box to the actual plot
    domain — and additionally clamp both dims so the whole plot fits in one
    viewport without scrolling. Falls back to (canvas_w, canvas_h) when axes
    aren't spatial or the data range is degenerate.
    """
    if not spatial_axes:
        return canvas_width, canvas_height
    x_span = x_range[1] - x_range[0]
    y_span = y_range[0] - y_range[1]  # y_range is inverted [y_max, y_min]
    if x_span <= 0 or y_span <= 0:
        return canvas_width, canvas_height
    aspect = x_span / y_span
    w, h = canvas_width, int(round(canvas_width / aspect))
    # Shrink (preserving aspect) until both dims fit the viewport caps.
    if h > _DISPLAY_MAX_HEIGHT:
        h = _DISPLAY_MAX_HEIGHT
        w = int(round(h * aspect))
    if w > _DISPLAY_MAX_WIDTH:
        w = _DISPLAY_MAX_WIDTH
        h = int(round(w / aspect))
    return max(w, 100), max(h, 100)


# Text in Plotly is sized in screen pixels with no native "data unit" mode, so
# to keep word labels true-to-scale we convert a real (monitor-pixel) font size
# into the figure's screen pixels using the same scale the boxes/fixations use.
_MIN_LABEL_PX = 1.0

# Advance-width / em of a typical monospaced font (DejaVu Sans Mono ≈ 0.6).
# Reading experiments use monospaced fonts (OneStop included), so we can cap the
# label size by the box *width* — stopping long words from colliding when the
# on-screen font is a touch wider than the one the experiment was rendered in.
_MONO_ASPECT = 0.6
_WIDTH_FIT_MARGIN = 0.92  # leave a sliver of horizontal padding inside each box


def _width_fit_font(words: pd.DataFrame) -> Optional[float]:
    """Largest font (data px) at which every word still fits its box width.

    Word boxes are drawn around the rendered text, so ``box_width / n_chars`` is
    the per-character advance; dividing by the monospace aspect recovers the em.
    The tightest words bind, so we take a low quantile (robust to one odd box).
    Returns None when there's no text/width to measure.
    """
    if "width" not in words.columns or "text" not in words.columns:
        return None
    w = pd.to_numeric(words["width"], errors="coerce")
    n = words["text"].astype(str).str.len().clip(lower=1)
    per_char = (w / n).replace([np.inf, -np.inf], np.nan).dropna()
    if per_char.empty:
        return None
    tight = float(per_char.quantile(0.05))
    return tight / _MONO_ASPECT * _WIDTH_FIT_MARGIN if tight > 0 else None


def _display_scale(x_range: list, y_range: list, fitted_w: int, fitted_h: int) -> float:
    """Screen px per data unit for a fixed-size, equal-aspect spatial plot.

    With ``scaleratio=1`` the x and y mappings are identical; we take the min so
    rounding can never make text/markers sized through this overflow the boxes.
    Returns 1.0 for degenerate ranges.
    """
    x_span = x_range[1] - x_range[0]
    y_span = y_range[0] - y_range[1]  # y_range is inverted [y_max, y_min]
    if x_span <= 0 or y_span <= 0:
        return 1.0
    return min(fitted_w / x_span, fitted_h / y_span)


def _word_label_font_px(
    words: pd.DataFrame,
    *,
    scale: float,
    line_spacing: float,
    manual_font_px: float,
    scale_text_to_boxes: bool,
) -> float:
    """Word-label font size in *screen* px so the text stays true-to-scale.

    The experiment's font lives in monitor pixels. To keep the rendered glyphs
    the same physical fraction of the (data-space) word boxes at any display
    size, the font is expressed in data px and multiplied by ``scale`` (screen
    px per data unit):

    - ``scale_text_to_boxes`` (default): one line of text fills
      ``1 / line_spacing`` of the line pitch, where the pitch is the median word
      box height from the data. For OneStop the boxes tile the lines
      (height == pitch) and ``line_spacing == 3`` (one blank line above + below),
      so the height budget is box_height / 3. The size is *also* capped so the
      longest words still fit their box width (see :func:`_width_fit_font`), which
      keeps the default monospaced font from colliding; the smaller of the two
      wins.
    - otherwise / no usable boxes: ``manual_font_px`` is treated as the real
      monitor font size and scaled the same way.
    """
    font_data_px = float(manual_font_px)
    if scale_text_to_boxes and not words.empty and "height" in words.columns:
        box_h = float(pd.to_numeric(words["height"], errors="coerce").median())
        height_fit = box_h / line_spacing if (box_h > 0 and line_spacing > 0) else None
        width_fit = _width_fit_font(words)
        candidates = [c for c in (height_fit, width_fit) if c and c > 0]
        if candidates:
            font_data_px = min(candidates)
    return max(font_data_px * scale, _MIN_LABEL_PX)


_QUALITATIVE_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _resolve_marker_colors(
    color_data: Optional[pd.Series], is_numeric_color: bool
) -> Tuple[object, list]:
    """Return (marker_color, category_legend) for the fixation scatter trace.

    - Numeric color_data is passed straight through (Plotly maps it via colorscale).
    - Categorical color_data is mapped to a discrete palette so the picker has
      visible effect; the returned legend is a list of (category, hex) pairs the
      caller can render as legend-only scatter traces.
    - Missing / unmappable color_data falls back to the first palette color.
    """
    if color_data is None:
        return _QUALITATIVE_PALETTE[0], []
    if is_numeric_color:
        return color_data, []
    series = color_data.fillna("(missing)").astype(str)
    unique_vals = list(pd.unique(series))
    cat_to_color = {
        val: _QUALITATIVE_PALETTE[i % len(_QUALITATIVE_PALETTE)]
        for i, val in enumerate(unique_vals)
    }
    marker_color = [cat_to_color[val] for val in series]
    legend = [(val, cat_to_color[val]) for val in unique_vals]
    return marker_color, legend


def _compute_marker_sizes(
    durations: pd.Series, size_range: Tuple[int, int] = DEFAULT_MARKER_SIZE_RANGE
) -> np.ndarray:
    """Map fixation durations to marker sizes by linear interpolation."""
    durations = pd.to_numeric(durations, errors="coerce").fillna(0)
    d_min, d_max = float(durations.min()), float(durations.max())
    min_size, max_size = size_range
    if d_max - d_min > 0:
        return np.interp(durations, (d_min, d_max), (min_size, max_size))
    return np.full(len(durations), (min_size + max_size) / 2)


def _saccade_segments(
    fix_df: pd.DataFrame, x_col: str, y_col: str
) -> Tuple[list, list]:
    """Return concatenated x/y arrays separated by None for a single saccade trace."""
    if len(fix_df) < 2:
        return [], []
    ordered = fix_df.sort_values("timestamp_ms")
    xs: list = []
    ys: list = []
    x_vals = ordered[x_col].tolist()
    y_vals = ordered[y_col].tolist()
    for i in range(len(ordered) - 1):
        xs.extend([x_vals[i], x_vals[i + 1], None])
        ys.extend([y_vals[i], y_vals[i + 1], None])
    return xs, ys


# Saccades shorter than this fraction of the fixation-extent diagonal get no
# direction arrow — their heading is sub-pixel noise (refixations on one word).
_ARROW_MIN_LEN_FRAC = 0.005


def _saccade_arrow_markers(
    fix_df: pd.DataFrame, x_col: str, y_col: str
) -> Tuple[list, list, list]:
    """Arrowhead position + rotation for each saccade, for a marker trace.

    Returns (mid_x, mid_y, angle_deg) with one entry per consecutive-fixation
    segment: a marker at the segment midpoint, rotated to point along the gaze
    direction. Angles follow Plotly's ``marker.angle`` convention (degrees
    clockwise from "up") and account for the reversed y-axis — data y grows
    downward on screen — so they read correctly on the plot.
    """
    if len(fix_df) < 2:
        return [], [], []
    ordered = fix_df.sort_values("timestamp_ms")
    xv = pd.to_numeric(ordered[x_col], errors="coerce").to_numpy()
    yv = pd.to_numeric(ordered[y_col], errors="coerce").to_numpy()
    # Suppress arrowheads on micro-saccades: a sub-pixel refixation has a
    # well-defined midpoint but its direction is just noise, so a full-size
    # arrow would point a random way. Threshold scales with the data extent so
    # it's dataset-agnostic.
    finite = np.isfinite(xv) & np.isfinite(yv)
    if finite.any():
        x_ext = float(np.nanmax(xv[finite]) - np.nanmin(xv[finite]))
        y_ext = float(np.nanmax(yv[finite]) - np.nanmin(yv[finite]))
        min_len = np.hypot(x_ext, y_ext) * _ARROW_MIN_LEN_FRAC
    else:
        min_len = 0.0
    mid_x: list = []
    mid_y: list = []
    angles: list = []
    for i in range(len(ordered) - 1):
        x0, y0, x1, y1 = xv[i], yv[i], xv[i + 1], yv[i + 1]
        if not np.isfinite((x0, y0, x1, y1)).all():
            continue
        dx, dy = x1 - x0, y1 - y0
        seg_len = float(np.hypot(dx, dy))
        if seg_len == 0.0 or seg_len < min_len:
            continue
        mid_x.append((x0 + x1) / 2.0)
        mid_y.append((y0 + y1) / 2.0)
        # marker.angle is clockwise from up; screen-up is decreasing data y
        # (the y-axis is drawn reversed), so negate dy.
        angles.append(float(np.degrees(np.arctan2(dx, -dy))))
    return mid_x, mid_y, angles


def build_word_boxes(words: pd.DataFrame, color: str = WORD_BOX_COLOR) -> list:
    shapes = []
    for row in words.itertuples():
        x0, y0 = row.x, row.y
        x1, y1 = row.x + row.width, row.y + row.height
        shapes.append(
            dict(
                type="rect",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line=dict(color=color, width=1),
                fillcolor="rgba(100,100,100,0.05)",
            )
        )
    return shapes


# Bold-frame overlay for critical-span words; rendered on top of regular word
# boxes only when the trial was shown with a preview question (Hunting condition).
_CRITICAL_FRAME_COLOR = "#000000"  # black — high-contrast frame, readable over heatmaps
_CRITICAL_FRAME_WIDTH = 2
_CRITICAL_TEXT_COLOR = (
    "#C8097C"  # dark pink — used when critical_span_style="Mark text"
)


def build_critical_span_overlay(words: pd.DataFrame) -> list:
    """Return outline shapes for the critical span (`is_in_aspan`).

    Each visual line that contains critical-span words gets its own outline
    rectangle, going from the *first* to the *last* critical word on that
    line (not the whole line). Returns [] when the column is missing or no
    words match.
    """
    if "is_in_aspan" not in words.columns:
        return []
    mask = words["is_in_aspan"].fillna(False).astype(bool)
    if not mask.any():
        return []
    span = words[mask].copy()

    # Cluster words into visual lines by y. `line_idx` upstream is often a
    # constant (no real per-word line numbers in OneStop IA exports), so we
    # group by y with a tolerance of ~half a word-height: rows whose y jumps
    # by more than that are on a new line.
    typical_h = float(span["height"].median() or 1.0)
    y_sorted = span["y"].sort_values()
    line_ids = (y_sorted.diff().fillna(0) > typical_h * 0.5).cumsum()
    span["_line_id"] = line_ids.reindex(span.index)

    shapes = []
    for _, group in span.groupby("_line_id"):
        x0 = float(group["x"].min())
        x1 = float((group["x"] + group["width"]).max())
        y0 = float(group["y"].min())
        y1 = float((group["y"] + group["height"]).max())
        shapes.append(
            dict(
                type="rect",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line=dict(color=_CRITICAL_FRAME_COLOR, width=_CRITICAL_FRAME_WIDTH),
                fillcolor="rgba(0,0,0,0)",
                layer="above",
            )
        )
    return shapes


def _add_word_label_trace(
    fig: go.Figure,
    words: pd.DataFrame,
    base_font_size: int,
    font_family: str,
    row: Optional[int] = None,
    col: Optional[int] = None,
    highlight_critical_text: bool = False,
) -> None:
    if words.empty or "text" not in words.columns:
        return
    customdata = None
    hover = "Word %{text}<extra></extra>"
    if "word_id" in words.columns and "line_idx" in words.columns:
        customdata = words[["word_id", "line_idx"]]
        hover = (
            "Word %{text}<br>Word ID %{customdata[0]}"
            "<br>Line %{customdata[1]}<extra></extra>"
        )
    # Per-word text color: dark pink for critical-span words when the caller
    # asks for "Mark text", default label color otherwise.
    if highlight_critical_text and "is_in_aspan" in words.columns:
        critical_mask = words["is_in_aspan"].fillna(False).astype(bool)
        text_color = [
            _CRITICAL_TEXT_COLOR if is_crit else WORD_LABEL_COLOR
            for is_crit in critical_mask
        ]
    else:
        text_color = WORD_LABEL_COLOR
    trace = go.Scatter(
        x=words["x"] + words["width"] / 2,
        y=words["y"] + words["height"] / 2,
        text=words["text"],
        mode="text",
        showlegend=False,
        textfont=dict(color=text_color, size=base_font_size, family=font_family),
        hovertemplate=hover,
        customdata=customdata,
        name="words",
    )
    if row is not None and col is not None:
        fig.add_trace(trace, row=row, col=col)
    else:
        fig.add_trace(trace)


def make_scanpath_figure(
    words: pd.DataFrame,
    fixations: pd.DataFrame,
    *,
    canvas_width: int,
    canvas_height: int,
    base_font_size: int,
    font_family: str,
    x_field: str,
    y_field: str,
    show_words: bool,
    show_word_labels: bool,
    show_fixations: bool,
    show_order: bool,
    show_saccades: bool,
    show_heatmap: bool,
    color_by: str,
    heatmap_metric: Optional[str],
    show_saccade_arrows: bool = False,
    heatmap_style: str = "Word boxes",
    marker_size_range: Tuple[int, int],
    order_font_size: int,
    order_font_color: str,
    show_colorbars: bool,
    fixation_color_range: Optional[Tuple[float, float]],
    heatmap_range: Optional[Tuple[float, float]],
    fixation_colorscale: str = DEFAULT_FIXATION_COLORSCALE,
    heatmap_colorscale: str = DEFAULT_HEATMAP_COLORSCALE,
    raw_gaze: Optional[pd.DataFrame] = None,
    show_raw_gaze: bool = False,
    critical_span_style: str = "Mark text",
    background_color: Optional[str] = None,
    color_by_line: bool = False,
    highlight_out_of_text: bool = False,
    line_spacing: float = DEFAULT_LINE_SPACING,
    scale_text_to_boxes: bool = True,
) -> go.Figure:
    fig = go.Figure()
    spatial_axes = x_field == "x" and y_field == "y"
    font_settings = dict(family=font_family or FONT_FAMILY, size=base_font_size)

    raw_for_range = raw_gaze if (show_raw_gaze and raw_gaze is not None) else None
    if spatial_axes:
        x_range, y_range, x_min_data, x_max_data, y_min_data, y_max_data = (
            _compute_axis_ranges(
                canvas_width,
                canvas_height,
                (fixations, x_field, y_field),
                (raw_for_range, "x", "y"),
                word_frames=[words] if not words.empty else [],
            )
        )
    else:
        x_range = [0, canvas_width]
        y_range = [canvas_height, 0]
        x_min_data = x_max_data = y_min_data = y_max_data = None

    # Fix the display size up front so the data->screen scale is known: word
    # labels are then sized in that scale (true-to-scale text), and the same
    # fitted_w/fitted_h drive the final layout below.
    fitted_w, fitted_h = _fit_display_size(
        canvas_width, canvas_height, x_range, y_range, spatial_axes
    )
    scale = (
        _display_scale(x_range, y_range, fitted_w, fitted_h) if spatial_axes else 1.0
    )
    label_font_px = _word_label_font_px(
        words,
        scale=scale,
        line_spacing=line_spacing,
        manual_font_px=base_font_size,
        scale_text_to_boxes=scale_text_to_boxes,
    )

    # Hunting/preview trials get the critical span marked one of two ways:
    #   - "Mark text": color the critical-span words dark pink (no border).
    #   - "Mark border": draw a thin black outline around the span.
    has_preview = (
        "question_preview" in words.columns
        and not words.empty
        and bool(words["question_preview"].fillna(False).astype(bool).iloc[0])
    )
    highlight_critical_text = has_preview and critical_span_style == "Mark text"

    if spatial_axes and not words.empty:
        if show_words:
            shapes = build_word_boxes(words)
            if has_preview and critical_span_style == "Mark border":
                shapes = shapes + build_critical_span_overlay(words)
            fig.update_layout(shapes=shapes)
        if show_word_labels:
            _add_word_label_trace(
                fig,
                words,
                label_font_px,
                font_settings["family"],
                highlight_critical_text=highlight_critical_text,
            )

    if show_raw_gaze and raw_gaze is not None and not raw_gaze.empty:
        if "timestamp_ms" in raw_gaze.columns:
            color_vals = raw_gaze["timestamp_ms"]
            colorscale = "Viridis"
        else:
            color_vals = "#888888"
            colorscale = None
        fig.add_trace(
            go.Scatter(
                x=raw_gaze["x"],
                y=raw_gaze["y"],
                mode="markers",
                marker=dict(
                    size=4,
                    color=color_vals,
                    colorscale=colorscale,
                    opacity=0.6,
                    showscale=False,
                ),
                hovertemplate=(
                    "Raw gaze<br>x: %{x:.1f}<br>y: %{y:.1f}"
                    "<br>t: %{customdata} ms<extra></extra>"
                ),
                customdata=raw_gaze["timestamp_ms"]
                if "timestamp_ms" in raw_gaze.columns
                else None,
                name="Raw gaze",
                showlegend=True,
            )
        )

    if spatial_axes and show_heatmap and not fixations.empty:
        weights = fixations["duration_ms"] if heatmap_metric == "duration_ms" else None
        x_min = (
            x_min_data if x_min_data is not None else float(fixations[x_field].min())
        )
        x_max = (
            x_max_data if x_max_data is not None else float(fixations[x_field].max())
        )
        y_min = (
            y_min_data if y_min_data is not None else float(fixations[y_field].min())
        )
        y_max = (
            y_max_data if y_max_data is not None else float(fixations[y_field].max())
        )
        if heatmap_style == "Interpolated":
            # Smooth, word-box-independent density over the fixations themselves.
            _add_interpolated_heatmap(
                fig,
                fixations,
                x_field=x_field,
                y_field=y_field,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                weights=weights,
                heatmap_colorscale=heatmap_colorscale,
                show_colorbars=show_colorbars,
            )
        elif not words.empty:
            _add_word_level_heatmap(
                fig,
                words,
                fixations,
                x_field=x_field,
                y_field=y_field,
                weights=weights,
                heatmap_colorscale=heatmap_colorscale,
                heatmap_range=heatmap_range,
                show_colorbars=show_colorbars,
            )
        else:
            _add_density_heatmap(
                fig,
                fixations,
                x_field=x_field,
                y_field=y_field,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                weights=weights,
                heatmap_colorscale=heatmap_colorscale,
                heatmap_range=heatmap_range,
                show_colorbars=show_colorbars,
            )
    elif spatial_axes and show_heatmap and not words.empty:
        # Words-only dataset (no fixation report): fall back to the words
        # frame's own pre-aggregated reading measures for the box heatmap.
        measure = (
            "total_fixation_duration_ms"
            if heatmap_metric == "duration_ms"
            else "n_fixations"
        )
        if measure in words.columns:
            _add_word_measure_heatmap(
                fig,
                words,
                measure,
                heatmap_colorscale=heatmap_colorscale,
                heatmap_range=heatmap_range,
                show_colorbars=show_colorbars,
            )

    if spatial_axes and show_saccades and len(fixations) > 1:
        sx, sy = _saccade_segments(fixations, x_field, y_field)
        if sx:
            fig.add_trace(
                go.Scatter(
                    x=sx,
                    y=sy,
                    mode="lines",
                    line=dict(color=SACCADE_COLOR, width=2),
                    hoverinfo="skip",
                    showlegend=False,
                    name="saccades",
                )
            )

    # Directional arrowheads on each saccade (opt-in). Independent of the line
    # above so it can be toggled separately; rendered before the fixation
    # markers so the dots sit on top.
    if spatial_axes and show_saccade_arrows and len(fixations) > 1:
        amx, amy, aang = _saccade_arrow_markers(fixations, x_field, y_field)
        if amx:
            fig.add_trace(
                go.Scatter(
                    x=amx,
                    y=amy,
                    mode="markers",
                    marker=dict(
                        symbol="arrow",
                        size=12,
                        angle=aang,
                        angleref="up",
                        color=SACCADE_COLOR,
                        line=dict(width=0),
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                    name="saccade direction",
                )
            )

    if show_fixations and not fixations.empty:
        ordered = fixations.sort_values("timestamp_ms")
        # "Color by line" overrides the chosen color field: each fixation is
        # tinted by the text line it lands on (lines inferred from word
        # geometry). Rendered as discrete categories so the legend reads
        # "line: Line 1", "line: Line 2", …
        if color_by_line and spatial_axes and not words.empty:
            from .measures import assign_fixation_lines

            line_ids = assign_fixation_lines(ordered, words)
            color_data = line_ids.map(
                lambda v: f"Line {int(v) + 1}" if pd.notna(v) else "(off-text)"
            )
            color_label = "line"
            is_numeric_color = False
        else:
            color_data = ordered[color_by] if color_by in ordered.columns else None
            color_label = color_by
            is_numeric_color = color_data is not None and pd.api.types.is_numeric_dtype(
                color_data
            )
        marker_color, category_legend = _resolve_marker_colors(
            color_data, is_numeric_color
        )
        sizes = _compute_marker_sizes(ordered["duration_ms"], marker_size_range)
        fig.add_trace(
            go.Scatter(
                x=ordered[x_field],
                y=ordered[y_field],
                mode="markers+text" if show_order else "markers",
                marker=dict(
                    size=sizes,
                    color=marker_color,
                    colorscale=fixation_colorscale if is_numeric_color else None,
                    showscale=show_colorbars and is_numeric_color,
                    colorbar=dict(
                        title=color_label.replace("_", " ").title(),
                        x=1.12,
                        lenmode="fraction",
                        len=COLORBAR_LEN_FRACTION,
                        y=0.5,
                        yanchor="middle",
                    )
                    if show_colorbars and is_numeric_color
                    else None,
                    cmin=fixation_color_range[0] if fixation_color_range else None,
                    cmax=fixation_color_range[1] if fixation_color_range else None,
                    line=dict(color=FIX_MARKER_OUTLINE, width=0.5),
                ),
                text=ordered["order_in_trial"] if show_order else None,
                textfont=dict(
                    color=order_font_color,
                    size=order_font_size,
                    family=font_settings["family"],
                ),
                textposition="top center",
                hovertemplate=(
                    "Fixation #%{customdata[0]}<br>"
                    "Duration %{customdata[1]} ms<br>"
                    "Word #%{customdata[2]}<br>"
                    "Pass #%{customdata[3]}<extra></extra>"
                ),
                customdata=np.stack(
                    [
                        ordered["order_in_trial"],
                        ordered["duration_ms"],
                        ordered.get("word_id", pd.Series([np.nan] * len(ordered))),
                        ordered.get("pass_index", pd.Series([np.nan] * len(ordered))),
                    ],
                    axis=1,
                ),
                name="Fixations",
                showlegend=False,
            )
        )
        legend_limit = len(_QUALITATIVE_PALETTE)
        truncated_legend = category_legend[:legend_limit]
        for category, color in truncated_legend:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=color,
                        line=dict(color=FIX_MARKER_OUTLINE, width=0.5),
                    ),
                    name=f"{color_label}: {category}",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )
        if len(category_legend) > legend_limit:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color="#cccccc"),
                    name=f"… +{len(category_legend) - legend_limit} more",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

        # Out-of-text overlay: mark fixations falling outside every word box
        # with a red ✕ on top of the regular marker. Requires word boxes +
        # spatial axes to define "in text".
        if highlight_out_of_text and spatial_axes and not words.empty:
            from .measures import fixation_in_text_mask

            off = ordered[~fixation_in_text_mask(ordered, words)]
            if not off.empty:
                fig.add_trace(
                    go.Scatter(
                        x=off[x_field],
                        y=off[y_field],
                        mode="markers",
                        marker=dict(
                            symbol="x",
                            size=13,
                            color=OUT_OF_TEXT_COLOR,
                            line=dict(color="#ffffff", width=1),
                        ),
                        name="Out-of-text",
                        showlegend=True,
                        hovertemplate=(
                            "Out-of-text fixation<br>x %{x:.0f}, y %{y:.0f}"
                            "<extra></extra>"
                        ),
                    )
                )

    xaxis_cfg = dict(showticklabels=False, showgrid=False, zeroline=False, title=None)
    yaxis_cfg = dict(showticklabels=False, showgrid=False, zeroline=False, title=None)
    if spatial_axes:
        xaxis_cfg.update(range=x_range, constrain="domain")
        yaxis_cfg.update(
            range=y_range, constrain="domain", scaleanchor="x", scaleratio=1
        )
    else:
        xaxis_cfg.update(
            showticklabels=True, showgrid=True, title=x_field.replace("_", " ").title()
        )
        yaxis_cfg.update(
            showticklabels=True, showgrid=True, title=y_field.replace("_", " ").title()
        )

    shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    if spatial_axes:
        shapes.append(
            dict(
                type="rect",
                x0=x_range[0],
                y0=y_range[1],
                x1=x_range[1],
                y1=y_range[0],
                line=dict(color="#000000", width=1),
                fillcolor="rgba(0,0,0,0)",
            )
        )

    # fitted_w / fitted_h were computed up front (so the label scale matched).
    fig.update_layout(
        height=fitted_h,
        width=fitted_w,
        autosize=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=xaxis_cfg,
        yaxis=yaxis_cfg,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        # None leaves the template's default white; a hex value paints both the
        # plotting area and the surrounding paper (e.g. a neutral gray).
        plot_bgcolor=background_color,
        paper_bgcolor=background_color,
        font=font_settings,
        shapes=shapes,
    )
    return fig


def _add_word_level_heatmap(
    fig: go.Figure,
    words: pd.DataFrame,
    fixations: pd.DataFrame,
    *,
    x_field: str,
    y_field: str,
    weights: Optional[pd.Series],
    heatmap_colorscale: str,
    heatmap_range: Optional[Tuple[float, float]],
    show_colorbars: bool,
) -> None:
    # Pull the fixation coordinates (and optional weights) into numpy arrays once,
    # then test box membership per word against the arrays. Same O(words × fix)
    # work as before but without rebuilding pandas Series each iteration, and with
    # O(fix) memory (no full words × fix matrix).
    fx = pd.to_numeric(fixations[x_field], errors="coerce").to_numpy(dtype=float)
    fy = pd.to_numeric(fixations[y_field], errors="coerce").to_numpy(dtype=float)
    w_arr = (
        pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
        if weights is not None
        else None
    )
    word_values = []
    for word_row in words.itertuples():
        wx0, wy0 = word_row.x, word_row.y
        wx1, wy1 = wx0 + word_row.width, wy0 + word_row.height
        in_word = (fx >= wx0) & (fx <= wx1) & (fy >= wy0) & (fy <= wy1)
        val = (
            float(np.nansum(w_arr[in_word]))
            if w_arr is not None
            else float(in_word.sum())
        )
        word_values.append(val)

    _draw_word_value_heatmap(
        fig,
        words,
        word_values,
        heatmap_colorscale=heatmap_colorscale,
        heatmap_range=heatmap_range,
        show_colorbars=show_colorbars,
        colorbar_title="Fixation count" if weights is None else "Duration (ms)",
    )


def _add_word_measure_heatmap(
    fig: go.Figure,
    words: pd.DataFrame,
    measure: str,
    *,
    heatmap_colorscale: str,
    heatmap_range: Optional[Tuple[float, float]],
    show_colorbars: bool,
) -> None:
    """Word-box heatmap from a pre-aggregated per-word measure column.

    Used for words-only datasets (IA report without a fixation report): the
    usual heatmap aggregates fixation durations/counts into the boxes, but
    with no fixations the dataset's own reading measures (e.g. total fixation
    duration) carry the same information."""
    values = pd.to_numeric(words[measure], errors="coerce").fillna(0.0)
    _draw_word_value_heatmap(
        fig,
        words,
        [float(v) for v in values],
        heatmap_colorscale=heatmap_colorscale,
        heatmap_range=heatmap_range,
        show_colorbars=show_colorbars,
        colorbar_title="Fixation count"
        if measure == "n_fixations"
        else "Duration (ms)",
    )


def _draw_word_value_heatmap(
    fig: go.Figure,
    words: pd.DataFrame,
    word_values: list,
    *,
    heatmap_colorscale: str,
    heatmap_range: Optional[Tuple[float, float]],
    show_colorbars: bool,
    colorbar_title: str,
) -> None:
    from plotly.colors import sample_colorscale

    nonzero_rows = [(wr, v) for wr, v in zip(words.itertuples(), word_values) if v > 0]
    if not nonzero_rows:
        return
    vals = [v for _, v in nonzero_rows]
    z_min = heatmap_range[0] if heatmap_range else float(min(vals))
    z_max = heatmap_range[1] if heatmap_range else float(max(vals))
    z_span = max(z_max - z_min, 1e-9)

    heatmap_shapes = []
    for wr, v in nonzero_rows:
        norm = max(0.0, min(1.0, (v - z_min) / z_span))
        color = sample_colorscale(heatmap_colorscale, [norm])[0]
        heatmap_shapes.append(
            dict(
                type="rect",
                x0=wr.x,
                y0=wr.y,
                x1=wr.x + wr.width,
                y1=wr.y + wr.height,
                line=dict(width=0),
                fillcolor=color,
                opacity=0.5,
                layer="below",
            )
        )
    existing = list(fig.layout.shapes) if fig.layout.shapes else []
    fig.update_layout(shapes=existing + heatmap_shapes)
    if show_colorbars:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    colorscale=heatmap_colorscale,
                    showscale=True,
                    cmin=z_min,
                    cmax=z_max,
                    colorbar=dict(
                        title=colorbar_title,
                        x=1.02,
                        lenmode="fraction",
                        len=COLORBAR_LEN_FRACTION,
                        y=0.5,
                        yanchor="middle",
                    ),
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )


def _add_density_heatmap(
    fig: go.Figure,
    fixations: pd.DataFrame,
    *,
    x_field: str,
    y_field: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    weights: Optional[pd.Series],
    heatmap_colorscale: str,
    heatmap_range: Optional[Tuple[float, float]],
    show_colorbars: bool,
) -> None:
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    fig.add_trace(
        go.Histogram2d(
            x=fixations[x_field],
            y=fixations[y_field],
            xbins=dict(start=x_min, end=x_max, size=x_span / 40.0),
            ybins=dict(start=y_min, end=y_max, size=y_span / 40.0),
            colorscale=heatmap_colorscale,
            opacity=0.35,
            showscale=show_colorbars,
            colorbar=dict(
                title="Fixation density" if weights is None else "Duration (ms)",
                x=1.02,
                lenmode="fraction",
                len=COLORBAR_LEN_FRACTION,
                y=0.5,
                yanchor="middle",
            ),
            histfunc="sum" if weights is not None else "count",
            z=weights,
            zmin=heatmap_range[0] if heatmap_range else None,
            zmax=heatmap_range[1] if heatmap_range else None,
        )
    )


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Normalized 1-D Gaussian kernel, truncated at 3 sigma."""
    radius = max(1, int(round(sigma * 3)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _gaussian_blur_2d(
    grid: np.ndarray, sigma_rows: float, sigma_cols: float
) -> np.ndarray:
    """Separable Gaussian blur (a numpy-only stand-in for scipy.ndimage)."""
    out = grid.astype(float)
    if sigma_rows and sigma_rows > 0:
        k = _gaussian_kernel_1d(sigma_rows)
        out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, out)
    if sigma_cols and sigma_cols > 0:
        k = _gaussian_kernel_1d(sigma_cols)
        out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, out)
    return out


# Interpolated-heatmap tuning. The Gaussian sigma defaults to a fraction of the
# larger data span — enough to merge a fixation cluster into one smooth blob
# without bleeding across neighbouring text lines.
_INTERP_GRID = 240  # cells along the wider axis
_INTERP_SIGMA_FRAC = 0.02  # sigma as a fraction of the larger data span
_INTERP_MIN_SIGMA_PX = 8.0
_INTERP_OPACITY = 0.45
_INTERP_FLOOR_FRAC = 0.02  # cells below this fraction of the peak render transparent


def _add_interpolated_heatmap(
    fig: go.Figure,
    fixations: pd.DataFrame,
    *,
    x_field: str,
    y_field: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    weights: Optional[pd.Series],
    heatmap_colorscale: str,
    show_colorbars: bool,
) -> None:
    """Smooth, word-box-independent fixation heatmap (Gaussian-interpolated).

    Bins the fixations onto a fine grid (weighted by duration when ``weights``
    is given), then blurs with a Gaussian — the classic eye-movement heatmap
    (cf. PyGaze's gaze plotter). Empty cells render transparent so the reading
    text stays legible underneath.
    """
    xs = pd.to_numeric(fixations[x_field], errors="coerce")
    ys = pd.to_numeric(fixations[y_field], errors="coerce")
    valid = xs.notna() & ys.notna()
    if not valid.any():
        return
    if weights is not None:
        w = (
            pd.to_numeric(weights, errors="coerce")
            .reindex(fixations.index)
            .fillna(0.0)[valid]
            .to_numpy()
        )
    else:
        w = np.ones(int(valid.sum()))
    xs = xs[valid].to_numpy()
    ys = ys[valid].to_numpy()

    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    nx = _INTERP_GRID
    ny = max(10, int(round(_INTERP_GRID * y_span / x_span)))
    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)
    # histogram2d returns shape (nx, ny); transpose so rows index y, cols index x
    # (the orientation go.Heatmap's z expects).
    hist, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges], weights=w)
    grid = hist.T

    sigma_px = max(_INTERP_MIN_SIGMA_PX, _INTERP_SIGMA_FRAC * max(x_span, y_span))
    blurred = _gaussian_blur_2d(
        grid, sigma_rows=sigma_px / (y_span / ny), sigma_cols=sigma_px / (x_span / nx)
    )
    peak = float(blurred.max())
    if peak <= 0:
        return
    # Near-zero cells -> NaN so Plotly renders them transparent (only populated
    # regions get tinted, keeping the text readable).
    z = np.where(blurred < peak * _INTERP_FLOOR_FRAC, np.nan, blurred)

    fig.add_trace(
        go.Heatmap(
            x=(x_edges[:-1] + x_edges[1:]) / 2.0,
            y=(y_edges[:-1] + y_edges[1:]) / 2.0,
            z=z,
            colorscale=heatmap_colorscale,
            opacity=_INTERP_OPACITY,
            showscale=show_colorbars,
            # z is a Gaussian-smoothed density in arbitrary (weighted) units, not
            # the per-word counts/ms the `heatmap_range` slider is calibrated for,
            # so it autoscales from 0 rather than borrowing that range.
            zmin=0.0,
            colorbar=dict(
                title="Dwell-time density"
                if weights is not None
                else "Fixation density",
                x=1.02,
                lenmode="fraction",
                len=COLORBAR_LEN_FRACTION,
                y=0.5,
                yanchor="middle",
            ),
            hoverinfo="skip",
            name="Fixation heatmap",
        )
    )


# =============================================================================
# Scanpath animation — one or two scanpaths on a shared real reading-time clock
# =============================================================================

# Floor on per-frame duration: ~one 60 fps display frame. Browsers can't redraw
# faster than this, so it's the lowest value at which the quoted playback time
# (n_frames * avg) still matches the observed runtime — going lower would just
# make the quote understate reality. Also keeps the briefest gaps perceptible.
_ANIM_MIN_FRAME_MS = 16

# Vertical space (px) reserved BELOW the animation plot for the transport
# controls (play / pause / restart buttons + the time slider with its "Elapsed"
# readout). The figure is grown by this much (plus a small safety buffer) and
# the controls are placed in the bottom margin, so Plotly's automargin never has
# to shrink the equal-aspect plot to fit them — which would make the word boxes
# smaller than the true-to-scale label font computed for fitted_h (the
# text-too-large bug). Keeps the animation plot the SAME size as the static one.
_CONTROLS_MARGIN_PX = 116
_CONTROLS_SAFETY_PX = 24


def _scanpath_anim_specs(entries, marker_size_range):
    """Build per-scanpath animation specs from (fixations, color, label) entries.

    Empty/None fixations are skipped. Onsets are the recorded ``timestamp_ms``
    rebased to each reading's first fixation, so multiple scanpaths share one
    *real reading-time* clock. When timestamps aren't real times — missing, or
    the 0,1,2,… row index ``data.normalize_fixations`` synthesises when the
    source has no timestamp column — fixations are instead laid out back-to-back
    by their durations. Marker sizes are scaled over the COMBINED durations so
    equal durations render at equal sizes across scanpaths.
    """
    from .measures import rebased_fixation_onsets

    specs = []
    for fix_df, color, label in entries:
        if fix_df is None or fix_df.empty:
            continue
        ordered = fix_df.sort_values("timestamp_ms").reset_index(drop=True)
        dur = pd.to_numeric(ordered["duration_ms"], errors="coerce").fillna(0)
        # Recorded-timestamp-vs-synthetic-index heuristic (shared with the
        # similarity time-curve): trust recorded timestamps only when they look
        # like real times, else lay fixations back-to-back by their durations.
        onsets = rebased_fixation_onsets(ordered)
        specs.append(
            dict(
                ordered=ordered,
                dur=dur,
                onsets=onsets,
                end=float(onsets[-1] + dur.iloc[-1]),
                color=color,
                label=label,
            )
        )
    if specs:
        combined = _compute_marker_sizes(
            pd.concat([s["dur"] for s in specs], ignore_index=True), marker_size_range
        )
        cursor = 0
        for s in specs:
            n = len(s["dur"])
            s["sizes"] = np.asarray(combined[cursor : cursor + n], dtype=float)
            cursor += n
    return specs


def _anim_timeline(specs, playback_speed):
    """Merged frame timeline across all scanpaths.

    Returns ``(onset_times, frame_durations_ms, avg_frame_duration,
    reading_span_ms)``. A frame is emitted at every distinct fixation onset
    across all scanpaths; each frame lasts the gap to the next onset divided by
    ``playback_speed``, floored at ``_ANIM_MIN_FRAME_MS``. ``reading_span_ms`` is
    the longest reading's real span (all readings are rebased to t=0).
    """
    onset_times = sorted({float(t) for s in specs for t in s["onsets"]})
    reading_span_ms = max((s["end"] for s in specs), default=0.0)
    frame_durations_ms = []
    for k, t in enumerate(onset_times):
        nxt = onset_times[k + 1] if k + 1 < len(onset_times) else reading_span_ms
        gap = max(nxt - t, 0.0)
        frame_durations_ms.append(
            int(max(gap / max(playback_speed, 1e-6), _ANIM_MIN_FRAME_MS))
        )
    avg = (
        max(int(np.mean(frame_durations_ms)), _ANIM_MIN_FRAME_MS)
        if frame_durations_ms
        else _ANIM_MIN_FRAME_MS
    )
    return onset_times, frame_durations_ms, avg, reading_span_ms


def animation_playback_ms(fixations_list, playback_speed):
    """Reading span and *actual* animation runtime for the given scanpath(s).

    Returns ``(reading_span_ms, playback_ms)``. ``playback_ms`` is the real
    runtime the Play button produces: Play advances every frame at the average
    frame duration, so the total is ``n_frames * avg`` — quoting that in the side
    panel makes the stated playback time match what the user actually observes.
    Both 0 when there are no fixations.
    """
    specs = _scanpath_anim_specs(
        [(f, None, None) for f in fixations_list], DEFAULT_MARKER_SIZE_RANGE
    )
    if not specs:
        return 0.0, 0.0
    onset_times, _frame_durations, avg, reading_span_ms = _anim_timeline(
        specs, playback_speed
    )
    return reading_span_ms, float(len(onset_times) * avg)


def _animation_play_buttons(frame_duration):
    """Play / Pause / Restart buttons.

    Transitions are 0 so frames snap into place: no tweening means the replay
    runs at exactly ``n_frames * frame_duration`` and new markers/order-numbers
    appear on their fixation instead of gliding in from the corner.
    """
    return [
        dict(
            type="buttons",
            showactive=False,
            # A horizontal row just below the plot (y=0 = plot bottom; yanchor=
            # "top" hangs the row into the bottom margin); the time slider sits
            # below it. direction="right" keeps the three buttons on one line.
            direction="right",
            y=0.0,
            x=0.0,
            xanchor="left",
            yanchor="top",
            pad=dict(t=8, l=8),
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=frame_duration, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=0),
                        ),
                    ],
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                            transition=dict(duration=0),
                        ),
                    ],
                ),
                dict(
                    label="⟲ Restart",
                    method="animate",
                    args=[
                        ["0"],
                        dict(
                            frame=dict(duration=0, redraw=True),
                            mode="immediate",
                            transition=dict(duration=0),
                        ),
                    ],
                ),
            ],
        )
    ]


def _animation_time_slider(onset_times):
    """Slider whose handle position maps to elapsed reading time (not fixation
    index), so scrubbing moves through seconds into the reading. Steps jump
    instantly (duration 0).

    A long reading has hundreds of fixations. Drawing a tick + time label under
    *every* step renders an illegible wall of overlapping numbers — but the
    per-step ``label`` is also what the single "Elapsed" readout shows, so we
    can't simply blank most of them (the readout would go empty between the few
    kept labels). Instead every step keeps its real time label — so the readout
    updates smoothly on every frame and the handle stays frame-accurate — while
    the per-step tick ruler (``ticklen``/``minorticklen=0``) and the per-step
    labels (transparent ``font``) are hidden. The "Elapsed: X.Xs" readout is then
    the one, uncluttered time display.
    """
    return [
        dict(
            active=0,
            # Sit below the plot (y=0 = plot bottom; yanchor="top" hangs it into
            # the bottom margin), under the play/pause/restart buttons.
            yanchor="top",
            xanchor="left",
            ticklen=0,
            minorticklen=0,
            # Per-step labels feed the "Elapsed" readout but must not pile up
            # under the track, so draw them fully transparent.
            font=dict(color="rgba(0,0,0,0)"),
            currentvalue=dict(
                font=dict(size=14, color="#444"),
                prefix="Elapsed: ",
                visible=True,
                xanchor="right",
            ),
            transition=dict(duration=0),
            pad=dict(t=48, b=10),
            len=0.9,
            x=0.05,
            y=0.0,
            steps=[
                dict(
                    args=[
                        [str(k)],
                        dict(
                            frame=dict(duration=0, redraw=True),
                            mode="immediate",
                            transition=dict(duration=0),
                        ),
                    ],
                    label=f"{onset_times[k] / 1000:.1f}s",
                    method="animate",
                )
                for k in range(len(onset_times))
            ],
        )
    ]


def make_scanpath_animation(
    words: pd.DataFrame,
    fixations: pd.DataFrame,
    *,
    canvas_width: int,
    canvas_height: int,
    base_font_size: int,
    font_family: str,
    playback_speed: float = 1.0,
    show_words: bool = True,
    show_word_labels: bool = True,
    show_saccades: bool = True,
    show_order: bool = True,
    marker_size_range: Tuple[int, int] = DEFAULT_MARKER_SIZE_RANGE,
    order_font_size: int = 10,
    order_font_color: str = "#000000",
    color_by: Optional[str] = None,
    color_by_line: bool = False,
    fixation_colorscale: str = DEFAULT_FIXATION_COLORSCALE,
    fixation_color_range: Optional[Tuple[float, float]] = None,
    show_colorbars: bool = False,
    background_color: Optional[str] = None,
    fixations_b: Optional[pd.DataFrame] = None,
    words_b: Optional[pd.DataFrame] = None,
    label_a: str = "Scanpath A",
    label_b: str = "Scanpath B",
    line_spacing: float = DEFAULT_LINE_SPACING,
    scale_text_to_boxes: bool = True,
) -> go.Figure:
    """Frame-by-frame scanpath replay on a real reading-time clock.

    Pass ``fixations_b`` (and optionally ``words_b``) to overlay a SECOND
    scanpath animated on the same clock. Every scanpath is rebased to its first
    fixation's ``timestamp_ms``, so they share *real reading time* including the
    saccade/blink gaps between fixations; a frame is emitted at every fixation
    onset across all scanpaths, and the shorter reading finishes first and holds
    while the longer keeps going. The Play button advances frames at the average
    frame duration, so the whole replay takes ``reading_span / playback_speed``
    — exactly what :func:`animation_playback_ms` reports (and the side panel
    quotes), so the stated time matches the observed runtime.

    With two scanpaths the trails take the two comparison colours, order numbers
    are tinted per-scanpath, and a legend names them; word boxes/labels come from
    ``words`` (scanpath A), so the overlay is meaningful for two readings of the
    same text. With one scanpath the behaviour matches the classic single replay
    (order numbers honour ``order_font_color``, no legend).

    The single replay honours the same fixation-colouring options as
    :func:`make_scanpath_figure`: ``color_by`` (numeric → ``fixation_colorscale``
    pinned to the whole trial's range so colours stay stable as the trail grows,
    categorical → discrete palette + legend), ``color_by_line``, and an optional
    colorbar. The dual overlay ignores them — there the flat A/B colours are
    what tells the two readings apart.
    """
    fig = go.Figure()
    font_settings = dict(family=font_family or FONT_FAMILY, size=base_font_size)

    word_frames = [w for w in (words, words_b) if w is not None and not w.empty]
    x_range, y_range, *_ = _compute_axis_ranges(
        canvas_width,
        canvas_height,
        (fixations, "x", "y"),
        (fixations_b, "x", "y"),
        word_frames=word_frames,
    )

    # Fix the display size first so word labels are sized in the data->screen
    # scale (true-to-scale text); the same fitted_w/fitted_h drive the layout.
    fitted_w, fitted_h = _fit_display_size(
        canvas_width, canvas_height, x_range, y_range, spatial_axes=True
    )
    scale = _display_scale(x_range, y_range, fitted_w, fitted_h)
    label_font_px = _word_label_font_px(
        words,
        scale=scale,
        line_spacing=line_spacing,
        manual_font_px=base_font_size,
        scale_text_to_boxes=scale_text_to_boxes,
    )

    shapes = build_word_boxes(words) if show_words and not words.empty else []
    if show_word_labels and not words.empty:
        _add_word_label_trace(fig, words, label_font_px, font_settings["family"])

    specs = _scanpath_anim_specs(
        [
            (fixations, COMPARISON_PALETTE[0], label_a),
            (fixations_b, COMPARISON_PALETTE[1], label_b),
        ],
        marker_size_range,
    )
    dual = len(specs) > 1
    if not dual and specs:
        # A lone scanpath always wears the canonical single-replay colour,
        # whether it arrived as `fixations` or (degenerately) only as
        # `fixations_b`, so the trail never silently renders in the B colour.
        specs[0]["color"] = COMPARISON_PALETTE[0]

    # Metric colouring, mirroring the static figure's fixation trace. Single
    # replay only: the dual overlay keeps its flat A/B colours (they're what
    # tells the readings apart). Numeric metrics map through
    # `fixation_colorscale` with cmin/cmax pinned to the WHOLE trial (or the
    # caller's range) up front — otherwise the scale would renormalise to the
    # partial trail on every frame and colours would drift during playback.
    for s in specs:
        s["marker_colors"] = None
        s["marker_extra"] = {}
    category_legend: list = []
    color_label = color_by or ""
    if not dual and specs and (color_by or color_by_line):
        ordered0 = specs[0]["ordered"]
        if color_by_line and not words.empty:
            from .measures import assign_fixation_lines

            line_ids = assign_fixation_lines(ordered0, words)
            color_data = line_ids.map(
                lambda v: f"Line {int(v) + 1}" if pd.notna(v) else "(off-text)"
            )
            color_label = "line"
            is_numeric_color = False
        else:
            color_data = ordered0[color_by] if color_by in ordered0.columns else None
            is_numeric_color = color_data is not None and pd.api.types.is_numeric_dtype(
                color_data
            )
        if color_data is not None:
            marker_color, category_legend = _resolve_marker_colors(
                color_data, is_numeric_color
            )
            specs[0]["marker_colors"] = list(marker_color)
            if is_numeric_color:
                rng = fixation_color_range or (
                    float(color_data.min()),
                    float(color_data.max()),
                )
                specs[0]["marker_extra"] = dict(
                    colorscale=fixation_colorscale,
                    cmin=rng[0],
                    cmax=rng[1],
                    showscale=show_colorbars,
                    colorbar=dict(
                        title=color_label.replace("_", " ").title(),
                        x=1.12,
                        lenmode="fraction",
                        len=COLORBAR_LEN_FRACTION,
                        y=0.5,
                        yanchor="middle",
                    )
                    if show_colorbars
                    else None,
                )

    def _trail_marker(s, sizes, kk):
        """Marker dict for the first ``kk`` fixations of a trail (base + frames).

        Frames replace the whole marker object, so every frame restates the
        colorscale/cmin/cmax/colorbar — not just the sliced colour values."""
        colors = s["marker_colors"]
        return dict(
            size=sizes,
            color=colors[:kk] if colors is not None else s["color"],
            line=dict(color=FIX_MARKER_OUTLINE, width=0.5),
            **s["marker_extra"],
        )

    # Base traces, with stable indices the frames update by position. The trail
    # carries the markers (and the fixation number in `text`, for hover only);
    # the visible order numbers live in a SEPARATE, constant-length text trace
    # (added just below) so they never glide in from the corner as the trail
    # grows.
    for s in specs:
        ordered = s["ordered"]
        n_total = len(ordered)
        all_x = ordered["x"].tolist()
        all_y = ordered["y"].tolist()
        first_size = float(s["sizes"][0])
        s["text_color"] = s["color"] if dual else order_font_color
        sac_color = s["color"] if dual else SACCADE_COLOR
        curr_outline = s["color"] if dual else CURRENT_FIX_OUTLINE
        curr_outline_w = 2.5 if dual else 2

        s["idx_trail"] = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=[all_x[0]],
                y=[all_y[0]],
                mode="markers",
                marker=_trail_marker(s, [first_size], 1),
                text=["1"],
                showlegend=dual,
                name=s["label"],
                legendgroup=s["label"],
                hovertemplate=(
                    (s["label"] + "<br>" if dual else "")
                    + "Fixation #%{text}<br>Duration %{customdata} ms<extra></extra>"
                ),
                customdata=[ordered["duration_ms"].iloc[0]],
            )
        )
        # Order numbers: a text-only trace that holds EVERY fixation's final
        # position in every frame, with not-yet-reached numbers blanked to "".
        # Because the node set and their coordinates never change frame to frame,
        # Plotly only rewrites each label's string in place — so a new number
        # snaps on at its fixation instead of animating in from the (0,0) corner
        # (the artifact you get when a `markers+text` trail grows a point at a
        # time and each new <text> node flashes at the origin before placement).
        if show_order:
            s["idx_order"] = len(fig.data)
            fig.add_trace(
                go.Scatter(
                    x=all_x,
                    y=all_y,
                    mode="text",
                    text=["1"] + [""] * (n_total - 1),
                    textfont=dict(
                        color=s["text_color"],
                        size=order_font_size,
                        family=font_settings["family"],
                    ),
                    textposition="top center",
                    showlegend=False,
                    legendgroup=s["label"],
                    hoverinfo="skip",
                )
            )
        else:
            s["idx_order"] = None
        if show_saccades:
            s["idx_sac"] = len(fig.data)
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="lines",
                    line=dict(color=sac_color, width=2),
                    showlegend=False,
                    legendgroup=s["label"],
                    hoverinfo="skip",
                )
            )
        else:
            s["idx_sac"] = None
        s["idx_curr"] = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=[ordered["x"].iloc[0]],
                y=[ordered["y"].iloc[0]],
                mode="markers",
                marker=dict(
                    size=[first_size + 8],
                    color=CURRENT_FIX_COLOR,
                    line=dict(color=curr_outline, width=curr_outline_w),
                ),
                showlegend=False,
                legendgroup=s["label"],
                hoverinfo="skip",
            )
        )

    # Categorical colour legend (single replay), as in the static figure. These
    # dummy traces sit AFTER the per-scanpath traces so the frame indices
    # recorded above stay valid; frames never touch them.
    legend_limit = len(_QUALITATIVE_PALETTE)
    for category, color in category_legend[:legend_limit]:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=10,
                    color=color,
                    line=dict(color=FIX_MARKER_OUTLINE, width=0.5),
                ),
                name=f"{color_label}: {category}",
                showlegend=True,
                hoverinfo="skip",
            )
        )
    if len(category_legend) > legend_limit:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color="#cccccc"),
                name=f"… +{len(category_legend) - legend_limit} more",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    onset_times, _frame_durations, avg_frame_duration, _span = _anim_timeline(
        specs, playback_speed
    )

    frames = []
    for k, t in enumerate(onset_times):
        traces_in_frame = []
        traces_idx_in_frame = []
        for s in specs:
            ordered = s["ordered"]
            n_total = len(ordered)
            # Fixations whose recorded onset has been reached by time t.
            kk = max(int(np.searchsorted(s["onsets"], t, side="right")), 1)
            xs = ordered["x"].iloc[:kk].tolist()
            ys = ordered["y"].iloc[:kk].tolist()
            szs = s["sizes"][:kk].tolist()
            sac_color = s["color"] if dual else SACCADE_COLOR
            curr_outline = s["color"] if dual else CURRENT_FIX_OUTLINE
            curr_outline_w = 2.5 if dual else 2

            traces_in_frame.append(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker=_trail_marker(s, szs, kk),
                    text=[str(j + 1) for j in range(kk)],
                    customdata=ordered["duration_ms"].iloc[:kk].tolist(),
                )
            )
            traces_idx_in_frame.append(s["idx_trail"])

            if show_order:
                # Same full-length positions as the base order trace; only the
                # blanked tail shrinks as the trail advances, so numbers appear
                # in place rather than sliding in from the corner.
                traces_in_frame.append(
                    go.Scatter(
                        x=ordered["x"].tolist(),
                        y=ordered["y"].tolist(),
                        mode="text",
                        text=[str(j + 1) if j < kk else "" for j in range(n_total)],
                        textfont=dict(
                            color=s["text_color"],
                            size=order_font_size,
                            family=font_settings["family"],
                        ),
                        textposition="top center",
                    )
                )
                traces_idx_in_frame.append(s["idx_order"])

            if show_saccades:
                sac_x: list = []
                sac_y: list = []
                for j in range(kk - 1):
                    sac_x.extend([xs[j], xs[j + 1], None])
                    sac_y.extend([ys[j], ys[j + 1], None])
                traces_in_frame.append(
                    go.Scatter(
                        x=sac_x,
                        y=sac_y,
                        mode="lines",
                        line=dict(color=sac_color, width=2),
                    )
                )
                traces_idx_in_frame.append(s["idx_sac"])

            ci = kk - 1
            traces_in_frame.append(
                go.Scatter(
                    x=[xs[ci]],
                    y=[ys[ci]],
                    mode="markers",
                    marker=dict(
                        size=[szs[ci] + 8],
                        color=CURRENT_FIX_COLOR,
                        line=dict(color=curr_outline, width=curr_outline_w),
                    ),
                )
            )
            traces_idx_in_frame.append(s["idx_curr"])

        frames.append(
            go.Frame(data=traces_in_frame, name=str(k), traces=traces_idx_in_frame)
        )
    fig.frames = frames

    shapes.append(
        dict(
            type="rect",
            x0=x_range[0],
            y0=y_range[1],
            x1=x_range[1],
            y1=y_range[0],
            line=dict(color="#000000", width=1),
            fillcolor="rgba(0,0,0,0)",
        )
    )

    sliders = _animation_time_slider(onset_times) if onset_times else []
    updatemenus = _animation_play_buttons(avg_frame_duration) if onset_times else []

    # fitted_w / fitted_h were computed up front (so the label scale matched).
    # ALL transport controls (play/pause/restart buttons + the time slider with
    # its "Elapsed" readout) sit BELOW the plot in the bottom margin. Critically,
    # the figure is made tall enough that the plot region stays >= fitted_h after
    # Plotly's automargin reserves space for those controls — otherwise the
    # equal-aspect (`scaleanchor`) plot would shrink to fit the leftover height,
    # making the word boxes smaller than the true-to-scale label font computed
    # for fitted_h (text-too-large bug). _CONTROLS_MARGIN_PX is that reserve.
    layout = dict(
        height=fitted_h + _CONTROLS_MARGIN_PX + _CONTROLS_SAFETY_PX,
        width=fitted_w,
        autosize=False,
        margin=dict(l=0, r=0, t=0, b=_CONTROLS_MARGIN_PX),
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title=None,
            range=x_range,
            constrain="domain",
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title=None,
            range=y_range,
            constrain="domain",
            scaleanchor="x",
            scaleratio=1,
        ),
        template="plotly_white",
        plot_bgcolor=background_color,
        paper_bgcolor=background_color,
        font=font_settings,
        shapes=shapes,
        sliders=sliders,
        updatemenus=updatemenus,
    )
    if dual or category_legend:
        layout["legend"] = dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="#cccccc",
            borderwidth=1,
        )
    fig.update_layout(**layout)
    return fig


def _resolve_trial_display_name(
    participant: str,
    trial_id: str,
    trial_words: pd.DataFrame,
    trial_labels: Optional[Tuple[str, str]],
    idx: int,
) -> str:
    if trial_labels is not None and len(trial_labels) > idx:
        return trial_labels[idx]
    text_id = None
    if "text_id" in trial_words.columns and not trial_words.empty:
        text_id = trial_words["text_id"].iloc[0]
    text_str = str(text_id) if text_id is not None else ""
    trial_str = str(trial_id)
    contains_text = text_str and text_str.lower() in trial_str.lower()
    if text_str:
        return (
            f"{text_str} · {participant}"
            if contains_text
            else f"{text_str} · {participant} (trial {trial_str})"
        )
    return f"{trial_str} · {participant}"


def _add_comparison_fixation_trace(
    fig: go.Figure,
    trial_fix: pd.DataFrame,
    display_name: str,
    color: str,
    font_settings: dict,
    marker_size_range: Tuple[int, int],
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> None:
    if trial_fix.empty:
        return
    sizes = _compute_marker_sizes(trial_fix["duration_ms"], marker_size_range)
    trace = go.Scatter(
        x=trial_fix["x"],
        y=trial_fix["y"],
        mode="markers+lines",
        marker=dict(
            size=sizes,
            color=color,
            line=dict(color=FIX_MARKER_OUTLINE, width=0.5),
        ),
        line=dict(color=color, width=2, dash="solid"),
        name=display_name,
        text=trial_fix["order_in_trial"],
        textposition="top center",
        textfont=font_settings,
        hovertemplate=(
            f"{display_name} "
            "Order %{text}<br>Time %{customdata[0]} ms<br>"
            "Duration %{customdata[1]} ms<extra></extra>"
        ),
        customdata=trial_fix[["timestamp_ms", "duration_ms"]],
    )
    if row is not None and col is not None:
        fig.add_trace(trace, row=row, col=col)
    else:
        fig.add_trace(trace)


def _make_split_comparison_figure(
    words: pd.DataFrame,
    fixations: pd.DataFrame,
    trial_a: Tuple[str, str],
    trial_b: Tuple[str, str],
    *,
    canvas_width: int,
    canvas_height: int,
    font_family: str,
    base_font_size: int,
    show_words: bool,
    show_word_labels: bool,
    trial_labels: Optional[Tuple[str, str]],
    orientation: str,
    marker_size_range: Tuple[int, int] = DEFAULT_MARKER_SIZE_RANGE,
    background_color: Optional[str] = None,
    line_spacing: float = DEFAULT_LINE_SPACING,
    scale_text_to_boxes: bool = True,
) -> go.Figure:
    """Two-panel comparison, either horizontal (side-by-side) or vertical (stacked)."""
    from plotly.subplots import make_subplots

    font_settings = dict(family=font_family or FONT_FAMILY, size=base_font_size)
    palette = COMPARISON_PALETTE
    is_stacked = orientation == "stacked"
    # Per-panel pixel size (approx; subplot spacing/titles shave a little) used to
    # size word labels true-to-scale within each panel.
    per_panel_w = canvas_width if is_stacked else canvas_width // 2

    trial_specs = []
    for idx, trial in enumerate([trial_a, trial_b]):
        participant, trial_id = trial
        trial_words = words[
            (words["participant_id"] == participant) & (words["trial_id"] == trial_id)
        ]
        trial_fix = fixations[
            (fixations["participant_id"] == participant)
            & (fixations["trial_id"] == trial_id)
        ].sort_values("timestamp_ms")
        display_name = _resolve_trial_display_name(
            participant, trial_id, trial_words, trial_labels, idx
        )
        trial_specs.append(
            dict(
                trial_words=trial_words,
                trial_fix=trial_fix,
                display_name=display_name,
                color=palette[idx],
            )
        )

    if is_stacked:
        fig = make_subplots(
            rows=2,
            cols=1,
            vertical_spacing=0.08,
            subplot_titles=[
                trial_specs[0]["display_name"],
                trial_specs[1]["display_name"],
            ],
        )
    else:
        fig = make_subplots(
            rows=1,
            cols=2,
            horizontal_spacing=0.04,
            subplot_titles=[
                trial_specs[0]["display_name"],
                trial_specs[1]["display_name"],
            ],
        )

    all_shapes: list = []
    for idx, spec in enumerate(trial_specs):
        if is_stacked:
            row, col = idx + 1, 1
            axis_suffix = "" if idx == 0 else str(idx + 1)
        else:
            row, col = 1, idx + 1
            axis_suffix = "" if idx == 0 else str(idx + 1)
        xref = f"x{axis_suffix}"
        yref = f"y{axis_suffix}"
        trial_words = spec["trial_words"]
        trial_fix = spec["trial_fix"]

        x_range, y_range, *_ = _compute_axis_ranges(
            canvas_width,
            canvas_height,
            (trial_fix, "x", "y"),
            word_frames=[trial_words] if not trial_words.empty else [],
        )

        if show_words and not trial_words.empty:
            for box in build_word_boxes(trial_words, color=spec["color"]):
                box = dict(box)
                box["xref"] = xref
                box["yref"] = yref
                all_shapes.append(box)

        all_shapes.append(
            dict(
                type="rect",
                xref=xref,
                yref=yref,
                x0=x_range[0],
                y0=y_range[1],
                x1=x_range[1],
                y1=y_range[0],
                line=dict(color="#000000", width=1),
                fillcolor="rgba(0,0,0,0)",
            )
        )

        _add_comparison_fixation_trace(
            fig,
            trial_fix,
            spec["display_name"],
            spec["color"],
            font_settings,
            marker_size_range,
            row=row,
            col=col,
        )

        if show_word_labels:
            pf_w, pf_h = _fit_display_size(
                per_panel_w, canvas_height, x_range, y_range, spatial_axes=True
            )
            panel_scale = _display_scale(x_range, y_range, pf_w, pf_h)
            _add_word_label_trace(
                fig,
                trial_words,
                _word_label_font_px(
                    trial_words,
                    scale=panel_scale,
                    line_spacing=line_spacing,
                    manual_font_px=base_font_size,
                    scale_text_to_boxes=scale_text_to_boxes,
                ),
                font_settings["family"],
                row=row,
                col=col,
            )

        xaxis_key = "xaxis" if idx == 0 else f"xaxis{idx + 1}"
        yaxis_key = "yaxis" if idx == 0 else f"yaxis{idx + 1}"
        fig.update_layout(
            **{
                xaxis_key: dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    title=None,
                    range=x_range,
                    constrain="domain",
                ),
                yaxis_key: dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    title=None,
                    range=y_range,
                    constrain="domain",
                    scaleanchor=xref,
                    scaleratio=1,
                ),
            }
        )

    # Fit the figure to the data aspect just like the single-trial plot.
    # `x_range` / `y_range` from the inner loop are per-trial; the two trials
    # being compared usually share the paragraph (same canvas), so re-using
    # the last loop iteration's range is fine. `per_panel_w` was set above.
    panel_w, panel_h = _fit_display_size(
        per_panel_w, canvas_height, x_range, y_range, spatial_axes=True
    )
    if is_stacked:
        total_width = panel_w
        total_height = panel_h * 2 + 40
    else:  # side-by-side
        total_width = panel_w * 2
        total_height = panel_h
    fig.update_layout(
        height=total_height,
        width=total_width,
        autosize=False,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        template="plotly_white",
        plot_bgcolor=background_color,
        paper_bgcolor=background_color,
        title="Stacked comparison" if is_stacked else "Side-by-side comparison",
        font=font_settings,
        shapes=all_shapes,
    )
    return fig


def make_comparison_figure(
    words: pd.DataFrame,
    fixations: pd.DataFrame,
    trial_a: Tuple[str, str],
    trial_b: Tuple[str, str],
    *,
    canvas_width: int,
    canvas_height: int,
    font_family: str,
    base_font_size: int,
    show_words: bool = True,
    show_word_labels: bool = False,
    trial_labels: Optional[Tuple[str, str]] = None,
    layout: str = "overlay",
    marker_size_range: Tuple[int, int] = DEFAULT_MARKER_SIZE_RANGE,
    background_color: Optional[str] = None,
    line_spacing: float = DEFAULT_LINE_SPACING,
    scale_text_to_boxes: bool = True,
) -> go.Figure:
    if layout in {"side_by_side", "stacked"}:
        return _make_split_comparison_figure(
            words,
            fixations,
            trial_a,
            trial_b,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            font_family=font_family,
            base_font_size=base_font_size,
            show_words=show_words,
            show_word_labels=show_word_labels,
            trial_labels=trial_labels,
            orientation=layout,
            marker_size_range=marker_size_range,
            background_color=background_color,
            line_spacing=line_spacing,
            scale_text_to_boxes=scale_text_to_boxes,
        )

    fig = go.Figure()
    font_settings = dict(family=font_family or FONT_FAMILY, size=base_font_size)
    palette = COMPARISON_PALETTE

    trial_specs = []
    for idx, trial in enumerate([trial_a, trial_b]):
        participant, trial_id = trial
        trial_words = words[
            (words["participant_id"] == participant) & (words["trial_id"] == trial_id)
        ]
        trial_fix = fixations[
            (fixations["participant_id"] == participant)
            & (fixations["trial_id"] == trial_id)
        ].sort_values("timestamp_ms")
        display_name = _resolve_trial_display_name(
            participant, trial_id, trial_words, trial_labels, idx
        )
        trial_specs.append(
            dict(
                trial_words=trial_words,
                trial_fix=trial_fix,
                display_name=display_name,
                color=palette[idx],
            )
        )

    x_range, y_range, *_ = _compute_axis_ranges(
        canvas_width,
        canvas_height,
        *((spec["trial_fix"], "x", "y") for spec in trial_specs),
        word_frames=[
            spec["trial_words"] for spec in trial_specs if not spec["trial_words"].empty
        ],
    )

    # Both trials are overlaid on one shared canvas, so one display scale sizes
    # every word label true-to-scale (geometry is identical across the readings).
    fitted_w, fitted_h = _fit_display_size(
        canvas_width, canvas_height, x_range, y_range, spatial_axes=True
    )
    overlay_scale = _display_scale(x_range, y_range, fitted_w, fitted_h)

    for spec in trial_specs:
        _add_comparison_fixation_trace(
            fig,
            spec["trial_fix"],
            spec["display_name"],
            spec["color"],
            font_settings,
            marker_size_range,
        )
        if show_words:
            existing = list(fig.layout.shapes) if fig.layout.shapes else []
            fig.update_layout(
                shapes=existing
                + build_word_boxes(spec["trial_words"], color=spec["color"])
            )
        if show_word_labels:
            _add_word_label_trace(
                fig,
                spec["trial_words"],
                _word_label_font_px(
                    spec["trial_words"],
                    scale=overlay_scale,
                    line_spacing=line_spacing,
                    manual_font_px=base_font_size,
                    scale_text_to_boxes=scale_text_to_boxes,
                ),
                font_settings["family"],
            )

    shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    shapes.append(
        dict(
            type="rect",
            x0=x_range[0],
            y0=y_range[1],
            x1=x_range[1],
            y1=y_range[0],
            line=dict(color="#000000", width=1),
            fillcolor="rgba(0,0,0,0)",
        )
    )

    # fitted_w / fitted_h were computed up front (so the label scale matched).
    fig.update_layout(
        height=fitted_h,
        width=fitted_w,
        autosize=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title=None,
            range=x_range,
            constrain="domain",
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title=None,
            range=y_range,
            constrain="domain",
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        plot_bgcolor=background_color,
        paper_bgcolor=background_color,
        title="Overlay comparison",
        font=font_settings,
        shapes=shapes,
    )
    return fig


# =============================================================================
# Reading-research figures: per-word bar, fixation-duration histogram
# =============================================================================


def make_word_measure_bar_figure(
    words: pd.DataFrame,
    *,
    measure: str,
    canvas_width: int,
    base_font_size: int,
    font_family: str,
    height: int = 360,
) -> go.Figure:
    """Vertical bar plot of a per-word measure, with word text on the x-axis."""
    fig = go.Figure()
    font_settings = dict(family=font_family or FONT_FAMILY, size=base_font_size)
    if words.empty or measure not in words.columns:
        fig.update_layout(
            template="plotly_white",
            font=font_settings,
            title=f"No data for '{measure}'",
            height=height,
        )
        return fig
    ordered = words.sort_values(["line_idx", "word_id"]).reset_index(drop=True)
    labels = [
        f"{int(wid)}: {txt}" if pd.notna(wid) else str(txt)
        for wid, txt in zip(ordered["word_id"], ordered.get("text", ordered["word_id"]))
    ]
    values = pd.to_numeric(ordered[measure], errors="coerce")
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=values,
                colorscale=DEFAULT_HEATMAP_COLORSCALE,
                showscale=True,
                colorbar=dict(title=measure.replace("_", " ").title()),
            ),
            hovertemplate="%{x}<br>" + measure + ": %{y}<extra></extra>",
        )
    )
    mean_value = float(values.dropna().mean()) if values.dropna().size else None
    if mean_value is not None:
        fig.add_hline(
            y=mean_value,
            line=dict(color=COMPARISON_PALETTE[1], width=2, dash="dot"),
            annotation_text=f"mean {mean_value:.2f}",
            annotation_position="top right",
        )
    fig.update_layout(
        height=height,
        width=canvas_width,
        autosize=False,
        margin=dict(l=40, r=10, t=40, b=80),
        template="plotly_white",
        font=font_settings,
        xaxis=dict(title="Word", tickangle=-45, automargin=True),
        yaxis=dict(title=measure.replace("_", " ").title()),
        title=f"Per-word {measure.replace('_', ' ')}",
    )
    return fig


def make_fixation_duration_histogram(
    fixations: pd.DataFrame,
    *,
    canvas_width: int,
    base_font_size: int,
    font_family: str,
    bins: int = 30,
    overlay_words: Optional[pd.DataFrame] = None,
    height: int = 320,
) -> go.Figure:
    """Histogram of fixation durations, optionally with overlaid summary stats."""
    fig = go.Figure()
    font_settings = dict(family=font_family or FONT_FAMILY, size=base_font_size)
    if fixations.empty:
        fig.update_layout(
            template="plotly_white",
            font=font_settings,
            title="Fixation duration distribution (no data)",
            height=height,
        )
        return fig
    durations = pd.to_numeric(fixations["duration_ms"], errors="coerce").dropna()

    # Pre-bin server-side and draw bars instead of go.Histogram, which would
    # serialize *every* raw value to the browser — prohibitive for millions of
    # fixations. All series share one set of bin edges so the overlays align.
    series_list = [("All fixations", durations.to_numpy(), COMPARISON_PALETTE[0], 1.0)]
    if overlay_words is not None and not overlay_words.empty:
        for name, col in (
            ("FFD", "first_fixation_ms"),
            ("FPRT", "first_pass_gaze_duration_ms"),
            ("TFD", "total_fixation_duration_ms"),
        ):
            if col in overlay_words.columns:
                vals = pd.to_numeric(overlay_words[col], errors="coerce").dropna()
                if not vals.empty:
                    series_list.append((name, vals.to_numpy(), None, 0.4))

    all_vals = np.concatenate([arr for _, arr, _, _ in series_list])
    lo = float(all_vals.min()) if all_vals.size else 0.0
    hi = float(all_vals.max()) if all_vals.size else 1.0
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bar_width = float(edges[1] - edges[0])

    for name, arr, color, opacity in series_list:
        counts, _ = np.histogram(arr, bins=edges)
        marker = (
            dict(color=color, line=dict(color="white", width=0.5))
            if color is not None
            else None
        )
        fig.add_trace(
            go.Bar(
                x=centers,
                y=counts,
                width=bar_width,
                name=name,
                opacity=opacity,
                marker=marker,
            )
        )

    mean_ms = float(durations.mean()) if len(durations) else 0.0
    median_ms = float(durations.median()) if len(durations) else 0.0
    fig.add_vline(
        x=mean_ms,
        line=dict(color=COMPARISON_PALETTE[1], width=2, dash="dash"),
        annotation_text=f"mean {mean_ms:.0f} ms",
        annotation_position="top right",
    )
    fig.add_vline(
        x=median_ms,
        line=dict(color=SACCADE_COLOR, width=2, dash="dot"),
        annotation_text=f"median {median_ms:.0f} ms",
        annotation_position="top left",
    )
    fig.update_layout(
        height=height,
        width=canvas_width,
        autosize=False,
        margin=dict(l=40, r=10, t=40, b=40),
        template="plotly_white",
        font=font_settings,
        xaxis=dict(title="Duration (ms)"),
        yaxis=dict(title="Count"),
        barmode="overlay",
        title="Fixation duration distribution",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def make_metric_convergence_figure(
    series: dict,
    *,
    x_title: str,
    y_title: str,
    title: str,
    canvas_width: int,
    base_font_size: int,
    font_family: str,
    height: int = 340,
    y_range: Tuple[float, float] = (0.0, 1.02),
    highlight_x_range: Optional[Tuple[float, float]] = None,
) -> go.Figure:
    """Line chart of a metric (one line per model) over a cumulative x axis.

    ``series`` maps a model name to ``(xs, ys)``. Used by the Multiple Comparison
    tab to show how each model's NLD vs. the real scanpath evolves as more of the
    reading is included — either by cumulative fixation index or by elapsed time.
    Optionally shades ``highlight_x_range`` (e.g. the selected fixation window).
    """
    fig = go.Figure()
    font_settings = dict(family=font_family or FONT_FAMILY, size=base_font_size)
    has_data = False
    for i, (name, xy) in enumerate(series.items()):
        xs, ys = xy
        if not len(xs):
            continue
        has_data = True
        color = _QUALITATIVE_PALETTE[i % len(_QUALITATIVE_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=list(xs),
                y=list(ys),
                mode="lines+markers",
                name=str(name),
                line=dict(color=color, width=2),
                marker=dict(size=4, color=color),
                hovertemplate=(
                    f"{name}<br>{x_title}: %{{x}}<br>{y_title}: %{{y:.3f}}"
                    "<extra></extra>"
                ),
            )
        )
    if has_data and highlight_x_range is not None:
        lo, hi = highlight_x_range
        if hi > lo:
            fig.add_vrect(x0=lo, x1=hi, fillcolor="#6c757d", opacity=0.10, line_width=0)
    fig.update_layout(
        height=height,
        width=canvas_width,
        autosize=False,
        margin=dict(l=55, r=10, t=40, b=45),
        template="plotly_white",
        font=font_settings,
        xaxis=dict(title=x_title),
        yaxis=dict(title=y_title, range=list(y_range)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=title,
    )
    if not has_data:
        fig.add_annotation(
            text="No data", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper"
        )
    return fig
