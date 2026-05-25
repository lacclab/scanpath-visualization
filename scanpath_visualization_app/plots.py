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
    DEFAULT_MARKER_SIZE_RANGE,
    FIX_MARKER_OUTLINE,
    FONT_FAMILY,
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


def _add_word_label_trace(
    fig: go.Figure,
    words: pd.DataFrame,
    base_font_size: int,
    font_family: str,
    row: Optional[int] = None,
    col: Optional[int] = None,
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
    trace = go.Scatter(
        x=words["x"] + words["width"] / 2,
        y=words["y"] + words["height"] / 2,
        text=words["text"],
        mode="text",
        showlegend=False,
        textfont=dict(color=WORD_LABEL_COLOR, size=base_font_size, family=font_family),
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

    if spatial_axes and not words.empty:
        if show_words:
            fig.update_layout(shapes=build_word_boxes(words))
        if show_word_labels:
            _add_word_label_trace(fig, words, base_font_size, font_settings["family"])

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
        if not words.empty:
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

    if show_fixations and not fixations.empty:
        ordered = fixations.sort_values("timestamp_ms")
        color_data = ordered[color_by] if color_by in ordered.columns else None
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
                        title=color_by.replace("_", " ").title(),
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
                    name=f"{color_by}: {category}",
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

    fig.update_layout(
        height=canvas_height,
        width=canvas_width,
        autosize=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=xaxis_cfg,
        yaxis=yaxis_cfg,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
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
    from plotly.colors import sample_colorscale

    word_values = []
    for word_row in words.itertuples():
        wx0, wy0 = word_row.x, word_row.y
        wx1, wy1 = wx0 + word_row.width, wy0 + word_row.height
        in_word = (
            (fixations[x_field] >= wx0)
            & (fixations[x_field] <= wx1)
            & (fixations[y_field] >= wy0)
            & (fixations[y_field] <= wy1)
        )
        val = (
            float(weights[in_word].sum())
            if weights is not None
            else float(in_word.sum())
        )
        word_values.append(val)

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
                        title="Fixation count" if weights is None else "Duration (ms)",
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
) -> go.Figure:
    """Frame-by-frame animation with per-fixation frame durations.

    Each frame stays on screen for that fixation's actual duration divided by
    playback_speed, clipped at 50 ms so the slowest frames remain perceptible.
    """
    fig = go.Figure()
    font_settings = dict(family=font_family or FONT_FAMILY, size=base_font_size)

    x_range, y_range, *_ = _compute_axis_ranges(
        canvas_width,
        canvas_height,
        (fixations, "x", "y"),
        word_frames=[words] if not words.empty else [],
    )

    shapes = build_word_boxes(words) if show_words and not words.empty else []

    if show_word_labels:
        _add_word_label_trace(fig, words, base_font_size, font_settings["family"])
    word_label_trace_idx = 0 if show_word_labels and not words.empty else -1

    if fixations.empty:
        return fig

    ordered = fixations.sort_values("timestamp_ms").reset_index(drop=True)
    n_fix = len(ordered)
    durations = pd.to_numeric(ordered["duration_ms"], errors="coerce").fillna(0)
    sizes = _compute_marker_sizes(durations, marker_size_range)
    frame_durations_ms = (
        (durations / max(playback_speed, 1e-6)).clip(lower=50).astype(int).tolist()
    )
    avg_frame_duration = max(int(np.mean(frame_durations_ms)), 50)
    marker_mode = "markers+text" if show_order else "markers"

    fig.add_trace(
        go.Scatter(
            x=[ordered.iloc[0]["x"]],
            y=[ordered.iloc[0]["y"]],
            mode=marker_mode,
            marker=dict(
                size=[sizes[0]],
                color=COMPARISON_PALETTE[0],
                line=dict(color=FIX_MARKER_OUTLINE, width=0.5),
            ),
            text=["1"] if show_order else None,
            textfont=dict(
                color=order_font_color,
                size=order_font_size,
                family=font_settings["family"],
            ),
            textposition="top center",
            showlegend=False,
            name="fixations",
            hovertemplate=(
                "Fixation #%{text}<br>Duration %{customdata} ms<extra></extra>"
            ),
            customdata=[ordered.iloc[0]["duration_ms"]],
        )
    )
    fix_trace_idx = 1 if word_label_trace_idx == 0 else 0

    if show_saccades:
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color=SACCADE_COLOR, width=2),
                showlegend=False,
                name="saccades",
                hoverinfo="skip",
            )
        )
        sac_trace_idx = fix_trace_idx + 1
    else:
        sac_trace_idx = None

    fig.add_trace(
        go.Scatter(
            x=[ordered.iloc[0]["x"]],
            y=[ordered.iloc[0]["y"]],
            mode="markers",
            marker=dict(
                size=sizes[0] + 8,
                color=CURRENT_FIX_COLOR,
                line=dict(color=CURRENT_FIX_OUTLINE, width=2),
            ),
            showlegend=False,
            name="current",
            hoverinfo="skip",
        )
    )
    curr_trace_idx = (sac_trace_idx if sac_trace_idx is not None else fix_trace_idx) + 1

    frames = []
    for i in range(n_fix):
        fix_x = ordered["x"].iloc[: i + 1].tolist()
        fix_y = ordered["y"].iloc[: i + 1].tolist()
        fix_sizes = sizes[: i + 1].tolist()
        fix_texts = [str(j + 1) for j in range(i + 1)] if show_order else None
        fix_durations = ordered["duration_ms"].iloc[: i + 1].tolist()

        traces_in_frame = []
        traces_idx_in_frame = []

        fix_scatter = go.Scatter(
            x=fix_x,
            y=fix_y,
            mode=marker_mode,
            marker=dict(
                size=fix_sizes,
                color=COMPARISON_PALETTE[0],
                line=dict(color=FIX_MARKER_OUTLINE, width=0.5),
            ),
            text=fix_texts,
            textfont=dict(
                color=order_font_color,
                size=order_font_size,
                family=font_settings["family"],
            ),
            textposition="top center",
            customdata=fix_durations,
        )
        traces_in_frame.append(fix_scatter)
        traces_idx_in_frame.append(fix_trace_idx)

        if show_saccades:
            sac_x: list = []
            sac_y: list = []
            for j in range(i):
                sac_x.extend([ordered["x"].iloc[j], ordered["x"].iloc[j + 1], None])
                sac_y.extend([ordered["y"].iloc[j], ordered["y"].iloc[j + 1], None])
            traces_in_frame.append(
                go.Scatter(
                    x=sac_x,
                    y=sac_y,
                    mode="lines",
                    line=dict(color=SACCADE_COLOR, width=2),
                )
            )
            traces_idx_in_frame.append(sac_trace_idx)

        traces_in_frame.append(
            go.Scatter(
                x=[ordered["x"].iloc[i]],
                y=[ordered["y"].iloc[i]],
                mode="markers",
                marker=dict(
                    size=[sizes[i] + 8],
                    color=CURRENT_FIX_COLOR,
                    line=dict(color=CURRENT_FIX_OUTLINE, width=2),
                ),
            )
        )
        traces_idx_in_frame.append(curr_trace_idx)

        frames.append(
            go.Frame(
                data=traces_in_frame,
                name=str(i),
                traces=traces_idx_in_frame,
            )
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

    sliders = [
        dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue=dict(
                font=dict(size=14),
                prefix="Fixation: ",
                visible=True,
                xanchor="right",
            ),
            transition=dict(duration=avg_frame_duration // 2, easing="cubic-in-out"),
            pad=dict(b=10, t=50),
            len=0.9,
            x=0.1,
            y=0,
            steps=[
                dict(
                    args=[
                        [str(i)],
                        dict(
                            frame=dict(duration=frame_durations_ms[i], redraw=True),
                            mode="immediate",
                            transition=dict(
                                duration=min(frame_durations_ms[i] // 2, 100)
                            ),
                        ),
                    ],
                    label=str(i + 1),
                    method="animate",
                )
                for i in range(n_fix)
            ],
        )
    ]

    updatemenus = [
        dict(
            type="buttons",
            showactive=False,
            y=0,
            x=0.05,
            xanchor="right",
            yanchor="top",
            pad=dict(t=50, r=10),
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=avg_frame_duration, redraw=True),
                            fromcurrent=True,
                            transition=dict(
                                duration=min(avg_frame_duration // 2, 100),
                                easing="cubic-in-out",
                            ),
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

    fig.update_layout(
        height=canvas_height + 80,
        width=canvas_width,
        autosize=False,
        margin=dict(l=0, r=0, t=0, b=80),
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
        font=font_settings,
        shapes=shapes,
        sliders=sliders,
        updatemenus=updatemenus,
    )
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
    if "paragraph_id" in trial_words.columns and not trial_words.empty:
        text_id = trial_words["paragraph_id"].iloc[0]
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
) -> go.Figure:
    """Two-panel comparison, either horizontal (side-by-side) or vertical (stacked)."""
    from plotly.subplots import make_subplots

    font_settings = dict(family=font_family or FONT_FAMILY, size=base_font_size)
    palette = COMPARISON_PALETTE
    is_stacked = orientation == "stacked"

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
            _add_word_label_trace(
                fig,
                trial_words,
                base_font_size,
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

    total_height = canvas_height * 2 + 40 if is_stacked else canvas_height
    fig.update_layout(
        height=total_height,
        width=canvas_width,
        autosize=False,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        template="plotly_white",
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
                fig, spec["trial_words"], base_font_size, font_settings["family"]
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

    fig.update_layout(
        height=canvas_height,
        width=canvas_width,
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
    fig.add_trace(
        go.Histogram(
            x=durations,
            nbinsx=bins,
            marker=dict(
                color=COMPARISON_PALETTE[0], line=dict(color="white", width=0.5)
            ),
            name="All fixations",
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
    overlay = []
    if overlay_words is not None and not overlay_words.empty:
        if "first_fixation_ms" in overlay_words.columns:
            overlay.append(("FFD", overlay_words["first_fixation_ms"]))
        if "first_pass_gaze_duration_ms" in overlay_words.columns:
            overlay.append(("FPRT", overlay_words["first_pass_gaze_duration_ms"]))
        if "total_fixation_duration_ms" in overlay_words.columns:
            overlay.append(("TFD", overlay_words["total_fixation_duration_ms"]))
    for name, series in overlay:
        vals = pd.to_numeric(series, errors="coerce").dropna()
        if vals.empty:
            continue
        fig.add_trace(
            go.Histogram(
                x=vals,
                nbinsx=bins,
                opacity=0.4,
                name=name,
            )
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
