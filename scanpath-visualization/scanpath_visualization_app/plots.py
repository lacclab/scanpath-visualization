from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .constants import FONT_FAMILY

COLORBAR_LEN_FRACTION = 0.33


def build_word_boxes(words: pd.DataFrame, color: str = "#6c757d") -> list:
    shapes = []
    for _, row in words.iterrows():
        x0, y0 = row["x"], row["y"]
        x1, y1 = row["x"] + row["width"], row["y"] + row["height"]
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
) -> go.Figure:
    fig = go.Figure()
    spatial_axes = x_field == "x" and y_field == "y"
    font_settings = dict(family=font_family or FONT_FAMILY, size=base_font_size)

    if spatial_axes and not words.empty:
        if show_words:
            fig.update_layout(shapes=build_word_boxes(words))
        if show_word_labels and "text" in words.columns:
            fig.add_trace(
                go.Scatter(
                    x=words["x"] + words["width"] / 2,
                    y=words["y"] + words["height"] / 2,
                    text=words["text"],
                    mode="text",
                    showlegend=False,
                    textfont=dict(color="#343a40", size=base_font_size, family=font_settings["family"]),
                    hovertemplate=(
                        "Word %{text}<br>Word ID %{customdata[0]}<br>Line %{customdata[1]}"
                        "<extra></extra>"
                    ),
                    customdata=words[["word_id", "line_idx"]],
                )
            )

    if spatial_axes and show_heatmap and not fixations.empty:
        weights = None
        if heatmap_metric == "duration_ms":
            weights = fixations["duration_ms"]
        histfunc = "sum" if weights is not None else "count"
        x_min = float((words["x"]).min()) if not words.empty else float(fixations[x_field].min())
        x_max = float((words["x"] + words["width"]).max()) if not words.empty else float(fixations[x_field].max())
        y_min = float((words["y"]).min()) if not words.empty else float(fixations[y_field].min())
        y_max = float((words["y"] + words["height"]).max()) if not words.empty else float(fixations[y_field].max())
        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1.0)
        if not words.empty:
            x_edges = np.unique(np.sort(np.concatenate([words["x"].values, (words["x"] + words["width"]).values])))
            y_edges = np.unique(np.sort(np.concatenate([words["y"].values, (words["y"] + words["height"]).values])))
            if len(x_edges) > 1 and len(y_edges) > 1:
                hist, _, _ = np.histogram2d(
                    fixations[x_field],
                    fixations[y_field],
                    bins=[x_edges, y_edges],
                    weights=weights,
                )
                z_vals = hist.T
                x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
                y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
                fig.add_trace(
                    go.Heatmap(
                        x=x_centers,
                        y=y_centers,
                        z=z_vals,
                        colorscale="Blues",
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
                        zmin=heatmap_range[0] if heatmap_range else None,
                        zmax=heatmap_range[1] if heatmap_range else None,
                        xgap=0,
                        ygap=0,
                        zsmooth=False,
                    )
                )
        else:
            x_bin_size = x_span / 40.0
            y_bin_size = y_span / 40.0
            fig.add_trace(
                go.Histogram2d(
                    x=fixations[x_field],
                    y=fixations[y_field],
                    xbins=dict(start=x_min, end=x_max, size=x_bin_size),
                    ybins=dict(start=y_min, end=y_max, size=y_bin_size),
                    colorscale="Blues",
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
                    histfunc=histfunc,
                    z=weights,
                    zmin=heatmap_range[0] if heatmap_range else None,
                    zmax=heatmap_range[1] if heatmap_range else None,
                )
            )

    if spatial_axes and show_saccades and len(fixations) > 1:
        ordered = fixations.sort_values("timestamp_ms")
        for (_, row_a), (_, row_b) in zip(ordered.iloc[:-1].iterrows(), ordered.iloc[1:].iterrows()):
            fig.add_trace(
                go.Scatter(
                    x=[row_a[x_field], row_b[x_field]],
                    y=[row_a[y_field], row_b[y_field]],
                    mode="lines",
                    line=dict(color="#6f42c1", width=2),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    if show_fixations and not fixations.empty:
        ordered = fixations.sort_values("timestamp_ms")
        color_data = ordered[color_by] if color_by in ordered.columns else None
        is_numeric_color = (
            color_data is not None and pd.api.types.is_numeric_dtype(color_data)
        )

        durations = ordered["duration_ms"].fillna(0)
        d_min, d_max = float(durations.min()), float(durations.max())
        min_size, max_size = marker_size_range
        if d_max - d_min > 0:
            sizes = np.interp(durations, (d_min, d_max), (min_size, max_size))
        else:
            sizes = np.full(len(durations), (min_size + max_size) / 2)

        fig.add_trace(
            go.Scatter(
                x=ordered[x_field],
                y=ordered[y_field],
                mode="markers+text" if show_order else "markers",
                marker=dict(
                    size=sizes,
                    color=color_data,
                    colorscale="Blues" if is_numeric_color else None,
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
                    line=dict(color="#111", width=0.5),
                ),
                text=ordered["order_in_trial"] if show_order else None,
                textfont=dict(color=order_font_color, size=order_font_size, family=font_settings["family"]),
                textposition="top center",
                hovertemplate=(
                    "Fixation #%{customdata[0]}<br>"
                    "Duration %{customdata[1]} ms<br>"
                    "Word #%{customdata[2]}<br>"
                    "Pass #%{customdata[3]}<br>"
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

    xaxis_cfg = dict(showticklabels=False, showgrid=False, zeroline=False, title=None)
    yaxis_cfg = dict(showticklabels=False, showgrid=False, zeroline=False, title=None)
    if spatial_axes:
        xaxis_cfg.update(range=[0, canvas_width], constrain="domain")
        yaxis_cfg.update(range=[canvas_height, 0], constrain="domain", scaleanchor="x", scaleratio=1)

    shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    if spatial_axes:
        shapes.append(
            dict(
                type="rect",
                x0=0,
                y0=0,
                x1=canvas_width,
                y1=canvas_height,
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
) -> go.Figure:
    fig = go.Figure()
    font_settings = dict(family=font_family or FONT_FAMILY, size=base_font_size)
    palette = ["#1f77b4", "#e45756"]
    for idx, trial in enumerate([trial_a, trial_b]):
        participant, trial_id = trial
        trial_words = words[
            (words["participant_id"] == participant) & (words["trial_id"] == trial_id)
        ]
        trial_fix = fixations[
            (fixations["participant_id"] == participant) & (fixations["trial_id"] == trial_id)
        ].sort_values("timestamp_ms")
        fig.add_trace(
            go.Scatter(
                x=trial_fix["x"],
                y=trial_fix["y"],
                mode="markers+lines",
                marker=dict(
                    size=9 + trial_fix["duration_ms"] * 0.04,
                    color=palette[idx],
                    line=dict(color="#111", width=0.5),
                ),
                line=dict(color=palette[idx], width=2, dash="solid"),
                name=f"{participant} – {trial_id}",
                text=trial_fix["order_in_trial"],
                textposition="top center",
                textfont=font_settings,
                hovertemplate=(
                    f"{participant}-{trial_id} "
                    "Order %{text}<br>Time %{customdata[0]} ms<br>Duration %{customdata[1]} ms<extra></extra>"
                ),
                customdata=trial_fix[["timestamp_ms", "duration_ms"]],
            )
        )
        existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
        fig.update_layout(
            shapes=existing_shapes + build_word_boxes(trial_words, color=palette[idx])
        )

    shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    shapes.append(
        dict(
            type="rect",
            x0=0,
            y0=0,
            x1=canvas_width,
            y1=canvas_height,
            line=dict(color="#000000", width=1),
            fillcolor="rgba(0,0,0,0)",
        )
    )

    fig.update_layout(
        height=canvas_height,
        width=canvas_width,
        autosize=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=None, range=[0, canvas_width], constrain="domain"),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=None, range=[canvas_height, 0], constrain="domain", scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        title="Overlay comparison",
        font=font_settings,
        shapes=shapes,
    )
    return fig
