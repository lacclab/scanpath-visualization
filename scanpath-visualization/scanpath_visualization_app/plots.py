from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .constants import FONT_FAMILY, DEFAULT_FIXATION_COLORSCALE, DEFAULT_HEATMAP_COLORSCALE

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
    fixation_colorscale: str = DEFAULT_FIXATION_COLORSCALE,
    heatmap_colorscale: str = DEFAULT_HEATMAP_COLORSCALE,
    raw_gaze: Optional[pd.DataFrame] = None,
    show_raw_gaze: bool = False,
) -> go.Figure:
    fig = go.Figure()
    spatial_axes = x_field == "x" and y_field == "y"
    font_settings = dict(family=font_family or FONT_FAMILY, size=base_font_size)
    x_range = [0, canvas_width]
    y_range = [canvas_height, 0]
    x_min_data = x_max_data = y_min_data = y_max_data = None

    if spatial_axes:
        x_candidates = []
        y_candidates = []
        if not words.empty:
            x_candidates.extend([words["x"].min(), (words["x"] + words["width"]).max()])
            y_candidates.extend([words["y"].min(), (words["y"] + words["height"]).max()])
        if not fixations.empty:
            x_candidates.extend([fixations[x_field].min(), fixations[x_field].max()])
            y_candidates.extend([fixations[y_field].min(), fixations[y_field].max()])
        if show_raw_gaze and raw_gaze is not None and not raw_gaze.empty:
            x_candidates.extend([raw_gaze["x"].min(), raw_gaze["x"].max()])
            y_candidates.extend([raw_gaze["y"].min(), raw_gaze["y"].max()])

        if x_candidates and y_candidates:
            x_min_data = float(np.nanmin(x_candidates))
            x_max_data = float(np.nanmax(x_candidates))
            y_min_data = float(np.nanmin(y_candidates))
            y_max_data = float(np.nanmax(y_candidates))

            x_span = max(x_max_data - x_min_data, 1.0)
            y_span = max(y_max_data - y_min_data, 1.0)
            pad_x = max(20.0, 0.05 * x_span)
            pad_y = max(20.0, 0.05 * y_span)
            x_range = [x_min_data - pad_x, x_max_data + pad_x]
            y_range = [y_max_data + pad_y, y_min_data - pad_y]

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

    # Plot raw gaze data as small dots (before heatmap and fixations so it appears in background)
    if spatial_axes and show_raw_gaze and raw_gaze is not None and not raw_gaze.empty:
        # Color by timestamp if available, otherwise uniform color
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
                    size=3,
                    color=color_vals,
                    colorscale=colorscale,
                    opacity=0.5,
                    showscale=False,
                ),
                hovertemplate=(
                    "Raw gaze<br>"
                    "x: %{x:.1f}<br>"
                    "y: %{y:.1f}<br>"
                    "t: %{customdata} ms<extra></extra>"
                ),
                customdata=raw_gaze["timestamp_ms"] if "timestamp_ms" in raw_gaze.columns else None,
                name="Raw gaze",
                showlegend=True,
            )
        )

    if spatial_axes and show_heatmap and not fixations.empty:
        weights = None
        if heatmap_metric == "duration_ms":
            weights = fixations["duration_ms"]
        histfunc = "sum" if weights is not None else "count"
        x_min = x_min_data if x_min_data is not None else float(fixations[x_field].min())
        x_max = x_max_data if x_max_data is not None else float(fixations[x_field].max())
        y_min = y_min_data if y_min_data is not None else float(fixations[y_field].min())
        y_max = y_max_data if y_max_data is not None else float(fixations[y_field].max())
        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1.0)
        if not words.empty:
            # Word-level heatmap: aggregate fixations per word
            word_values = []
            for _, word_row in words.iterrows():
                wx0, wy0 = word_row["x"], word_row["y"]
                wx1, wy1 = wx0 + word_row["width"], wy0 + word_row["height"]
                # Find fixations within this word's bounding box
                in_word = (
                    (fixations[x_field] >= wx0) & (fixations[x_field] <= wx1) &
                    (fixations[y_field] >= wy0) & (fixations[y_field] <= wy1)
                )
                if weights is not None:
                    val = float(weights[in_word].sum())
                else:
                    val = float(in_word.sum())
                word_values.append(val)

            words_with_vals = words.copy()
            words_with_vals["heatmap_val"] = word_values

            # Only show words with non-zero values
            words_nonzero = words_with_vals[words_with_vals["heatmap_val"] > 0]
            if not words_nonzero.empty:
                z_min = heatmap_range[0] if heatmap_range else float(words_nonzero["heatmap_val"].min())
                z_max = heatmap_range[1] if heatmap_range else float(words_nonzero["heatmap_val"].max())
                z_range = max(z_max - z_min, 1e-9)

                # Use shapes for word-level heatmap cells
                from plotly.colors import sample_colorscale
                heatmap_shapes = []
                for _, wr in words_nonzero.iterrows():
                    norm_val = (wr["heatmap_val"] - z_min) / z_range
                    norm_val = max(0.0, min(1.0, norm_val))
                    color = sample_colorscale(heatmap_colorscale, [norm_val])[0]
                    heatmap_shapes.append(
                        dict(
                            type="rect",
                            x0=wr["x"],
                            y0=wr["y"],
                            x1=wr["x"] + wr["width"],
                            y1=wr["y"] + wr["height"],
                            line=dict(width=0),
                            fillcolor=color,
                            opacity=0.5,
                            layer="below",
                        )
                    )
                existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
                fig.update_layout(shapes=existing_shapes + heatmap_shapes)

                # Add invisible scatter for colorbar
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
        else:
            x_bin_size = x_span / 40.0
            y_bin_size = y_span / 40.0
            fig.add_trace(
                go.Histogram2d(
                    x=fixations[x_field],
                    y=fixations[y_field],
                    xbins=dict(start=x_min, end=x_max, size=x_bin_size),
                    ybins=dict(start=y_min, end=y_max, size=y_bin_size),
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
        xaxis_cfg.update(range=x_range, constrain="domain")
        yaxis_cfg.update(range=y_range, constrain="domain", scaleanchor="x", scaleratio=1)
    else:
        # Non-spatial axes: add trendline and show axis labels/grid
        xaxis_cfg.update(showticklabels=True, showgrid=True, title=x_field.replace("_", " ").title())
        yaxis_cfg.update(showticklabels=True, showgrid=True, title=y_field.replace("_", " ").title())
        
        if not fixations.empty and len(fixations) > 1:
            # Fit linear trendline using least squares
            x_vals = fixations[x_field].dropna()
            y_vals = fixations[y_field].dropna()
            # Align indices for valid pairs
            valid_idx = x_vals.index.intersection(y_vals.index)
            if len(valid_idx) > 1:
                x_clean = x_vals.loc[valid_idx].values
                y_clean = y_vals.loc[valid_idx].values
                # Linear regression: y = mx + b
                coeffs = np.polyfit(x_clean, y_clean, 1)
                x_trend = np.array([x_clean.min(), x_clean.max()])
                y_trend = np.polyval(coeffs, x_trend)
                fig.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=y_trend,
                        mode="lines",
                        line=dict(color="#dc3545", width=2, dash="dash"),
                        name="Trendline",
                        showlegend=True,
                        hovertemplate=f"y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}<extra>Trendline</extra>",
                    )
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
    marker_size_range: Tuple[int, int] = (8, 30),
    order_font_size: int = 10,
    order_font_color: str = "#000000",
) -> go.Figure:
    """Create an animated scanpath figure that shows fixations progressing over time.
    
    Each fixation is displayed for its actual duration divided by the playback_speed.
    For example, playback_speed=2.0 means 2x speed (fixations shown for half their duration).
    """
    fig = go.Figure()
    font_settings = dict(family=font_family or FONT_FAMILY, size=base_font_size)
    
    # Calculate ranges
    x_candidates = []
    y_candidates = []
    if not words.empty:
        x_candidates.extend([words["x"].min(), (words["x"] + words["width"]).max()])
        y_candidates.extend([words["y"].min(), (words["y"] + words["height"]).max()])
    if not fixations.empty:
        x_candidates.extend([fixations["x"].min(), fixations["x"].max()])
        y_candidates.extend([fixations["y"].min(), fixations["y"].max()])

    x_range = [0, canvas_width]
    y_range = [canvas_height, 0]
    if x_candidates and y_candidates:
        x_min = float(np.nanmin(x_candidates))
        x_max = float(np.nanmax(x_candidates))
        y_min = float(np.nanmin(y_candidates))
        y_max = float(np.nanmax(y_candidates))

        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1.0)
        pad_x = max(20.0, 0.05 * x_span)
        pad_y = max(20.0, 0.05 * y_span)
        x_range = [x_min - pad_x, x_max + pad_x]
        y_range = [y_max + pad_y, y_min - pad_y]

    # Add word boxes as shapes
    shapes = []
    if show_words and not words.empty:
        shapes = build_word_boxes(words)

    # Add word labels as initial trace
    if show_word_labels and not words.empty and "text" in words.columns:
        fig.add_trace(
            go.Scatter(
                x=words["x"] + words["width"] / 2,
                y=words["y"] + words["height"] / 2,
                text=words["text"],
                mode="text",
                showlegend=False,
                textfont=dict(color="#343a40", size=base_font_size, family=font_settings["family"]),
                hoverinfo="skip",
                name="words",
            )
        )

    # Order fixations by timestamp
    if fixations.empty:
        return fig
    
    ordered = fixations.sort_values("timestamp_ms").reset_index(drop=True)
    n_fixations = len(ordered)
    
    # Compute marker sizes based on duration
    durations = ordered["duration_ms"].fillna(0)
    d_min, d_max = float(durations.min()), float(durations.max())
    min_size, max_size = marker_size_range
    if d_max - d_min > 0:
        sizes = np.interp(durations, (d_min, d_max), (min_size, max_size))
    else:
        sizes = np.full(len(durations), (min_size + max_size) / 2)
    
    # Calculate frame durations based on actual fixation durations and playback speed
    # Each frame shows for the fixation's duration divided by playback speed
    frame_durations_ms = (durations / playback_speed).clip(lower=50).astype(int).tolist()
    
    # For slider and play button, use average duration as default
    avg_frame_duration = int(np.mean(frame_durations_ms))
    
    # Determine display mode based on show_order
    marker_mode = "markers+text" if show_order else "markers"
    
    # Create initial empty traces for fixations and saccades
    # Fixation markers trace
    fig.add_trace(
        go.Scatter(
            x=[ordered.iloc[0]["x"]],
            y=[ordered.iloc[0]["y"]],
            mode=marker_mode,
            marker=dict(
                size=[sizes[0]],
                color="#1f77b4",
                line=dict(color="#111", width=0.5),
            ),
            text=["1"] if show_order else None,
            textfont=dict(color=order_font_color, size=order_font_size, family=font_settings["family"]),
            textposition="top center",
            showlegend=False,
            name="fixations",
            hovertemplate=(
                "Fixation #%{text}<br>"
                "Duration %{customdata} ms<br>"
                "<extra></extra>"
            ),
            customdata=[ordered.iloc[0]["duration_ms"]],
        )
    )
    
    # Saccade lines trace
    if show_saccades:
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="#6f42c1", width=2),
                showlegend=False,
                name="saccades",
                hoverinfo="skip",
            )
        )
    
    # Current fixation highlight trace (larger, different color)
    fig.add_trace(
        go.Scatter(
            x=[ordered.iloc[0]["x"]],
            y=[ordered.iloc[0]["y"]],
            mode="markers",
            marker=dict(
                size=sizes[0] + 8,
                color="rgba(255, 127, 14, 0.6)",
                line=dict(color="#ff7f0e", width=2),
            ),
            showlegend=False,
            name="current",
            hoverinfo="skip",
        )
    )
    
    # Create frames for animation
    frames = []
    for i in range(n_fixations):
        # Accumulated fixations up to this point
        fix_x = ordered.iloc[:i+1]["x"].tolist()
        fix_y = ordered.iloc[:i+1]["y"].tolist()
        fix_sizes = sizes[:i+1].tolist()
        fix_texts = [str(j+1) for j in range(i+1)] if show_order else None
        fix_durations = ordered.iloc[:i+1]["duration_ms"].tolist()
        
        # Saccade lines: connect consecutive fixations
        sac_x = []
        sac_y = []
        if show_saccades:
            for j in range(i):
                sac_x.extend([ordered.iloc[j]["x"], ordered.iloc[j+1]["x"], None])
                sac_y.extend([ordered.iloc[j]["y"], ordered.iloc[j+1]["y"], None])
        
        # Current fixation position
        curr_x = [ordered.iloc[i]["x"]]
        curr_y = [ordered.iloc[i]["y"]]
        curr_size = [sizes[i] + 8]
        
        frame_data = [
            go.Scatter(
                x=fix_x,
                y=fix_y,
                mode=marker_mode,
                marker=dict(
                    size=fix_sizes,
                    color="#1f77b4",
                    line=dict(color="#111", width=0.5),
                ),
                text=fix_texts,
                textfont=dict(color=order_font_color, size=order_font_size, family=font_settings["family"]),
                textposition="top center",
                customdata=fix_durations,
            ),
        ]
        
        if show_saccades:
            frame_data.append(
                go.Scatter(
                    x=sac_x,
                    y=sac_y,
                    mode="lines",
                    line=dict(color="#6f42c1", width=2),
                )
            )
        
        frame_data.append(
            go.Scatter(
                x=curr_x,
                y=curr_y,
                mode="markers",
                marker=dict(
                    size=curr_size,
                    color="rgba(255, 127, 14, 0.6)",
                    line=dict(color="#ff7f0e", width=2),
                ),
            )
        )
        
        # If word labels exist, keep them in each frame
        if show_word_labels and not words.empty and "text" in words.columns:
            frame_data.insert(0, go.Scatter(
                x=words["x"] + words["width"] / 2,
                y=words["y"] + words["height"] / 2,
                text=words["text"],
                mode="text",
                textfont=dict(color="#343a40", size=base_font_size, family=font_settings["family"]),
            ))
        
        frames.append(go.Frame(
            data=frame_data,
            name=str(i),
            traces=list(range(len(frame_data))),
        ))
    
    fig.frames = frames
    
    # Add border shape
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
    
    # Create slider steps - each step uses its own frame duration
    sliders = [dict(
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
                args=[[str(i)], dict(
                    frame=dict(duration=frame_durations_ms[i], redraw=True),
                    mode="immediate",
                    transition=dict(duration=min(frame_durations_ms[i] // 2, 100)),
                )],
                label=str(i + 1),
                method="animate",
            )
            for i in range(n_fixations)
        ],
    )]
    
    # Update buttons for play/pause
    # For play, we use average frame duration since Plotly animation doesn't support per-frame durations in auto-play
    updatemenus = [dict(
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
                        transition=dict(duration=min(avg_frame_duration // 2, 100), easing="cubic-in-out"),
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
        ],
    )]
    
    fig.update_layout(
        height=canvas_height + 80,  # Extra space for slider
        width=canvas_width,
        autosize=False,
        margin=dict(l=0, r=0, t=0, b=80),
        xaxis=dict(
            showticklabels=False, 
            showgrid=False, 
            zeroline=False, 
            title=None, 
            range=x_range, 
            constrain="domain"
        ),
        yaxis=dict(
            showticklabels=False, 
            showgrid=False, 
            zeroline=False, 
            title=None, 
            range=y_range, 
            constrain="domain", 
            scaleanchor="x", 
            scaleratio=1
        ),
        template="plotly_white",
        font=font_settings,
        shapes=shapes,
        sliders=sliders,
        updatemenus=updatemenus,
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
    x_candidates = []
    y_candidates = []
    for idx, trial in enumerate([trial_a, trial_b]):
        participant, trial_id = trial
        trial_words = words[
            (words["participant_id"] == participant) & (words["trial_id"] == trial_id)
        ]
        trial_fix = fixations[
            (fixations["participant_id"] == participant) & (fixations["trial_id"] == trial_id)
        ].sort_values("timestamp_ms")
        if not trial_words.empty:
            x_candidates.extend([trial_words["x"].min(), (trial_words["x"] + trial_words["width"]).max()])
            y_candidates.extend([trial_words["y"].min(), (trial_words["y"] + trial_words["height"]).max()])
        if not trial_fix.empty:
            x_candidates.extend([trial_fix["x"].min(), trial_fix["x"].max()])
            y_candidates.extend([trial_fix["y"].min(), trial_fix["y"].max()])
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

    x_range = [0, canvas_width]
    y_range = [canvas_height, 0]
    if x_candidates and y_candidates:
        x_min = float(np.nanmin(x_candidates))
        x_max = float(np.nanmax(x_candidates))
        y_min = float(np.nanmin(y_candidates))
        y_max = float(np.nanmax(y_candidates))

        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1.0)
        pad_x = max(20.0, 0.05 * x_span)
        pad_y = max(20.0, 0.05 * y_span)
        x_range = [x_min - pad_x, x_max + pad_x]
        y_range = [y_max + pad_y, y_min - pad_y]

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
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=None, range=x_range, constrain="domain"),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=None, range=y_range, constrain="domain", scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        title="Overlay comparison",
        font=font_settings,
        shapes=shapes,
    )
    return fig
