"""Headless programmatic API for scanpath-studio.

The Streamlit app and this module share one pipeline (``data`` → ``measures``
→ ``plots``), so a figure produced here is pixel-identical to the app's
canonical visualization. Typical use::

    import scanpath_studio as sps

    words, fixations = sps.load_scanpath_data("ia.csv", "fixations.csv")
    print(sps.list_trials(words, fixations))
    fig = sps.plot_scanpath(words, fixations, participant="p1", trial="t1")
    sps.save_figure(fig, "scanpath.html")   # or .png/.svg/.pdf (needs Chrome)

Every keyword accepted by :func:`plots.make_scanpath_figure` /
:func:`plots.make_scanpath_animation` can be overridden through
``plot_scanpath`` / ``animate_scanpath`` (e.g. ``show_heatmap=False``).
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
import plotly.graph_objects as go

# Outside a Streamlit runtime the @st.cache_data decorators in `data` fall
# back to bare-mode caching and log a "No runtime found" warning per cached
# function — harmless but noisy for library/CLI users, so quiet those loggers.
# Order matters twice over: streamlit must be imported first (its get_logger()
# sets each module logger's level at import, clobbering anything set earlier),
# and `.data` must be imported after (its decorators fire the warnings at
# import time). Inside the app a runtime exists and these warnings never fire.
import streamlit as _st  # noqa: F401  (imported for its logging side effect)

for _name in (
    "streamlit.runtime.caching.cache_data_api",
    "streamlit.runtime.scriptrunner_utils.script_run_context",
):
    logging.getLogger(_name).setLevel(logging.ERROR)

from . import data as _data  # noqa: E402
from .constants import (  # noqa: E402
    DEFAULT_BACKGROUND_COLOR,
    DEFAULT_FIXATION_COLORSCALE,
    DEFAULT_HEATMAP_COLORSCALE,
    DEFAULT_LINE_SPACING,
    DEFAULT_MARKER_SIZE_RANGE,
    DEFAULT_ORDER_FONT_COLOR,
    FONT_FAMILY,
)
from .plots import make_scanpath_animation, make_scanpath_figure  # noqa: E402

TableLike = Union[pd.DataFrame, str, Path]
TablesLike = Union[TableLike, "list[TableLike]"]

# Mirrors the app's sidebar defaults (controls.sidebar_controls) — the
# "canonical" scanpath rendering. `heatmap_metric="counts"` is translated to
# the figure-level `None` in _figure_kwargs, like tabs._build_figure_settings.
CANONICAL_FIGURE_DEFAULTS: dict = dict(
    show_words=True,
    show_word_labels=True,
    show_fixations=True,
    show_order=True,
    show_saccades=True,
    show_saccade_arrows=False,
    show_heatmap=True,
    heatmap_style="Word boxes",
    color_by="duration_ms",
    heatmap_metric="duration_ms",
    marker_size_range=DEFAULT_MARKER_SIZE_RANGE,
    order_font_size=16,
    order_font_color=DEFAULT_ORDER_FONT_COLOR,
    show_colorbars=False,
    fixation_color_range=None,
    heatmap_range=None,
    fixation_colorscale=DEFAULT_FIXATION_COLORSCALE,
    heatmap_colorscale=DEFAULT_HEATMAP_COLORSCALE,
    critical_span_style="Mark text",
    background_color=DEFAULT_BACKGROUND_COLOR,
    color_by_line=False,
    highlight_out_of_text=False,
    line_spacing=DEFAULT_LINE_SPACING,
    scale_text_to_boxes=True,
)


def _as_dataframe(table: TablesLike, label: str) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table
    items = _data.expand_table_inputs(table)
    for item in items:
        if not isinstance(item, pd.DataFrame) and not Path(item).is_file():
            raise FileNotFoundError(f"{label} table not found: {item}")
    return _data.read_tables(items)


def load_scanpath_data(
    words: Optional[TablesLike] = None,
    fixations: Optional[TablesLike] = None,
    *,
    word_schema: Optional[dict] = None,
    fix_schema: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and normalize a words/IA table and/or a fixations table.

    ``words`` / ``fixations`` may be DataFrames, paths to ``.csv`` / ``.tsv``
    / ``.parquet`` / ``.feather`` files, glob patterns, or lists of paths —
    multi-file datasets (one file per participant and/or text) are
    concatenated, with each file's stem kept in a ``source_file`` column.
    Column schemas are auto-detected (EyeLink, Gazepoint, and snake_case
    names); pass ``word_schema`` / ``fix_schema`` mappings (field → column
    name, see ``controls.WORD_FIELD_SPECS``) to override detection. For
    per-word reading measures, pass the result to :func:`compute_word_metrics`.

    Either table may be omitted for datasets that ship only one report: the
    missing side comes back as an empty canonical frame and the plots simply
    skip that layer. Words without a participant column (stimulus-level AoIs)
    are broadcast across the participants found in the fixations, and
    fixations without x/y but with a word/AoI ID are placed at word-box
    centers.

    Returns the normalized ``(words, fixations)`` frames the plotting
    functions expect. Raises ``ValueError`` if required fields can't be found.
    """
    if words is None and fixations is None:
        raise ValueError("Provide at least one of words= or fixations=.")

    if words is not None:
        words_df = _as_dataframe(words, "words/IA")
        word_schema = word_schema or _data.propose_word_schema(words_df)
        problems = _data.validate_word_schema(word_schema)
        if problems:
            raise ValueError(f"Words/IA schema problems: {'; '.join(problems)}")
        words_norm = _data.normalize_words(words_df, word_schema)
    else:
        words_norm = _data.empty_words_frame()

    if fixations is not None:
        fixations_df = _as_dataframe(fixations, "fixations")
        fix_schema = fix_schema or _data.propose_fix_schema(fixations_df)
        problems = _data.validate_fix_schema(fix_schema)
        if problems:
            raise ValueError(f"Fixations schema problems: {'; '.join(problems)}")
        fixations_norm = _data.normalize_fixations(fixations_df, fix_schema)
    else:
        fixations_norm = _data.empty_fixations_frame()

    return _data.harmonize_frames(words_norm, fixations_norm)


def load_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return the bundled 3-participant OneStop demo, normalized and ready to plot."""
    return load_scanpath_data(*_data.load_sample_data())


def compute_word_metrics(words: pd.DataFrame, fixations: pd.DataFrame) -> pd.DataFrame:
    """Per-word reading measures (FFD/FPRT/RPD/TFD, skips, regressions, …).

    Pre-aggregated columns in ``words`` (EyeLink IA exports) are preserved;
    anything missing is computed from fixations + word bounding boxes. Takes
    the normalized frames from :func:`load_scanpath_data`."""
    return _data.compute_word_metrics(words, fixations)


def list_trials(words: pd.DataFrame, fixations: pd.DataFrame) -> pd.DataFrame:
    """Plottable ``(participant_id, trial_id)`` combos.

    Combos present in both frames when both are loaded; for single-report
    datasets (words-only or fixations-only), combos from whichever frame has
    data."""
    cols = ["participant_id", "trial_id"]
    if words.empty or fixations.empty:
        present = fixations if words.empty else words
        combos = present[cols].drop_duplicates()
    else:
        combos = words[cols].drop_duplicates().merge(fixations[cols].drop_duplicates())
    return combos.sort_values(cols).reset_index(drop=True)


def _resolve_trial(
    words: pd.DataFrame,
    fixations: pd.DataFrame,
    participant: Optional[str],
    trial: Optional[str],
    *,
    default_first: bool = False,
) -> Tuple[str, str]:
    """Resolve to one (participant_id, trial_id), validating what was given.

    A nonexistent participant/trial always raises. An underspecified selection
    matching several trials raises too, unless ``default_first`` picks the
    first match (the CLI's behavior, mirroring the app's default selection).
    """
    combos = list_trials(words, fixations)
    if combos.empty:
        raise ValueError("No (participant, trial) combo exists in the data.")
    if participant is not None:
        combos = combos[combos["participant_id"] == str(participant)]
    if trial is not None:
        combos = combos[combos["trial_id"] == str(trial)]
    if combos.empty:
        raise ValueError(
            f"No trial matches participant={participant!r}, trial={trial!r}. "
            "Use list_trials(words, fixations) to see what's available."
        )
    if len(combos) > 1 and not default_first:
        preview = combos.head(5).to_records(index=False).tolist()
        raise ValueError(
            f"Ambiguous selection: {len(combos)} trials match "
            f"(first few: {preview}). Pass participant= and trial=."
        )
    row = combos.iloc[0]
    return str(row["participant_id"]), str(row["trial_id"])


def _select_trial(
    words: pd.DataFrame,
    fixations: pd.DataFrame,
    participant: Optional[str],
    trial: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    pid, tid = _resolve_trial(words, fixations, participant, trial)
    trial_words, trial_fixations = _data.filter_data(
        words, fixations, {"participants": [pid], "trials": [tid]}
    )
    if not trial_fixations.empty and trial_fixations["x"].isna().all():
        # AOI-sequence fixations whose coordinates couldn't be reconstructed:
        # either no words table was given, or the word/AoI ids matched no box.
        raise ValueError(
            f"Fixations for participant={pid!r}, trial={tid!r} have no usable "
            "coordinates. AOI-sequence datasets (no x/y) need a words table "
            "whose word/AoI ids match the fixations' so fixations can be "
            "placed at word-box centers."
        )
    return trial_words, trial_fixations, pid, tid


def _figure_kwargs(overrides: dict) -> dict:
    settings = {**CANONICAL_FIGURE_DEFAULTS, **overrides}
    if settings.get("heatmap_metric") == "counts":
        settings["heatmap_metric"] = None
    return settings


def plot_scanpath(
    words: pd.DataFrame,
    fixations: pd.DataFrame,
    participant: Optional[str] = None,
    trial: Optional[str] = None,
    *,
    canvas_size: Optional[Tuple[int, int]] = None,
    base_font_size: int = 16,
    font_family: str = FONT_FAMILY,
    raw_gaze: Optional[pd.DataFrame] = None,
    **figure_overrides,
) -> go.Figure:
    """Build the canonical scanpath figure for one trial.

    ``words`` / ``fixations`` are normalized frames from
    :func:`load_scanpath_data`. ``participant`` / ``trial`` may be omitted when
    the frames contain exactly one combo. ``canvas_size`` is the monitor size
    in px; by default it is estimated from the data extents — pass the real
    monitor resolution (e.g. ``(2560, 1440)`` for OneStop) to keep coordinates
    true to scale. ``raw_gaze`` is a normalized frame (see
    :func:`data.normalize_raw_gaze`) and is filtered to the selected trial.
    Remaining keywords override the app's defaults and are forwarded to
    :func:`plots.make_scanpath_figure` (e.g. ``show_heatmap=False``,
    ``color_by="pass_index"``, ``x_field="order_in_trial"``).
    """
    trial_words, trial_fixations, pid, tid = _select_trial(
        words, fixations, participant, trial
    )
    if canvas_size is None:
        canvas_size = _data.compute_canvas_size(trial_words, trial_fixations)
    settings = _figure_kwargs(figure_overrides)
    # Spatial fields are explicit kwargs of make_scanpath_figure, so they can't
    # ride along in **settings without a "multiple values" TypeError.
    x_field = settings.pop("x_field", "x")
    y_field = settings.pop("y_field", "y")
    if raw_gaze is not None:
        raw_gaze = _data.filter_raw_gaze(raw_gaze, [pid], [tid])
        settings.setdefault("show_raw_gaze", True)
    return make_scanpath_figure(
        trial_words,
        trial_fixations,
        canvas_width=int(canvas_size[0]),
        canvas_height=int(canvas_size[1]),
        base_font_size=int(base_font_size),
        font_family=font_family,
        x_field=x_field,
        y_field=y_field,
        raw_gaze=raw_gaze,
        **settings,
    )


def animate_scanpath(
    words: pd.DataFrame,
    fixations: pd.DataFrame,
    participant: Optional[str] = None,
    trial: Optional[str] = None,
    *,
    canvas_size: Optional[Tuple[int, int]] = None,
    base_font_size: int = 16,
    font_family: str = FONT_FAMILY,
    playback_speed: float = 1.0,
    **animation_overrides,
) -> go.Figure:
    """Build the animated scanpath replay for one trial.

    Same trial selection and canvas semantics as :func:`plot_scanpath`. The
    returned Plotly figure plays in real reading time scaled by
    ``playback_speed``; save it as interactive HTML with :func:`save_figure`,
    or rasterize to GIF/MP4 with :func:`animation_export.export_animation`.

    The animation builder accepts a subset of the static figure's options
    (``show_words``, ``show_word_labels``, ``show_saccades``, ``show_order``,
    styling, and second-scanpath overlays) — an unsupported key raises a
    ``ValueError`` naming the valid ones rather than an opaque ``TypeError``.
    """
    valid = set(inspect.signature(make_scanpath_animation).parameters) - {
        "words",
        "fixations",
        "canvas_width",
        "canvas_height",
        "base_font_size",
        "font_family",
        "playback_speed",
    }
    unknown = set(animation_overrides) - valid
    if unknown:
        raise ValueError(
            f"Options not supported by the animation: {sorted(unknown)}. "
            f"Valid overrides: {sorted(valid)}."
        )
    trial_words, trial_fixations, _, _ = _select_trial(
        words, fixations, participant, trial
    )
    if canvas_size is None:
        canvas_size = _data.compute_canvas_size(trial_words, trial_fixations)
    return make_scanpath_animation(
        trial_words,
        trial_fixations,
        canvas_width=int(canvas_size[0]),
        canvas_height=int(canvas_size[1]),
        base_font_size=int(base_font_size),
        font_family=font_family,
        playback_speed=playback_speed,
        **animation_overrides,
    )


def save_figure(fig: go.Figure, path: Union[str, Path], *, scale: int = 2) -> Path:
    """Save a figure by extension: ``.html`` (interactive, browser-free) or
    ``.png``/``.svg``/``.pdf`` (static via Kaleido — needs a Chrome/Chromium;
    run ``plotly_get_chrome -y`` once if missing). Returns the written path."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".html":
        fig.write_html(str(path))
        return path
    if suffix in (".png", ".svg", ".pdf"):
        try:
            fig.write_image(str(path), scale=scale)
        except OSError:
            raise  # filesystem problem — the original error says it best
        except Exception as exc:  # Kaleido raises various types
            raise RuntimeError(
                f"Static {suffix} export failed: {exc} — if Kaleido can't find "
                "a Chrome/Chromium binary, run `plotly_get_chrome -y` once, or "
                "save as .html instead."
            ) from exc
        return path
    raise ValueError(
        f"Unsupported extension {suffix!r} — use .html, .png, .svg, or .pdf."
    )
