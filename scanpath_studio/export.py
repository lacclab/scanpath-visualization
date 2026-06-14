"""Configurable bulk export of figures and tabular data for filtered trials.

This module powers the "Bulk export" button. Users pick which artifacts they
want per trial (PNG, SVG, JSON plot config, fixations CSV/Parquet, per-word
measures CSV/Parquet) plus an optional aggregated mega-table across all
selected trials. Everything is packaged into a single zip archive with a
clean folder structure:

    bulk_export_<timestamp>.zip
    ├─ per_trial/
    │  ├─ <participant>__<trial>/
    │  │  ├─ figure.png
    │  │  ├─ figure.svg
    │  │  ├─ plot_config.json
    │  │  ├─ fixations.csv (and/or .parquet)
    │  │  └─ measures.csv (and/or .parquet)
    │  ├─ ...
    └─ aggregate/
       ├─ all_fixations.csv (and/or .parquet)
       └─ all_measures.csv (and/or .parquet)
"""

from __future__ import annotations

import io
import json
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd

from .constants import CITATION, DEFAULT_LINE_SPACING
from .data import compute_word_metrics
from .plots import make_scanpath_figure


@dataclass
class ExportOptions:
    """User-chosen export artifacts.

    Defaults: figures on (PNG + SVG), tabular data off. The scope fields
    narrow the set of trials when not "all".
    """

    include_png: bool = True
    include_svg: bool = True
    include_pdf: bool = False
    include_plot_config: bool = True
    include_fixations: bool = False
    include_measures: bool = False
    include_mega_table: bool = False
    table_format: str = "csv"  # "csv" | "parquet" | "both"
    png_scale: int = 2
    # When True, export operates on the whole loaded dataset, ignoring the
    # sidebar "Filter trials" panel; the caller supplies the unfiltered frames.
    export_unfiltered: bool = False
    scope: str = "all"  # "all" | "trial" | "participant" | "text"
    scope_participant: Optional[str] = None
    scope_trial: Optional[str] = None
    scope_text: Optional[str] = None

    def any_table(self) -> bool:
        return (
            self.include_fixations or self.include_measures or self.include_mega_table
        )

    def table_formats(self) -> List[str]:
        if self.table_format == "both":
            return ["csv", "parquet"]
        return [self.table_format]

    def figure_formats(self) -> List[str]:
        formats: List[str] = []
        if self.include_png:
            formats.append("png")
        if self.include_svg:
            formats.append("svg")
        if self.include_pdf:
            formats.append("pdf")
        return formats


@dataclass
class ExportProgress:
    total_trials: int
    finished_trials: int = 0
    bytes_written: int = 0
    errors: List[str] = field(default_factory=list)


def _safe_id(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(text))


def _write_table(zf: zipfile.ZipFile, path: str, df: pd.DataFrame, fmt: str) -> int:
    if fmt == "parquet":
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        data = buf.getvalue()
    else:
        data = df.to_csv(index=False).encode("utf-8")
    zf.writestr(path, data)
    return len(data)


@contextmanager
def _figure_renderer(enabled: bool):
    """Yield ``render(fig, fmt, width, height, scale) -> bytes``.

    When ``enabled`` and Kaleido starts, every trial's figure is rasterized
    through one persistent Kaleido browser (``calc_fig_sync``) instead of
    cold-starting a fresh Chrome on each ``fig.to_image`` call — the cold start
    is the "Resorting to unclean kill browser." log noise and ~seconds-per-trial
    latency. Falls back to per-call ``to_image`` if the warm server can't start
    (or no figures were requested), so behavior is unchanged when Kaleido/Chrome
    is unavailable — the per-trial failure is still surfaced as an export error.
    """
    server = None
    if enabled:
        try:
            import kaleido

            kaleido.start_sync_server(silence_warnings=True)
            server = kaleido
        except Exception:
            server = None

    def render(fig, fmt: str, width: int, height: int, scale: int) -> bytes:
        if server is not None:
            data = server.calc_fig_sync(
                fig,
                opts={
                    "format": fmt,
                    "width": int(width),
                    "height": int(height),
                    "scale": scale,
                },
            )
            return bytes(data)
        return fig.to_image(
            format=fmt, width=int(width), height=int(height), scale=scale
        )

    try:
        yield render
    finally:
        if server is not None:
            try:
                server.stop_sync_server(silence_warnings=True)
            except Exception:  # pragma: no cover - best-effort teardown
                pass


def _plot_config_dict(
    participant: str,
    trial: str,
    canvas_width: int,
    canvas_height: int,
    x_field: str,
    y_field: str,
    settings: dict,
) -> dict:
    return {
        "selection": {"participant_id": participant, "trial_id": trial},
        "canvas_px": {"width": int(canvas_width), "height": int(canvas_height)},
        "axes": {"x_field": x_field, "y_field": y_field},
        "layers": {
            "words": settings.get("show_words"),
            "word_labels": settings.get("show_word_labels"),
            "fixations": settings.get("show_fixations"),
            "order_labels": settings.get("show_order"),
            "saccades": settings.get("show_saccades"),
            "saccade_arrows": settings.get("show_saccade_arrows", False),
            "heatmap": settings.get("show_heatmap"),
            "raw_gaze": settings.get("show_raw_gaze"),
        },
        "coloring": {
            "color_by": settings.get("color_by"),
            "heatmap_metric": settings.get("heatmap_metric"),
            "heatmap_style": settings.get("heatmap_style", "Word boxes"),
            "fixation_colorscale": settings.get("fixation_colorscale"),
            "heatmap_colorscale": settings.get("heatmap_colorscale"),
        },
        "sizing": {
            "marker_size_range": list(settings.get("marker_size_range", [])),
            "order_font_size": settings.get("order_font_size"),
        },
        # True-to-scale reading text: records how the word labels were sized so
        # the figure can be reproduced exactly (see plots._word_label_font_px).
        "text": {
            "scale_text_to_boxes": settings.get("scale_text_to_boxes", True),
            "line_spacing": settings.get("line_spacing", DEFAULT_LINE_SPACING),
        },
    }


def _render_scope_picker(
    st,
    combos: pd.DataFrame,
    key_prefix: str,
    combos_all: Optional[pd.DataFrame] = None,
) -> tuple[str, Optional[str], Optional[str], Optional[str], bool]:
    """Render the scope radio + dependent selectors.

    Returns ``(scope, pid, trial, text, export_unfiltered)``. The whole-dataset
    choice lives inside the "Trials to include" radio (an extra "All" option that
    ignores the sidebar filter) rather than as a separate checkbox.
    """
    # Build the ordered radio: label -> (scope, export_unfiltered). Both "All"
    # (the whole dataset, ignoring the sidebar filter) and "All filtered trials"
    # (the current sidebar selection) are always offered — they coincide only
    # when no filter is active.
    options_map: dict[str, tuple[str, bool]] = {
        "All": ("all", True),
        "All filtered trials": ("all", False),
    }
    options_map["A single trial"] = ("trial", False)
    options_map["All trials of one participant"] = ("participant", False)
    options_map["All trials of one text"] = ("text", False)

    # Default to the filtered subset (respect what the user narrowed to).
    default_index = 1
    scope_label = st.radio(
        "Trials to include",
        options=list(options_map),
        index=default_index,
        key=f"{key_prefix}_scope",
        horizontal=True,
        help="Limit the export to a subset of trials. **All** exports every "
        "trial in the dataset, ignoring the **Filter trials** sidebar panel.",
    )
    scope, export_unfiltered = options_map[scope_label]
    active = combos_all if (export_unfiltered and combos_all is not None) else combos

    scope_participant: Optional[str] = None
    scope_trial: Optional[str] = None
    scope_text: Optional[str] = None
    text_col = (
        "unique_text_id"
        if "unique_text_id" in active.columns
        else ("text_id" if "text_id" in active.columns else None)
    )

    if scope == "trial" and not active.empty:
        participants = sorted(active["participant_id"].dropna().astype(str).unique())
        scope_participant = st.selectbox(
            "Participant", options=participants, key=f"{key_prefix}_scope_pid"
        )
        trials_for_pid = (
            active.loc[
                active["participant_id"].astype(str) == str(scope_participant),
                "trial_id",
            ]
            .astype(str)
            .unique()
        )
        scope_trial = st.selectbox(
            "Trial", options=sorted(trials_for_pid), key=f"{key_prefix}_scope_trial"
        )
    elif scope == "participant" and not active.empty:
        participants = sorted(active["participant_id"].dropna().astype(str).unique())
        scope_participant = st.selectbox(
            "Participant", options=participants, key=f"{key_prefix}_scope_pid"
        )
    elif scope == "text" and not active.empty:
        if text_col is None:
            st.info("No text id is available in this dataset.")
        else:
            texts = sorted(active[text_col].dropna().astype(str).unique())
            scope_text = st.selectbox(
                "Text", options=texts, key=f"{key_prefix}_scope_text"
            )

    # Close the Scope section with a live count of what will be exported.
    n_export = len(
        _scope_frame(active, scope, scope_participant, scope_trial, scope_text)
    )
    n_total = len(combos_all) if combos_all is not None else len(combos)
    st.caption(f"**{n_export:,}** of **{n_total:,}** trials will be exported.")

    return scope, scope_participant, scope_trial, scope_text, export_unfiltered


def render_export_options(
    st_module,
    combos: pd.DataFrame,
    key_prefix: str = "export",
    combos_all: Optional[pd.DataFrame] = None,
) -> ExportOptions:
    """Render the bulk-export options UI and return a populated ExportOptions.

    ``combos`` is the currently filtered trial pool; ``combos_all`` (when given)
    is the whole loaded dataset. Picking the "All" scope switches the scope
    picker — and the export itself — to ``combos_all`` so the sidebar filters
    are ignored.
    """
    st = st_module
    # No expander — the options are always displayed (TODO 2.1).
    with st.container():
        st.markdown(
            "Choose which trials to include and which artifacts to bundle. "
            "Everything is packaged into a single zip you can download."
        )

        st.markdown("##### Scope")
        # The whole-dataset choice now lives inside the scope radio (TODO 1).
        (
            scope,
            scope_pid,
            scope_trial,
            scope_text,
            export_unfiltered,
        ) = _render_scope_picker(st, combos, key_prefix, combos_all=combos_all)

        st.markdown("##### Figures")
        # One checkbox per row, vector-first (TODO 3); PDF + Config on by
        # default (TODO 2).
        include_pdf = st.checkbox("PDF (vector)", value=True, key=f"{key_prefix}_pdf")
        include_svg = st.checkbox("SVG (vector)", value=False, key=f"{key_prefix}_svg")
        include_png = st.checkbox("PNG (raster)", value=False, key=f"{key_prefix}_png")
        # Only surface the scale stepper when PNG is on, and keep it narrow —
        # it's a single small number, not a full-width control.
        if include_png:
            scale_col, _ = st.columns([1, 4])
            png_scale = scale_col.number_input(
                "PNG scale",
                min_value=1,
                max_value=4,
                value=2,
                key=f"{key_prefix}_scale",
                help="Higher → better quality and larger files. 1 = 1×, 2 = retina, 4 = poster.",
            )
        else:
            png_scale = int(st.session_state.get(f"{key_prefix}_scale", 2))
        st.caption(
            "The plot config is a JSON snapshot of every plot setting (layers, "
            "colors, sizing, text scaling) — bundle it to reproduce or restore "
            "these exact figures later."
        )
        include_plot_config = st.checkbox("Config", value=True, key=f"{key_prefix}_cfg")

        st.markdown("##### Tabular data")
        include_fixations = st.checkbox(
            "Per-trial fixations", value=False, key=f"{key_prefix}_fix"
        )
        include_measures = st.checkbox(
            "Per-trial word measures (FFD/FPRT/RPD/TFD/...)",
            value=False,
            key=f"{key_prefix}_mes",
        )
        include_mega_table = st.checkbox(
            "Aggregated mega-table across selected trials",
            value=False,
            key=f"{key_prefix}_mega",
        )
        any_table = include_fixations or include_measures or include_mega_table
        table_format = st.radio(
            "Table format",
            options=["csv", "parquet", "both"],
            index=0,
            key=f"{key_prefix}_fmt",
            horizontal=True,
            disabled=not any_table,
            help=(
                "Tick at least one tabular option above to enable this."
                if not any_table
                else None
            ),
        )

    return ExportOptions(
        include_png=include_png,
        include_svg=include_svg,
        include_pdf=include_pdf,
        include_plot_config=include_plot_config,
        include_fixations=include_fixations,
        include_measures=include_measures,
        include_mega_table=include_mega_table,
        table_format=table_format,
        png_scale=int(png_scale),
        export_unfiltered=export_unfiltered,
        scope=scope,
        scope_participant=scope_pid,
        scope_trial=scope_trial,
        scope_text=scope_text,
    )


def _scope_frame(
    combos: pd.DataFrame,
    scope: str,
    scope_participant: Optional[str],
    scope_trial: Optional[str],
    scope_text: Optional[str],
) -> pd.DataFrame:
    """Filter combos to the chosen scope (pure helper, no ExportOptions needed)."""
    if scope == "trial" and scope_participant and scope_trial:
        return combos[
            (combos["participant_id"].astype(str) == str(scope_participant))
            & (combos["trial_id"].astype(str) == str(scope_trial))
        ]
    if scope == "participant" and scope_participant:
        return combos[combos["participant_id"].astype(str) == str(scope_participant)]
    if scope == "text" and scope_text:
        text_col = (
            "unique_text_id"
            if "unique_text_id" in combos.columns
            else ("text_id" if "text_id" in combos.columns else None)
        )
        if text_col is None:
            return combos
        return combos[combos[text_col].astype(str) == str(scope_text)]
    return combos


def _apply_scope(combos: pd.DataFrame, options: ExportOptions) -> pd.DataFrame:
    """Filter combos according to options.scope."""
    return _scope_frame(
        combos,
        options.scope,
        options.scope_participant,
        options.scope_trial,
        options.scope_text,
    )


def bulk_export(
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
    settings: dict,
    options: ExportOptions,
    progress_callback=None,
) -> tuple[bytes, ExportProgress]:
    """Build a zip archive of selected artifacts and return its bytes.

    progress_callback (if given) is invoked with an ExportProgress after every
    trial so the UI can update a progress bar.
    """
    combos = _apply_scope(combos, options)
    progress = ExportProgress(total_trials=len(combos))
    buf = io.BytesIO()
    zf = zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED)

    mega_fixations: list[pd.DataFrame] = []
    mega_measures: list[pd.DataFrame] = []

    readme_lines = [
        "# Bulk export",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        f"Authors: {CITATION['authors']}",
        f"Tool: {CITATION['title']}",
        "",
        "## Layout",
        "- `per_trial/<participant>__<trial>/` holds artifacts for each trial.",
        "- `aggregate/` holds long-form tables across every trial in this run.",
        "",
        "## Data dictionary",
        "Canonical column names from the visualization tool:",
        "- participant_id, trial_id, text_id, word_id",
        "- x, y, width, height (word bounding boxes in screen px)",
        "- x, y, duration_ms, timestamp_ms (fixations)",
        "- first_fixation_ms (FFD), first_pass_gaze_duration_ms (FPRT / gaze duration)",
        "- regression_path_duration_ms (RPD / go-past)",
        "- total_fixation_duration_ms (TFD / dwell), n_fixations",
        "- skip_flag, regression_in_flag, regression_out_flag",
        "",
        f"Demo corpus note: {CITATION['corpus_note']}",
    ]
    zf.writestr("README.md", "\n".join(readme_lines))

    # One warm Kaleido browser for every trial's figure (see _figure_renderer)
    # instead of cold-starting Chrome on each render.
    figure_formats = options.figure_formats()
    with _figure_renderer(bool(figure_formats)) as render_figure:
        for combo in combos.itertuples(index=False):
            participant = getattr(combo, "participant_id")
            trial = getattr(combo, "trial_id")
            slug = f"{_safe_id(participant)}__{_safe_id(trial)}"
            prefix = f"per_trial/{slug}/"

            trial_words = words[
                (words["participant_id"] == participant) & (words["trial_id"] == trial)
            ]
            trial_fix = fixations[
                (fixations["participant_id"] == participant)
                & (fixations["trial_id"] == trial)
            ]

            if trial_words.empty or trial_fix.empty:
                progress.finished_trials += 1
                progress.errors.append(f"{slug}: empty data, skipped")
                if progress_callback:
                    progress_callback(progress)
                continue

            if figure_formats:
                try:
                    fig = make_scanpath_figure(
                        trial_words,
                        trial_fix,
                        canvas_width=int(canvas_width),
                        canvas_height=int(canvas_height),
                        base_font_size=int(base_font_size),
                        font_family=font_family,
                        x_field=x_field,
                        y_field=y_field,
                        show_words=settings.get("show_words", True),
                        show_word_labels=settings.get("show_word_labels", True),
                        show_fixations=settings.get("show_fixations", True),
                        show_order=settings.get("show_order", True),
                        show_saccades=settings.get("show_saccades", True),
                        show_saccade_arrows=settings.get("show_saccade_arrows", False),
                        show_heatmap=settings.get("show_heatmap", False),
                        heatmap_style=settings.get("heatmap_style", "Word boxes"),
                        color_by=settings.get("color_by", "duration_ms"),
                        heatmap_metric=settings.get("heatmap_metric"),
                        marker_size_range=tuple(
                            settings.get("marker_size_range", (8, 24))
                        ),
                        order_font_size=int(settings.get("order_font_size", 10)),
                        order_font_color=settings.get("order_font_color", "#111111"),
                        show_colorbars=settings.get("show_colorbars", False),
                        fixation_color_range=settings.get("fixation_color_range"),
                        heatmap_range=settings.get("heatmap_range"),
                        fixation_colorscale=settings.get(
                            "fixation_colorscale", "Blues"
                        ),
                        heatmap_colorscale=settings.get(
                            "heatmap_colorscale", "Oranges"
                        ),
                        background_color=settings.get("background_color"),
                        color_by_line=settings.get("color_by_line", False),
                        highlight_out_of_text=settings.get(
                            "highlight_out_of_text", False
                        ),
                        line_spacing=settings.get("line_spacing", DEFAULT_LINE_SPACING),
                        scale_text_to_boxes=settings.get("scale_text_to_boxes", True),
                    )
                    # Render at the figure's own fitted size (not the raw
                    # monitor canvas) so the exported reading text matches the
                    # on-screen scale.
                    out_w = int(fig.layout.width or canvas_width)
                    out_h = int(fig.layout.height or canvas_height)
                    for fmt in figure_formats:
                        scale = options.png_scale if fmt == "png" else 1
                        data = render_figure(fig, fmt, out_w, out_h, scale)
                        zf.writestr(f"{prefix}figure.{fmt}", data)
                        progress.bytes_written += len(data)
                except Exception as exc:
                    progress.errors.append(f"{slug}: figure export failed ({exc})")

            if options.include_plot_config:
                cfg = _plot_config_dict(
                    participant,
                    trial,
                    canvas_width,
                    canvas_height,
                    x_field,
                    y_field,
                    settings,
                )
                data = json.dumps(cfg, indent=2).encode("utf-8")
                zf.writestr(f"{prefix}plot_config.json", data)
                progress.bytes_written += len(data)

            per_trial_measures = (
                compute_word_metrics(trial_words, trial_fix)
                if options.include_measures or options.include_mega_table
                else None
            )

            for fmt in options.table_formats():
                if options.include_fixations:
                    progress.bytes_written += _write_table(
                        zf, f"{prefix}fixations.{fmt}", trial_fix, fmt
                    )
                if options.include_measures and per_trial_measures is not None:
                    progress.bytes_written += _write_table(
                        zf, f"{prefix}measures.{fmt}", per_trial_measures, fmt
                    )

            if options.include_mega_table:
                mega_fixations.append(trial_fix)
                if per_trial_measures is not None:
                    mega_measures.append(per_trial_measures)

            progress.finished_trials += 1
            if progress_callback:
                progress_callback(progress)

    if options.include_mega_table and (mega_fixations or mega_measures):
        for fmt in options.table_formats():
            if mega_fixations:
                progress.bytes_written += _write_table(
                    zf,
                    f"aggregate/all_fixations.{fmt}",
                    pd.concat(mega_fixations, ignore_index=True),
                    fmt,
                )
            if mega_measures:
                progress.bytes_written += _write_table(
                    zf,
                    f"aggregate/all_measures.{fmt}",
                    pd.concat(mega_measures, ignore_index=True),
                    fmt,
                )

    zf.close()
    buf.seek(0)
    return buf.getvalue(), progress
