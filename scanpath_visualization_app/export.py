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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd

from .constants import CITATION
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


def _figure_bytes(fig, fmt: str, width: int, height: int, scale: int) -> bytes:
    return fig.to_image(format=fmt, width=int(width), height=int(height), scale=scale)


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
            "heatmap": settings.get("show_heatmap"),
            "raw_gaze": settings.get("show_raw_gaze"),
        },
        "coloring": {
            "color_by": settings.get("color_by"),
            "heatmap_metric": settings.get("heatmap_metric"),
            "fixation_colorscale": settings.get("fixation_colorscale"),
            "heatmap_colorscale": settings.get("heatmap_colorscale"),
        },
        "sizing": {
            "marker_size_range": list(settings.get("marker_size_range", [])),
            "order_font_size": settings.get("order_font_size"),
        },
    }


_SCOPE_LABELS = {
    "all": "All filtered trials",
    "trial": "A single trial",
    "participant": "All trials of one participant",
    "text": "All trials of one text",
}


def _render_scope_picker(
    st, combos: pd.DataFrame, key_prefix: str
) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Render the scope radio + dependent selectors, return (scope, pid, trial, text)."""
    scope_choices = list(_SCOPE_LABELS.values())
    scope_label = st.radio(
        "Trials to include",
        options=scope_choices,
        index=0,
        key=f"{key_prefix}_scope",
        horizontal=True,
        help="Limit the export to a subset of the currently filtered trials.",
    )
    scope = next(k for k, v in _SCOPE_LABELS.items() if v == scope_label)

    scope_participant: Optional[str] = None
    scope_trial: Optional[str] = None
    scope_text: Optional[str] = None
    text_col = (
        "unique_paragraph_id"
        if "unique_paragraph_id" in combos.columns
        else ("paragraph_id" if "paragraph_id" in combos.columns else None)
    )

    if scope == "trial" and not combos.empty:
        participants = sorted(combos["participant_id"].dropna().astype(str).unique())
        scope_participant = st.selectbox(
            "Participant", options=participants, key=f"{key_prefix}_scope_pid"
        )
        trials_for_pid = (
            combos.loc[
                combos["participant_id"].astype(str) == str(scope_participant),
                "trial_id",
            ]
            .astype(str)
            .unique()
        )
        scope_trial = st.selectbox(
            "Trial", options=sorted(trials_for_pid), key=f"{key_prefix}_scope_trial"
        )
    elif scope == "participant" and not combos.empty:
        participants = sorted(combos["participant_id"].dropna().astype(str).unique())
        scope_participant = st.selectbox(
            "Participant", options=participants, key=f"{key_prefix}_scope_pid"
        )
    elif scope == "text" and not combos.empty:
        if text_col is None:
            st.info("No text/paragraph id is available in this dataset.")
        else:
            texts = sorted(combos[text_col].dropna().astype(str).unique())
            scope_text = st.selectbox(
                "Text", options=texts, key=f"{key_prefix}_scope_text"
            )

    return scope, scope_participant, scope_trial, scope_text


def render_export_options(
    st_module, combos: pd.DataFrame, key_prefix: str = "export"
) -> ExportOptions:
    """Render the bulk-export options UI and return a populated ExportOptions."""
    st = st_module
    with st.expander("Bulk export options", expanded=False):
        st.markdown(
            "Choose which trials to include and which artifacts to bundle. "
            "Everything is packaged into a single zip you can download."
        )

        st.markdown("##### Scope")
        scope, scope_pid, scope_trial, scope_text = _render_scope_picker(
            st, combos, key_prefix
        )

        st.markdown("##### Figures")
        fig_cols = st.columns([1.3, 1, 1, 1.3])
        with fig_cols[0]:
            include_png = st.checkbox(
                "PNG (raster)", value=True, key=f"{key_prefix}_png"
            )
            png_scale = st.number_input(
                "PNG scale",
                min_value=1,
                max_value=4,
                value=2,
                key=f"{key_prefix}_scale",
                help="Higher → better quality and larger files. 1 = 1×, 2 = retina, 4 = poster.",
                disabled=not include_png,
            )
        with fig_cols[1]:
            include_svg = st.checkbox(
                "SVG (vector)", value=True, key=f"{key_prefix}_svg"
            )
        with fig_cols[2]:
            include_pdf = st.checkbox(
                "PDF (vector)", value=False, key=f"{key_prefix}_pdf"
            )
        with fig_cols[3]:
            include_plot_config = st.checkbox(
                "Plot config JSON", value=True, key=f"{key_prefix}_cfg"
            )

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
        scope=scope,
        scope_participant=scope_pid,
        scope_trial=scope_trial,
        scope_text=scope_text,
    )


def _apply_scope(combos: pd.DataFrame, options: ExportOptions) -> pd.DataFrame:
    """Filter combos according to options.scope."""
    if options.scope == "trial" and options.scope_participant and options.scope_trial:
        return combos[
            (combos["participant_id"].astype(str) == str(options.scope_participant))
            & (combos["trial_id"].astype(str) == str(options.scope_trial))
        ]
    if options.scope == "participant" and options.scope_participant:
        return combos[
            combos["participant_id"].astype(str) == str(options.scope_participant)
        ]
    if options.scope == "text" and options.scope_text:
        text_col = (
            "unique_paragraph_id"
            if "unique_paragraph_id" in combos.columns
            else ("paragraph_id" if "paragraph_id" in combos.columns else None)
        )
        if text_col is None:
            return combos
        return combos[combos[text_col].astype(str) == str(options.scope_text)]
    return combos


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
        "- participant_id, trial_id, paragraph_id, word_id",
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

        figure_formats = options.figure_formats()
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
                    show_heatmap=settings.get("show_heatmap", False),
                    color_by=settings.get("color_by", "duration_ms"),
                    heatmap_metric=settings.get("heatmap_metric"),
                    marker_size_range=tuple(settings.get("marker_size_range", (8, 24))),
                    order_font_size=int(settings.get("order_font_size", 10)),
                    order_font_color=settings.get("order_font_color", "#111111"),
                    show_colorbars=settings.get("show_colorbars", False),
                    fixation_color_range=settings.get("fixation_color_range"),
                    heatmap_range=settings.get("heatmap_range"),
                    fixation_colorscale=settings.get("fixation_colorscale", "Blues"),
                    heatmap_colorscale=settings.get("heatmap_colorscale", "Oranges"),
                    background_color=settings.get("background_color"),
                    color_by_line=settings.get("color_by_line", False),
                    highlight_out_of_text=settings.get("highlight_out_of_text", False),
                )
                for fmt in figure_formats:
                    scale = options.png_scale if fmt == "png" else 1
                    data = _figure_bytes(fig, fmt, canvas_width, canvas_height, scale)
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
