"""Tests for the bulk-export module."""

from __future__ import annotations

import io
import json
import zipfile

import pandas as pd
import pytest

from scanpath_studio.export import ExportOptions, bulk_export


@pytest.fixture
def minimal_combos():
    return pd.DataFrame(
        {
            "participant_id": ["p1", "p1"],
            "trial_id": ["t1", "t2"],
            "paragraph_id": ["para1", "para2"],
        }
    )


@pytest.fixture
def minimal_words():
    return pd.DataFrame(
        {
            "participant_id": ["p1"] * 4,
            "trial_id": ["t1", "t1", "t2", "t2"],
            "paragraph_id": ["para1", "para1", "para2", "para2"],
            "word_id": [1, 2, 1, 2],
            "text": ["the", "cat", "the", "dog"],
            "line_idx": [1, 1, 1, 1],
            "x": [100, 200, 100, 200],
            "y": [50, 50, 50, 50],
            "width": [80, 80, 80, 80],
            "height": [40, 40, 40, 40],
        }
    )


@pytest.fixture
def minimal_fixations():
    return pd.DataFrame(
        {
            "participant_id": ["p1"] * 4,
            "trial_id": ["t1", "t1", "t2", "t2"],
            "paragraph_id": ["para1", "para1", "para2", "para2"],
            "x": [140, 240, 140, 240],
            "y": [70, 70, 70, 70],
            "duration_ms": [200, 250, 220, 230],
            "timestamp_ms": [0, 200, 0, 220],
            "order_in_trial": [1, 2, 1, 2],
        }
    )


@pytest.fixture
def base_settings():
    return dict(
        show_words=True,
        show_word_labels=False,
        show_fixations=True,
        show_order=False,
        show_saccades=True,
        show_heatmap=False,
        color_by="duration_ms",
        heatmap_metric=None,
        marker_size_range=(8, 24),
        order_font_size=10,
        order_font_color="#111111",
        show_colorbars=False,
        fixation_color_range=None,
        heatmap_range=None,
        fixation_colorscale="Blues",
        heatmap_colorscale="Oranges",
    )


class TestBulkExport:
    def test_tabular_only_export(
        self, minimal_combos, minimal_words, minimal_fixations, base_settings
    ):
        opts = ExportOptions(
            include_png=False,
            include_svg=False,
            include_plot_config=True,
            include_fixations=True,
            include_measures=True,
            include_mega_table=True,
            table_format="csv",
        )
        zip_bytes, progress = bulk_export(
            minimal_combos,
            minimal_words,
            minimal_fixations,
            canvas_width=800,
            canvas_height=400,
            base_font_size=14,
            font_family="monospace",
            x_field="x",
            y_field="y",
            settings=base_settings,
            options=opts,
        )
        assert progress.finished_trials == 2
        assert progress.errors == []
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = set(zf.namelist())
            assert "README.md" in names
            assert "aggregate/all_fixations.csv" in names
            assert "aggregate/all_measures.csv" in names
            assert "per_trial/p1__t1/fixations.csv" in names
            assert "per_trial/p1__t1/measures.csv" in names
            assert "per_trial/p1__t1/plot_config.json" in names
            # No figures
            assert not any(n.endswith(".png") for n in names)
            assert not any(n.endswith(".svg") for n in names)
            cfg = json.loads(zf.read("per_trial/p1__t1/plot_config.json"))
            assert cfg["selection"]["participant_id"] == "p1"
            assert cfg["selection"]["trial_id"] == "t1"

    def test_parquet_format(
        self, minimal_combos, minimal_words, minimal_fixations, base_settings
    ):
        opts = ExportOptions(
            include_png=False,
            include_svg=False,
            include_plot_config=False,
            include_fixations=True,
            include_measures=False,
            include_mega_table=False,
            table_format="parquet",
        )
        zip_bytes, _ = bulk_export(
            minimal_combos,
            minimal_words,
            minimal_fixations,
            canvas_width=800,
            canvas_height=400,
            base_font_size=14,
            font_family="monospace",
            x_field="x",
            y_field="y",
            settings=base_settings,
            options=opts,
        )
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            assert "per_trial/p1__t1/fixations.parquet" in zf.namelist()
            assert "per_trial/p1__t1/fixations.csv" not in zf.namelist()

    def test_skips_empty_trial(self, minimal_words, minimal_fixations, base_settings):
        combos = pd.DataFrame(
            {
                "participant_id": ["p1", "p999"],
                "trial_id": ["t1", "tNONE"],
                "paragraph_id": ["para1", "paraX"],
            }
        )
        opts = ExportOptions(
            include_png=False,
            include_svg=False,
            include_plot_config=True,
            include_fixations=False,
            include_measures=False,
            include_mega_table=False,
            table_format="csv",
        )
        _, progress = bulk_export(
            combos,
            minimal_words,
            minimal_fixations,
            canvas_width=800,
            canvas_height=400,
            base_font_size=14,
            font_family="monospace",
            x_field="x",
            y_field="y",
            settings=base_settings,
            options=opts,
        )
        assert progress.finished_trials == 2
        # The unknown participant trial should be reported as an error
        assert any("p999__tNONE" in e for e in progress.errors)

    def test_progress_callback_invoked(
        self, minimal_combos, minimal_words, minimal_fixations, base_settings
    ):
        seen = []

        def cb(p):
            seen.append(p.finished_trials)

        opts = ExportOptions(
            include_png=False,
            include_svg=False,
            include_plot_config=False,
            include_fixations=False,
            include_measures=False,
            include_mega_table=False,
            table_format="csv",
        )
        bulk_export(
            minimal_combos,
            minimal_words,
            minimal_fixations,
            canvas_width=800,
            canvas_height=400,
            base_font_size=14,
            font_family="monospace",
            x_field="x",
            y_field="y",
            settings=base_settings,
            options=opts,
            progress_callback=cb,
        )
        assert seen == [1, 2]
