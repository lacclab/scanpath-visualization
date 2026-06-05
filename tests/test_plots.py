"""Tests for plots.py module."""

import pandas as pd
import plotly.graph_objects as go

from scanpath_visualization_app.plots import (
    build_word_boxes,
    make_comparison_figure,
    make_dual_scanpath_animation,
    make_scanpath_animation,
    make_scanpath_figure,
)


class TestBuildWordBoxes:
    """Tests for build_word_boxes function."""

    def test_build_word_boxes(self, normalized_words_df):
        shapes = build_word_boxes(normalized_words_df)
        assert len(shapes) == len(normalized_words_df)
        assert all(shape["type"] == "rect" for shape in shapes)
        assert all("x0" in shape for shape in shapes)
        assert all("y0" in shape for shape in shapes)

    def test_build_word_boxes_empty(self):
        empty_df = pd.DataFrame()
        shapes = build_word_boxes(empty_df)
        assert shapes == []


class TestMakeScanpathFigure:
    """Tests for make_scanpath_figure function."""

    def test_make_scanpath_figure_basic(
        self, normalized_words_df, normalized_fixations_df
    ):
        fig = make_scanpath_figure(
            normalized_words_df,
            normalized_fixations_df,
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
            x_field="x",
            y_field="y",
            show_words=True,
            show_word_labels=True,
            show_fixations=True,
            show_order=True,
            show_saccades=True,
            show_heatmap=False,
            color_by="duration_ms",
            heatmap_metric=None,
            marker_size_range=(8, 24),
            order_font_size=10,
            order_font_color="#000000",
            show_colorbars=False,
            fixation_color_range=None,
            heatmap_range=None,
        )
        assert isinstance(fig, go.Figure)
        # Figure now shrinks to the data's aspect ratio (see _fit_display_size);
        # dimensions are capped at the requested canvas, not pinned to it.
        assert 0 < fig.layout.width <= 800
        assert 0 < fig.layout.height <= 600

    def test_make_scanpath_figure_with_heatmap(
        self, normalized_words_df, normalized_fixations_df
    ):
        fig = make_scanpath_figure(
            normalized_words_df,
            normalized_fixations_df,
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
            x_field="x",
            y_field="y",
            show_words=True,
            show_word_labels=False,
            show_fixations=True,
            show_order=False,
            show_saccades=False,
            show_heatmap=True,
            color_by="duration_ms",
            heatmap_metric="duration_ms",
            marker_size_range=(8, 24),
            order_font_size=10,
            order_font_color="#000000",
            show_colorbars=True,
            fixation_color_range=None,
            heatmap_range=None,
        )
        assert isinstance(fig, go.Figure)

    def test_make_scanpath_figure_empty_fixations(self, normalized_words_df):
        empty_fixations = pd.DataFrame()
        fig = make_scanpath_figure(
            normalized_words_df,
            empty_fixations,
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
            x_field="x",
            y_field="y",
            show_words=True,
            show_word_labels=True,
            show_fixations=True,
            show_order=True,
            show_saccades=True,
            show_heatmap=False,
            color_by="duration_ms",
            heatmap_metric=None,
            marker_size_range=(8, 24),
            order_font_size=10,
            order_font_color="#000000",
            show_colorbars=False,
            fixation_color_range=None,
            heatmap_range=None,
        )
        assert isinstance(fig, go.Figure)

    def test_make_scanpath_figure_with_raw_gaze(
        self, normalized_words_df, normalized_fixations_df
    ):
        raw_gaze = pd.DataFrame(
            {
                "participant_id": ["p1", "p1"],
                "trial_id": ["t1", "t1"],
                "x": [120, 125],
                "y": [70, 75],
                "timestamp_ms": [0, 1],
            }
        )
        fig = make_scanpath_figure(
            normalized_words_df,
            normalized_fixations_df,
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
            x_field="x",
            y_field="y",
            show_words=True,
            show_word_labels=True,
            show_fixations=True,
            show_order=True,
            show_saccades=True,
            show_heatmap=False,
            color_by="duration_ms",
            heatmap_metric=None,
            marker_size_range=(8, 24),
            order_font_size=10,
            order_font_color="#000000",
            show_colorbars=False,
            fixation_color_range=None,
            heatmap_range=None,
            raw_gaze=raw_gaze,
            show_raw_gaze=True,
        )
        assert isinstance(fig, go.Figure)

    def test_make_scanpath_figure_non_spatial_axes(self, normalized_fixations_df):
        # Test with non-spatial axes (e.g., timestamp vs duration)
        empty_words = pd.DataFrame()
        fig = make_scanpath_figure(
            empty_words,
            normalized_fixations_df,
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
            x_field="timestamp_ms",
            y_field="duration_ms",
            show_words=False,
            show_word_labels=False,
            show_fixations=True,
            show_order=False,
            show_saccades=False,
            show_heatmap=False,
            color_by="duration_ms",
            heatmap_metric=None,
            marker_size_range=(8, 24),
            order_font_size=10,
            order_font_color="#000000",
            show_colorbars=False,
            fixation_color_range=None,
            heatmap_range=None,
        )
        assert isinstance(fig, go.Figure)


class TestMakeScanpathAnimation:
    """Tests for make_scanpath_animation function."""

    def test_make_scanpath_animation_basic(
        self, normalized_words_df, normalized_fixations_df
    ):
        fig = make_scanpath_animation(
            normalized_words_df,
            normalized_fixations_df,
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
            playback_speed=1.0,
            show_words=True,
            show_word_labels=True,
            show_saccades=True,
            show_order=True,
            marker_size_range=(8, 24),
            order_font_size=10,
            order_font_color="#000000",
        )
        assert isinstance(fig, go.Figure)
        assert hasattr(fig, "frames")
        assert len(fig.frames) == len(normalized_fixations_df)

    def test_make_scanpath_animation_empty_fixations(self, normalized_words_df):
        empty_fixations = pd.DataFrame()
        fig = make_scanpath_animation(
            normalized_words_df,
            empty_fixations,
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
        )
        assert isinstance(fig, go.Figure)

    def test_make_scanpath_animation_playback_speed(
        self, normalized_words_df, normalized_fixations_df
    ):
        fig = make_scanpath_animation(
            normalized_words_df,
            normalized_fixations_df,
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
            playback_speed=2.0,
        )
        assert isinstance(fig, go.Figure)


class TestMakeDualScanpathAnimation:
    """Tests for make_dual_scanpath_animation function."""

    @staticmethod
    def _second_fixations():
        # Onsets are recorded timestamps rebased to t=0: [0, 300]. Paired with
        # scanpath A's [0, 200, 450] the merged onset set is {0, 200, 300, 450}
        # → 4 frames.
        return pd.DataFrame(
            {
                "participant_id": ["p2", "p2"],
                "trial_id": ["t1", "t1"],
                "x": [130, 230],
                "y": [80, 80],
                "duration_ms": [300, 100],
                "timestamp_ms": [0, 300],
                "order_in_trial": [1, 2],
            }
        )

    def test_dual_animation_basic(self, normalized_words_df, normalized_fixations_df):
        fig = make_dual_scanpath_animation(
            normalized_words_df,
            normalized_fixations_df,
            self._second_fixations(),
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
            playback_speed=1.0,
        )
        assert isinstance(fig, go.Figure)
        assert hasattr(fig, "frames")
        # One frame per distinct fixation onset across both scanpaths.
        assert len(fig.frames) == 4
        # Both trails appear in the legend so the two readers are tellable apart.
        legend_names = [t.name for t in fig.data if t.showlegend]
        assert len(legend_names) == 2

    def test_dual_animation_uses_real_timestamps(self, normalized_words_df):
        # The shared clock must come from recorded timestamp_ms (rebased), NOT
        # cumulative durations — otherwise readings with saccade/blink gaps
        # desync. Here fixation 2 starts at t=1000ms but lasts only 100ms, so a
        # duration-based clock would place its onset at 100ms. The elapsed-time
        # slider label proves which clock is used.
        fix_a = pd.DataFrame(
            {
                "participant_id": ["p1", "p1"],
                "trial_id": ["t1", "t1"],
                "x": [120, 220],
                "y": [70, 70],
                "duration_ms": [100, 100],
                "timestamp_ms": [0, 1000],
                "order_in_trial": [1, 2],
            }
        )
        fix_b = fix_a.iloc[:1].copy()  # single fixation at t=0
        fig = make_dual_scanpath_animation(
            normalized_words_df,
            fix_a,
            fix_b,
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
        )
        labels = [s.label for s in fig.layout.sliders[0].steps]
        # Real-timestamp clock → onset of fixation 2 at 1.0s; a duration clock
        # would have shown 0.1s.
        assert labels == ["0.0s", "1.0s"], labels
        assert fig.layout.sliders[0].currentvalue.prefix == "Elapsed: "

    def test_dual_animation_identical_inputs(
        self, normalized_words_df, normalized_fixations_df
    ):
        # Identical scanpaths share onsets, so the merged frame count collapses
        # to a single scanpath's fixation count.
        fig = make_dual_scanpath_animation(
            normalized_words_df,
            normalized_fixations_df,
            normalized_fixations_df,
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
        )
        assert len(fig.frames) == len(normalized_fixations_df)

    def test_dual_animation_one_empty(
        self, normalized_words_df, normalized_fixations_df
    ):
        # A robust no-op for the empty side: still animates the populated one.
        fig = make_dual_scanpath_animation(
            normalized_words_df,
            normalized_fixations_df,
            pd.DataFrame(),
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == len(normalized_fixations_df)

    def test_dual_animation_both_empty(self, normalized_words_df):
        fig = make_dual_scanpath_animation(
            normalized_words_df,
            pd.DataFrame(),
            pd.DataFrame(),
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 0


class TestMakeComparisonFigure:
    """Tests for make_comparison_figure function."""

    def test_make_comparison_figure(self, normalized_words_df, normalized_fixations_df):
        # Create data for two trials
        words_multi = pd.concat(
            [
                normalized_words_df.assign(participant_id="p1", trial_id="t1"),
                normalized_words_df.assign(participant_id="p2", trial_id="t1"),
            ]
        )
        fixations_multi = pd.concat(
            [
                normalized_fixations_df.assign(participant_id="p1", trial_id="t1"),
                normalized_fixations_df.assign(participant_id="p2", trial_id="t1"),
            ]
        )

        fig = make_comparison_figure(
            words_multi,
            fixations_multi,
            trial_a=("p1", "t1"),
            trial_b=("p2", "t1"),
            canvas_width=800,
            canvas_height=600,
            font_family="Arial",
            base_font_size=12,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Two traces for two trials


class TestPlotEnhancements:
    """Background color, out-of-text highlight, and color-by-line options."""

    def _figure(self, words, fixations, **overrides):
        kwargs = dict(
            canvas_width=500,
            canvas_height=300,
            base_font_size=12,
            font_family="Arial",
            x_field="x",
            y_field="y",
            show_words=True,
            show_word_labels=True,
            show_fixations=True,
            show_order=True,
            show_saccades=True,
            show_heatmap=False,
            color_by="duration_ms",
            heatmap_metric=None,
            marker_size_range=(8, 24),
            order_font_size=10,
            order_font_color="#000000",
            show_colorbars=False,
            fixation_color_range=None,
            heatmap_range=None,
        )
        kwargs.update(overrides)
        return make_scanpath_figure(words, fixations, **kwargs)

    def test_background_color_applied(self, synthetic_words_df, synthetic_fixations_df):
        fig = self._figure(
            synthetic_words_df, synthetic_fixations_df, background_color="#bdbdbd"
        )
        assert fig.layout.plot_bgcolor == "#bdbdbd"
        assert fig.layout.paper_bgcolor == "#bdbdbd"

    def test_background_color_default_is_none(
        self, synthetic_words_df, synthetic_fixations_df
    ):
        # Default leaves the template's background untouched.
        fig = self._figure(synthetic_words_df, synthetic_fixations_df)
        assert fig.layout.plot_bgcolor is None

    def test_out_of_text_overlay_trace(
        self, synthetic_words_df, synthetic_fixations_df
    ):
        # The synthetic trial has exactly one out-of-text fixation.
        fig = self._figure(
            synthetic_words_df, synthetic_fixations_df, highlight_out_of_text=True
        )
        oot = [t for t in fig.data if t.name == "Out-of-text"]
        assert len(oot) == 1
        assert len(oot[0].x) == 1  # one off-text fixation

    def test_out_of_text_overlay_absent_by_default(
        self, synthetic_words_df, synthetic_fixations_df
    ):
        fig = self._figure(synthetic_words_df, synthetic_fixations_df)
        assert not any(t.name == "Out-of-text" for t in fig.data)

    def test_color_by_line_legend(self, synthetic_words_df, synthetic_fixations_df):
        # Two lines in the layout -> two "line:" legend entries.
        fig = self._figure(
            synthetic_words_df, synthetic_fixations_df, color_by_line=True
        )
        line_traces = [t for t in fig.data if str(t.name).startswith("line:")]
        assert len(line_traces) == 2
