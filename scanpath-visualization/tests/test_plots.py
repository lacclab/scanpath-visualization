"""Tests for plots.py module."""

import pandas as pd
import pytest
import plotly.graph_objects as go

from scanpath_visualization_app.plots import (
    build_word_boxes,
    make_scanpath_figure,
    make_scanpath_animation,
    make_comparison_figure,
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

    def test_make_scanpath_figure_basic(self, normalized_words_df, normalized_fixations_df):
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
        assert fig.layout.width == 800
        assert fig.layout.height == 600

    def test_make_scanpath_figure_with_heatmap(self, normalized_words_df, normalized_fixations_df):
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

    def test_make_scanpath_figure_with_raw_gaze(self, normalized_words_df, normalized_fixations_df):
        raw_gaze = pd.DataFrame({
            "participant_id": ["p1", "p1"],
            "trial_id": ["t1", "t1"],
            "x": [120, 125],
            "y": [70, 75],
            "timestamp_ms": [0, 1],
        })
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

    def test_make_scanpath_animation_basic(self, normalized_words_df, normalized_fixations_df):
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

    def test_make_scanpath_animation_playback_speed(self, normalized_words_df, normalized_fixations_df):
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


class TestMakeComparisonFigure:
    """Tests for make_comparison_figure function."""

    def test_make_comparison_figure(self, normalized_words_df, normalized_fixations_df):
        # Create data for two trials
        words_multi = pd.concat([
            normalized_words_df.assign(participant_id="p1", trial_id="t1"),
            normalized_words_df.assign(participant_id="p2", trial_id="t1"),
        ])
        fixations_multi = pd.concat([
            normalized_fixations_df.assign(participant_id="p1", trial_id="t1"),
            normalized_fixations_df.assign(participant_id="p2", trial_id="t1"),
        ])
        
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
