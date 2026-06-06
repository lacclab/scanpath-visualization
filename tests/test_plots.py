"""Tests for plots.py module."""

import pandas as pd
import plotly.graph_objects as go
import pytest

from scanpath_visualization_app.plots import (
    _width_fit_font,
    _word_label_font_px,
    animation_playback_ms,
    build_word_boxes,
    make_comparison_figure,
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

    def test_slider_declutters_long_reading_but_keeps_elapsed(
        self, normalized_words_df
    ):
        # A long reading must not draw a tick + time label per frame (illegible
        # smear). Every step keeps its real time label (so the "Elapsed" readout
        # updates on every frame and stays frame-accurate), while the per-step
        # ticks and labels are hidden — the readout is the one time display.
        n = 60
        fixations = pd.DataFrame(
            {
                "participant_id": ["p1"] * n,
                "trial_id": ["t1"] * n,
                "x": [100 + (i % 3) * 50 for i in range(n)],
                "y": [50] * n,
                "duration_ms": [100] * n,
                "timestamp_ms": list(range(0, n * 100, 100)),
                "word_id": [1] * n,
                "order_in_trial": list(range(1, n + 1)),
                "pass_index": [1] * n,
            }
        )
        fig = make_scanpath_animation(
            normalized_words_df,
            fixations,
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
        )
        slider = fig.layout.sliders[0]
        assert len(slider.steps) == n  # every frame scrubbable
        # Every step labelled -> the Elapsed readout shows a time at any position.
        assert all(s.label for s in slider.steps)
        assert slider.currentvalue.visible
        # Tick ruler hidden, and per-step labels drawn transparent.
        assert slider.ticklen == 0
        assert slider.minorticklen == 0
        assert "0)" in slider.font.color or "rgba" in str(slider.font.color)


class TestDualScanpathAnimation:
    """Tests for the dual path of make_scanpath_animation (the fixations_b arg)."""

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
        fig = make_scanpath_animation(
            normalized_words_df,
            normalized_fixations_df,
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
            playback_speed=1.0,
            fixations_b=self._second_fixations(),
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
        fig = make_scanpath_animation(
            normalized_words_df,
            fix_a,
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
            fixations_b=fix_a.iloc[:1].copy(),  # single fixation at t=0
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
        fig = make_scanpath_animation(
            normalized_words_df,
            normalized_fixations_df,
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
            fixations_b=normalized_fixations_df,
        )
        assert len(fig.frames) == len(normalized_fixations_df)

    def test_dual_animation_one_empty_falls_back(
        self, normalized_words_df, normalized_fixations_df
    ):
        # An empty second scanpath degrades to the single replay (no legend).
        fig = make_scanpath_animation(
            normalized_words_df,
            normalized_fixations_df,
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
            fixations_b=pd.DataFrame(),
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == len(normalized_fixations_df)
        assert [t for t in fig.data if t.showlegend] == []

    def test_dual_animation_both_empty(self, normalized_words_df):
        fig = make_scanpath_animation(
            normalized_words_df,
            pd.DataFrame(),
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
            fixations_b=pd.DataFrame(),
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 0


class TestAnimationPlaybackTiming:
    """The side panel must quote the *actual* animation runtime."""

    def test_playback_ms_matches_play_button(
        self, normalized_words_df, normalized_fixations_df
    ):
        # animation_playback_ms must equal what Play actually runs: Play advances
        # all frames at a single frame-duration, so runtime == n_frames * that.
        speed = 2.0
        fig = make_scanpath_animation(
            normalized_words_df,
            normalized_fixations_df,
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
            playback_speed=speed,
        )
        play_btn = fig.layout.updatemenus[0].buttons[0]
        frame_ms = play_btn.args[1]["frame"]["duration"]
        expected = len(fig.frames) * frame_ms
        _span, playback_ms = animation_playback_ms([normalized_fixations_df], speed)
        assert playback_ms == expected

    def test_playback_ms_empty(self):
        assert animation_playback_ms([], 1.0) == (0.0, 0.0)

    def test_frame_floor_clamps_tiny_gaps(self, normalized_words_df):
        # Gaps below the frame floor are clamped up so frames stay renderable
        # (browsers cap ~60fps); the Play frame duration is the floor itself.
        from scanpath_visualization_app.plots import _ANIM_MIN_FRAME_MS

        fix = pd.DataFrame(
            {
                "participant_id": ["p1", "p1", "p1"],
                "trial_id": ["t1", "t1", "t1"],
                "x": [100, 200, 300],
                "y": [50, 50, 50],
                "duration_ms": [5, 5, 5],
                "timestamp_ms": [0, 10, 20],  # 10 ms gaps, below the floor
                "order_in_trial": [1, 2, 3],
            }
        )
        fig = make_scanpath_animation(
            normalized_words_df,
            fix,
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
            playback_speed=1.0,
        )
        play_ms = fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"]
        assert play_ms == _ANIM_MIN_FRAME_MS

    def test_fake_index_timestamps_fall_back_to_durations(self, normalized_words_df):
        # data.normalize_fixations synthesizes timestamp_ms = 0,1,2,... when the
        # source has no timestamp column. Those row indices must NOT be read as
        # milliseconds; the clock falls back to fixation durations laid out
        # back-to-back, so reading time ~ sum(durations), not a couple of ms.
        fix = pd.DataFrame(
            {
                "participant_id": ["p1", "p1", "p1"],
                "trial_id": ["t1", "t1", "t1"],
                "x": [120, 220, 320],
                "y": [70, 70, 70],
                "duration_ms": [200, 250, 180],
                "timestamp_ms": [0, 1, 2],  # row-index sentinel, not real ms
                "order_in_trial": [1, 2, 3],
            }
        )
        span_ms, _playback = animation_playback_ms([fix], 1.0)
        assert span_ms == 630  # 200+250+180, NOT ~2 ms

    def test_single_mode_color_is_canonical_from_b_slot(
        self, normalized_words_df, normalized_fixations_df
    ):
        # Degenerate direct call: only the second slot is populated. The lone
        # trail must still be the canonical single-replay blue, never the red
        # "B" colour.
        from scanpath_visualization_app.constants import COMPARISON_PALETTE

        fig = make_scanpath_animation(
            normalized_words_df,
            pd.DataFrame(),
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
            fixations_b=normalized_fixations_df,
        )
        marker_colors = [
            t.marker.color
            for t in fig.data
            if t.marker is not None and isinstance(t.marker.color, str)
        ]
        assert COMPARISON_PALETTE[0] in marker_colors
        assert COMPARISON_PALETTE[1] not in marker_colors

    def test_animation_transitions_are_zero(
        self, normalized_words_df, normalized_fixations_df
    ):
        # Zero transition = labels/markers appear on their fixation instead of
        # gliding in from the corner, and the runtime stays exact.
        fig = make_scanpath_animation(
            normalized_words_df,
            normalized_fixations_df,
            canvas_width=800,
            canvas_height=600,
            base_font_size=12,
            font_family="Arial",
            show_order=True,
        )
        play_btn = fig.layout.updatemenus[0].buttons[0]
        assert play_btn.args[1]["transition"]["duration"] == 0
        assert fig.layout.sliders[0].transition.duration == 0


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


class TestTrueToScaleText:
    """Word-label text is sized true-to-scale from the box geometry.

    The reading text must track the word boxes (so it always fills the real
    line slot) instead of being a fixed screen-pixel size. See
    `plots._word_label_font_px`.
    """

    def _label_size(self, fig):
        """Pixel size of the word-label text trace in a scanpath figure."""
        trace = next(t for t in fig.data if t.name == "words")
        return float(trace.textfont.size)

    def _figure(self, words, fixations, **overrides):
        kwargs = dict(
            canvas_width=800,
            canvas_height=600,
            base_font_size=16,
            font_family="monospace",
            x_field="x",
            y_field="y",
            show_words=True,
            show_word_labels=True,
            show_fixations=False,
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
        kwargs.update(overrides)
        return make_scanpath_figure(words, fixations, **kwargs)

    # -- _word_label_font_px unit behaviour -------------------------------

    def test_autofit_uses_box_height_over_line_spacing(self, normalized_words_df):
        # 50px boxes; line_spacing 3 budgets 50/3 height. width_fit (5-char
        # words in 50px boxes) is the tighter bound here, so it wins — but the
        # result is always <= the height budget and scales linearly with `scale`.
        font = _word_label_font_px(
            normalized_words_df,
            scale=1.0,
            line_spacing=3.0,
            manual_font_px=99,
            scale_text_to_boxes=True,
        )
        assert 0 < font <= 50 / 3 + 1e-6
        # Linear in scale.
        font2 = _word_label_font_px(
            normalized_words_df,
            scale=2.0,
            line_spacing=3.0,
            manual_font_px=99,
            scale_text_to_boxes=True,
        )
        assert font2 == pytest.approx(2 * font)

    def test_autofit_shrinks_with_larger_line_spacing(self, normalized_words_df):
        # Past the point where the height budget is the binding constraint,
        # a bigger line spacing yields strictly smaller text.
        big = _word_label_font_px(
            normalized_words_df,
            scale=1.0,
            line_spacing=3.0,
            manual_font_px=0,
            scale_text_to_boxes=True,
        )
        bigger = _word_label_font_px(
            normalized_words_df,
            scale=1.0,
            line_spacing=10.0,
            manual_font_px=0,
            scale_text_to_boxes=True,
        )
        assert bigger < big
        assert bigger == pytest.approx(50 / 10)  # height budget binds

    def test_manual_mode_is_real_font_times_scale(self, normalized_words_df):
        # scale_text_to_boxes off -> manual font (monitor px) * display scale,
        # independent of line_spacing.
        font = _word_label_font_px(
            normalized_words_df,
            scale=0.5,
            line_spacing=3.0,
            manual_font_px=20,
            scale_text_to_boxes=False,
        )
        assert font == pytest.approx(10.0)

    def test_falls_back_to_manual_without_boxes(self):
        empty = pd.DataFrame()
        font = _word_label_font_px(
            empty,
            scale=0.5,
            line_spacing=3.0,
            manual_font_px=20,
            scale_text_to_boxes=True,
        )
        assert font == pytest.approx(10.0)

    def test_width_fit_recovers_per_char_advance(self, normalized_words_df):
        # 5-char words in 50px boxes -> 10px/char advance; /0.6 monospace aspect.
        wf = _width_fit_font(normalized_words_df)
        assert wf == pytest.approx(10 / 0.6 * 0.92, rel=1e-6)

    # -- integration through the figure builder ---------------------------

    def test_label_font_tracks_line_spacing(
        self, normalized_words_df, normalized_fixations_df
    ):
        small_spacing = self._figure(
            normalized_words_df, normalized_fixations_df, line_spacing=3.0
        )
        large_spacing = self._figure(
            normalized_words_df, normalized_fixations_df, line_spacing=10.0
        )
        assert self._label_size(large_spacing) < self._label_size(small_spacing)

    def test_label_font_independent_of_line_spacing_when_manual(
        self, normalized_words_df, normalized_fixations_df
    ):
        a = self._figure(
            normalized_words_df,
            normalized_fixations_df,
            scale_text_to_boxes=False,
            line_spacing=3.0,
        )
        b = self._figure(
            normalized_words_df,
            normalized_fixations_df,
            scale_text_to_boxes=False,
            line_spacing=10.0,
        )
        assert self._label_size(a) == pytest.approx(self._label_size(b))
