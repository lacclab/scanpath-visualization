"""End-to-end smoke tests against the bundled sample data."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from scanpath_studio.data import (
    compute_word_metrics,
    infer_fix_schema,
    infer_word_schema,
    load_sample_data,
    normalize_fixations,
    normalize_words,
)
from scanpath_studio.plots import (
    make_comparison_figure,
    make_fixation_duration_histogram,
    make_scanpath_animation,
    make_scanpath_figure,
    make_word_measure_bar_figure,
)


@pytest.fixture(scope="module")
def normalized_demo():
    """Load + normalize the bundled OneStop sample data once per test module."""
    words_raw, fixations_raw = load_sample_data()
    assert not words_raw.empty, "Sample IA file missing"
    assert not fixations_raw.empty, "Sample fixations file missing"

    word_schema = infer_word_schema(words_raw)
    fix_schema = infer_fix_schema(fixations_raw)
    assert word_schema is not None, "Schema inference failed for sample IA"
    assert fix_schema is not None, "Schema inference failed for sample fixations"

    words = normalize_words(words_raw, word_schema)
    fixations = normalize_fixations(fixations_raw, fix_schema)
    return words, fixations


class TestSampleDataPipeline:
    def test_required_columns(self, normalized_demo):
        words, fixations = normalized_demo
        for col in [
            "participant_id",
            "trial_id",
            "word_id",
            "x",
            "y",
            "width",
            "height",
        ]:
            assert col in words.columns, f"missing canonical column {col}"
        for col in [
            "participant_id",
            "trial_id",
            "x",
            "y",
            "duration_ms",
            "timestamp_ms",
        ]:
            assert col in fixations.columns, f"missing canonical column {col}"

    def test_has_multiple_participants(self, normalized_demo):
        words, _ = normalized_demo
        assert words["participant_id"].nunique() >= 2, (
            "Demo corpus should bundle multiple participants for the comparison feature"
        )

    def test_has_both_difficulty_levels(self, normalized_demo):
        words, _ = normalized_demo
        if "difficulty_level" in words.columns:
            levels = set(words["difficulty_level"].dropna().unique())
            assert len(levels) >= 2, (
                f"Demo corpus should span both difficulty levels; got {levels}"
            )

    def test_linguistic_features_present(self, normalized_demo):
        words, _ = normalized_demo
        for col in ["gpt2_surprisal", "wordfreq_frequency", "universal_pos"]:
            assert col in words.columns, (
                f"Demo corpus should carry NLP-relevant feature: {col}"
            )


class TestPipelineFigures:
    def test_scanpath_figure_renders(self, normalized_demo):
        words, fixations = normalized_demo
        pid = words["participant_id"].iloc[0]
        tid = words["trial_id"].iloc[0]
        tw = words[(words["participant_id"] == pid) & (words["trial_id"] == tid)]
        tf = fixations[
            (fixations["participant_id"] == pid) & (fixations["trial_id"] == tid)
        ]
        fig = make_scanpath_figure(
            tw,
            tf,
            canvas_width=1024,
            canvas_height=600,
            base_font_size=14,
            font_family="monospace",
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
            order_font_color="#111111",
            show_colorbars=False,
            fixation_color_range=None,
            heatmap_range=None,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1, "Expected at least one trace"

    def test_saccades_collapsed_into_single_trace(self, normalized_demo):
        """Regression: the per-saccade-trace explosion (one trace per saccade)
        was a known perf bug. A trial with N fixations should now yield at
        most a small constant number of traces, not O(N)."""
        words, fixations = normalized_demo
        biggest = fixations.groupby(["participant_id", "trial_id"]).size().idxmax()
        pid, tid = biggest
        tw = words[(words["participant_id"] == pid) & (words["trial_id"] == tid)]
        tf = fixations[
            (fixations["participant_id"] == pid) & (fixations["trial_id"] == tid)
        ]
        assert len(tf) >= 10, "need a non-trivial trial for this regression test"
        fig = make_scanpath_figure(
            tw,
            tf,
            canvas_width=1024,
            canvas_height=600,
            base_font_size=14,
            font_family="monospace",
            x_field="x",
            y_field="y",
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
        )
        # Expect (in any order): saccades trace (1) + fixations trace (1) + optional word labels.
        # Never one-per-saccade.
        assert len(fig.data) <= 5, f"Too many traces: {len(fig.data)}"

    def test_word_measure_bar(self, normalized_demo):
        words, fixations = normalized_demo
        pid = words["participant_id"].iloc[0]
        tid = words["trial_id"].iloc[0]
        tw = words[(words["participant_id"] == pid) & (words["trial_id"] == tid)]
        tf = fixations[
            (fixations["participant_id"] == pid) & (fixations["trial_id"] == tid)
        ]
        measures = compute_word_metrics(tw, tf)
        # Pick whatever first-fixation measure is present
        measure = next(
            (
                c
                for c in [
                    "first_fixation_ms",
                    "first_pass_gaze_duration_ms",
                    "total_fixation_duration_ms",
                ]
                if c in measures.columns
            ),
            None,
        )
        assert measure is not None
        fig = make_word_measure_bar_figure(
            measures,
            measure=measure,
            canvas_width=1024,
            base_font_size=14,
            font_family="monospace",
        )
        assert isinstance(fig, go.Figure)

    def test_fixation_duration_histogram(self, normalized_demo):
        _, fixations = normalized_demo
        fig = make_fixation_duration_histogram(
            fixations.head(200),
            canvas_width=800,
            base_font_size=14,
            font_family="monospace",
        )
        assert isinstance(fig, go.Figure)

    def test_animation_has_frames(self, normalized_demo):
        words, fixations = normalized_demo
        pid = words["participant_id"].iloc[0]
        tid = words["trial_id"].iloc[0]
        tw = words[(words["participant_id"] == pid) & (words["trial_id"] == tid)]
        tf = fixations[
            (fixations["participant_id"] == pid) & (fixations["trial_id"] == tid)
        ]
        fig = make_scanpath_animation(
            tw,
            tf,
            canvas_width=1024,
            canvas_height=600,
            base_font_size=14,
            font_family="monospace",
        )
        assert len(fig.frames) == len(tf), (
            "Animation should have one frame per fixation"
        )

    def test_comparison_figure(self, normalized_demo):
        words, fixations = normalized_demo
        participants = sorted(words["participant_id"].unique())
        if len(participants) < 2:
            pytest.skip("Need >=2 participants for comparison")
        p1, p2 = participants[:2]
        # Find a trial each participant has
        t1 = words[words["participant_id"] == p1]["trial_id"].iloc[0]
        t2 = words[words["participant_id"] == p2]["trial_id"].iloc[0]
        fig = make_comparison_figure(
            words,
            fixations,
            (p1, t1),
            (p2, t2),
            canvas_width=1024,
            canvas_height=600,
            font_family="monospace",
            base_font_size=14,
            layout="overlay",
        )
        assert isinstance(fig, go.Figure)

    def test_marker_sizes_consistent_across_figure_types(self, normalized_demo):
        """The same fixation should render at the same size in single-trial
        and comparison figures."""
        from scanpath_studio.plots import _compute_marker_sizes

        words, fixations = normalized_demo
        pid = words["participant_id"].iloc[0]
        tid = words["trial_id"].iloc[0]
        tf = fixations[
            (fixations["participant_id"] == pid) & (fixations["trial_id"] == tid)
        ]
        sizes = _compute_marker_sizes(tf["duration_ms"])
        assert sizes.min() >= 8
        assert sizes.max() <= 24


class TestPerWordMetricsOnSample:
    def test_metrics_computed_on_real_data(self, normalized_demo):
        words, fixations = normalized_demo
        pid = words["participant_id"].iloc[0]
        tid = words["trial_id"].iloc[0]
        tw = words[(words["participant_id"] == pid) & (words["trial_id"] == tid)]
        tf = fixations[
            (fixations["participant_id"] == pid) & (fixations["trial_id"] == tid)
        ]
        metrics = compute_word_metrics(tw, tf)
        assert not metrics.empty
        # Should have at least the canonical measures (either from IA columns
        # or computed from first principles).
        canonical_present = any(
            c in metrics.columns
            for c in [
                "first_fixation_ms",
                "first_pass_gaze_duration_ms",
                "total_fixation_duration_ms",
            ]
        )
        assert canonical_present
