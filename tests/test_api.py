"""Tests for the headless programmatic API (scanpath_studio.api)."""

import pandas as pd
import plotly.graph_objects as go
import pytest

import scanpath_studio as sps
from scanpath_studio import api
from scanpath_studio import data as data_module


@pytest.fixture(scope="module")
def sample():
    """Normalized bundled demo data, loaded once per module."""
    return sps.load_sample_data()


def test_load_sample_data_is_normalized(sample):
    words, fixations = sample
    for col in ("participant_id", "trial_id", "x", "y", "text"):
        assert col in words.columns
    for col in ("participant_id", "trial_id", "x", "y", "duration_ms"):
        assert col in fixations.columns


def test_top_level_exports():
    for name in (
        "load_scanpath_data",
        "load_sample_data",
        "list_trials",
        "compute_word_metrics",
        "plot_scanpath",
        "animate_scanpath",
        "save_figure",
    ):
        assert callable(getattr(sps, name))
    with pytest.raises(AttributeError):
        sps.does_not_exist


def test_list_trials(sample):
    combos = sps.list_trials(*sample)
    assert list(combos.columns) == ["participant_id", "trial_id"]
    assert len(combos) > 1
    assert not combos.duplicated().any()


def test_load_scanpath_data_from_files(tmp_path):
    words_raw, fix_raw = data_module.load_sample_data()
    words_path = tmp_path / "ia.csv"
    fix_path = tmp_path / "fixations.csv"
    words_raw.to_csv(words_path, index=False)
    fix_raw.to_csv(fix_path, index=False)

    words, fixations = sps.load_scanpath_data(words_path, fix_path)
    assert "trial_id" in words.columns
    assert "duration_ms" in fixations.columns


def test_load_scanpath_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        sps.load_scanpath_data(tmp_path / "nope.csv", tmp_path / "nope2.csv")


def test_load_scanpath_data_bad_schema():
    junk = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="schema problems"):
        sps.load_scanpath_data(junk, junk)


def test_plot_scanpath_returns_figure(sample):
    words, fixations = sample
    pid, tid = sps.list_trials(words, fixations).iloc[0]
    fig = sps.plot_scanpath(words, fixations, pid, tid, canvas_size=(2560, 1440))
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    # Word boxes arrive as layout shapes (canonical defaults show them).
    assert len(fig.layout.shapes) > 0


def test_plot_scanpath_overrides(sample):
    words, fixations = sample
    pid, tid = sps.list_trials(words, fixations).iloc[0]
    default_fig = sps.plot_scanpath(words, fixations, pid, tid)
    fig = sps.plot_scanpath(
        words,
        fixations,
        pid,
        tid,
        show_words=False,
        show_heatmap=False,
        heatmap_metric="counts",
    )
    assert isinstance(fig, go.Figure)
    # Word boxes gone: only the canvas border rect remains, vs one shape per
    # word (plus border) in the canonical default.
    assert len(fig.layout.shapes or ()) < len(default_fig.layout.shapes)


def test_plot_scanpath_axis_field_override(sample):
    # Regression: x_field/y_field used to collide with the explicitly passed
    # kwargs and raise "got multiple values for keyword argument".
    words, fixations = sample
    pid, tid = sps.list_trials(words, fixations).iloc[0]
    fig = sps.plot_scanpath(words, fixations, pid, tid, x_field="order_in_trial")
    assert isinstance(fig, go.Figure)


def test_plot_scanpath_filters_raw_gaze(sample):
    # Regression: raw_gaze used to be forwarded unfiltered, overlaying gaze
    # points from every other trial on the single-trial figure.
    words, fixations = sample
    combos = sps.list_trials(words, fixations)
    pid, tid = combos.iloc[0]
    other_pid, other_tid = combos.iloc[1]
    raw_gaze = pd.DataFrame(
        {
            "participant_id": [pid, pid, other_pid],
            "trial_id": [tid, tid, other_tid],
            "x": [100.0, 110.0, 5000.0],
            "y": [100.0, 105.0, 5000.0],
            "timestamp_ms": [0, 1, 0],
        }
    )
    fig = sps.plot_scanpath(words, fixations, pid, tid, raw_gaze=raw_gaze)
    raw_traces = [t for t in fig.data if t.name == "Raw gaze"]
    assert raw_traces and len(raw_traces[0].x) == 2


def test_animate_scanpath_rejects_static_only_options(sample):
    # Regression: static-only keys used to surface as an opaque TypeError.
    # (color_by no longer qualifies — the replay honours it like the static
    # figure; the heatmap overlay is still static-only.)
    words, fixations = sample
    pid, tid = sps.list_trials(words, fixations).iloc[0]
    with pytest.raises(ValueError, match="not supported by the animation"):
        sps.animate_scanpath(words, fixations, pid, tid, show_heatmap=True)


def test_resolve_trial_default_first(sample):
    words, fixations = sample
    combos = sps.list_trials(words, fixations)
    pid, tid = api._resolve_trial(words, fixations, None, None, default_first=True)
    assert (pid, tid) == tuple(combos.iloc[0])
    # default_first never excuses a nonexistent id.
    with pytest.raises(ValueError, match="No trial matches"):
        api._resolve_trial(words, fixations, None, "no_such_trial", default_first=True)


def test_dir_lists_lazy_exports():
    assert "plot_scanpath" in dir(sps)


def test_plot_scanpath_ambiguous_raises(sample):
    with pytest.raises(ValueError, match="Ambiguous"):
        sps.plot_scanpath(*sample)


def test_plot_scanpath_unknown_trial_raises(sample):
    with pytest.raises(ValueError, match="No trial matches"):
        sps.plot_scanpath(*sample, participant="nobody", trial="nothing")


def test_animate_scanpath_returns_frames(sample):
    words, fixations = sample
    pid, tid = sps.list_trials(words, fixations).iloc[0]
    fig = sps.animate_scanpath(words, fixations, pid, tid, canvas_size=(2560, 1440))
    assert isinstance(fig, go.Figure)
    assert len(fig.frames) > 0


def test_compute_word_metrics(sample):
    metrics = sps.compute_word_metrics(*sample)
    assert "total_fixation_duration_ms" in metrics.columns
    assert len(metrics) > 0


def test_save_figure_html(sample, tmp_path):
    words, fixations = sample
    pid, tid = sps.list_trials(words, fixations).iloc[0]
    fig = sps.plot_scanpath(words, fixations, pid, tid)
    out = sps.save_figure(fig, tmp_path / "fig.html")
    assert out.is_file()
    assert out.stat().st_size > 0


def test_save_figure_bad_extension(sample, tmp_path):
    fig = go.Figure()
    with pytest.raises(ValueError, match="Unsupported extension"):
        sps.save_figure(fig, tmp_path / "fig.docx")
