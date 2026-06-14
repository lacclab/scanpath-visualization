"""Restoring an uploaded plot config seeds the visualization widgets' state.

Exercises ``app._restore_plot_config`` / ``_apply_uploaded_plot_config`` — the
inverse of the config written by ``tabs._render_plot_config_expander`` — covering
the round-trip mapping, data-dependent validation, clamping, and tolerance of a
hand-edited / malformed upload (it should skip bad fields, never crash).
"""

from __future__ import annotations

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest


# AppTest execs each app function as a standalone script, so anything it needs
# (the fixtures frame below) must be built inline — module-level helpers aren't
# in scope. x/y/duration_ms/pass_index are numeric (axis + color-by options);
# trial_id doubles as unique_trial_id so the none/Trial selection branch runs.
_FIX_COLUMNS = {
    "participant_id": ["p1", "p1", "p2"],
    "trial_id": ["t1", "t2", "t1"],
    "unique_trial_id": ["t1", "t2", "t1"],
    "paragraph_id": ["pA", "pB", "pA"],
    "x": [1.0, 2.0, 3.0],
    "y": [4.0, 5.0, 6.0],
    "duration_ms": [100, 200, 300],
    "pass_index": [1, 1, 2],
}


def _restore_app():
    """Run ``_restore_plot_config`` over the fixed dataset + the seeded config."""
    import pandas as pd
    import streamlit as st

    from scanpath_studio.app import _restore_plot_config
    from scanpath_studio.utils import build_combo_options

    fixations = pd.DataFrame(st.session_state["_fix"])
    combos, _, _ = build_combo_options(fixations)
    applied, skipped = _restore_plot_config(
        st.session_state["_config"], combos, fixations
    )
    st.session_state["_applied"] = applied
    st.session_state["_skipped"] = skipped


def _apply_app():
    """Drive ``_apply_uploaded_plot_config`` through a fake uploaded file."""
    import pandas as pd
    import streamlit as st

    from scanpath_studio.app import _apply_uploaded_plot_config
    from scanpath_studio.utils import build_combo_options

    class _FakeUpload:
        def __init__(self, data: bytes):
            self._data = data
            self.name = "cfg.json"
            self.size = len(data)

        def getvalue(self) -> bytes:
            return self._data

    fixations = pd.DataFrame(st.session_state["_fix"])
    combos, _, _ = build_combo_options(fixations)
    st.session_state["plot_config_upload"] = _FakeUpload(st.session_state["_bytes"])
    _apply_uploaded_plot_config(combos, fixations)


def _run(app, **state):
    at = AppTest.from_function(app)
    at.session_state["_fix"] = _FIX_COLUMNS
    for key, value in state.items():
        at.session_state[key] = value
    at.run(timeout=20)
    assert not at.exception, at.exception
    return at


def _full_config() -> dict:
    """A config in the shape ``tabs._render_plot_config_expander`` writes."""
    return {
        "selection": {"participant_id": "p1", "trial_id": "t2"},
        "canvas_px": {"width": 1920, "height": 1080},
        "axes": {"x_field": "x", "y_field": "y"},
        "layers": {
            "words": False,
            "word_labels": True,
            "fixations": True,
            "order_labels": False,
            "saccades": True,
            "saccade_arrows": True,
            "heatmap": False,
            "raw_gaze": False,
        },
        "coloring": {
            "color_by": "pass_index",
            "heatmap_metric": "counts",
            "heatmap_style": "Interpolated",
            "show_colorbars": True,
            "fixation_range": [150.0, 250.0],
            "heatmap_range": [120.0, 280.0],
            "fixation_colorscale": "Viridis",
            "heatmap_colorscale": "Plasma",
        },
        "sizing": {
            "marker_size_range": [10, 30],
            "order_font_size": 40,
            "order_font_color": "#aa0000",
            "base_font_size": 28,
        },
    }


@pytest.mark.timeout(60)
class TestPlotConfigRestore:
    def test_round_trip_sets_all_widget_state(self):
        ss = _run(_restore_app, _config=_full_config()).session_state
        # layers
        assert ss["global_show_words"] is False
        assert ss["global_show_heatmap"] is False
        assert ss["global_show_saccade_arrows"] is True
        assert ss["global_show_order"] is False
        # coloring
        assert ss["global_heatmap_style"] == "Interpolated"
        assert ss["global_color_by"] == "pass_index"
        assert ss["global_heatmap_metric"] == "counts"
        assert ss["global_show_colorbars"] is True
        assert ss["global_fixation_colorscale"] == "Viridis"
        assert ss["global_heatmap_colorscale"] == "Plasma"
        assert ss["global_fixation_color_range"] == (150.0, 250.0)
        assert ss["global_heatmap_color_range"] == (120.0, 280.0)
        # sizing
        assert ss["global_marker_size_range"] == (10, 30)
        assert ss["global_order_font_size"] == 40
        assert ss["global_order_font_color"] == "#aa0000"
        assert ss["global_base_font_size"] == 28
        # canvas / axes
        assert ss["global_canvas_width"] == 1920
        assert ss["global_canvas_height"] == 1080
        assert ss["global_x_field"] == "x"
        assert ss["global_y_field"] == "y"
        # selection (none/Trial mode → single_trial_id holds the option value)
        assert ss["single_select_trial_mode"] == "Trial"
        assert ss["single_trial_id"] == "t2"
        assert ss["_skipped"] == []

    def test_restores_text_highlighting_and_annotations(self):
        # The merged "Save & restore" config (schema 2) also carries text
        # sizing, highlighting, and annotations — all re-applied on restore.
        config = dict(_full_config())
        config["schema"] = 2
        config["coloring"] = dict(config["coloring"], color_by="line")  # synthetic opt
        config["text"] = {
            "scale_text_to_boxes": False,
            "line_spacing": 2.5,
            "font_family": "Courier New",
        }
        config["highlighting"] = {
            "critical_span_style": "Mark border",
            "highlight_out_of_text": True,
            "background_color": "#222222",
        }
        config["annotations"] = [
            {
                "participant_id": "p1",
                "trial_id": "t2",
                "star": True,
                "tags": ["Review"],
                "note": "hi",
            }
        ]
        ss = _run(_restore_app, _config=config).session_state
        assert ss["global_color_by"] == "line"  # "line" is a valid option now
        assert ss["global_scale_text_to_boxes"] is False
        assert ss["global_line_spacing"] == 2.5
        assert ss["global_font_family"] == "Courier New"
        assert ss["global_critical_span_style"] == "Mark border"
        assert ss["global_highlight_out_of_text"] is True
        assert ss["global_bg_choice"] == "Custom…"
        assert ss["global_bg_custom"] == "#222222"
        store = ss["trial_annotations"]
        assert ("p1", "t2") in store
        assert store[("p1", "t2")]["star"] is True
        assert store[("p1", "t2")]["tags"] == ["Review"]

    def test_invalid_fields_are_skipped_not_applied(self):
        config = {
            "coloring": {"color_by": "does_not_exist", "heatmap_style": "Bogus"},
            "axes": {"x_field": "nope", "y_field": "y"},
            "selection": {"participant_id": "p9", "trial_id": "missing"},
        }
        ss = _run(_restore_app, _config=config).session_state
        assert "global_color_by" not in ss
        assert "global_heatmap_style" not in ss
        assert "global_x_field" not in ss
        assert ss["global_y_field"] == "y"  # the valid one still applies
        skipped = ss["_skipped"]
        for label in (
            "color-by field",
            "heatmap style",
            "X axis field",
            "trial selection",
        ):
            assert label in skipped

    def test_numeric_values_are_clamped_to_widget_bounds(self):
        config = {
            "canvas_px": {"width": 99999, "height": 1},
            "sizing": {"marker_size_range": [1, 99], "base_font_size": 999},
        }
        ss = _run(_restore_app, _config=config).session_state
        assert ss["global_canvas_width"] == 10000
        assert ss["global_canvas_height"] == 100
        assert ss["global_marker_size_range"] == (4, 40)
        assert ss["global_base_font_size"] == 72

    def test_malformed_numeric_fields_skipped_without_crashing(self):
        # Guards the bug where int("abc") / int(None) raised and surfaced a
        # Streamlit error page instead of skipping the field.
        config = {
            "layers": {"words": False},
            "coloring": {"heatmap_style": "Interpolated", "fixation_range": ["lo", 9]},
            "canvas_px": {"width": "abc", "height": None},
            "sizing": {"base_font_size": "huge", "marker_size_range": ["x", 30]},
        }
        ss = _run(_restore_app, _config=config).session_state  # must not raise
        # good fields still applied
        assert ss["global_show_words"] is False
        assert ss["global_heatmap_style"] == "Interpolated"
        # malformed numerics skipped, not applied
        for key in (
            "global_canvas_width",
            "global_canvas_height",
            "global_base_font_size",
            "global_marker_size_range",
            "global_fixation_color_range",
        ):
            assert key not in ss
        for label in (
            "canvas width",
            "canvas height",
            "figure font size",
            "marker size range",
            "fixation color range",
        ):
            assert label in ss["_skipped"]

    def test_malformed_sections_are_tolerated(self):
        # Sections of the wrong type must coerce to empty, not crash.
        config = {
            "layers": "nope",
            "coloring": [1, 2],
            "sizing": 5,
            "canvas_px": None,
            "axes": "x",
            "selection": [],
        }
        ss = _run(_restore_app, _config=config).session_state  # must not raise
        assert ss["_applied"] == 0


@pytest.mark.timeout(60)
class TestApplyUploadedPlotConfig:
    def test_valid_upload_applies_and_records_skips(self):
        import json

        data = json.dumps(
            {"layers": {"heatmap": False}, "axes": {"x_field": "bogus"}}
        ).encode("utf-8")
        ss = _run(_apply_app, _bytes=data).session_state
        assert ss["global_show_heatmap"] is False
        assert ss["_plot_config_skipped"] == ["X axis field"]

    def test_malformed_json_does_not_crash(self):
        ss = _run(_apply_app, _bytes=b"{not valid json").session_state
        # nothing applied, no widget state written, no exception
        assert "global_show_heatmap" not in ss

    def test_non_object_json_does_not_crash(self):
        ss = _run(_apply_app, _bytes=b"[1, 2, 3]").session_state
        assert "global_show_heatmap" not in ss


def test_build_studio_config_includes_provenance_and_round_trips():
    """The Save & restore config builder records provenance (app version, data
    source, column mapping) + annotations and is JSON-serializable (TODO 4.1)."""
    import json

    import pandas as pd

    from scanpath_studio.tabs import _build_studio_config

    figure_settings = {
        "show_words": True,
        "show_word_labels": True,
        "show_fixations": True,
        "show_order": False,
        "show_saccades": True,
        "show_saccade_arrows": True,
        "show_heatmap": False,
        "show_raw_gaze": False,
        "color_by": "line",
        "heatmap_style": "Word boxes",
        "show_colorbars": False,
        "fixation_color_range": None,
        "heatmap_range": None,
        "fixation_colorscale": "Blues",
        "heatmap_colorscale": "Oranges",
        "marker_size_range": (8, 24),
        "order_font_size": 14,
        "order_font_color": "#111111",
        "scale_text_to_boxes": True,
        "line_spacing": 3.0,
        "critical_span_style": "Mark border",
        "highlight_out_of_text": True,
        "background_color": "#222222",
    }
    cfg = _build_studio_config(
        selected_participant="p1",
        selected_trial="t1",
        canvas_width=2560,
        canvas_height=1440,
        x_field="x",
        y_field="y",
        figure_settings=figure_settings,
        viz_settings={"heatmap_metric": "duration_ms"},
        base_font_size=16,
        trial_raw_gaze=pd.DataFrame(),
        font_family="Courier New",
        annotation_records=[
            {
                "participant_id": "p1",
                "trial_id": "t1",
                "star": True,
                "tags": [],
                "note": "",
            }
        ],
        column_mapping={"col_map_fix_x": "CURRENT_FIX_X"},
        data_source="Use bundled demo",
        app_version="9.9.9",
    )
    assert cfg["schema"] == 2
    assert cfg["app"] == {"name": "Scanpath Studio", "version": "9.9.9"}
    assert cfg["data_source"] == "Use bundled demo"
    assert cfg["column_mapping"] == {"col_map_fix_x": "CURRENT_FIX_X"}
    assert cfg["selection"] == {"participant_id": "p1", "trial_id": "t1"}
    assert cfg["coloring"]["color_by"] == "line"
    assert cfg["text"]["font_family"] == "Courier New"
    assert cfg["highlighting"]["background_color"] == "#222222"
    assert len(cfg["annotations"]) == 1
    json.dumps(cfg)  # must be JSON-serializable


def _collect_mapping_app():
    """Seed session_state like an active upload, then collect the mapping."""
    import streamlit as st

    from scanpath_studio.tabs import _collect_column_mapping

    class _FakeUpload:
        name = "data.csv.zip"

    # Real mapping selections.
    st.session_state["col_map_fix_x"] = "CURRENT_FIX_X"
    st.session_state["col_map_words_box_format"] = "edges"
    # File-uploader widgets share the col_map_* prefix — must NOT be collected.
    st.session_state["col_map_fix_upload"] = [_FakeUpload()]
    st.session_state["col_map_words_upload"] = _FakeUpload()
    st.session_state["col_map_raw_gaze_upload"] = None
    # Unrelated key — also excluded.
    st.session_state["other_key"] = "ignored"
    st.session_state["_mapping"] = _collect_column_mapping()


def test_collect_column_mapping_excludes_uploader_widgets():
    """The mapping sweep must drop the upload boxes' file_uploader widgets, whose
    UploadedFile values aren't JSON-serializable (regression: Save & restore
    download crashed with an active upload)."""
    import json

    at = AppTest.from_function(_collect_mapping_app)
    at.run()
    assert not at.exception
    mapping = at.session_state["_mapping"]
    assert mapping == {
        "col_map_fix_x": "CURRENT_FIX_X",
        "col_map_words_box_format": "edges",
    }
    json.dumps(mapping)  # must be JSON-serializable
