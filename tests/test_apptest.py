"""Streamlit AppTest integration: spin up the app and assert key surfaces render."""

from __future__ import annotations

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest


SYNTHETIC_SOURCE = "Synthetic test trial"


def _make_apptest(*, synthetic: bool = False) -> "AppTest":
    """Build an AppTest for the app.

    Booting the bundled demo renders every tab over a large dataset (~5s).
    ``synthetic=True`` pre-seeds the "Synthetic test trial" source *before* the
    first ``run()``, so the app boots straight into the tiny synthetic trial
    (6 words / 9 fixations) — the same surfaces render ~10x faster and there's
    no throwaway demo render. Use it for tests that don't assert on the demo's
    specific richness; a few launch/multi-trial tests stay on the demo as the
    real-default-experience guardrails.
    """
    at = AppTest.from_file("streamlit_app.py")
    if synthetic:
        at.session_state["data_source_choice"] = SYNTHETIC_SOURCE
    return at


@pytest.mark.timeout(60)
class TestAppLaunches:
    def test_app_launches_with_bundled_demo(self):
        # The bundled demo must boot the full five-tab UI cleanly: no Python
        # exceptions and no st.error surfaced on the default render.
        at = _make_apptest()
        at.run(timeout=30)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert at.error == [], f"st.error calls: {[e.value for e in at.error]}"

    def test_title_present(self):
        at = _make_apptest(synthetic=True)
        at.run(timeout=30)
        titles = [t.value for t in at.title]
        assert any("Scanpath Studio" in v for v in titles)

    def test_synthetic_data_source_renders(self):
        # The "Synthetic test trial" source should load + render without error.
        at = _make_apptest(synthetic=True)
        at.run(timeout=30)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert at.error == [], f"st.error calls: {[e.value for e in at.error]}"

    def test_multiple_comparison_tab_renders(self):
        # Exercise the Multiple Comparison tab: change the grid columns and
        # bump the regenerate nonce, then confirm no exceptions / errors.
        at = _make_apptest(synthetic=True)
        at.session_state["multi_n_cols"] = 2
        at.session_state["multi_nonce"] = 1
        at.run(timeout=30)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert at.error == [], f"st.error calls: {[e.value for e in at.error]}"

    def test_multiple_comparison_fixation_range(self):
        # Narrow the fixation-index window and confirm the slice path (figures +
        # snapshot table + convergence plots) rebuilds without exceptions.
        at = _make_apptest(synthetic=True)
        at.session_state["multi_fix_range"] = (3, 10)
        at.run(timeout=30)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert at.error == [], f"st.error calls: {[e.value for e in at.error]}"

    def test_multiple_comparison_range_clamped_when_max_shrinks(self):
        # A persisted fixation window must not crash when max_fix shrinks (e.g.
        # a much shorter trial / fewer models): the stored value is clamped.
        # A deliberately huge window; clamp must pull it into range on boot.
        at = _make_apptest(synthetic=True)
        at.session_state["multi_fix_range"] = (900, 1000)
        at.run(timeout=30)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert at.error == [], f"st.error calls: {[e.value for e in at.error]}"

    def test_animation_export_controls_render(self):
        # Animation is now a checkbox in the Scanpath Visualization tab; with it
        # on, the Export toggle offers the HTML/GIF/MP4 selector. Selecting a
        # rasterized format must render its options + Render button without
        # crashing (and without triggering the expensive Kaleido render).
        at = _make_apptest(synthetic=True)
        at.run(timeout=30)
        at.session_state["single_animate"] = True
        at.run(timeout=30)
        fmt_radios = [r for r in at.radio if list(r.options) == ["HTML", "GIF", "MP4"]]
        assert fmt_radios, "animation export-format radio not found"
        at.session_state["anim_export_format"] = "MP4"
        at.run(timeout=30)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert at.error == [], f"st.error calls: {[e.value for e in at.error]}"

    def test_single_trial_participant_mode_skips_slider(self):
        # The synthetic source has a single participant, so the picker no longer
        # offers a Participant mode (it would be a no-op) and collapses to a
        # plain Trial dropdown — which in particular never instantiates a
        # one-option st.select_slider (that crashes the browser with a
        # `RangeError: min (0) is equal/bigger than max (0)`).
        at = _make_apptest(synthetic=True)
        at.session_state["single_select_trial_mode"] = "Participant"
        at.run(timeout=30)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        slider_keys = [s.key for s in at.select_slider]
        assert "single_slider" not in slider_keys, (
            "single-option select_slider rendered — this crashes the browser"
        )

    def test_single_trial_all_selection_modes(self):
        # Trial / Text / Participant must all resolve the lone synthetic trial.
        for mode in ["Trial", "Text", "Participant"]:
            at = _make_apptest(synthetic=True)
            at.session_state["single_select_trial_mode"] = mode
            at.run(timeout=30)
            assert not at.exception, f"{mode}: {at.exception}"
            assert at.error == [], f"{mode}: {[e.value for e in at.error]}"
            markdown = " ".join(m.value for m in at.markdown)
            assert "synthetic_2line_demo" in markdown, (
                f"{mode} mode did not resolve the single trial"
            )

    def test_multi_trial_participant_mode_keeps_slider(self):
        # Guard the fix's other side: with several trials per participant
        # (bundled demo), Participant mode still gets its select_slider.
        at = _make_apptest()
        at.run(timeout=30)
        at.session_state["single_select_trial_mode"] = "Participant"
        at.run(timeout=30)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        slider_keys = [s.key for s in at.select_slider]
        assert "single_slider" in slider_keys

    def test_new_viz_toggles_build_without_error(self):
        # Flip the new plot options (color-by-line — now the "line" option in the
        # color-by selector — out-of-text, gray background) and confirm those
        # code paths build without error. These pre-set values mimic a Save &
        # restore: the widgets must honour them (no inline value=/index= override
        # — see _VIZ_WIDGET_DEFAULTS) rather than reset to their hardcoded default.
        at = _make_apptest(synthetic=True)
        at.session_state["global_color_by"] = "line"
        at.session_state["global_highlight_out_of_text"] = True
        at.session_state["global_critical_span_style"] = "Mark border"
        at.session_state["global_bg_choice"] = "Gray"
        at.run(timeout=30)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert at.error == [], f"st.error calls: {[e.value for e in at.error]}"
        out_of_text = {c.key: c.value for c in at.checkbox if c.key}
        assert out_of_text.get("global_highlight_out_of_text") is True, (
            "restored out-of-text value was overridden by an inline default"
        )

    def test_animate_checkbox_renders_animation(self):
        # The Scanpath Visualization tab's Animate checkbox folds the former
        # animation tab in: the playback-speed slider must appear without error.
        at = _make_apptest(synthetic=True)
        at.session_state["single_animate"] = True
        at.run(timeout=30)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert at.error == [], f"st.error calls: {[e.value for e in at.error]}"
        assert "single_playback_speed" in [s.key for s in at.select_slider]

    def test_bulk_export_whole_dataset_option(self):
        # The "Trials to include" scope radio always offers both "All" (every
        # trial, ignoring the sidebar filter) first and "All filtered trials"
        # (the current sidebar selection) second.
        at = _make_apptest()  # bundled 3-pid demo
        at.run(timeout=30)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        scope_radios = [r for r in at.radio if r.key == "bulk_export_scope"]
        assert scope_radios, "bulk-export scope radio missing"
        assert scope_radios[0].options[:2] == ["All", "All filtered trials"], (
            f"unexpected scope options: {scope_radios[0].options}"
        )

        # Narrow to a single participant, pick "All" (whole dataset), build clean.
        sel = [m for m in at.multiselect if m.key == "filter_participants"]
        assert sel, "participant filter missing"
        sel[0].set_value([sel[0].options[0]])
        at.session_state["bulk_export_scope"] = "All"
        at.run(timeout=30)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert at.error == [], f"st.error calls: {[e.value for e in at.error]}"

    def test_participant_mode_sub_selection_methods(self):
        # Participant mode offers a trial-index / text / trial-id sub-selector
        # (TODO 1.15); picking Trial ID resolves a trial without error.
        at = _make_apptest()
        at.run(timeout=30)
        at.session_state["single_select_trial_mode"] = "Participant"
        at.run(timeout=30)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        sub = [
            r for r in at.radio if set(r.options) == {"Trial index", "Text", "Trial ID"}
        ]
        assert sub, "participant-mode sub-selection radio not found"
        sub[0].set_value("Trial ID")
        at.run(timeout=30)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert at.error == [], f"st.error calls: {[e.value for e in at.error]}"

    def test_participant_mode_stale_sub_method_does_not_crash(self):
        # A stale "Trial index" sub-method pick must not crash when the current
        # data offers only Text / Trial ID (the synthetic source has no
        # trial-index column) — the radio's stale value is dropped, not raised.
        at = _make_apptest(synthetic=True)
        at.session_state["single_select_trial_mode"] = "Participant"
        at.session_state["single_participant_by"] = "Trial index"  # unavailable here
        at.run(timeout=30)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert at.error == [], f"st.error calls: {[e.value for e in at.error]}"


@pytest.mark.timeout(90)
class TestSingleReportDatasets:
    """The whole app pipeline (load → normalize → filter → all five tabs)
    must run when a dataset ships only one of the two reports."""

    def _run_with_demo_override(self, monkeypatch, make_frames):
        import pandas as pd

        from scanpath_studio import app, data

        words_raw, fix_raw = data.load_sample_data()
        words_raw = words_raw[words_raw["participant_id"] == "l37_1129"]
        fix_raw = fix_raw[fix_raw["participant_id"] == "l37_1129"]
        override = make_frames(words_raw, fix_raw, pd.DataFrame())
        monkeypatch.setattr(app, "load_sample_data", lambda *a, **k: override)

        at = _make_apptest()
        at.run(timeout=60)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert at.error == [], f"st.error calls: {[e.value for e in at.error]}"

    def test_fixations_only_dataset(self, monkeypatch):
        self._run_with_demo_override(
            monkeypatch, lambda words, fix, empty: (empty, fix)
        )

    def test_words_only_dataset(self, monkeypatch):
        self._run_with_demo_override(
            monkeypatch, lambda words, fix, empty: (words, empty)
        )


@pytest.mark.timeout(90)
class TestUnmappedRawDataView:
    """When a required column is unmapped, the app must show the raw uploaded
    data (so the user can pick the mapping) instead of halting."""

    def test_raw_data_shown_when_mapping_incomplete(self, monkeypatch):
        import pandas as pd

        from scanpath_studio import app

        # A words table whose columns match neither a word/IA id nor box
        # coordinates — exactly the screenshot's failure. Injected through the
        # per-table upload seam (AppTest can't drive st.file_uploader); the
        # Words/IA group gets it, the others stay empty.
        raw_words = pd.DataFrame(
            {"reader": ["r0", "r0"], "stim": ["b0", "b0"], "token": ["Um", "das"]}
        )
        monkeypatch.setattr(
            app,
            "_read_uploaded_frame",
            lambda **kw: (
                raw_words if kw["state_prefix"] == "col_map_words" else pd.DataFrame()
            ),
        )

        at = _make_apptest()
        at.session_state["data_source_choice"] = app.UPLOAD_CHOICE
        # Past the setup wizard (collapsed "Data & mapping" panel), an incomplete
        # mapping still surfaces the raw data so the user can fix it.
        at.session_state["setup_complete"] = True
        at.run(timeout=60)

        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        # The guidance banner names the missing field…
        warnings = " ".join(w.value for w in at.warning)
        assert "column mapping" in warnings.lower()
        assert "Word/IA ID" in warnings
        # …and the raw uploaded table is rendered so the user can inspect it.
        frames = [df.value for df in at.dataframe]
        assert any(list(f.columns) == ["reader", "stim", "token"] for f in frames), (
            "raw uploaded columns should be visible in the Raw Data tab"
        )

    def test_public_datasets_hidden_by_default(self, monkeypatch):
        """The Public datasets source is feature-flagged off until release."""
        from scanpath_studio import app

        monkeypatch.delenv("SCANPATH_PUBLIC_DATASETS", raising=False)
        at = _make_apptest(synthetic=True)
        at.run(timeout=30)
        source_radio = [r for r in at.radio if r.key == "data_source_choice"]
        assert source_radio, "data source radio not found"
        assert app.PUBLIC_DATASETS_CHOICE not in source_radio[0].options

    def test_potec_source_renders(self, monkeypatch):
        """Public datasets → PoTeC loads through the same pipeline as an upload.

        The source is behind the SCANPATH_PUBLIC_DATASETS flag until release —
        enabled here so the whole path stays tested."""
        import pandas as pd

        from scanpath_studio import app, datasets

        monkeypatch.setenv("SCANPATH_PUBLIC_DATASETS", "1")
        words = pd.DataFrame(
            {
                "aoi": [1, 2],
                "start_x": [80.0, 115.0],
                "start_y": [21.0, 21.0],
                "end_x": [115.0, 189.0],
                "end_y": [99.0, 99.0],
                "word": ["Um", "null"],
                "text_id": ["b0", "b0"],
                "line": [1, 1],
            }
        )
        fixations = pd.DataFrame(
            {
                "reader_id": [0, 0],
                "text_id": ["b0", "b0"],
                "fixation_duration": [210, 190],
                "fixation_index": [1, 2],
                "word_index_in_text": [1, 2],
                "x": [97.5, 152.0],
                "y": [60.0, 60.0],
            }
        )
        monkeypatch.setattr(
            datasets, "potec_raw_frames", lambda *a, **k: (words, fixations)
        )

        at = _make_apptest()
        at.session_state["data_source_choice"] = app.PUBLIC_DATASETS_CHOICE
        at.run(timeout=60)

        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert at.error == [], f"st.error calls: {[e.value for e in at.error]}"
        # The dataset picker renders the registry entries (PoTeC today).
        pickers = [s for s in at.selectbox if s.label == "Dataset"]
        assert pickers, "expected a Dataset selectbox under Public datasets"
        assert pickers[0].options == list(app.PUBLIC_DATASET_REGISTRY)


class TestGroupedUploadMapping:
    """Upload source: each table's mapping renders under its own upload box, and
    raw gaze is a first-class peer (its mapping panel is always available)."""

    def test_upload_renders_all_three_mapping_panels(self, monkeypatch):
        import pandas as pd

        from scanpath_studio import app

        # Inject the bundled raw frames through the per-table upload seam.
        sample_words, sample_fix = app.load_sample_data()
        sample_rg = app.load_sample_raw_gaze()
        frames = {
            "col_map_words": sample_words,
            "col_map_fix": sample_fix,
            "col_map_raw_gaze": sample_rg,
        }
        monkeypatch.setattr(
            app,
            "_read_uploaded_frame",
            lambda **kw: frames.get(kw["state_prefix"], pd.DataFrame()),
        )

        at = _make_apptest()
        at.session_state["data_source_choice"] = app.UPLOAD_CHOICE
        at.run(timeout=60)

        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert at.error == [], f"st.error: {[e.value for e in at.error]}"
        # All three tables rendered their own mapping panel (a Participant field
        # each) — words, fixations, and raw gaze as peers.
        keys = {s.key for s in at.selectbox}
        assert "col_map_words_participant" in keys
        assert "col_map_fix_participant" in keys
        assert "col_map_raw_gaze_participant" in keys

    def test_words_only_upload_renders(self, monkeypatch):
        # Single-report upload: only a Words/IA table — the missing fixations
        # side becomes a canonical empty frame and the app still renders.
        import pandas as pd

        from scanpath_studio import app

        sample_words, _ = app.load_sample_data()
        monkeypatch.setattr(
            app,
            "_read_uploaded_frame",
            lambda **kw: (
                sample_words
                if kw["state_prefix"] == "col_map_words"
                else pd.DataFrame()
            ),
        )

        at = _make_apptest()
        at.session_state["data_source_choice"] = app.UPLOAD_CHOICE
        at.run(timeout=60)

        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert at.error == [], f"st.error: {[e.value for e in at.error]}"
        keys = {s.key for s in at.selectbox}
        # The Words/IA mapping renders; the (un-uploaded) Fixations table gets no
        # mapping panel, and the app got past mapping (no unmapped-view warning).
        assert "col_map_words_participant" in keys
        assert "col_map_fix_participant" not in keys
        warnings = " ".join(w.value for w in at.warning)
        assert "Finish the column mapping" not in warnings

    def test_raw_gaze_only_renders(self, monkeypatch):
        # A raw-gaze-only dataset (no words, no fixations) must load and render
        # the gaze instead of falling back to the demo or halting on "no data".
        import pandas as pd

        from scanpath_studio import app

        sample_rg = app.load_sample_raw_gaze()
        monkeypatch.setattr(
            app,
            "_read_uploaded_frame",
            lambda **kw: (
                sample_rg
                if kw["state_prefix"] == "col_map_raw_gaze"
                else pd.DataFrame()
            ),
        )

        at = _make_apptest()
        at.session_state["data_source_choice"] = app.UPLOAD_CHOICE
        at.session_state["setup_complete"] = True  # past the setup wizard
        at.run(timeout=60)

        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert at.error == [], f"st.error: {[e.value for e in at.error]}"
        keys = {s.key for s in at.selectbox}
        # Raw-gaze mapping rendered; words/fixations did not; and the app neither
        # halted on "no data" nor fell back to the demo.
        assert "col_map_raw_gaze_participant" in keys
        assert "col_map_words_participant" not in keys
        warnings = " ".join(w.value for w in at.warning)
        assert "No data after filtering" not in warnings
        # The raw-gaze overlay is defaulted on so the Interactive Plot isn't blank.
        assert at.session_state["global_show_raw_gaze"] is True


def _box_mapping_script():
    """Render only the Words/IA column-mapping panel for an edges-format table,
    stashing the resulting mapping in session_state for assertions."""
    import pandas as pd
    import streamlit as st

    from scanpath_studio.controls import WORD_FIELD_SPECS, column_mapping_ui
    from scanpath_studio.data import propose_word_schema

    df = pd.DataFrame(
        {
            "participant_id": ["p1"],
            "trial_id": ["t1"],
            "IA_ID": [1],
            "IA_LEFT": [100],
            "IA_RIGHT": [150],
            "IA_TOP": [50],
            "IA_BOTTOM": [100],
        }
    )
    st.session_state["_box_mapping"] = column_mapping_ui(
        df,
        table_label="Words/IA",
        state_key_prefix="col_map_words",
        field_specs=WORD_FIELD_SPECS,
        proposed=propose_word_schema(df),
    )


class TestColumnMappingBoxWidget:
    """The Words/IA box mapping shows one coordinate-format radio + 4 fields, yet
    still returns all eight box keys (the inactive four set to None)."""

    def test_edges_default_and_full_mapping(self):
        from scanpath_studio.controls import BOX_FORMAT_EDGES

        at = AppTest.from_function(_box_mapping_script).run(timeout=30)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"

        box_radios = [r for r in at.radio if r.key == "col_map_words_box_format"]
        assert box_radios, "box-format radio not rendered"
        assert box_radios[0].value == BOX_FORMAT_EDGES

        mapping = at.session_state["_box_mapping"]
        assert mapping["left"] == "IA_LEFT"
        assert mapping["right"] == "IA_RIGHT"
        assert mapping["top"] == "IA_TOP"
        assert mapping["bottom"] == "IA_BOTTOM"
        assert mapping["x"] is None and mapping["y"] is None
        assert mapping["width"] is None and mapping["height"] is None

    def test_switching_to_origin_drops_edge_columns(self):
        from scanpath_studio.controls import BOX_FORMAT_ORIGIN

        at = AppTest.from_function(_box_mapping_script).run(timeout=30)
        box_radio = [r for r in at.radio if r.key == "col_map_words_box_format"][0]
        box_radio.set_value(BOX_FORMAT_ORIGIN)
        at.run(timeout=30)

        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        mapping = at.session_state["_box_mapping"]
        # Origin fields are now the active four (None here — this table has no
        # x/y/w/h columns); the edge keys are no longer mapped.
        assert mapping["left"] is None and mapping["right"] is None
        assert mapping["top"] is None and mapping["bottom"] is None


@pytest.mark.timeout(90)
class TestWelcomeTour:
    """Dialog-style tour (TOUR_STYLE="dialog"): opens once per session,
    navigates, and stays out of the way of embeds / deep links."""

    @staticmethod
    def _tour_buttons(at):
        return {b.key for b in at.button if b.key and b.key.startswith("tour_")}

    @pytest.fixture(autouse=True)
    def _dialog_style(self, monkeypatch):
        from scanpath_studio import tour

        monkeypatch.setattr(tour, "TOUR_STYLE", "dialog")

    def test_tour_opens_on_first_run_only(self):
        at = _make_apptest(synthetic=True)
        at.run(timeout=30)
        assert at.session_state["tour_seen"] is True
        assert "tour_next" in self._tour_buttons(at)
        # Any later full rerun must NOT reopen the dialog (e.g. after the
        # user dismissed it with X) — only the replay button may.
        at.run(timeout=30)
        assert "tour_next" not in self._tour_buttons(at)

    def test_tour_next_advances_step(self):
        at = _make_apptest(synthetic=True)
        at.run(timeout=30)
        at.button(key="tour_next").click()
        at.run(timeout=30)
        assert at.session_state["tour_step"] == 1
        assert not at.exception, f"Streamlit exceptions: {at.exception}"

    def test_tour_suppressed_for_embed_and_deep_link(self):
        # AppTest can't inject query params into st.query_params, so the
        # suppression predicate is tested directly (it takes a mapping).
        from scanpath_studio.tour import tour_suppressed

        assert tour_suppressed({"embed": "true"})
        assert tour_suppressed({"embed": "1"})
        assert tour_suppressed({"participant": "l37_1129"})
        assert tour_suppressed({"source": "onestop"})
        assert tour_suppressed({"trial": "3"})
        assert tour_suppressed({"tab": "animation"})
        assert not tour_suppressed({})
        assert not tour_suppressed({"embed": "false"})

    def test_tour_replay_button_reopens(self):
        at = _make_apptest(synthetic=True)
        at.run(timeout=30)  # first run: auto-open, marks tour_seen
        at.run(timeout=30)  # second run: dialog gone
        at.session_state["tour_step"] = 3
        at.button(key="tour_replay").click()
        at.run(timeout=30)
        assert "tour_next" in self._tour_buttons(at)
        assert at.session_state["tour_step"] == 0  # replay restarts the tour
        assert not at.exception, f"Streamlit exceptions: {at.exception}"


@pytest.mark.timeout(90)
class TestSpotlightTour:
    """Spotlight-style tour (the default): floating card + per-step highlight,
    armed once per session via tour_mode, dismissed by Exit/Done."""

    @staticmethod
    def _sp_buttons(at):
        return {b.key for b in at.button if b.key and b.key.startswith("tour_sp_")}

    def test_spotlight_arms_on_first_run(self):
        at = _make_apptest(synthetic=True)
        at.run(timeout=30)
        assert at.session_state["tour_seen"] is True
        assert at.session_state["tour_mode"] == "spotlight"
        assert "tour_sp_next" in self._sp_buttons(at)
        # The dialog style must NOT also open.
        assert not any(b.key == "tour_next" for b in at.button)
        # The welcome step renders centered with a dimmed backdrop.
        markdown = " ".join(m.value for m in at.markdown)
        assert "tour-backdrop" in markdown

    def test_spotlight_navigates_and_exits(self):
        from scanpath_studio.tour import _SPOTLIGHT_STEPS

        at = _make_apptest(synthetic=True)
        at.run(timeout=30)
        at.button(key="tour_sp_next").click()
        at.run(timeout=30)
        assert at.session_state["tour_step"] == 1
        # From step 2 on, the card drops to the corner: no backdrop.
        markdown = " ".join(m.value for m in at.markdown)
        assert "tour-backdrop" not in markdown
        at.button(key="tour_sp_exit").click()
        at.run(timeout=30)
        assert at.session_state["tour_mode"] is None
        assert self._sp_buttons(at) == set(), "card must vanish after Exit"
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        # Step list sanity: every selector-bearing step targets a keyed
        # wrapper or a stable testid that exists in the app.
        selectors = [s["selector"] for s in _SPOTLIGHT_STEPS if s["selector"]]
        assert all(
            sel.startswith(".st-key-tour_grp_") or "data-testid" in sel
            for sel in selectors
        )

    def test_spotlight_done_on_last_step(self):
        from scanpath_studio.tour import _SPOTLIGHT_STEPS

        at = _make_apptest(synthetic=True)
        at.run(timeout=30)
        at.session_state["tour_step"] = len(_SPOTLIGHT_STEPS) - 1
        at.run(timeout=30)
        assert "tour_sp_done" in self._sp_buttons(at)
        at.button(key="tour_sp_done").click()
        at.run(timeout=30)
        assert at.session_state["tour_mode"] is None
        assert not at.exception, f"Streamlit exceptions: {at.exception}"

    def test_spotlight_replay(self):
        at = _make_apptest(synthetic=True)
        at.run(timeout=30)
        at.button(key="tour_sp_exit").click()
        at.run(timeout=30)
        assert self._sp_buttons(at) == set()
        at.button(key="tour_replay").click()
        at.run(timeout=30)
        assert "tour_sp_next" in self._sp_buttons(at)
        assert at.session_state["tour_step"] == 0
        assert not at.exception, f"Streamlit exceptions: {at.exception}"


class TestSetupWizard:
    """The hybrid setup wizard for the Upload source."""

    @staticmethod
    def _inject(monkeypatch):
        import pandas as pd

        from scanpath_studio import app

        raw_words = pd.DataFrame(
            {
                "participant_id": ["p1", "p1", "p2", "p2"],
                "trial_id": ["t1", "t1", "t1", "t1"],
                "word_id": [1, 2, 1, 2],
                "IA_LEFT": [0, 10, 0, 10],
                "IA_RIGHT": [10, 20, 10, 20],
                "IA_TOP": [0, 0, 0, 0],
                "IA_BOTTOM": [10, 10, 10, 10],
                "IA_LABEL": ["a", "b", "a", "b"],
                "difficulty_level": ["Adv", "Adv", "Ele", "Ele"],
                "junk_col": [9, 9, 9, 9],
            }
        )
        raw_fix = pd.DataFrame(
            {
                "participant_id": ["p1", "p1", "p2"],
                "trial_id": ["t1", "t1", "t1"],
                "CURRENT_FIX_X": [5.0, 15.0, 5.0],
                "CURRENT_FIX_Y": [5.0, 5.0, 5.0],
                "CURRENT_FIX_DURATION": [100, 120, 90],
            }
        )
        monkeypatch.setattr(
            app,
            "_read_uploaded_frame",
            lambda **kw: (
                raw_words
                if kw["state_prefix"] == "col_map_words"
                else raw_fix
                if kw["state_prefix"] == "col_map_fix"
                else pd.DataFrame()
            ),
        )
        return app

    def test_add_data_button_enters_wizard(self):
        """Clicking '➕ Add data' from a built-in source switches into the upload
        wizard. Regression: the handler reassigned the data_source_choice radio
        key inline, which Streamlit forbids after the radio is instantiated — it
        must run in an on_click callback instead."""
        from scanpath_studio import app

        at = _make_apptest(synthetic=True)
        at.run(timeout=60)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        add = [b for b in at.button if b.key == "add_data_btn"]
        assert add, "Add data button not rendered"
        add[0].click()
        at.run(timeout=60)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert at.session_state["data_source_choice"] == app.UPLOAD_CHOICE
        assert at.session_state["setup_complete"] is False

    def test_wizard_active_then_finalize_renders_tabs(self, monkeypatch):
        app = self._inject(monkeypatch)
        at = _make_apptest()
        at.session_state["data_source_choice"] = app.UPLOAD_CHOICE
        at.run(timeout=60)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        # Wizard is active: the finalize button is shown and tabs are not yet up.
        assert any(b.key == "wizard_finalize" for b in at.button)
        assert "single_trial_id" not in {s.key for s in at.selectbox}

        # Finalizing reveals the visualization (collapsed Data & mapping panel +
        # the trial picker), and the kept columns are pruned.
        at.session_state["setup_complete"] = True
        at.run(timeout=60)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert any(b.key == "wizard_reconfigure" for b in at.button)

    def test_wizard_prunes_unkept_columns(self, monkeypatch):
        app = self._inject(monkeypatch)
        at = _make_apptest()
        at.session_state["data_source_choice"] = app.UPLOAD_CHOICE
        at.run(timeout=60)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        # Finalize with defaults (keep detected optional fields, drop unclaimed),
        # then confirm the stored frame is actually thinned.
        [b for b in at.button if b.key == "wizard_finalize"][0].click()
        at.run(timeout=60)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        words = at.session_state["_datasets"][at.session_state["data_source_choice"]][
            "words"
        ]
        # junk_col is unclaimed and not kept -> pruned; difficulty_level is a
        # detected condition kept by default -> survives.
        assert "junk_col" not in words.columns
        assert "difficulty_level" in words.columns

    def test_unified_trial_picker_and_setup_step(self, monkeypatch):
        """Group A + C: one shared Trial ID picker (not per-table) and the
        inline Experimental Setup controls both render in the active wizard."""
        app = self._inject(monkeypatch)
        at = _make_apptest()
        at.session_state["data_source_choice"] = app.UPLOAD_CHOICE
        at.run(timeout=60)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        # The unified trial multiselect is rendered; per-table trial widgets are
        # not (single shared mapping by default).
        ms_keys = {m.key for m in at.multiselect}
        assert "col_map_trial_unified" in ms_keys
        assert "col_map_fix_trial" not in ms_keys
        assert "col_map_words_trial" not in ms_keys
        # Per-table override toggle is offered (both tables present).
        assert any(t.key == "wizard_trial_per_table" for t in at.toggle)
        # Display calibration moved into the loading flow.
        assert any(n.key == "global_canvas_width" for n in at.number_input)

    def test_per_table_trial_toggle_reveals_per_table_pickers(self, monkeypatch):
        app = self._inject(monkeypatch)
        at = _make_apptest()
        at.session_state["data_source_choice"] = app.UPLOAD_CHOICE
        at.session_state["wizard_trial_per_table"] = True
        at.run(timeout=60)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        ms_keys = {m.key for m in at.multiselect}
        assert "col_map_fix_trial" in ms_keys
        assert "col_map_words_trial" in ms_keys

    def test_finalize_button_stores_named_dataset(self, monkeypatch):
        """Group B.4: clicking 'Use this dataset' persists the normalized frames
        under a name and switches the data source to it (no re-upload to switch)."""
        app = self._inject(monkeypatch)
        at = _make_apptest()
        at.session_state["data_source_choice"] = app.UPLOAD_CHOICE
        at.run(timeout=60)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        finalize = [b for b in at.button if b.key == "wizard_finalize"]
        assert finalize, "finalize button not rendered for a valid mapping"
        finalize[0].click()
        at.run(timeout=60)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        stored = at.session_state["_datasets"]
        assert stored, "no dataset was stored on finalize"
        name = at.session_state["data_source_choice"]
        assert name in stored
        # The stored entry holds the already-normalized frames, so switching to
        # it later needs no re-upload / re-mapping, and the source switched away
        # from Upload onto the new named dataset.
        assert not stored[name]["fixations"].empty
        assert not stored[name]["words"].empty
        assert name != app.UPLOAD_CHOICE
        assert at.session_state["setup_complete"] is True

    def test_finalize_selects_new_dataset_in_sidebar(self, monkeypatch):
        """Regression: after '➕ Add data' → 'Use this dataset', the new dataset
        must appear in the sidebar Data-source radio AND be the selected value.
        The real flow renders the radio first (on a built-in source), so the
        finalize switch must not be lost to the radio's frontend reconciliation."""
        self._inject(monkeypatch)
        at = _make_apptest(synthetic=True)
        at.run(timeout=60)
        # Real flow: enter the wizard via the button (radio already rendered).
        [b for b in at.button if b.key == "add_data_btn"][0].click()
        at.run(timeout=60)
        [b for b in at.button if b.key == "wizard_finalize"][0].click()
        at.run(timeout=60)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        name = at.session_state["data_source_choice"]
        assert name in at.session_state["_datasets"]
        radios = [r for r in at.radio if r.key == "data_source_choice"]
        assert radios, "data-source radio not rendered after finalize"
        assert name in list(radios[0].options), (name, list(radios[0].options))
        assert radios[0].value == name

    def test_stored_dataset_loads_full_app_without_wizard(self, monkeypatch):
        """Group B.4: selecting a stored dataset reloads the whole app from the
        persisted frames — no wizard, no re-upload — and renders the trial picker."""
        app = self._inject(monkeypatch)
        # First finalize a dataset to capture its persisted store entry.
        at = _make_apptest()
        at.session_state["data_source_choice"] = app.UPLOAD_CHOICE
        at.run(timeout=60)
        [b for b in at.button if b.key == "wizard_finalize"][0].click()
        at.run(timeout=60)
        stored = dict(at.session_state["_datasets"])
        name = at.session_state["data_source_choice"]

        # Fresh session: point straight at the stored dataset (no upload monkeypatch
        # needed — the stored branch never reads uploads).
        at2 = _make_apptest()
        at2.session_state["_datasets"] = stored
        at2.session_state["data_source_choice"] = name
        at2.run(timeout=60)
        assert not at2.exception, f"Streamlit exceptions: {at2.exception}"
        assert not any(b.key == "wizard_finalize" for b in at2.button)
        keys = {w.key for w in list(at2.selectbox) + list(at2.radio) if w.key}
        assert any(k.startswith("single") for k in keys), keys

    def test_differing_trial_counts_are_emphasized(self, monkeypatch):
        """Group C.1c: when the per-table trial coverage differs the wizard says so
        (an info note when the tables still overlap, not a single green count)."""
        import pandas as pd

        from scanpath_studio import app

        # Words cover t1,t2,t3; fixations only t1,t2 — overlapping but unequal.
        raw_words = pd.DataFrame(
            {
                "participant_id": ["p1", "p1", "p1"],
                "trial_id": ["t1", "t2", "t3"],
                "word_id": [1, 1, 1],
                "IA_LEFT": [0, 0, 0],
                "IA_RIGHT": [10, 10, 10],
                "IA_TOP": [0, 0, 0],
                "IA_BOTTOM": [10, 10, 10],
                "IA_LABEL": ["a", "b", "c"],
            }
        )
        raw_fix = pd.DataFrame(
            {
                "participant_id": ["p1", "p1"],
                "trial_id": ["t1", "t2"],
                "CURRENT_FIX_X": [5.0, 5.0],
                "CURRENT_FIX_Y": [5.0, 5.0],
                "CURRENT_FIX_DURATION": [100, 120],
            }
        )
        monkeypatch.setattr(
            app,
            "_read_uploaded_frame",
            lambda **kw: (
                raw_words
                if kw["state_prefix"] == "col_map_words"
                else raw_fix
                if kw["state_prefix"] == "col_map_fix"
                else pd.DataFrame()
            ),
        )
        at = _make_apptest()
        at.session_state["data_source_choice"] = app.UPLOAD_CHOICE
        at.run(timeout=60)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        info_text = " ".join(e.value for e in at.info)
        assert "Trial coverage differs" in info_text, info_text

    def test_disjoint_trial_ids_warn(self, monkeypatch):
        """Group C.1c: when the tables share no trial ids at all (a likely mapping
        error), the wizard warns rather than just noting differing coverage."""
        import pandas as pd

        from scanpath_studio import app

        raw_words = pd.DataFrame(
            {
                "participant_id": ["p1", "p1"],
                "trial_id": ["w1", "w2"],
                "word_id": [1, 1],
                "IA_LEFT": [0, 0],
                "IA_RIGHT": [10, 10],
                "IA_TOP": [0, 0],
                "IA_BOTTOM": [10, 10],
                "IA_LABEL": ["a", "b"],
            }
        )
        raw_fix = pd.DataFrame(
            {
                "participant_id": ["p1", "p1"],
                "trial_id": ["f1", "f2"],
                "CURRENT_FIX_X": [5.0, 5.0],
                "CURRENT_FIX_Y": [5.0, 5.0],
                "CURRENT_FIX_DURATION": [100, 120],
            }
        )
        monkeypatch.setattr(
            app,
            "_read_uploaded_frame",
            lambda **kw: (
                raw_words
                if kw["state_prefix"] == "col_map_words"
                else raw_fix
                if kw["state_prefix"] == "col_map_fix"
                else pd.DataFrame()
            ),
        )
        at = _make_apptest()
        at.session_state["data_source_choice"] = app.UPLOAD_CHOICE
        at.run(timeout=60)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        warn_text = " ".join(e.value for e in at.warning)
        assert "No trial ids are shared" in warn_text, warn_text

    def test_raw_gaze_only_incomplete_mapping_blocks_finalize(self, monkeypatch):
        """Bug fix: a raw-gaze-only upload with an unmappable trial id must block
        finalize (raw-gaze problems are folded in) instead of storing an empty
        dataset."""
        import pandas as pd

        from scanpath_studio import app

        # No participant/trial/x/y the schema can auto-detect.
        raw_gaze = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        monkeypatch.setattr(
            app,
            "_read_uploaded_frame",
            lambda **kw: (
                raw_gaze if kw["state_prefix"] == "col_map_raw_gaze" else pd.DataFrame()
            ),
        )
        at = _make_apptest()
        at.session_state["data_source_choice"] = app.UPLOAD_CHOICE
        at.run(timeout=60)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        # Finalize is shown but disabled; the blocking problem mentions raw gaze.
        finalize = [b for b in at.button if b.key == "wizard_finalize"]
        assert finalize and finalize[0].disabled
        warn_text = " ".join(e.value for e in at.warning)
        assert "Raw gaze" in warn_text, warn_text

    def test_composite_trial_dataset_restores_picker_on_switch_back(self, monkeypatch):
        """Regression for the review's HIGH finding: a stored dataset whose trial
        id is composite must restore _composite_trial_columns (and its cascading
        picker) on reselect, overriding whatever source was loaded last."""
        import pandas as pd

        from scanpath_studio import app

        # Both a paragraph- and a text-level id present → unified default composes
        # them (a composite trial id).
        raw_words = pd.DataFrame(
            {
                "participant_id": ["p1", "p1", "p1"],
                "paragraph_id": ["A", "A", "B"],
                "text_id": ["1", "1", "1"],
                "word_id": [1, 2, 1],
                "IA_LEFT": [0, 10, 0],
                "IA_RIGHT": [10, 20, 10],
                "IA_TOP": [0, 0, 0],
                "IA_BOTTOM": [10, 10, 10],
                "IA_LABEL": ["a", "b", "a"],
            }
        )
        raw_fix = pd.DataFrame(
            {
                "participant_id": ["p1", "p1", "p1"],
                "paragraph_id": ["A", "A", "B"],
                "text_id": ["1", "1", "1"],
                "CURRENT_FIX_X": [5.0, 15.0, 5.0],
                "CURRENT_FIX_Y": [5.0, 5.0, 5.0],
                "CURRENT_FIX_DURATION": [100, 120, 90],
            }
        )
        monkeypatch.setattr(
            app,
            "_read_uploaded_frame",
            lambda **kw: (
                raw_words
                if kw["state_prefix"] == "col_map_words"
                else raw_fix
                if kw["state_prefix"] == "col_map_fix"
                else pd.DataFrame()
            ),
        )
        at = _make_apptest()
        at.session_state["data_source_choice"] = app.UPLOAD_CHOICE
        at.run(timeout=60)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert set(at.session_state["_composite_trial_columns"]) == {
            "paragraph_id",
            "text_id",
        }
        [b for b in at.button if b.key == "wizard_finalize"][0].click()
        at.run(timeout=60)
        stored = dict(at.session_state["_datasets"])
        name = at.session_state["data_source_choice"]
        assert set(stored[name]["composite_trial_columns"]) == {
            "paragraph_id",
            "text_id",
        }

        # Fresh session with STALE composite state (as if Demo was loaded last).
        at2 = _make_apptest()
        at2.session_state["_datasets"] = stored
        at2.session_state["data_source_choice"] = name
        at2.session_state["_composite_trial_columns"] = None
        at2.run(timeout=60)
        assert not at2.exception, f"Streamlit exceptions: {at2.exception}"
        assert set(at2.session_state["_composite_trial_columns"]) == {
            "paragraph_id",
            "text_id",
        }
        keys = {w.key for w in at2.selectbox if w.key}
        assert any(k.startswith("single_composite_") for k in keys), keys

    def test_experimental_setup_control_writes_shared_global_key(self, monkeypatch):
        """Group A: the Display & experiment setup controls live inside the wizard
        and write the shared global_* keys the sidebar later reads."""
        app = self._inject(monkeypatch)
        at = _make_apptest()
        at.session_state["data_source_choice"] = app.UPLOAD_CHOICE
        at.run(timeout=60)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        width = [n for n in at.number_input if n.key == "global_canvas_width"]
        assert width, "Experimental Setup monitor-width control not in the wizard"
        width[0].set_value(1999)
        at.run(timeout=60)
        assert at.session_state["global_canvas_width"] == 1999
