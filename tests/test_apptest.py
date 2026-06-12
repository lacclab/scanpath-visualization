"""Streamlit AppTest integration: spin up the app and assert key surfaces render."""

from __future__ import annotations

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest


def _make_apptest() -> "AppTest":
    return AppTest.from_file("streamlit_app.py")


@pytest.mark.timeout(60)
class TestAppLaunches:
    def test_app_launches_with_bundled_demo(self):
        at = _make_apptest()
        at.run(timeout=30)
        # The five-tab UI should render without exceptions
        assert not at.exception, f"Streamlit exceptions: {at.exception}"

    def test_title_present(self):
        at = _make_apptest()
        at.run(timeout=30)
        titles = [t.value for t in at.title]
        assert any("Scanpath Studio" in v for v in titles)

    def test_no_streamlit_errors(self):
        at = _make_apptest()
        at.run(timeout=30)
        assert at.error == [], f"st.error calls: {[e.value for e in at.error]}"

    def test_synthetic_data_source_renders(self):
        # The "Synthetic test trial" source should load + render without error.
        at = _make_apptest()
        at.run(timeout=30)
        at.session_state["data_source_choice"] = "Synthetic test trial"
        at.run(timeout=30)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert at.error == [], f"st.error calls: {[e.value for e in at.error]}"

    def test_multiple_comparison_tab_renders(self):
        # Exercise the Multiple Comparison tab: change the grid columns and
        # bump the regenerate nonce, then confirm no exceptions / errors.
        at = _make_apptest()
        at.run(timeout=30)
        at.session_state["multi_n_cols"] = 2
        at.session_state["multi_nonce"] = 1
        at.run(timeout=30)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert at.error == [], f"st.error calls: {[e.value for e in at.error]}"

    def test_multiple_comparison_fixation_range(self):
        # Narrow the fixation-index window and confirm the slice path (figures +
        # snapshot table + convergence plots) rebuilds without exceptions.
        at = _make_apptest()
        at.run(timeout=30)
        at.session_state["multi_fix_range"] = (3, 10)
        at.run(timeout=30)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert at.error == [], f"st.error calls: {[e.value for e in at.error]}"

    def test_multiple_comparison_range_clamped_when_max_shrinks(self):
        # A persisted fixation window must not crash when max_fix shrinks (e.g.
        # a much shorter trial / fewer models): the stored value is clamped.
        at = _make_apptest()
        at.run(timeout=30)
        # A deliberately huge window; clamp must pull it into range on rerun.
        at.session_state["multi_fix_range"] = (900, 1000)
        at.run(timeout=30)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert at.error == [], f"st.error calls: {[e.value for e in at.error]}"

    def test_animation_export_controls_render(self):
        # The animated-scanpath export selector offers HTML/GIF/MP4; selecting a
        # rasterized format must render its options + Render button without
        # crashing (and without triggering the expensive Kaleido render).
        at = _make_apptest()
        at.run(timeout=30)
        fmt_radios = [r for r in at.radio if list(r.options) == ["HTML", "GIF", "MP4"]]
        assert fmt_radios, "animation export-format radio not found"
        at.session_state["anim_export_format"] = "MP4"
        at.run(timeout=30)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        assert at.error == [], f"st.error calls: {[e.value for e in at.error]}"

    def test_single_trial_participant_mode_skips_slider(self):
        # With a single trial (the synthetic source), Participant mode used to
        # instantiate st.select_slider with ONE option — the Python side
        # accepts it, but the browser slider throws `RangeError: min (0) is
        # equal/bigger than max (0)` and the tab dies. The picker must render
        # the lone value as static text instead of a slider.
        at = _make_apptest()
        at.run(timeout=30)
        at.session_state["data_source_choice"] = "Synthetic test trial"
        at.run(timeout=30)
        at.session_state["single_select_trial_mode"] = "Participant"
        at.run(timeout=30)
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
        slider_keys = [s.key for s in at.select_slider]
        assert "single_slider" not in slider_keys, (
            "single-option select_slider rendered — this crashes the browser"
        )
        captions = [c.value for c in at.caption]
        assert any("only one available" in c for c in captions)

    def test_single_trial_all_selection_modes(self):
        # Trial / Text / Participant must all resolve the lone synthetic trial.
        for mode in ["Trial", "Text", "Participant"]:
            at = _make_apptest()
            at.run(timeout=30)
            at.session_state["data_source_choice"] = "Synthetic test trial"
            at.run(timeout=30)
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
        # Flip the new plot options (color-by-line, out-of-text, gray
        # background) and re-run the whole app to exercise those code paths.
        at = _make_apptest()
        at.run(timeout=30)
        at.session_state["global_color_by_line"] = True
        at.session_state["global_highlight_out_of_text"] = True
        at.session_state["global_bg_choice"] = "Gray"
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
        # coordinates — exactly the screenshot's failure.
        raw_words = pd.DataFrame(
            {"reader": ["r0", "r0"], "stim": ["b0", "b0"], "token": ["Um", "das"]}
        )
        monkeypatch.setattr(
            app,
            "load_words_and_fixations",
            lambda *a, **k: (raw_words, pd.DataFrame()),
        )

        at = _make_apptest()
        at.session_state["data_source_choice"] = app.UPLOAD_CHOICE
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
        at = _make_apptest()
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
        at = _make_apptest()
        at.run(timeout=30)
        assert at.session_state["tour_seen"] is True
        assert "tour_next" in self._tour_buttons(at)
        # Any later full rerun must NOT reopen the dialog (e.g. after the
        # user dismissed it with X) — only the replay button may.
        at.run(timeout=30)
        assert "tour_next" not in self._tour_buttons(at)

    def test_tour_next_advances_step(self):
        at = _make_apptest()
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
        at = _make_apptest()
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
        at = _make_apptest()
        at.run(timeout=30)
        assert at.session_state["tour_seen"] is True
        assert at.session_state["tour_mode"] == "spotlight"
        assert "tour_sp_next" in self._sp_buttons(at)
        # The dialog style must NOT also open.
        assert not any(b.key == "tour_next" for b in at.button)

    def test_spotlight_navigates_and_exits(self):
        from scanpath_studio.tour import _SPOTLIGHT_STEPS

        at = _make_apptest()
        at.run(timeout=30)
        at.button(key="tour_sp_next").click()
        at.run(timeout=30)
        assert at.session_state["tour_step"] == 1
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

        at = _make_apptest()
        at.run(timeout=30)
        at.session_state["tour_step"] = len(_SPOTLIGHT_STEPS) - 1
        at.run(timeout=30)
        assert "tour_sp_done" in self._sp_buttons(at)
        at.button(key="tour_sp_done").click()
        at.run(timeout=30)
        assert at.session_state["tour_mode"] is None
        assert not at.exception, f"Streamlit exceptions: {at.exception}"

    def test_spotlight_replay(self):
        at = _make_apptest()
        at.run(timeout=30)
        at.button(key="tour_sp_exit").click()
        at.run(timeout=30)
        assert self._sp_buttons(at) == set()
        at.button(key="tour_replay").click()
        at.run(timeout=30)
        assert "tour_sp_next" in self._sp_buttons(at)
        assert at.session_state["tour_step"] == 0
        assert not at.exception, f"Streamlit exceptions: {at.exception}"
