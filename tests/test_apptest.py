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
