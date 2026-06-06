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
        # The four-tab UI should render without exceptions
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
