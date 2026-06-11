"""The trial picker breaks a composite trial id into one selector per part.

Mirrors the Text / Participant cascading modes instead of one opaque
``a_b_c`` dropdown (utils._select_trial_composite_mode).
"""

from __future__ import annotations

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest


def _picker_app():
    """Render select_trial over a small composite-trial combos frame."""
    import pandas as pd
    import streamlit as st

    from scanpath_studio.utils import build_combo_options, select_trial

    # Two texts, two participants, one repeated reading — trial id composed of
    # unique_paragraph_id + participant_id + repeated_reading_trial.
    fixations = pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2", "p1"],
            "unique_paragraph_id": ["A", "B", "A", "A"],
            "repeated_reading_trial": [False, False, False, True],
        }
    )
    fixations["trial_id"] = (
        fixations[["unique_paragraph_id", "participant_id", "repeated_reading_trial"]]
        .astype(str)
        .agg("_".join, axis=1)
    )
    fixations["unique_trial_id"] = fixations["trial_id"]
    fixations["paragraph_id"] = fixations["unique_paragraph_id"]

    st.session_state["_composite_trial_columns"] = [
        "unique_paragraph_id",
        "participant_id",
        "repeated_reading_trial",
    ]
    combos, _, _ = build_combo_options(fixations)
    participant, trial, mode, text = select_trial(combos, key_prefix="single")
    st.session_state["_picked"] = (participant, trial, mode, text)


@pytest.mark.timeout(60)
class TestCompositeTrialPicker:
    def test_renders_one_selector_per_component(self):
        at = AppTest.from_function(_picker_app)
        at.run(timeout=15)
        assert not at.exception
        labels = [s.label for s in at.selectbox]
        # Canonical components reuse the friendly Participant / Text wording;
        # the extra component shows its raw column name.
        assert "Text" in labels
        assert "Participant" in labels
        assert "repeated_reading_trial" in labels

    def test_no_opaque_unique_trial_dropdown(self):
        at = AppTest.from_function(_picker_app)
        at.run(timeout=15)
        labels = [s.label for s in at.selectbox]
        assert "Unique trial id" not in labels

    def test_cascading_selection_resolves_a_trial(self):
        at = AppTest.from_function(_picker_app)
        at.run(timeout=15)
        at.selectbox(key="single_composite_unique_paragraph_id").set_value("B").run(
            timeout=15
        )
        participant, trial, mode, text = at.session_state["_picked"]
        assert mode == "Trial"
        assert trial == "B_p1_False"
        assert participant == "p1"
        assert text == "B"

    def test_later_selector_narrows_to_valid_options(self):
        # Text A has participants p1 and p2; Text B only p1. After picking B the
        # Participant selector must drop p2 (no stale value, no crash).
        at = AppTest.from_function(_picker_app)
        at.run(timeout=15)
        at.selectbox(key="single_composite_participant_id").set_value("p2").run(
            timeout=15
        )
        at.selectbox(key="single_composite_unique_paragraph_id").set_value("B").run(
            timeout=15
        )
        assert not at.exception
        participant = at.selectbox(key="single_composite_participant_id").value
        assert participant == "p1"
        _, trial, _, _ = at.session_state["_picked"]
        assert trial == "B_p1_False"

    def test_single_column_mapping_keeps_plain_dropdown(self):
        # Sanity: with no composite columns flagged, Trial mode still shows the
        # single unique-trial dropdown.
        def _plain_app():
            import pandas as pd
            import streamlit as st

            from scanpath_studio.utils import build_combo_options, select_trial

            fixations = pd.DataFrame(
                {
                    "participant_id": ["p1", "p2"],
                    "trial_id": ["t1", "t2"],
                    "paragraph_id": ["A", "B"],
                }
            )
            st.session_state["_composite_trial_columns"] = None
            combos, _, _ = build_combo_options(fixations)
            select_trial(combos, key_prefix="single")

        at = AppTest.from_function(_plain_app)
        at.run(timeout=15)
        assert not at.exception
        labels = [s.label for s in at.selectbox]
        assert "Unique trial id" in labels
