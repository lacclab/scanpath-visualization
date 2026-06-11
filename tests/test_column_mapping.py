"""Tests for the column-mapping sidebar UI (controls.column_mapping_ui).

The Trial ID field is multi-capable: selecting several columns means "build a
unique trial ID on the fly by joining their values" (see data.trial_id_series).
"""

from __future__ import annotations

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest


def _mapping_app():
    """Minimal app rendering the Fixations column-mapping UI over a frame
    that has no recognizable unique-trial column."""
    import pandas as pd
    import streamlit as st

    from scanpath_studio.controls import FIX_FIELD_SPECS, column_mapping_ui
    from scanpath_studio.data import propose_fix_schema

    df = pd.DataFrame(
        {
            "participant_id": ["p1"],
            "paragraph": ["A"],
            "repeated": [False],
            "CURRENT_FIX_X": [1.0],
            "CURRENT_FIX_Y": [2.0],
            "CURRENT_FIX_DURATION": [100],
        }
    )
    mapping = column_mapping_ui(
        df,
        table_label="Fixations",
        state_key_prefix="col_map_fix",
        field_specs=FIX_FIELD_SPECS,
        proposed=propose_fix_schema(df),
    )
    st.session_state["_result_mapping"] = mapping


@pytest.mark.timeout(60)
class TestTrialMappingMultiselect:
    def test_trial_field_renders_as_multiselect(self):
        at = AppTest.from_function(_mapping_app)
        at.run(timeout=15)
        assert not at.exception
        keys = [m.key for m in at.multiselect]
        assert "col_map_fix_trial" in keys

    def test_single_selection_returns_plain_string(self):
        at = AppTest.from_function(_mapping_app)
        at.run(timeout=15)
        at.multiselect(key="col_map_fix_trial").set_value(["paragraph"]).run(timeout=15)
        mapping = at.session_state["_result_mapping"]
        assert mapping["trial"] == "paragraph"

    def test_multi_selection_returns_column_list(self):
        at = AppTest.from_function(_mapping_app)
        at.run(timeout=15)
        at.multiselect(key="col_map_fix_trial").set_value(
            ["participant_id", "paragraph", "repeated"]
        ).run(timeout=15)
        mapping = at.session_state["_result_mapping"]
        assert mapping["trial"] == ["participant_id", "paragraph", "repeated"]

    def test_empty_selection_returns_none_and_fails_validation(self):
        from scanpath_studio.data import validate_fix_schema

        at = AppTest.from_function(_mapping_app)
        at.run(timeout=15)
        at.multiselect(key="col_map_fix_trial").set_value([]).run(timeout=15)
        mapping = at.session_state["_result_mapping"]
        assert mapping["trial"] is None
        assert any("Trial" in p for p in validate_fix_schema(mapping))

    def test_composite_mapping_normalizes_end_to_end(self):
        """The list mapping returned by the UI feeds normalize_fixations and
        yields the joined on-the-fly unique trial id."""
        import pandas as pd

        from scanpath_studio.data import normalize_fixations

        at = AppTest.from_function(_mapping_app)
        at.run(timeout=15)
        at.multiselect(key="col_map_fix_trial").set_value(
            ["participant_id", "paragraph", "repeated"]
        ).run(timeout=15)
        mapping = at.session_state["_result_mapping"]

        df = pd.DataFrame(
            {
                "participant_id": ["p1", "p1"],
                "paragraph": ["A", "A"],
                "repeated": [False, True],
                "CURRENT_FIX_X": [1.0, 2.0],
                "CURRENT_FIX_Y": [2.0, 3.0],
                "CURRENT_FIX_DURATION": [100, 90],
            }
        )
        result = normalize_fixations(df, mapping)
        assert result["trial_id"].tolist() == ["p1_A_False", "p1_A_True"]
        assert (result["unique_trial_id"] == result["trial_id"]).all()
