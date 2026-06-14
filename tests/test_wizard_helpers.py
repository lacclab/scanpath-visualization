"""Unit tests for the setup-wizard's pure helper functions (Group B/C).

These are deterministic and dependency-light, so they're tested directly instead
of through the heavy AppTest path.
"""

from __future__ import annotations

import pandas as pd

from scanpath_studio import app


class TestDefaultTrialColumns:
    def test_composes_paragraph_and_text_when_both_present(self):
        # The user's "default to both paragraph id and text id".
        proposed = {"trial": "trial_id", "text_id": "text_id"}
        present = ["paragraph_id", "text_id", "participant_id"]
        assert app._default_trial_columns(proposed, present) == [
            "paragraph_id",
            "text_id",
        ]

    def test_prefers_single_unique_trial_over_redundant_composite(self):
        # OneStop-shaped upload: a precomputed unique_trial_id plus a paragraph id.
        # Pairing them would force opaque composite ids for no benefit.
        proposed = {"trial": "unique_trial_id", "text_id": "unique_paragraph_id"}
        present = ["unique_trial_id", "unique_paragraph_id", "participant_id"]
        assert app._default_trial_columns(proposed, present) == ["unique_trial_id"]

    def test_paragraph_only_falls_back_to_trial_proposal(self):
        proposed = {"trial": "trial_id", "text_id": None}
        present = ["trial_id", "participant_id"]
        assert app._default_trial_columns(proposed, present) == ["trial_id"]

    def test_restricted_to_present_columns(self):
        # A proposed trial column absent from the common columns is dropped.
        proposed = {"trial": "trial_id", "text_id": "text_id"}
        present = ["participant_id"]  # neither trial nor text present
        assert app._default_trial_columns(proposed, present) == []


class TestTrialIdValues:
    def test_single_column_mapping(self):
        raw = pd.DataFrame({"trial_id": ["a", "a", "b"]})
        assert app._trial_id_values(raw, {"trial": "trial_id"}) == {"a", "b"}

    def test_composite_mapping_joins_components(self):
        raw = pd.DataFrame({"p": ["x", "x"], "q": ["1", "2"]})
        assert app._trial_id_values(raw, {"trial": ["p", "q"]}) == {"x_1", "x_2"}

    def test_absent_column_returns_none(self):
        raw = pd.DataFrame({"trial_id": ["a"]})
        assert app._trial_id_values(raw, {"trial": "missing"}) is None

    def test_unmapped_trial_returns_none(self):
        raw = pd.DataFrame({"trial_id": ["a"]})
        assert app._trial_id_values(raw, {}) is None


class TestSafeDatasetName:
    def test_reserved_label_is_suffixed(self):
        assert (
            app._safe_dataset_name(app.DEMO_CHOICE) == f"{app.DEMO_CHOICE} (uploaded)"
        )

    def test_plain_name_passes_through_trimmed(self):
        assert app._safe_dataset_name("  My data  ") == "My data"
