"""Pytest configuration and fixtures for scanpath visualization tests."""

import pandas as pd
import pytest


@pytest.fixture
def sample_words_df():
    """Create a sample words/IA dataframe for testing."""
    return pd.DataFrame({
        "participant_id": ["p1", "p1", "p1", "p2", "p2"],
        "trial_id": ["t1", "t1", "t1", "t1", "t1"],
        "word_id": [1, 2, 3, 1, 2],
        "IA_LEFT": [100, 200, 300, 100, 200],
        "IA_RIGHT": [150, 250, 350, 150, 250],
        "IA_TOP": [50, 50, 50, 50, 50],
        "IA_BOTTOM": [100, 100, 100, 100, 100],
        "IA_LABEL": ["word1", "word2", "word3", "word1", "word2"],
        "paragraph_id": ["para1", "para1", "para1", "para1", "para1"],
    })


@pytest.fixture
def sample_fixations_df():
    """Create a sample fixations dataframe for testing."""
    return pd.DataFrame({
        "participant_id": ["p1", "p1", "p1", "p2", "p2"],
        "trial_id": ["t1", "t1", "t1", "t1", "t1"],
        "CURRENT_FIX_X": [125, 225, 325, 125, 225],
        "CURRENT_FIX_Y": [75, 75, 75, 75, 75],
        "CURRENT_FIX_DURATION": [200, 250, 180, 220, 190],
        "CURRENT_FIX_START": [0, 200, 450, 0, 220],
        "CURRENT_FIX_INTEREST_AREA_ID": [1, 2, 3, 1, 2],
    })


@pytest.fixture
def sample_raw_gaze_df():
    """Create a sample raw gaze dataframe for testing."""
    return pd.DataFrame({
        "participant_id": ["p1", "p1", "p1", "p1", "p1"],
        "trial_id": ["t1", "t1", "t1", "t1", "t1"],
        "x": [120, 125, 130, 220, 225],
        "y": [70, 75, 80, 70, 75],
        "timestamp": [0, 1, 2, 3, 4],
    })


@pytest.fixture
def normalized_words_df():
    """Create a normalized words dataframe for testing."""
    return pd.DataFrame({
        "participant_id": ["p1", "p1", "p1"],
        "trial_id": ["t1", "t1", "t1"],
        "word_id": [1, 2, 3],
        "x": [100, 200, 300],
        "y": [50, 50, 50],
        "width": [50, 50, 50],
        "height": [50, 50, 50],
        "text": ["word1", "word2", "word3"],
        "paragraph_id": ["para1", "para1", "para1"],
        "line_idx": [1, 1, 1],
    })


@pytest.fixture
def normalized_fixations_df():
    """Create a normalized fixations dataframe for testing."""
    return pd.DataFrame({
        "participant_id": ["p1", "p1", "p1"],
        "trial_id": ["t1", "t1", "t1"],
        "x": [125, 225, 325],
        "y": [75, 75, 75],
        "duration_ms": [200, 250, 180],
        "timestamp_ms": [0, 200, 450],
        "word_id": [1, 2, 3],
        "order_in_trial": [1, 2, 3],
        "pass_index": [1, 1, 1],
        "saccade_type": ["RIGHT", "RIGHT", "LEFT"],
        "eye": ["Both", "Both", "Both"],
        "noise_flag": [False, False, False],
    })
