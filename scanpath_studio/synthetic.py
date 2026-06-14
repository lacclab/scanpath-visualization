"""A small, fully-specified synthetic scanpath — ground truth for testing + a
visualizable in-app data source ("Synthetic test trial").

Real eye-tracking exports are noisy and their reading measures can only be
checked approximately. This module ships a tiny hand-built trial whose geometry
and reading sequence are simple enough that *every* canonical measure has an
exact, hand-traced expected value (see ``EXPECTED``). The app exposes it as a
data source so you can eyeball the scanpath against that ground truth; the test
suite (``tests/test_synthetic.py``) asserts the measures match.

Layout — 6 words, 2 lines of 3, 50x20 px boxes on a clean grid::

        x=100..150   x=200..250   x=300..350
    y=100..120   [0 The]      [1 cat]      [2 sat]      (line 0, y-center 110)
    y=200..220   [3 on ]      [4 the]      [5 mat]      (line 1, y-center 210)

Reading sequence (9 fixations) — first-pass refixation on word 0, a regression
from word 2 back to word 1, and one deliberately out-of-text fixation::

    #  t(ms)   (x,  y)     dur  lands on
    1     0   (115,110)   100   word 0   (first fixation)
    2   100   (135,110)    50   word 0   (refixation, same first pass)
    3   150   (225,110)   150   word 1
    4   300   (325,110)   120   word 2
    5   420   (225,110)    80   word 1   (regression back)
    6   500   (125,210)   200   word 3
    7   700   (225,210)    90   word 4
    8   790   (700,700)    60   --       (out of text; word_id NaN)
    9   850   (325,210)   110   word 5
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Human-friendly ids so the trial reads clearly in the app's trial picker.
PARTICIPANT = "synthetic"
TRIAL = "synthetic_2line_demo"
PARAGRAPH = "synthetic_2line_demo"

_WORD_TEXT = ["The", "cat", "sat", "on", "the", "mat"]
_WORD_X = [100, 200, 300, 100, 200, 300]
_WORD_Y = [100, 100, 100, 200, 200, 200]
_BOX_W = 50
_BOX_H = 20


def make_synthetic_words() -> pd.DataFrame:
    """Return the normalized words/IA frame for the synthetic trial."""
    n = len(_WORD_TEXT)
    return pd.DataFrame(
        {
            "participant_id": [PARTICIPANT] * n,
            "trial_id": [TRIAL] * n,
            "text_id": [PARAGRAPH] * n,
            "word_id": list(range(n)),
            "text": list(_WORD_TEXT),
            # line_idx is intentionally a constant: real OneStop IA exports do
            # not carry a usable per-word line number, so the by-line feature
            # must infer lines from geometry, not trust this column.
            "line_idx": [1] * n,
            "x": list(_WORD_X),
            "y": list(_WORD_Y),
            "width": [_BOX_W] * n,
            "height": [_BOX_H] * n,
        }
    )


# (x, y, duration_ms, timestamp_ms) for each fixation in temporal order.
_FIX_ROWS = [
    (115, 110, 100, 0),
    (135, 110, 50, 100),
    (225, 110, 150, 150),
    (325, 110, 120, 300),
    (225, 110, 80, 420),
    (125, 210, 200, 500),
    (225, 210, 90, 700),
    (700, 700, 60, 790),
    (325, 210, 110, 850),
]


def make_synthetic_fixations() -> pd.DataFrame:
    """Return the normalized fixations frame for the synthetic trial.

    ``word_id`` is left NaN on purpose so that fixation-to-word assignment runs
    and can itself be tested / exercised.
    """
    n = len(_FIX_ROWS)
    return pd.DataFrame(
        {
            "participant_id": [PARTICIPANT] * n,
            "trial_id": [TRIAL] * n,
            "text_id": [PARAGRAPH] * n,
            "x": [r[0] for r in _FIX_ROWS],
            "y": [r[1] for r in _FIX_ROWS],
            "duration_ms": [r[2] for r in _FIX_ROWS],
            "timestamp_ms": [r[3] for r in _FIX_ROWS],
            "word_id": [np.nan] * n,
            "order_in_trial": list(range(1, n + 1)),
            "pass_index": [1] * n,
        }
    )


def load_synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(words, fixations) for the "Synthetic test trial" data source."""
    return make_synthetic_words(), make_synthetic_fixations()


# Hand-traced expected values. Keyed by word_id (0..5) unless noted.
EXPECTED = {
    # Word-box line clustering (top-to-bottom, 0-based), per word_id 0..5.
    "word_line": [0, 0, 0, 1, 1, 1],
    # Fixation -> word_id via bbox containment (NaN for the out-of-text fix).
    "fixation_word_id": [0, 0, 1, 2, 1, 3, 4, np.nan, 5],
    # Fixation -> nearest text line (the out-of-text fix snaps to line 1).
    "fixation_line": [0, 0, 0, 0, 0, 1, 1, 1, 1],
    # In-text mask (True = inside some word box): exactly one out-of-text fix.
    "in_text": [True, True, True, True, True, True, True, False, True],
    "out_of_text_count": 1,
    # Per-word reading measures.
    "first_fixation_ms": {0: 100, 1: 150, 2: 120, 3: 200, 4: 90, 5: 110},
    "first_pass_gaze_duration_ms": {0: 150, 1: 150, 2: 120, 3: 200, 4: 90, 5: 110},
    "regression_path_duration_ms": {0: 150, 1: 150, 2: 200, 3: 200, 4: 90, 5: 110},
    "total_fixation_duration_ms": {0: 150, 1: 230, 2: 120, 3: 200, 4: 90, 5: 110},
    "n_fixations": {0: 2, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1},
    "skip_flag": {0: False, 1: False, 2: False, 3: False, 4: False, 5: False},
    "regression_in_flag": {0: False, 1: True, 2: False, 3: False, 4: False, 5: False},
    "regression_out_flag": {0: False, 1: False, 2: True, 3: False, 4: False, 5: False},
}
