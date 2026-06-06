"""Re-export the synthetic ground-truth trial from the package.

The dataset now lives in ``scanpath_studio.synthetic`` so it is a
single source of truth shared by the test suite and the in-app "Synthetic test
trial" data source. This shim keeps the historical ``tests.synthetic_data``
import path working.
"""

from __future__ import annotations

from scanpath_studio.synthetic import (
    EXPECTED,
    PARAGRAPH,
    PARTICIPANT,
    TRIAL,
    make_synthetic_fixations,
    make_synthetic_words,
)

__all__ = [
    "EXPECTED",
    "PARAGRAPH",
    "PARTICIPANT",
    "TRIAL",
    "make_synthetic_fixations",
    "make_synthetic_words",
]
