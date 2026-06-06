"""Scanpath similarity metrics for comparing scanpaths over the same text.

The reference use case is the **Multiple Comparison** tab: score how close each
model-generated scanpath is to the real reading of the same paragraph.

Currently one metric is implemented for real:

- **NLD** — Normalized Levenshtein Distance on the area-of-interest (word-ID)
  sequence. A standard scanpath-similarity measure (edit distance over AOI/word-
  index sequences predates and is used well beyond any one model), and the
  primary evaluation metric reported by Eyettention (Deng, Reich, Prasse, Haller,
  Scheffer & Jäger, 2023, *Eyettention: An Attention-based Dual-Sequence Model for
  Predicting Human Scanpaths during Reading*). Each scanpath is reduced to the
  ordered list of word indices its fixations land on — each fixation's own
  recorded word/AOI label when present, otherwise a bounding-box assignment with a
  small line-misregistration tolerance. NLD is the edit distance between two such
  lists divided by the longer list's length, so it sits in ``[0, 1]`` (0 =
  identical sequence, 1 = maximally different). Lower is better.

Three further metrics are registered as **labeled placeholders** (``fn=None``)
so the comparison table is laid out and ready for ScanMatch / MultiMatch /
Scasim (etc.) to be dropped in later — they currently render as "—".

The module is deliberately pure (no Streamlit, no plotting) so the metrics can
be unit-tested against the hand-traced :mod:`scanpath_studio.synthetic` trial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .measures import (
    LINE_MISREGISTRATION_PX,
    _assign_word_ids_single,
    rebased_fixation_onsets,
)


# -----------------------------------------------------------------------------
# Edit distance
# -----------------------------------------------------------------------------


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Levenshtein (edit) distance between two sequences of hashable items.

    Classic two-row dynamic program, O(len(a) * len(b)) time and O(len(b))
    space. Scanpaths are tens-to-low-hundreds of fixations long, so this is
    plenty fast.
    """
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[m]


def normalized_levenshtein(a: Sequence, b: Sequence) -> float:
    """NLD in ``[0, 1]``: edit distance divided by the longer sequence's length.

    Two empty sequences are treated as identical (0.0); one empty and one
    non-empty are maximally different (1.0, since the distance equals the
    non-empty length).
    """
    denom = max(len(a), len(b))
    if denom == 0:
        return 0.0
    return levenshtein(a, b) / float(denom)


# -----------------------------------------------------------------------------
# Fixation -> word (AOI) sequence
# -----------------------------------------------------------------------------


def assign_single_trial_word_ids(
    fixations: pd.DataFrame,
    words: pd.DataFrame,
    *,
    nearest_within_px: float = LINE_MISREGISTRATION_PX,
) -> np.ndarray:
    """Word id for every fixation via bounding-box containment, single trial.

    Thin wrapper over :func:`scanpath_studio.measures._assign_word_ids_single`
    that does **not** group by ``participant_id`` / ``trial_id`` — the generated
    model scanpaths carry synthetic ids (e.g. ``"Model 1"``) that don't match the
    real trial's, so the grouped :func:`measures.assign_fixations_to_words` would
    find no matching word boxes and return all-NaN. Here every fixation is tested
    against the one ``words`` frame passed in. Fixations outside every box snap to
    the nearest word centre within ``nearest_within_px`` (line-misregistration
    tolerance), else NaN.

    Returns a float array aligned to ``fixations`` rows (NaN = out of text).
    """
    if fixations.empty or words.empty:
        return np.full(len(fixations), np.nan)
    return _assign_word_ids_single(fixations, words, nearest_within_px)


def _ordered_fixations(fixations: pd.DataFrame) -> pd.DataFrame:
    """Fixations in reading order (by ``timestamp_ms`` when available)."""
    if "timestamp_ms" in fixations.columns and not fixations.empty:
        return fixations.sort_values("timestamp_ms")
    return fixations


def ordered_word_ids(
    fixations: pd.DataFrame,
    words: pd.DataFrame,
    *,
    nearest_within_px: float = 50.0,
) -> np.ndarray:
    """Per-fixation word id in reading order; NaN where out of text.

    Prefers each fixation's own ``word_id`` (the corpus AOI label, or — for
    generated scanpaths — the word it was drawn over); only fixations without one
    are mapped geometrically via :func:`assign_single_trial_word_ids`.
    """
    ordered = _ordered_fixations(fixations)
    if ordered.empty or words.empty:
        return np.array([], dtype=float)
    geometric = assign_single_trial_word_ids(
        ordered, words, nearest_within_px=nearest_within_px
    )
    if "word_id" in ordered.columns:
        existing = pd.to_numeric(ordered["word_id"], errors="coerce").to_numpy(
            dtype=float
        )
        return np.where(np.isnan(existing), geometric, existing)
    return geometric


def aoi_sequence(
    fixations: pd.DataFrame,
    words: pd.DataFrame,
    *,
    nearest_within_px: float = 50.0,
) -> List[int]:
    """Temporal sequence of fixated word ids (out-of-text fixations dropped).

    Fixations are read in ``timestamp_ms`` order when that column is present
    (it always is after normalization), so the sequence reflects reading order.
    """
    word_ids = ordered_word_ids(fixations, words, nearest_within_px=nearest_within_px)
    return [int(w) for w in word_ids if pd.notna(w)]


# -----------------------------------------------------------------------------
# Metric registry
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanpathMetric:
    """One column of the similarity table.

    ``fn`` maps ``(reference_ctx, generated_ctx)`` to a float; both contexts are
    the dicts built by :func:`_context` (precomputed AOI sequence + the raw
    frames). ``fn=None`` marks a placeholder — the column is shown but no value
    is computed yet.
    """

    key: str
    label: str
    description: str
    lower_is_better: bool
    fn: Optional[Callable[[dict, dict], float]]


def _nld_metric(reference: dict, generated: dict) -> float:
    return normalized_levenshtein(reference["aoi"], generated["aoi"])


# Metric 1 is real (NLD); 2-4 are placeholders with the standard scanpath
# metrics they're earmarked for, so plugging in a real implementation is just
# swapping ``fn=None`` for the function.
METRICS: List[ScanpathMetric] = [
    ScanpathMetric(
        key="nld",
        label="NLD",
        description=(
            "Normalized Levenshtein Distance on the word-index (AOI) sequence — "
            "a standard scanpath-similarity measure and the primary metric "
            "reported by Eyettention (Deng et al. 2023). 0 = identical sequence, "
            "1 = maximally different. Lower is better."
        ),
        lower_is_better=True,
        fn=_nld_metric,
    ),
    ScanpathMetric(
        key="scanmatch",
        label="ScanMatch",
        description=(
            "Placeholder — ScanMatch (Cristino et al. 2010): Needleman–Wunsch "
            "alignment with a spatial substitution matrix. Not yet computed."
        ),
        lower_is_better=False,
        fn=None,
    ),
    ScanpathMetric(
        key="multimatch",
        label="MultiMatch",
        description=(
            "Placeholder — MultiMatch (Dewhurst et al. 2012): vector, direction, "
            "length, position and duration sub-scores. Not yet computed."
        ),
        lower_is_better=False,
        fn=None,
    ),
    ScanpathMetric(
        key="scasim",
        label="Scasim",
        description=(
            "Placeholder — Scasim (von der Malsburg & Vasishth 2011): a "
            "saccade-sensitive, duration-aware similarity. Not yet computed."
        ),
        lower_is_better=True,
        fn=None,
    ),
]


def _context(fixations: pd.DataFrame, words: pd.DataFrame) -> dict:
    """Precompute the per-scanpath inputs the metric functions share."""
    return {
        "fix": fixations,
        "words": words,
        "aoi": aoi_sequence(fixations, words),
    }


def compute_similarity_table(
    reference_fixations: pd.DataFrame,
    model_fixations: Dict[str, pd.DataFrame],
    words: pd.DataFrame,
) -> pd.DataFrame:
    """Score every model scanpath against the reference, one row per model.

    Args:
        reference_fixations: the real scanpath's fixations (single trial).
        model_fixations: ordered ``{model_name: fixations_df}``.
        words: the shared word boxes (the real trial's words).

    Returns:
        DataFrame with a ``Model`` column plus one column per metric label.
        Placeholder metrics are filled with ``NaN``; a metric that raises is
        also recorded as ``NaN`` rather than aborting the whole table.
    """
    reference_ctx = _context(reference_fixations, words)
    rows = []
    for name, fix in model_fixations.items():
        generated_ctx = _context(fix, words)
        row: Dict[str, object] = {"Model": name}
        for metric in METRICS:
            if metric.fn is None:
                row[metric.label] = np.nan
                continue
            try:
                row[metric.label] = float(metric.fn(reference_ctx, generated_ctx))
            except Exception:
                row[metric.label] = np.nan
        rows.append(row)
    columns = ["Model"] + [m.label for m in METRICS]
    return pd.DataFrame(rows, columns=columns)


# -----------------------------------------------------------------------------
# Cumulative metric curves (for the Multiple Comparison convergence plots)
# -----------------------------------------------------------------------------


def _eval_indices(n: int, max_points: int) -> List[int]:
    """Up to ``max_points`` prefix lengths in 1..n (all of them when n is small),
    always including 1 and n. Keeps the convergence sweep fast on long trials."""
    if n <= 0:
        return []
    if n <= max_points:
        return list(range(1, n + 1))
    return sorted({int(round(v)) for v in np.linspace(1, n, max_points)})


def nld_by_fixation_index(
    reference_fixations: pd.DataFrame,
    model_fixations: pd.DataFrame,
    words: pd.DataFrame,
    *,
    max_points: int = 80,
) -> Tuple[List[int], List[float]]:
    """NLD between the first-*k*-fixations prefixes of the two scanpaths, vs k.

    For each prefix length k the reference and model are each truncated to their
    first k fixations (reading order) and scored. Returns ``(ks, nlds)``; for
    long scanpaths the k axis is subsampled to ``max_points`` evenly spaced
    values (kept exhaustive when the scanpath is short).
    """
    ref_w = ordered_word_ids(reference_fixations, words)
    mod_w = ordered_word_ids(model_fixations, words)
    n = max(len(ref_w), len(mod_w))
    ks = _eval_indices(n, max_points)
    nlds = []
    for k in ks:
        ref_seq = [int(w) for w in ref_w[:k] if pd.notna(w)]
        mod_seq = [int(w) for w in mod_w[:k] if pd.notna(w)]
        nlds.append(normalized_levenshtein(ref_seq, mod_seq))
    return ks, nlds


def _rebased_onsets(fixations: pd.DataFrame) -> np.ndarray:
    """Fixation onset times (ms) rebased so the first fixation is t=0.

    Orders the fixations by ``timestamp_ms`` and delegates the recorded-vs-
    synthetic-timestamp heuristic to
    :func:`scanpath_studio.measures.rebased_fixation_onsets` (shared with the
    animation clock).
    """
    return rebased_fixation_onsets(_ordered_fixations(fixations))


def nld_by_time(
    reference_fixations: pd.DataFrame,
    model_fixations: pd.DataFrame,
    words: pd.DataFrame,
    *,
    max_points: int = 80,
) -> Tuple[List[float], List[float]]:
    """NLD between the prefixes up to elapsed reading time t, vs t in seconds.

    Both scanpaths are rebased to their first fixation; at each sample time the
    reference and model are truncated to the fixations whose onset ≤ t and
    scored. Sample times are the union of both scanpaths' fixation onsets
    (subsampled to ``max_points`` for long readings). Returns ``(t_seconds,
    nlds)``.
    """
    # ordered_word_ids and _rebased_onsets each order by timestamp_ms internally
    # (stable sort), so their outputs stay row-aligned without a pre-sort here.
    ref_w = ordered_word_ids(reference_fixations, words)
    mod_w = ordered_word_ids(model_fixations, words)
    ref_on = _rebased_onsets(reference_fixations)
    mod_on = _rebased_onsets(model_fixations)
    times = sorted({float(t) for t in (*ref_on, *mod_on)})
    if not times:
        return [], []
    if len(times) > max_points:
        picks = np.linspace(0, len(times) - 1, max_points)
        times = sorted({times[int(round(i))] for i in picks})
    xs, nlds = [], []
    for t in times:
        ref_seq = [int(w) for w, o in zip(ref_w, ref_on) if o <= t and pd.notna(w)]
        mod_seq = [int(w) for w, o in zip(mod_w, mod_on) if o <= t and pd.notna(w)]
        xs.append(t / 1000.0)
        nlds.append(normalized_levenshtein(ref_seq, mod_seq))
    return xs, nlds
