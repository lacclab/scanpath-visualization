"""Synthetic "model-generated" scanpaths over a real text's word boxes.

This is **placeholder data** for the *Multiple Comparison* tab. The intended
end state is: for a given participant reading a given text, several scanpath
*models* each predict a scanpath, and the tab shows the real reading alongside
each model's prediction plus similarity scores. Until those real model outputs
are connected, this module fabricates a reading-like scanpath per "model" so the
whole tab — grid layout, similarity table — can be built and demoed.

Design goals:

- **Reading-like, not uniform noise.** Each scanpath is a left-to-right walk
  over the word boxes with model-specific skip / regression / refixation rates,
  so the panels actually look like scanpaths and their NLD vs. the real reading
  spreads out across models.
- **Reproducible.** A scanpath is deterministic in ``(trial_id, model_index,
  nonce)``. Streamlit reruns the whole script on every interaction, so a
  non-seeded generator would reshuffle every panel on each click. The tab's
  "Regenerate" button bumps the ``nonce`` to deliberately re-draw.
- **Canonical output.** Frames carry the same columns
  :func:`scanpath_studio.data.normalize_fixations` produces, so they drop
  straight into :func:`scanpath_studio.plots.make_scanpath_figure` and the
  similarity metrics without special-casing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .measures import cluster_word_lines


@dataclass(frozen=True)
class ModelProfile:
    """Behaviour knobs for one synthetic model.

    ``skip_prob`` / ``regress_prob`` / ``refix_prob`` are per-step probabilities
    of skipping the next word, regressing to an earlier word, or refixating the
    current word. ``mean_dur_ms`` / ``dur_sd_ms`` shape fixation durations;
    ``jitter_frac`` is the Gaussian landing-position jitter as a fraction of the
    word-box size.
    """

    name: str
    skip_prob: float
    regress_prob: float
    refix_prob: float
    mean_dur_ms: float
    dur_sd_ms: float
    jitter_frac: float


# Ordered from most reading-like (Model 1: few skips/regressions, durations near
# typical reading) to progressively noisier. The tab slices this to the
# requested model count. Tuned so the panels look visibly different and the NLD
# vs. the real scanpath spreads across a useful range.
MODEL_PROFILES = [
    ModelProfile("Model 1", 0.08, 0.06, 0.10, 230, 60, 0.12),
    ModelProfile("Model 2", 0.15, 0.10, 0.12, 215, 70, 0.16),
    ModelProfile("Model 3", 0.22, 0.14, 0.10, 250, 80, 0.20),
    ModelProfile("Model 4", 0.30, 0.05, 0.18, 200, 65, 0.18),
    ModelProfile("Model 5", 0.18, 0.22, 0.14, 240, 90, 0.24),
    ModelProfile("Model 6", 0.35, 0.18, 0.20, 190, 75, 0.26),
    ModelProfile("Model 7", 0.12, 0.30, 0.16, 260, 85, 0.22),
    ModelProfile("Model 8", 0.45, 0.25, 0.22, 175, 70, 0.30),
]

MAX_MODELS = len(MODEL_PROFILES)
DEFAULT_N_MODELS = 6

# Inter-fixation gap folded into timestamps, approximating saccade + planning
# time so the synthetic real-time clock reads plausibly in the animation/metrics.
SACCADE_GAP_MS = 25

# Fixation durations are clamped to a plausible reading band (ms).
_MIN_DUR_MS = 50
_MAX_DUR_MS = 800

# Canonical fixation columns produced by data.normalize_fixations that the
# figure builder and metrics rely on. Kept here so an empty frame still has the
# right shape.
_FIX_COLUMNS = [
    "participant_id",
    "trial_id",
    "paragraph_id",
    "x",
    "y",
    "duration_ms",
    "timestamp_ms",
    "fixation_id",
    "word_id",
    "pass_index",
    "order_in_trial",
    "eye",
    "noise_flag",
    "saccade_type",
]


def _seed(trial_id: object, model_index: int, nonce: int) -> int:
    """Stable 32-bit seed from (trial, model, nonce).

    Uses md5 rather than the builtin ``hash`` because ``hash`` of a str is salted
    per process, which would make scanpaths non-reproducible across reruns.
    """
    raw = f"{trial_id}|{model_index}|{nonce}".encode("utf-8")
    return int(hashlib.md5(raw).hexdigest()[:8], 16)


def _empty_fix_frame() -> pd.DataFrame:
    return pd.DataFrame({col: [] for col in _FIX_COLUMNS})


def _ordered_word_rows(words: pd.DataFrame) -> pd.DataFrame:
    """Words in reading order: by ``word_id`` when usable, else line-then-x.

    ``word_id`` is reading order in OneStop. When it's missing or has gaps we
    fall back to clustering rows into visual lines by ``y`` (tolerance ~half a
    word height) and reading each line left-to-right.
    """
    if "word_id" in words.columns and words["word_id"].notna().all():
        return words.sort_values("word_id").reset_index(drop=True)
    # Cluster rows into visual lines by vertical position (shared with the
    # reading-measure geometry), then read each line left-to-right. NaN-safe
    # line pitch and the all-NaN-height guard live in ``cluster_word_lines``.
    w = words.copy()
    w["_line"] = cluster_word_lines(w)
    return w.sort_values(["_line", "x"]).drop(columns="_line").reset_index(drop=True)


def _walk_word_indices(n_words: int, profile: ModelProfile, rng) -> list[int]:
    """Random reading-like walk producing a list of fixated word positions.

    Positions index into the reading-ordered word list (0..n_words-1). Advances
    one word by default; with the profile's probabilities it skips ahead two,
    regresses one or two words, or refixates the current word. Bounded by a cap
    so heavy refixation/regression can't loop forever.
    """
    if n_words <= 0:
        return [0]
    seq: list[int] = []
    pos = 0
    cap = max(8, n_words * 3)
    consecutive_refix = 0
    while pos < n_words and len(seq) < cap:
        seq.append(pos)
        # Refixation: stay on the same word (bounded run so we always progress).
        if rng.random() < profile.refix_prob and consecutive_refix < 2:
            consecutive_refix += 1
            continue
        consecutive_refix = 0
        roll = rng.random()
        if roll < profile.regress_prob and pos > 0:
            pos = max(0, pos - int(rng.integers(1, 3)))
        elif roll < profile.regress_prob + profile.skip_prob:
            pos += 2
        else:
            pos += 1
    return seq or [0]


def generate_model_scanpath(
    words: pd.DataFrame,
    profile: ModelProfile,
    *,
    model_index: int,
    reference_trial_id: object,
    paragraph_id: Optional[object] = None,
    nonce: int = 0,
) -> pd.DataFrame:
    """One model's synthetic scanpath over ``words``, in canonical fixation form.

    Deterministic in ``(reference_trial_id, model_index, nonce)``.
    """
    if words is None or words.empty:
        return _empty_fix_frame()

    rng = np.random.default_rng(_seed(reference_trial_id, model_index, nonce))
    ordered = _ordered_word_rows(words)
    # Drop words whose box geometry failed to export (NaN x/y/width/height) so
    # generated fixations always land at finite coordinates — np.maximum(nan, 1)
    # is nan, so the jitter floor wouldn't otherwise sanitize them.
    finite_geom = (
        pd.to_numeric(ordered["x"], errors="coerce").notna()
        & pd.to_numeric(ordered["y"], errors="coerce").notna()
        & pd.to_numeric(ordered["width"], errors="coerce").notna()
        & pd.to_numeric(ordered["height"], errors="coerce").notna()
    )
    ordered = ordered[finite_geom.to_numpy()].reset_index(drop=True)
    if ordered.empty:
        return _empty_fix_frame()
    n_words = len(ordered)

    x_box = pd.to_numeric(ordered["x"], errors="coerce").to_numpy(dtype=float)
    y_box = pd.to_numeric(ordered["y"], errors="coerce").to_numpy(dtype=float)
    w_box = pd.to_numeric(ordered["width"], errors="coerce").to_numpy(dtype=float)
    h_box = pd.to_numeric(ordered["height"], errors="coerce").to_numpy(dtype=float)
    if "text" in ordered.columns:
        text_len = ordered["text"].astype(str).str.len().to_numpy(dtype=float)
    else:
        text_len = np.full(n_words, 5.0)
    word_ids = (
        ordered["word_id"].to_numpy()
        if "word_id" in ordered.columns
        else np.full(n_words, np.nan)
    )

    seq = _walk_word_indices(n_words, profile, rng)
    seq_idx = np.asarray(seq, dtype=int)
    k = len(seq_idx)

    # Landing position: word-box centre + Gaussian jitter. Vertical jitter is
    # scaled down (box height encodes the line pitch, ~3 text lines in OneStop)
    # so a fixation stays near its own line rather than drifting onto neighbours.
    cx = x_box[seq_idx] + w_box[seq_idx] / 2.0
    cy = y_box[seq_idx] + h_box[seq_idx] / 2.0
    jitter_x = rng.normal(0.0, profile.jitter_frac * np.maximum(w_box[seq_idx], 1.0))
    jitter_y = rng.normal(
        0.0, profile.jitter_frac * np.maximum(h_box[seq_idx], 1.0) / 3.0
    )
    fix_x = cx + jitter_x
    fix_y = cy + jitter_y

    # Duration: profile mean ± sd, nudged longer for longer words, then clamped.
    mean_len = float(text_len.mean()) if text_len.size else 5.0
    durations = rng.normal(profile.mean_dur_ms, profile.dur_sd_ms, size=k)
    durations = durations + (text_len[seq_idx] - mean_len) * 4.0
    durations = np.clip(durations, _MIN_DUR_MS, _MAX_DUR_MS).round().astype(int)

    # Cumulative timestamps with a saccade gap between fixations.
    timestamps = np.empty(k, dtype=int)
    acc = 0
    for i in range(k):
        timestamps[i] = acc
        acc += int(durations[i]) + SACCADE_GAP_MS

    para = paragraph_id if paragraph_id is not None else reference_trial_id
    return pd.DataFrame(
        {
            "participant_id": [profile.name] * k,
            "trial_id": [str(reference_trial_id)] * k,
            "paragraph_id": [str(para)] * k,
            "x": fix_x,
            "y": fix_y,
            "duration_ms": durations,
            "timestamp_ms": timestamps,
            "fixation_id": np.arange(1, k + 1),
            "word_id": word_ids[seq_idx],
            "pass_index": [1] * k,
            "order_in_trial": np.arange(1, k + 1),
            "eye": ["Both"] * k,
            "noise_flag": [False] * k,
            "saccade_type": ["unknown"] * k,
        }
    )


def generate_model_scanpaths(
    words: pd.DataFrame,
    *,
    n_models: int = DEFAULT_N_MODELS,
    reference_trial_id: object,
    paragraph_id: Optional[object] = None,
    nonce: int = 0,
) -> Dict[str, pd.DataFrame]:
    """Generate ``n_models`` synthetic scanpaths over ``words``.

    Returns an insertion-ordered ``{model_name: fixations_df}`` (Python dicts
    preserve order), so the tab grid and the similarity table list models in the
    same Model 1..N order.
    """
    count = max(1, min(int(n_models), MAX_MODELS))
    profiles = MODEL_PROFILES[:count]
    return {
        profile.name: generate_model_scanpath(
            words,
            profile,
            model_index=index,
            reference_trial_id=reference_trial_id,
            paragraph_id=paragraph_id,
            nonce=nonce,
        )
        for index, profile in enumerate(profiles)
    }
