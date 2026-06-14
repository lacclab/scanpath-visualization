"""Build a small, representative demo subset from the OneStop eye-tracking exports.

The bundled sample data ships with the wheel and powers the "Bundled Demo"
mode. We aim for a corpus that lets users actually exercise every feature
without uploading their own data: a handful of participants reading the same
paragraphs at both Adv and Ele difficulty levels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
DEFAULT_SOURCE_DIR = DATA_DIR / "OneStop"
DEFAULT_OUTPUT_DIR = DATA_DIR / "sample_data"

# Read at most this many rows from each giant source CSV; large enough to span
# many participants and trials while staying memory-friendly.
DEFAULT_MAX_ROWS = 300_000

IA_KEEP_COLUMNS = [
    "participant_id",
    "TRIAL_INDEX",
    "trial_index",
    "article_batch",
    "article_id",
    "paragraph_id",
    "difficulty_level",
    "repeated_reading_trial",
    "article_title",
    "question",
    "selected_answer",
    "is_correct",
    "IA_ID",
    "IA_LABEL",
    "IA_LEFT",
    "IA_RIGHT",
    "IA_TOP",
    "IA_BOTTOM",
    "IA_DWELL_TIME",
    "IA_FIRST_FIXATION_DURATION",
    "IA_FIRST_RUN_DWELL_TIME",
    "IA_SECOND_RUN_DWELL_TIME",
    "IA_LAST_RUN_DWELL_TIME",
    "IA_FIXATION_COUNT",
    "IA_SKIP",
    "IA_REGRESSION_IN",
    "IA_REGRESSION_IN_COUNT",
    "IA_REGRESSION_OUT",
    "IA_REGRESSION_OUT_COUNT",
    "IA_REGRESSION_PATH_DURATION",
    "TRIAL_DWELL_TIME",
    "TRIAL_FIXATION_COUNT",
    "TRIAL_IA_COUNT",
    "word_length",
    "word_length_no_punctuation",
    "wordfreq_frequency",
    "subtlex_frequency",
    "gpt2_surprisal",
    "universal_pos",
    "ptb_pos",
    "Reduced_POS",
    "dependency_relation",
    "head_word_index",
    "distance_to_head",
    "morphological_features",
    "entity_type",
    # Critical-/distractor-span columns enable the Hunting-condition overlay
    # in plots.py / tabs.py. Bool flags drive the per-word coloring; the
    # ind_* + indices columns are normalized through but not yet read by the
    # app — they're kept for parity with the OneStop server bundle so reviewers
    # comparing demo vs production see the same metadata table.
    "question_preview",
    "is_in_aspan",
    "is_in_dspan",
    "aspan_ind_start",
    "aspan_ind_end",
    "dspan_ind_start",
    "dspan_ind_end",
    "critical_span_indices",
    "distractor_span_indices",
]

FIXATION_KEEP_COLUMNS = [
    "participant_id",
    "TRIAL_INDEX",
    "trial_index",
    "article_batch",
    "article_id",
    "paragraph_id",
    "difficulty_level",
    "repeated_reading_trial",
    "article_title",
    "question",
    "selected_answer",
    "is_correct",
    "CURRENT_FIX_INDEX",
    "CURRENT_FIX_START",
    "CURRENT_FIX_END",
    "CURRENT_FIX_DURATION",
    "CURRENT_FIX_X",
    "CURRENT_FIX_Y",
    "CURRENT_FIX_INTEREST_AREA_ID",
    "CURRENT_FIX_INTEREST_AREA_LABEL",
    "CURRENT_FIX_VALIDITY",
    "NEXT_SAC_DIRECTION",
    "NEXT_SAC_AMPLITUDE",
    "EYE_TRACKED",
    "question_preview",  # needed by normalize_fixations so trial-level Hunting flag survives
]


def _resolve_csv_path(source_csv: Path) -> Path:
    """Accept either `foo.csv` or `foo.csv.zip` — fall back to the zipped
    variant when the plain CSV isn't on disk. Lacclab exports ship zipped."""
    if source_csv.exists():
        return source_csv
    zipped = source_csv.with_suffix(source_csv.suffix + ".zip")
    if zipped.exists():
        return zipped
    raise FileNotFoundError(f"Neither {source_csv} nor {zipped} found.")


def load_subset(
    source_csv: Path, preferred_columns: Iterable[str], max_rows: int
) -> pd.DataFrame:
    source_csv = _resolve_csv_path(source_csv)
    available_cols = pd.read_csv(source_csv, nrows=0, low_memory=False).columns
    use_cols = [col for col in preferred_columns if col in available_cols]
    missing_core = [
        col
        for col in ["participant_id", "TRIAL_INDEX", "repeated_reading_trial"]
        if col not in use_cols
    ]
    if missing_core:
        raise RuntimeError(f"Missing required columns {missing_core} in {source_csv}")
    return pd.read_csv(source_csv, usecols=use_cols, nrows=max_rows, low_memory=False)


def normalize_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["repeated_reading_trial"] = (
        df["repeated_reading_trial"].fillna(False).astype(bool)
    )
    return df


def add_unique_ids(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "article_batch",
        "article_id",
        "paragraph_id",
        "difficulty_level",
        "participant_id",
        "repeated_reading_trial",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns for id creation: {missing}")
    df = normalize_flags(df)
    df["unique_paragraph_id"] = (
        df["article_batch"].astype(str)
        + "_"
        + df["article_id"].astype(str)
        + "_"
        + df["paragraph_id"].astype(str)
        + "_"
        + df["difficulty_level"].astype(str)
    )
    df["unique_trial_id"] = (
        df["participant_id"].astype(str)
        + "_"
        + df["unique_paragraph_id"]
        + "_r"
        + df["repeated_reading_trial"].astype(int).astype(str)
    )
    return df


def pick_demo_slice(
    ia_df: pd.DataFrame,
    n_participants: int,
    n_articles: int,
    seed: int,
) -> tuple[list[str], list[tuple[int, int]]]:
    """Pick participants + articles so each participant reads every article,
    and across participants both Adv and Ele appear for each article.

    Reading-research observation: in OneStop each participant reads a given
    article at exactly one difficulty. To still demo Adv/Ele contrasts, we
    need different participants reading the same article at different levels.
    """
    rng_seed = seed
    first_reading = ia_df[~ia_df["repeated_reading_trial"].astype(bool)].copy()
    first_reading["article"] = list(
        zip(first_reading["article_batch"], first_reading["article_id"])
    )

    # Participant → set of (article, difficulty) tuples
    pa_diffs = (
        first_reading.groupby(["participant_id", "article", "difficulty_level"])
        .size()
        .reset_index(name="n_paragraphs")
    )

    # Greedy search: try random participant samples, pick the one whose joint
    # article coverage spans both difficulties on the most articles.
    candidates = sorted(first_reading["participant_id"].astype(str).unique())
    # Prefer participants who have at least one Hunting (preview) trial — the
    # demo is the only way for someone without OneStop access to exercise the
    # critical-span overlay, and only Hunting trials trigger it. ~43% of L2
    # pids have Hunting, so this leaves plenty of search room. Falls back to
    # the full candidate pool if `question_preview` isn't in the data.
    if "question_preview" in first_reading.columns:
        hunters = set(
            first_reading[first_reading["question_preview"].fillna(False).astype(bool)][
                "participant_id"
            ]
            .astype(str)
            .unique()
        )
        candidates = [c for c in candidates if c in hunters] or candidates
    if len(candidates) < n_participants:
        raise RuntimeError(
            f"Only {len(candidates)} participants in slice; need {n_participants}."
        )

    rng = pd.Series(candidates)
    best_score = -1
    best_participants: list[str] = []
    best_articles: list = []
    for trial in range(200):
        sample = rng.sample(n=n_participants, random_state=rng_seed + trial).tolist()
        sample_diffs = pa_diffs[pa_diffs["participant_id"].isin(sample)]
        # For each article, how many distinct difficulties appear in this sample?
        per_article = sample_diffs.groupby("article")["difficulty_level"].nunique()
        spanning = per_article[per_article >= 2]
        # Articles all sample participants read (joint coverage)
        per_p_articles = sample_diffs.groupby("participant_id")["article"].apply(set)
        if len(per_p_articles) < n_participants:
            continue
        joint = set.intersection(*per_p_articles.values)
        useful = [a for a in joint if a in spanning.index]
        score = len(useful)
        if score > best_score:
            best_score = score
            best_participants = sorted(sample)
            best_articles = sorted(useful)
        if best_score >= n_articles:
            break

    if best_score < n_articles:
        raise RuntimeError(
            f"Could not find {n_participants} participants jointly reading "
            f"{n_articles} articles that span Adv+Ele. Best found: {best_score}. "
            "Try increasing --max-rows or relaxing constraints."
        )

    return best_participants, list(best_articles[:n_articles])


def filter_demo(
    df: pd.DataFrame,
    participants: list[str],
    articles: list[tuple[int, int]],
) -> pd.DataFrame:
    article_tuples = set(articles)
    mask = df["participant_id"].isin(participants) & (
        ~df["repeated_reading_trial"].astype(bool)
    )
    df = df[mask].copy()
    df["_article"] = list(zip(df["article_batch"], df["article_id"]))
    df = df[df["_article"].isin(article_tuples)].drop(columns=["_article"])
    return df


def write_outputs(df: pd.DataFrame, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base.with_suffix(".csv"), index=False)
    try:
        df.to_parquet(base.with_suffix(".parquet"), index=False)
    except Exception as exc:
        print(f"Parquet write failed for {base}: {exc}")


# ---------------------------------------------------------------------------
# Raw-gaze sample synthesis.
#
# OneStop's exports are IA/fixation reports only — there are no raw, sample-
# level (ms) gaze recordings. To still let the bundled demo exercise the
# "Show raw gaze data" overlay, we synthesize an illustrative gaze path for ONE
# real bundled trial: a small jittered cloud around each fixation for its
# duration, joined by short interpolated saccade segments. It's keyed to a
# (participant, trial) present in BOTH the ia and fixation samples so it lines
# up with that trial's scanpath and isn't dropped by the trial filter.
# ---------------------------------------------------------------------------

RAW_GAZE_SAMPLE_DT_MS = 10  # ~100 Hz effective sampling
RAW_GAZE_JITTER_PX = 4.0
RAW_GAZE_SACCADE_POINTS = 6
RAW_GAZE_MAX_POINTS = 2500

_RAW_GAZE_COLUMNS = ["participant_id", "unique_trial_id", "x", "y", "timestamp_ms"]


def synthesize_raw_gaze(
    fixations: pd.DataFrame, ia: pd.DataFrame, *, seed: int = 20260520
) -> pd.DataFrame:
    """Build an illustrative ms-level gaze path for one real bundled trial.

    Returns an empty frame (with the canonical columns) if the required
    ``CURRENT_FIX_*`` columns are missing or no trial is shared between the two
    tables.
    """
    empty = pd.DataFrame(columns=_RAW_GAZE_COLUMNS)
    needed = {
        "participant_id",
        "unique_trial_id",
        "CURRENT_FIX_X",
        "CURRENT_FIX_Y",
        "CURRENT_FIX_DURATION",
        "CURRENT_FIX_START",
    }
    if fixations.empty or ia.empty or not needed.issubset(fixations.columns):
        return empty

    fix = fixations.assign(
        _p=fixations["participant_id"].astype(str),
        _t=fixations["unique_trial_id"].astype(str),
    )
    ia_keys = set(
        zip(ia["participant_id"].astype(str), ia["unique_trial_id"].astype(str))
    )
    sizes = fix.groupby(["_p", "_t"]).size().reset_index(name="n")
    sizes = sizes[[(p, t) in ia_keys for p, t in zip(sizes["_p"], sizes["_t"])]]
    if sizes.empty:
        return empty
    # Deterministic pick: most fixations, then lexicographic id tie-break.
    sizes = sizes.sort_values(["n", "_p", "_t"], ascending=[False, True, True])
    pid, tid = str(sizes.iloc[0]["_p"]), str(sizes.iloc[0]["_t"])

    trial = fix[(fix["_p"] == pid) & (fix["_t"] == tid)].sort_values(
        "CURRENT_FIX_START"
    )
    xs = pd.to_numeric(trial["CURRENT_FIX_X"], errors="coerce").to_numpy(float)
    ys = pd.to_numeric(trial["CURRENT_FIX_Y"], errors="coerce").to_numpy(float)
    durs = (
        pd.to_numeric(trial["CURRENT_FIX_DURATION"], errors="coerce")
        .fillna(0)
        .to_numpy(float)
    )
    starts = (
        pd.to_numeric(trial["CURRENT_FIX_START"], errors="coerce")
        .fillna(0)
        .to_numpy(float)
    )

    rng = np.random.default_rng(seed)
    px: list[float] = []
    py: list[float] = []
    pt: list[float] = []
    n = len(trial)
    for i in range(n):
        n_dwell = max(1, int(round(durs[i] / RAW_GAZE_SAMPLE_DT_MS)))
        px.extend(xs[i] + rng.normal(0.0, RAW_GAZE_JITTER_PX, n_dwell))
        py.extend(ys[i] + rng.normal(0.0, RAW_GAZE_JITTER_PX, n_dwell))
        pt.extend(starts[i] + np.linspace(0.0, durs[i], n_dwell))
        if i < n - 1:
            seg = np.linspace(0.0, 1.0, RAW_GAZE_SACCADE_POINTS + 2)[1:-1]
            px.extend(xs[i] + seg * (xs[i + 1] - xs[i]))
            py.extend(ys[i] + seg * (ys[i + 1] - ys[i]))
            t0 = starts[i] + durs[i]
            t1 = starts[i + 1] if starts[i + 1] > t0 else t0 + RAW_GAZE_SACCADE_POINTS
            pt.extend(t0 + seg * (t1 - t0))

    out = pd.DataFrame(
        {
            "participant_id": pid,
            "unique_trial_id": tid,
            "x": np.round(px, 1),
            "y": np.round(py, 1),
            "timestamp_ms": np.round(pt).astype(int),
        }
    )
    if len(out) > RAW_GAZE_MAX_POINTS:
        step = int(np.ceil(len(out) / RAW_GAZE_MAX_POINTS))
        out = out.iloc[::step].reset_index(drop=True)
    return out


def _read_bundle(output_dir: Path, name: str) -> pd.DataFrame:
    """Load a bundled sample table, preferring Parquet over CSV."""
    parquet = output_dir / f"{name}.parquet"
    csv = output_dir / f"{name}.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv, low_memory=False)
    raise FileNotFoundError(f"Neither {parquet} nor {csv} found.")


def regenerate_raw_gaze_from_bundle(output_dir: Path, *, seed: int = 20260520) -> None:
    """(Re)build raw_gaze.{csv,parquet} from the already-written ia/fixation
    samples — no access to the multi-GB source CSVs required."""
    ia = _read_bundle(output_dir, "ia")
    fix = _read_bundle(output_dir, "fixations")
    raw_gaze = synthesize_raw_gaze(fix, ia, seed=seed)
    if raw_gaze.empty:
        print("Raw-gaze synthesis produced no rows (no shared ia/fixation trial).")
        return
    write_outputs(raw_gaze, output_dir / "raw_gaze")
    print(
        f"Raw-gaze sample written: {len(raw_gaze)} pts for "
        f"{raw_gaze['participant_id'].iloc[0]} / {raw_gaze['unique_trial_id'].iloc[0]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--participants", type=int, default=3, help="Number of participants to bundle."
    )
    parser.add_argument(
        "--articles",
        type=int,
        default=2,
        help="Number of articles (each spanning Adv + Ele) to bundle.",
    )
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help="Cap on rows read from each source CSV.",
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--raw-gaze-only",
        action="store_true",
        help="Skip the source rebuild; only regenerate raw_gaze.{csv,parquet} "
        "from the already-bundled ia/fixation samples.",
    )
    args = parser.parse_args()

    if args.raw_gaze_only:
        regenerate_raw_gaze_from_bundle(args.output_dir, seed=args.seed)
        return

    ia_full = add_unique_ids(
        load_subset(
            args.source_dir / "ia_Paragraph.csv", IA_KEEP_COLUMNS, args.max_rows
        )
    )
    fix_full = add_unique_ids(
        load_subset(
            args.source_dir / "fixations_Paragraph.csv",
            FIXATION_KEEP_COLUMNS,
            args.max_rows,
        )
    )

    participants, articles = pick_demo_slice(
        ia_full, args.participants, args.articles, args.seed
    )

    ia_sample = filter_demo(ia_full, participants, articles)
    fix_sample = filter_demo(fix_full, participants, articles)

    write_outputs(ia_sample, args.output_dir / "ia")
    write_outputs(fix_sample, args.output_dir / "fixations")

    # Synthesize a raw-gaze overlay sample tied to one of these trials.
    regenerate_raw_gaze_from_bundle(args.output_dir, seed=args.seed)

    # Drop the legacy *_full CSVs so they don't bloat the wheel.
    for legacy in [
        args.output_dir / "ia_full.csv",
        args.output_dir / "fixations_full.csv",
    ]:
        if legacy.exists():
            legacy.unlink()

    print(
        "Demo slice written:\n"
        f"  participants: {participants}\n"
        f"  articles: {articles}\n"
        f"  ia rows: {len(ia_sample):,}\n"
        f"  fixation rows: {len(fix_sample):,}\n"
        f"  output: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
