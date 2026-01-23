"""Build trimmed sample CSV tables from the OneStop exports."""

from pathlib import Path
from typing import Iterable

import pandas as pd


DATA_DIR = Path(__file__).parent
SOURCE_DIR = DATA_DIR / "OneStop"
OUTPUT_DIR = DATA_DIR / "sample_data"

# Limit how much of the giant source csvs we touch; keep this large enough to span
# multiple participants and trials while staying memory-friendly.
MAX_ROWS = 200_000

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
    "TRIAL_DWELL_TIME",
    "TRIAL_FIXATION_COUNT",
    "TRIAL_IA_COUNT",
    "word_length",
    "word_length_no_punctuation",
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
    "CURRENT_FIX_DURATION",
    "CURRENT_FIX_X",
    "CURRENT_FIX_Y",
    "CURRENT_FIX_INTEREST_AREA_ID",
    "CURRENT_FIX_VALIDITY",
    "NEXT_SAC_DIRECTION",
    "EYE_TRACKED",
]


def load_subset(source_csv: Path, preferred_columns: Iterable[str]) -> pd.DataFrame:
    """Read only the columns we need (plus MAX_ROWS cap) to keep memory down."""
    available_cols = pd.read_csv(source_csv, nrows=0, low_memory=False).columns
    use_cols = [col for col in preferred_columns if col in available_cols]
    missing_core = [col for col in ["participant_id", "TRIAL_INDEX", "repeated_reading_trial"] if col not in use_cols]
    if missing_core:
        raise RuntimeError(f"Missing required columns {missing_core} in {source_csv}")
    return pd.read_csv(source_csv, usecols=use_cols, nrows=MAX_ROWS, low_memory=False)


def normalize_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["repeated_reading_trial"] = df["repeated_reading_trial"].fillna(False).astype(bool)
    return df


def add_unique_ids(df: pd.DataFrame) -> pd.DataFrame:
    required = ["article_batch", "article_id", "paragraph_id", "difficulty_level", "participant_id", "repeated_reading_trial"]
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


def choose_participants(df: pd.DataFrame, count: int = 3) -> list[str]:
    unique = df["participant_id"].dropna().astype(str).unique().tolist()
    if len(unique) < count:
        raise RuntimeError(f"Found only {len(unique)} participants in provided slice: {unique}")
    return unique[:count]


def choose_trials(df: pd.DataFrame, participants: list[str], trials_per_participant: int = 3) -> list[str]:
    trials: list[str] = []
    for pid in participants:
        subset = df[(df["participant_id"] == pid) & (df["repeated_reading_trial"] == False)]  # noqa: E712
        if subset.empty:
            raise RuntimeError(f"No trials found for participant {pid}")
        ordered = (
            subset[["unique_trial_id", "TRIAL_INDEX"]]
            .drop_duplicates()
            .sort_values(by=["TRIAL_INDEX", "unique_trial_id"])
        )
        chosen = ordered["unique_trial_id"].head(trials_per_participant).tolist()
        if len(chosen) < trials_per_participant:
            raise RuntimeError(f"Only found {len(chosen)} trials for participant {pid}")
        trials.extend(chosen)
    return trials


def trim_columns(df: pd.DataFrame, keep: Iterable[str]) -> pd.DataFrame:
    keep_cols = [col for col in keep if col in df.columns]
    keep_cols.extend([col for col in ["unique_paragraph_id", "unique_trial_id"] if col in df.columns and col not in keep_cols])
    return df[keep_cols].copy()


def filter_and_save(df: pd.DataFrame, participants: list[str], trials: list[str], output_csv: Path) -> int:
    if "unique_trial_id" not in df.columns:
        raise RuntimeError("Expected unique_trial_id column in data.")

    filtered = df[
        df["participant_id"].isin(participants)
        & df["unique_trial_id"].isin(trials)
        & (df["repeated_reading_trial"] == False)  # noqa: E712
    ]

    if filtered.empty:
        raise RuntimeError(f"No rows matched participants {participants} and trials {trials}.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)
    return len(filtered)


def main() -> None:
    ia_df = add_unique_ids(load_subset(SOURCE_DIR / "ia_Paragraph.csv", IA_KEEP_COLUMNS))
    fix_df = add_unique_ids(load_subset(SOURCE_DIR / "fixations_Paragraph.csv", FIXATION_KEEP_COLUMNS))

    ia_trimmed = trim_columns(ia_df, IA_KEEP_COLUMNS)
    fix_trimmed = trim_columns(fix_df, FIXATION_KEEP_COLUMNS)

    # Save the full (trimmed) slices alongside the smaller samples.
    ia_full_path = OUTPUT_DIR / "ia_full.csv"
    fix_full_path = OUTPUT_DIR / "fixations_full.csv"
    ia_full_path.parent.mkdir(parents=True, exist_ok=True)
    ia_trimmed.to_csv(ia_full_path, index=False)
    fix_trimmed.to_csv(fix_full_path, index=False)

    participants = choose_participants(ia_trimmed)
    trials = choose_trials(ia_trimmed, participants)

    ia_rows = filter_and_save(ia_trimmed, participants, trials, OUTPUT_DIR / "ia.csv")
    print(
        f"Wrote {ia_rows} IA rows for participants {participants} and trials {trials} "
        f"to {OUTPUT_DIR / 'ia.csv'}"
    )

    fix_rows = filter_and_save(fix_trimmed, participants, trials, OUTPUT_DIR / "fixations.csv")
    print(
        f"Wrote {fix_rows} fixation rows for participants {participants} and trials {trials} "
        f"to {OUTPUT_DIR / 'fixations.csv'}"
    )


if __name__ == "__main__":
    main()
