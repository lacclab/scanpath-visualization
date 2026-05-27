"""One-time prep: shard the OneStop lacclab CSV exports into per-participant Parquet.

The full IA + fixations reports are ~15GB CSV (5GB zipped) and take ~3 min to
load into pandas. The scanpath app only ever shows ONE participant at a time
when deep-linked from an external review tool (see
`data.load_onestop_server_bundle` and `app._apply_url_preset`), so loading
the whole cohort is pure overhead.

This script writes:

    <ONESTOP_DATA_DIR>/by_pid/
        ia/<pid>.parquet
        fixations/<pid>.parquet

…one file per participant_id. After running, opening a deep-linked scanpath
page loads in under a second instead of ~3 min.

Usage:

    python -m scanpath_visualization_app.onestop_shard \\
        --data-dir /path/to/onestop_<cohort>/reports/<source>/<date>/full/

By default skips pids whose shards already exist; pass `--rebuild` to force.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


_BIG_FILES = {
    # output subdir → source CSV.zip name
    "ia": "ia_Paragraph.csv.zip",
    "fixations": "fixations_Paragraph.csv.zip",
}


def _shard_one(
    csv_path: Path,
    out_dir: Path,
    kind: str,
    rebuild: bool,
) -> tuple[int, int, int]:
    """Read one big CSV.zip, write one Parquet per participant_id.

    Returns (written_count, skipped_count, total_pid_count).
    """
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"  [{kind}] reading {csv_path.name} ({csv_path.stat().st_size / 1e6:.0f} MB zipped)…",
        flush=True,
    )
    t0 = time.time()
    # low_memory=False so mixed-type columns don't crash the writer later.
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  [{kind}] loaded {len(df):,} rows in {time.time() - t0:.0f}s", flush=True)

    if "participant_id" not in df.columns:
        raise ValueError(f"{csv_path.name}: no participant_id column")

    pids = df["participant_id"].dropna().astype(str).str.lower().unique()
    written, skipped = 0, 0
    for i, pid in enumerate(sorted(pids), 1):
        out_path = out_dir / f"{pid}.parquet"
        if out_path.exists() and not rebuild:
            skipped += 1
            continue
        sub = df[df["participant_id"].astype(str).str.lower() == pid]
        # pyarrow doesn't like fully-empty object columns; coerce them to str
        # so write_parquet works without a schema dance.
        sub.to_parquet(out_path, index=False)
        written += 1
        if i % 50 == 0 or i == len(pids):
            print(
                f"  [{kind}] {i}/{len(pids)} pids · wrote={written} skipped={skipped}",
                flush=True,
            )
    return written, skipped, len(pids)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        dest="data_dir",
        required=True,
        type=Path,
        help="OneStop lacclab export folder containing ia_Paragraph.csv.zip + "
        "fixations_Paragraph.csv.zip. Shards land under <data_dir>/by_pid/.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rewrite shards even if they already exist.",
    )
    parser.add_argument(
        "--only",
        choices=list(_BIG_FILES.keys()),
        help="Shard only one of {ia, fixations}.",
    )
    args = parser.parse_args(argv)

    base = args.data_dir.resolve()
    if not base.is_dir():
        parser.error(f"--data-dir not a directory: {base}")

    by_pid_root = base / "by_pid"
    targets = [args.only] if args.only else list(_BIG_FILES.keys())

    print(f"sharding into {by_pid_root}/ …")
    t_total = time.time()
    summary = []
    for kind in targets:
        out_dir = by_pid_root / kind
        csv_path = base / _BIG_FILES[kind]
        try:
            w, s, n = _shard_one(csv_path, out_dir, kind, rebuild=args.rebuild)
        except Exception as e:
            print(f"  [{kind}] FAILED: {e!r}", file=sys.stderr)
            sys.exit(1)
        summary.append((kind, w, s, n))

    print()
    print(f"done in {time.time() - t_total:.0f}s")
    for kind, w, s, n in summary:
        print(f"  {kind:10s}: {n} pids · wrote {w} · skipped {s}")


if __name__ == "__main__":
    main()
