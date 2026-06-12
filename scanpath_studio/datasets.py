"""Loaders for public eye-tracking-while-reading corpora.

Currently: PoTeC (Potsdam Textbook Corpus, Jakobi et al. 2024,
https://github.com/DiLi-Lab/PoTeC) — a German corpus of 75 readers × 12
textbook texts, and a working example of two dataset shapes the generic
pipeline supports:

* **multi-file** — fixations ship as one TSV per reader × text
  (``reader0_b0_scanpath.tsv`` … 900 files), concatenated on load;
* **stimulus-level AoIs** — word bounding boxes ship once per *text* with no
  participant column, and are broadcast across readers
  (``data.broadcast_stimulus_words``).

PoTeC fixations carry no pixel coordinates — only the fixated character's
index. The loader reconstructs (x, y) as the center of that character's
bounding box from the per-text ``.ias`` AOI files, giving within-word landing
positions. (For AOI-sequence datasets *without* character AOIs, the generic
fallback places fixations at word-box centers instead.)

Typical use::

    from scanpath_studio.datasets import load_potec

    words, fixations = load_potec("data/PoTeC", download=True)
    fig = scanpath_studio.plot_scanpath(words, fixations, participant="0", trial="b0")
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd

# PoTeC text p3 contains the German word "null" — pandas' default NA list
# would turn it into NaN (see the PoTeC README), so every PoTeC table is read
# with keep_default_na=False and this explicit list.
_POTEC_NA_VALUES = [
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NaN",
    "None",
    "n/a",
    "nan",
    "",
]

_POTEC_TEXTS = [f"{domain}{i}" for domain in ("b", "p") for i in range(6)]

# OSF storage ids from the PoTeC repo's download_data_files.py.
_POTEC_OSF_URL = "https://osf.io/download/{resource}"
_POTEC_OSF_RESOURCES = {
    "scanpaths": "thgv2",
    "fixations": "53zwb",
    "reading_measures": "g5jds",
}
_POTEC_RAW_URL = "https://raw.githubusercontent.com/DiLi-Lab/PoTeC/main/{path}"


def _read_potec_tsv(path) -> pd.DataFrame:
    return pd.read_csv(
        path, sep="\t", keep_default_na=False, na_values=_POTEC_NA_VALUES
    )


def download_potec(root, *, fixation_source: str = "scanpaths") -> Path:
    """Download the PoTeC files :func:`load_potec` needs into ``root``.

    Fetches the per-trial eye-tracking archive (~45 MB zip) from PoTeC's OSF
    repository and the 24 per-text AOI files (word boxes + character boxes)
    from the PoTeC GitHub repo. Skips anything already present, so it's safe
    to call repeatedly (and it's a no-op on a full clone of the PoTeC repo
    where ``download_data_files.py`` has been run).

    ``fixation_source`` is ``"scanpaths"`` (default; temporally ordered
    fixations with word indices) or ``"fixations"``.
    """
    if fixation_source not in _POTEC_OSF_RESOURCES:
        raise ValueError(
            f"fixation_source must be one of {sorted(_POTEC_OSF_RESOURCES)}, "
            f"got {fixation_source!r}"
        )
    root = Path(root)

    eyetracking_dir = root / "eyetracking_data" / fixation_source
    if not eyetracking_dir.is_dir():
        url = _POTEC_OSF_URL.format(resource=_POTEC_OSF_RESOURCES[fixation_source])
        print(f"Downloading PoTeC {fixation_source} from {url} …")
        with urllib.request.urlopen(url) as response:
            payload = response.read()
        (root / "eyetracking_data").mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [
                m
                for m in archive.namelist()
                # The OSF zips carry macOS resource-fork cruft; keep only the
                # real per-trial TSVs.
                if m.startswith(f"{fixation_source}/") and m.endswith(".tsv")
            ]
            archive.extractall(root / "eyetracking_data", members=members)

    for text_id in _POTEC_TEXTS:
        for rel in (
            f"stimuli/word_aoi_texts/word_aoi_{text_id}.tsv",
            f"stimuli/aoi_texts/{text_id}.ias",
        ):
            dest = root / rel
            if dest.is_file():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = _POTEC_RAW_URL.format(path=rel)
            print(f"Downloading {url} …")
            with urllib.request.urlopen(url) as response:
                dest.write_bytes(response.read())
    return root


def _potec_words(root: Path, texts: Iterable[str]) -> pd.DataFrame:
    """Stimulus-level word table: one row per word per text, with boxes.

    PoTeC keys the text id in the AOI *filename* only; it becomes a regular
    ``text_id`` column here. Line indices come from the character-level
    ``.ias`` files (the word AOI files don't carry them) via the lines' y
    positions."""
    frames = []
    for text_id in texts:
        path = root / "stimuli" / "word_aoi_texts" / f"word_aoi_{text_id}.tsv"
        if not path.is_file():
            raise FileNotFoundError(
                f"PoTeC word AOI file not found: {path} — pass download=True "
                "or run PoTeC's download_data_files.py in a repo clone."
            )
        words = _read_potec_tsv(path)
        words["text_id"] = text_id

        ias = _read_potec_ias(root, text_id)
        # Character boxes on the same text line share start_y, so the char
        # AOIs give an exact y → line lookup for the word boxes.
        y_to_line = ias.drop_duplicates("start_y").set_index("start_y")["line"]
        words["line"] = words["start_y"].map(y_to_line)
        frames.append(words)
    return pd.concat(frames, ignore_index=True)


def _read_potec_ias(root: Path, text_id: str) -> pd.DataFrame:
    path = root / "stimuli" / "aoi_texts" / f"{text_id}.ias"
    if not path.is_file():
        raise FileNotFoundError(
            f"PoTeC character AOI file not found: {path} — pass download=True "
            "or run PoTeC's download_data_files.py in a repo clone."
        )
    return _read_potec_tsv(path)


def _potec_fixations(
    root: Path,
    texts: Iterable[str],
    readers: Optional[Iterable] = None,
) -> pd.DataFrame:
    """Concatenated per-trial fixation files with reconstructed coordinates.

    Prefers ``eyetracking_data/scanpaths/`` (fixations in temporal order, with
    word indices) and falls back to ``eyetracking_data/fixations/``. Each
    fixation's (x, y) is the center of the fixated character's box from the
    per-text ``.ias`` file — PoTeC discards the original screen coordinates."""
    base = root / "eyetracking_data"
    source = next(
        (s for s in ("scanpaths", "fixations") if (base / s).is_dir()), None
    )
    if source is None:
        raise FileNotFoundError(
            f"No PoTeC fixation data under {base} — expected a 'scanpaths' or "
            "'fixations' folder. Pass download=True, or run PoTeC's "
            "download_data_files.py in a repo clone."
        )
    suffix = "scanpath" if source == "scanpaths" else "fixations"

    reader_set = None if readers is None else {str(r) for r in readers}
    frames = []
    for text_id in texts:
        char_boxes = _read_potec_ias(root, text_id)
        char_x = (char_boxes["start_x"] + char_boxes["end_x"]) / 2.0
        char_y = (char_boxes["start_y"] + char_boxes["end_y"]) / 2.0
        centers = pd.DataFrame(
            {"aoi": char_boxes["aoi"], "x": char_x, "y": char_y}
        ).drop_duplicates("aoi")

        for path in sorted((base / source).glob(f"reader*_{text_id}_{suffix}.tsv")):
            reader_id = path.stem.removeprefix("reader").split("_")[0]
            if reader_set is not None and reader_id not in reader_set:
                continue
            fixations = _read_potec_tsv(path)
            fixations = fixations.merge(centers, on="aoi", how="left")
            frames.append(fixations)
    if not frames:
        raise FileNotFoundError(
            f"No PoTeC fixation files matched the requested readers/texts "
            f"under {base / source}."
        )
    return pd.concat(frames, ignore_index=True, sort=False)


# Column mappings from the raw PoTeC frames to the canonical schema. Explicit
# (rather than relying on auto-detection) so the loader stays stable even if
# PoTeC adds columns. No participant on words: the word boxes are
# stimulus-level and get broadcast across readers. Shared by load_potec and
# the app's PoTeC data source (which auto-detects, but these document intent).
POTEC_WORD_SCHEMA = dict(
    participant=None,
    trial="text_id",
    word_id="aoi",
    text="word",
    line="line",
    left="start_x",
    right="end_x",
    top="start_y",
    bottom="end_y",
)
POTEC_FIX_SCHEMA = dict(
    participant="reader_id",
    trial="text_id",
    duration="fixation_duration",
    x="x",
    y="y",
    fixation_id="fixation_index",
    word_id="word_index_in_text",
)

# PoTeC presentation monitor (DELL P2210, 60 Hz). Pass as ``canvas_size`` to
# plot_scanpath for true-to-scale rendering.
POTEC_MONITOR = (1680, 1050)


def potec_raw_frames(
    root,
    *,
    readers: Optional[Iterable] = None,
    texts: Optional[Iterable[str]] = None,
    download: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Raw (pre-normalization) PoTeC ``(words, fixations)`` frames.

    Same inputs as :func:`load_potec`, but returns the frames *before* schema
    normalization — for callers that run their own auto-detection / column
    mapping (e.g. the Streamlit app's PoTeC data source). Word boxes are
    stimulus-level (one row per word per text, ``text_id`` column, no
    participant); fixations carry reconstructed ``x``/``y`` from the fixated
    character's box center. Use :func:`load_potec` for the normalized,
    ready-to-plot frames.
    """
    root = Path(root)
    if download:
        download_potec(root)
    texts = list(texts) if texts is not None else list(_POTEC_TEXTS)
    unknown = sorted(set(texts) - set(_POTEC_TEXTS))
    if unknown:
        raise ValueError(f"Unknown PoTeC text ids: {unknown} (valid: {_POTEC_TEXTS})")
    return _potec_words(root, texts), _potec_fixations(root, texts, readers)


def load_potec(
    root,
    *,
    readers: Optional[Iterable] = None,
    texts: Optional[Iterable[str]] = None,
    download: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load PoTeC as normalized ``(words, fixations)`` frames, ready to plot.

    ``root`` is a clone of the PoTeC repo (with the eye-tracking data
    downloaded) or any folder; with ``download=True`` the needed files are
    fetched into it on first use (~45 MB). Narrow the load with ``readers``
    (e.g. ``[0, 1]``) and/or ``texts`` (e.g. ``["b0", "p3"]``) — the full
    corpus is 75 readers × 12 texts = 900 trials.

    Participants are PoTeC reader ids (as strings), trials are text ids
    (``b0``–``b5`` biology, ``p0``–``p5`` physics)::

        words, fixations = load_potec("data/PoTeC", readers=[0], texts=["b0"])
        fig = scanpath_studio.plot_scanpath(words, fixations)

    The PoTeC monitor was 1680×1050 (DELL P2210, 60 Hz); pass that as
    ``canvas_size`` to :func:`scanpath_studio.plot_scanpath` for true-to-scale
    rendering.
    """
    words_raw, fixations_raw = potec_raw_frames(
        root, readers=readers, texts=texts, download=download
    )

    from . import api

    return api.load_scanpath_data(
        words=words_raw,
        fixations=fixations_raw,
        word_schema=dict(POTEC_WORD_SCHEMA),
        fix_schema=dict(
            POTEC_FIX_SCHEMA,
            word_id=(
                "word_index_in_text"
                if "word_index_in_text" in fixations_raw.columns
                else None
            ),
        ),
    )
