"""Tests for flexible dataset support: multi-file inputs, single-report
datasets (words-only / fixations-only), stimulus-level word tables, AOI-only
fixations, and the PoTeC loader."""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

import scanpath_studio as sps
from scanpath_studio import data as data_module
from scanpath_studio import datasets as datasets_module
from scanpath_studio.plots import make_scanpath_figure


# ---------------------------------------------------------------------------
# Multi-file reading
# ---------------------------------------------------------------------------


def _write_fix_csv(path, participant, trial, n=3):
    pd.DataFrame(
        {
            "participant_id": [participant] * n,
            "trial_id": [trial] * n,
            "x": np.linspace(100, 300, n),
            "y": [80.0] * n,
            "duration_ms": [200.0] * n,
        }
    ).to_csv(path, index=False)


def test_read_table_tsv(tmp_path):
    path = tmp_path / "words.tsv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(path, sep="\t", index=False)
    df = data_module.read_table(path)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


class _NamedBytesIO(io.BytesIO):
    """A BytesIO that carries a ``name`` like Streamlit's UploadedFile, so we
    can exercise the upload path (where pandas can't infer compression)."""

    def __init__(self, data, name):
        super().__init__(data)
        self.name = name


def _zip_bytes(member_name, data: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member_name, data)
    return buf.getvalue()


def test_read_table_csv_zip_path(tmp_path):
    path = tmp_path / "words.csv.zip"
    path.write_bytes(_zip_bytes("words.csv", b"a,b\n1,x\n2,y\n"))
    df = data_module.read_table(path)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_read_table_csv_zip_uploaded_file_like():
    # The real bug: an in-memory upload named *.csv.zip — pandas infers
    # compression only from string paths, so this must be handled explicitly.
    upload = _NamedBytesIO(_zip_bytes("data.csv", b"a,b\n1,x\n2,y\n"), "data.csv.zip")
    df = data_module.read_table(upload)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_read_table_tsv_zip():
    upload = _NamedBytesIO(_zip_bytes("d.tsv", b"a\tb\n1\tx\n"), "d.tsv.zip")
    df = data_module.read_table(upload)
    assert list(df.columns) == ["a", "b"]


def test_read_table_zip_ignores_macosx_cruft():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("real.csv", b"a,b\n1,2\n")
        zf.writestr("__MACOSX/._real.csv", b"junk")
        zf.writestr(".DS_Store", b"junk")
    upload = _NamedBytesIO(buf.getvalue(), "bundle.zip")
    df = data_module.read_table(upload)
    assert list(df.columns) == ["a", "b"]


def test_read_table_zip_multiple_members_concatenates():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("reader0.csv", b"a,b\n1,x\n2,y\n")
        zf.writestr("reader1.csv", b"a,b\n3,z\n")
    upload = _NamedBytesIO(buf.getvalue(), "bundle.zip")
    df = data_module.read_table(upload)
    assert len(df) == 3
    # Each member's rows are traceable via the source_file stem.
    assert set(df["source_file"]) == {"reader0", "reader1"}


def test_read_table_zip_mixed_formats_concatenates():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.csv", b"x\n1\n")
        zf.writestr("b.tsv", b"x\n2\n")
    upload = _NamedBytesIO(buf.getvalue(), "mixed.zip")
    df = data_module.read_table(upload)
    assert sorted(df["x"]) == [1, 2]


def test_read_table_zip_no_data_files_raises():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("__MACOSX/._x", b"junk")
        zf.writestr(".DS_Store", b"junk")
    upload = _NamedBytesIO(buf.getvalue(), "empty.zip")
    with pytest.raises(ValueError, match="no readable table files"):
        data_module.read_table(upload)


def test_read_tables_list_adds_source_file(tmp_path):
    p1, p2 = tmp_path / "reader0_t1.csv", tmp_path / "reader1_t1.csv"
    _write_fix_csv(p1, "p0", "t1")
    _write_fix_csv(p2, "p1", "t1")
    df = data_module.read_tables([p1, p2])
    assert len(df) == 6
    assert set(df["source_file"]) == {"reader0_t1", "reader1_t1"}


def test_read_tables_single_file_no_source_column(tmp_path):
    p1 = tmp_path / "only.csv"
    _write_fix_csv(p1, "p0", "t1")
    df = data_module.read_tables(p1)
    assert "source_file" not in df.columns


def test_read_tables_glob(tmp_path):
    for i in range(3):
        _write_fix_csv(tmp_path / f"reader{i}_fix.csv", f"p{i}", "t1")
    df = data_module.read_tables(str(tmp_path / "reader*_fix.csv"))
    assert len(df) == 9
    assert df["source_file"].nunique() == 3


def test_read_tables_glob_no_match_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No files match"):
        data_module.read_tables(str(tmp_path / "nope*.csv"))


def test_load_scanpath_data_accepts_path_list(tmp_path):
    p1, p2 = tmp_path / "f1.csv", tmp_path / "f2.csv"
    _write_fix_csv(p1, "p0", "t1")
    _write_fix_csv(p2, "p1", "t1")
    words, fixations = sps.load_scanpath_data(fixations=[p1, p2])
    assert words.empty
    assert set(fixations["participant_id"]) == {"p0", "p1"}
    # the origin file survives normalization for traceability
    assert set(fixations["source_file"]) == {"f1", "f2"}


# ---------------------------------------------------------------------------
# Single-report datasets
# ---------------------------------------------------------------------------


def test_load_scanpath_data_requires_some_input():
    with pytest.raises(ValueError, match="at least one"):
        sps.load_scanpath_data()


def test_fixations_only_load_list_and_plot(tmp_path):
    path = tmp_path / "fix.csv"
    _write_fix_csv(path, "p0", "t1")
    words, fixations = sps.load_scanpath_data(fixations=path)
    assert words.empty and not fixations.empty

    combos = sps.list_trials(words, fixations)
    assert combos.to_records(index=False).tolist() == [("p0", "t1")]

    fig = sps.plot_scanpath(words, fixations, "p0", "t1")
    assert len(fig.data) > 0


def test_words_only_load_list_and_plot(sample_words_df):
    words, fixations = sps.load_scanpath_data(words=sample_words_df)
    assert fixations.empty and not words.empty

    combos = sps.list_trials(words, fixations)
    assert ("p1", "t1") in {tuple(r) for r in combos.to_numpy()}

    fig = sps.plot_scanpath(words, fixations, "p1", "t1")
    assert len(fig.data) > 0


def test_words_only_heatmap_uses_preaggregated_measures(sample_words_df):
    sample_words_df = sample_words_df.copy()
    sample_words_df["IA_DWELL_TIME"] = [500, 250, 0, 100, 100]
    words, fixations = sps.load_scanpath_data(words=sample_words_df)
    trial_words = words[words["participant_id"] == "p1"]

    def n_shapes(show_heatmap):
        fig = make_scanpath_figure(
            trial_words,
            fixations,
            canvas_width=800,
            canvas_height=600,
            base_font_size=14,
            font_family="Arial",
            x_field="x",
            y_field="y",
            show_words=True,
            show_word_labels=False,
            show_fixations=False,
            show_order=False,
            show_saccades=False,
            show_heatmap=show_heatmap,
            color_by="duration_ms",
            heatmap_metric="duration_ms",
            marker_size_range=(6, 30),
            order_font_size=12,
            order_font_color="#000",
            show_colorbars=False,
            fixation_color_range=None,
            heatmap_range=None,
        )
        return len(fig.layout.shapes or [])

    # word boxes only vs. word boxes + heatmap rectangles for the two
    # words with nonzero pre-aggregated dwell time
    assert n_shapes(True) == n_shapes(False) + 2


def test_default_filters_fixations_only(tmp_path):
    path = tmp_path / "fix.csv"
    _write_fix_csv(path, "p0", "t1")
    words, fixations = sps.load_scanpath_data(fixations=path)
    filters = data_module.default_filters(words, fixations)
    assert filters["participants"] == ["p0"]
    assert filters["trials"] == ["t1"]
    w, f = data_module.filter_data(words, fixations, filters)
    assert len(f) == 3


# ---------------------------------------------------------------------------
# Stimulus-level words + AOI-only fixations
# ---------------------------------------------------------------------------


@pytest.fixture
def stimulus_words_df():
    """Word boxes keyed by text only — no participant column."""
    return pd.DataFrame(
        {
            "text_id": ["t1", "t1", "t2"],
            "word_id": [1, 2, 1],
            "word": ["Hello", "world", "Bye"],
            "left": [100, 200, 100],
            "right": [180, 280, 160],
            "top": [50, 50, 50],
            "bottom": [100, 100, 100],
        }
    )


@pytest.fixture
def aoi_fixations_df():
    """AOI-sequence fixations: word ids but no pixel coordinates."""
    return pd.DataFrame(
        {
            "reader_id": [7, 7, 7, 8],
            "text_id": ["t1", "t1", "t1", "t2"],
            "fixation_duration": [180, 220, 150, 200],
            "word_index": [1, 2, 1, 1],
        }
    )


def test_stimulus_words_broadcast_and_aoi_xy(stimulus_words_df, aoi_fixations_df):
    words, fixations = sps.load_scanpath_data(
        words=stimulus_words_df, fixations=aoi_fixations_df
    )
    # words replicated across the readers that read each text
    assert set(words["participant_id"]) == {"7", "8"}
    t1_words = words[words["trial_id"] == "t1"]
    assert len(t1_words) == 2  # only reader 7 read t1
    # t2 words exist only for reader 8
    assert set(words[words["trial_id"] == "t2"]["participant_id"]) == {"8"}

    # fixation coordinates = word box centers
    assert fixations["x"].tolist() == [140.0, 240.0, 140.0, 130.0]
    assert fixations["y"].tolist() == [75.0] * 4

    fig = sps.plot_scanpath(words, fixations, "7", "t1")
    assert len(fig.data) > 0


def test_stimulus_words_without_fixations_get_synthetic_participant(stimulus_words_df):
    words, fixations = sps.load_scanpath_data(words=stimulus_words_df)
    # No fixations to broadcast across → a single anonymous reader.
    assert (words["participant_id"] == data_module.SYNTHETIC_PARTICIPANT).all()
    assert data_module.STIMULUS_WORDS_FLAG not in words.columns


def test_aoi_fixations_without_words_raise_on_plot(aoi_fixations_df):
    words, fixations = sps.load_scanpath_data(fixations=aoi_fixations_df)
    assert fixations["x"].isna().all()
    with pytest.raises(ValueError, match="no usable coordinates"):
        sps.plot_scanpath(words, fixations, "7", "t1")


def test_fix_schema_requires_xy_or_word_id():
    no_position = pd.DataFrame(
        {"participant_id": ["p"], "trial_id": ["t"], "duration_ms": [100]}
    )
    with pytest.raises(ValueError, match="Word/IA ID"):
        sps.load_scanpath_data(fixations=no_position)


def test_participant_less_fixations_get_synthetic_participant():
    """A fixations table with no participant column loads (participant is now
    optional) and every row is stamped with the synthetic participant."""
    fixations = pd.DataFrame(
        {
            "trial_id": ["t1", "t1", "t2"],
            "x": [10.0, 20.0, 30.0],
            "y": [5.0, 5.0, 5.0],
            "duration_ms": [100, 120, 90],
        }
    )
    _words, fix = sps.load_scanpath_data(fixations=fixations)
    assert (fix["participant_id"] == data_module.SYNTHETIC_PARTICIPANT).all()


def test_asymmetric_participant_reconciles_word_boxes():
    """Words carry a participant id but fixations don't — the boxes must be
    re-keyed to the synthetic participant the trial picker uses, or they'd be
    silently invisible (extract_trial would find none)."""
    words = pd.DataFrame(
        {
            "participant_id": ["sub1", "sub1"],
            "trial_id": ["t1", "t1"],
            "word_id": [1, 2],
            "word": ["Hello", "world"],
            "left": [100, 200],
            "right": [180, 280],
            "top": [50, 50],
            "bottom": [100, 100],
        }
    )
    fixations = pd.DataFrame(
        {
            "trial_id": ["t1", "t1"],
            "x": [140.0, 240.0],
            "y": [75.0, 75.0],
            "duration_ms": [180, 200],
        }
    )
    words_n, fix_n = sps.load_scanpath_data(words=words, fixations=fixations)
    assert set(fix_n["participant_id"]) == {data_module.SYNTHETIC_PARTICIPANT}
    assert set(words_n["participant_id"]) == {data_module.SYNTHETIC_PARTICIPANT}
    # The boxes for the trial the picker offers ('(all)', 't1') are now reachable.
    fig = sps.plot_scanpath(words_n, fix_n, data_module.SYNTHETIC_PARTICIPANT, "t1")
    assert len(fig.data) > 0


def test_frame_fingerprint_distinguishes_unhashable_columns():
    """Two frames identical in shape + columns but differing in a list-valued
    (unhashable) column must not collapse to the same cache key."""
    a = pd.DataFrame({"x": [1, 2], "spans": [[1, 2], [3, 4]]})
    b = pd.DataFrame({"x": [1, 2], "spans": [[9, 9], [8, 8]]})
    assert data_module.frame_fingerprint(a) != data_module.frame_fingerprint(b)


# ---------------------------------------------------------------------------
# PoTeC loader (against a tiny synthesized PoTeC-format tree, no download)
# ---------------------------------------------------------------------------


@pytest.fixture
def potec_root(tmp_path):
    """A minimal PoTeC-shaped directory: one text (b0), two readers (0, 1)."""
    aoi_dir = tmp_path / "stimuli" / "aoi_texts"
    word_dir = tmp_path / "stimuli" / "word_aoi_texts"
    scan_dir = tmp_path / "eyetracking_data" / "scanpaths"
    for d in (aoi_dir, word_dir, scan_dir):
        d.mkdir(parents=True)

    # text "Um null" — two words, char AOIs 1..7 (space belongs to no AOI in
    # real PoTeC, but a simple consecutive layout is fine here). "null" guards
    # the keep_default_na handling (PoTeC text p3 contains the word "null").
    chars = pd.DataFrame(
        {
            "aoi_type": ["0 RECTANGLE"] * 7,
            "aoi": range(1, 8),
            "start_x": [80, 93, 115, 137, 150, 163, 176],
            "start_y": [21] * 7,
            "end_x": [93, 115, 137, 150, 163, 176, 189],
            "end_y": [99] * 7,
            "character": list("Um") + [" "] + list("null"),
            "line": [1] * 7,
        }
    )
    chars.to_csv(aoi_dir / "b0.ias", sep="\t", index=False)

    words = pd.DataFrame(
        {
            "aoi_type": ["0 RECTANGLE"] * 2,
            "aoi": [1.0, 2.0],
            "start_x": [80.0, 115.0],
            "start_y": [21.0, 21.0],
            "end_x": [115.0, 189.0],
            "end_y": [99.0, 99.0],
            "word": ["Um", "null"],
        }
    )
    words.to_csv(word_dir / "word_aoi_b0.tsv", sep="\t", index=False)

    for reader in (0, 1):
        pd.DataFrame(
            {
                "fixation_index": [1, 2, 3],
                "fixation_duration": [210, 190, 250],
                "line": [1, 1, 1],
                "aoi": [2, 5, 1],  # chars m, u, U
                "reader_id": [reader] * 3,
                "text_id": ["b0"] * 3,
                "word_index_in_text": [1, 2, 1],
                "word": ["Um", "null", "Um"],
            }
        ).to_csv(scan_dir / f"reader{reader}_b0_scanpath.tsv", sep="\t", index=False)
    return tmp_path


def test_load_potec(potec_root):
    words, fixations = datasets_module.load_potec(potec_root, texts=["b0"])

    # stimulus words broadcast across both readers
    assert set(words["participant_id"]) == {"0", "1"}
    assert len(words) == 4  # 2 words x 2 readers
    assert set(words["text"]) == {"Um", "null"}  # "null" must stay a string
    assert words["line_idx"].eq(1).all()

    # fixation x/y reconstructed from the character AOI centers
    reader0 = fixations[fixations["participant_id"] == "0"]
    assert reader0["x"].tolist() == [104.0, 156.5, 86.5]
    assert reader0["y"].tolist() == [60.0] * 3
    assert reader0["duration_ms"].tolist() == [210.0, 190.0, 250.0]
    # word ids link fixations to the word AOIs
    assert reader0["word_id"].tolist() == [1.0, 2.0, 1.0]

    fig = sps.plot_scanpath(words, fixations, "1", "b0", canvas_size=(1680, 1050))
    assert len(fig.data) > 0


def test_load_potec_reader_subset(potec_root):
    words, fixations = datasets_module.load_potec(potec_root, readers=[1], texts=["b0"])
    assert set(fixations["participant_id"]) == {"1"}
    assert set(words["participant_id"]) == {"1"}


def test_load_potec_unknown_text(potec_root):
    with pytest.raises(ValueError, match="Unknown PoTeC text ids"):
        datasets_module.load_potec(potec_root, texts=["z9"])


def test_load_potec_missing_data_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="download=True"):
        datasets_module.load_potec(tmp_path, texts=["b0"])


# ---------------------------------------------------------------------------
# Column keep-list / pruning (perf core)
# ---------------------------------------------------------------------------


def test_keep_columns_prunes_normalized_frame():
    words = pd.DataFrame(
        {
            "participant_id": ["p1", "p1"],
            "trial_id": ["t1", "t1"],
            "word_id": [1, 2],
            "IA_LEFT": [0, 10],
            "IA_RIGHT": [10, 20],
            "IA_TOP": [0, 0],
            "IA_BOTTOM": [10, 10],
            "IA_LABEL": ["a", "b"],
            "gpt2_surprisal": [1.0, 2.0],
            "difficulty_level": ["Adv", "Adv"],
            "junk": [9, 9],
        }
    )
    schema = data_module.propose_word_schema(words)

    # Default (keep_columns=None): all detected optional fields kept, junk dropped.
    full = data_module.normalize_words(words, schema)
    assert "gpt2_surprisal" in full.columns
    assert "difficulty_level" in full.columns
    assert "junk" not in full.columns

    # Pruned: only the chosen optional + explicit extra keep survive.
    keep = data_module.compute_keep_columns(
        schema, optional_sources=["gpt2_surprisal"], keep_columns=["junk"]
    )
    thin = data_module.normalize_words(words, schema, keep_columns=keep)
    assert "gpt2_surprisal" in thin.columns
    assert "junk" in thin.columns  # carried verbatim
    assert "difficulty_level" not in thin.columns  # detected but not chosen


def test_categorize_columns_splits_mapped_detected_unclaimed():
    words = pd.DataFrame(
        {
            "participant_id": ["p1"],
            "trial_id": ["t1"],
            "word_id": [1],
            "x": [0],
            "y": [0],
            "width": [1],
            "height": [1],
            "gpt2_surprisal": [1.0],
            "my_custom_col": [3],
        }
    )
    schema = data_module.propose_word_schema(words)
    cats = data_module.categorize_columns(
        words, schema, data_module.WORD_OPTIONAL_FIELDS
    )
    assert "participant_id" in cats["mapped"]
    assert any(d["source"] == "gpt2_surprisal" for d in cats["detected_optional"])
    assert "my_custom_col" in cats["unclaimed"]
