"""Tests for the command-line interface (scanpath_studio.cli)."""

import pytest

from scanpath_studio import __version__, cli


def test_version(capsys):
    cli.main(["--version"])
    assert capsys.readouterr().out.strip() == __version__


def test_help(capsys):
    cli.main(["--help"])
    out = capsys.readouterr().out
    assert "render" in out
    assert "streamlit run" in out


def test_default_launches_app(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "launch_app", lambda args: calls.append(args))
    cli.main([])
    cli.main(["run", "--server.port", "8502"])
    # Backward compat: bare streamlit flags forward to the app launcher.
    cli.main(["--server.port", "8502"])
    assert calls == [[], ["--server.port", "8502"], ["--server.port", "8502"]]


def test_render_requires_input_choice():
    with pytest.raises(SystemExit):
        cli.main(["render", "-o", "out.html"])  # neither --sample nor files
    with pytest.raises(SystemExit):
        cli.main(["render", "--sample", "--words", "w.csv", "-o", "out.html"])


def test_render_requires_output():
    with pytest.raises(SystemExit):
        cli.main(["render", "--sample"])


def test_render_list_trials(capsys):
    cli.main(["render", "--sample", "--list-trials"])
    out = capsys.readouterr().out
    assert "participant_id" in out
    assert "trial_id" in out


def test_render_sample_html(tmp_path, capsys):
    out_file = tmp_path / "scanpath.html"
    cli.main(["render", "--sample", "-o", str(out_file)])
    assert out_file.is_file()
    err = capsys.readouterr().err
    assert "Rendering participant=" in err


def test_render_explicit_trial_with_flags(tmp_path):
    import scanpath_studio as sps

    pid, tid = sps.list_trials(*sps.load_sample_data()).iloc[0]
    out_file = tmp_path / "scanpath.html"
    cli.main(
        [
            "render",
            "--sample",
            "-p",
            pid,
            "-t",
            tid,
            "--no-heatmap",
            "--saccade-arrows",
            "--canvas",
            "2560x1440",
            "-o",
            str(out_file),
        ]
    )
    assert out_file.is_file()


def test_render_animate_html(tmp_path):
    out_file = tmp_path / "anim.html"
    cli.main(["render", "--sample", "--animate", "-o", str(out_file)])
    assert out_file.is_file()


def test_render_animate_rejects_non_html(tmp_path):
    with pytest.raises(SystemExit, match="html"):
        cli.main(["render", "--sample", "--animate", "-o", str(tmp_path / "a.png")])


def test_render_unknown_trial_exits():
    with pytest.raises(SystemExit):
        cli.main(
            ["render", "--sample", "-p", "nobody", "-t", "nothing", "-o", "x.html"]
        )


def test_render_unknown_trial_without_participant_exits(tmp_path):
    # Regression: a mistyped -t without -p must error, not silently render
    # the dataset's first trial.
    out_file = tmp_path / "x.html"
    with pytest.raises(SystemExit, match="No trial matches"):
        cli.main(["render", "--sample", "-t", "no_such_trial", "-o", str(out_file)])
    assert not out_file.exists()


def test_render_trial_only_resolves_matching_participant(tmp_path, capsys):
    # A valid -t without -p picks a participant that actually has that trial.
    import scanpath_studio as sps

    combos = sps.list_trials(*sps.load_sample_data())
    pid, tid = combos.iloc[-1]
    out_file = tmp_path / "x.html"
    cli.main(["render", "--sample", "-t", tid, "-o", str(out_file)])
    assert out_file.is_file()
    assert f"trial={tid}" in capsys.readouterr().err


def test_render_bad_canvas_exits():
    with pytest.raises(SystemExit, match="--canvas"):
        cli.main(["render", "--sample", "--canvas", "huge", "-o", "x.html"])
    with pytest.raises(SystemExit, match="positive"):
        cli.main(["render", "--sample", "--canvas", "0x1440", "-o", "x.html"])


def test_render_animate_warns_on_unsupported_flags(tmp_path, capsys):
    out_file = tmp_path / "anim.html"
    cli.main(
        [
            "render",
            "--sample",
            "--animate",
            "--no-heatmap",
            "--color-by",
            "pass_index",
            "-o",
            str(out_file),
        ]
    )
    assert out_file.is_file()
    err = capsys.readouterr().err
    assert "ignoring" in err
    assert "color_by" in err and "show_heatmap" in err


def test_render_from_files(tmp_path):
    from scanpath_studio import data as data_module

    words_raw, fix_raw = data_module.load_sample_data()
    words_path = tmp_path / "ia.csv"
    fix_path = tmp_path / "fix.csv"
    words_raw.to_csv(words_path, index=False)
    fix_raw.to_csv(fix_path, index=False)

    out_file = tmp_path / "out.html"
    cli.main(
        [
            "render",
            "--words",
            str(words_path),
            "--fixations",
            str(fix_path),
            "-o",
            str(out_file),
        ]
    )
    assert out_file.is_file()


def test_render_fixations_only_multifile(tmp_path):
    """Fixations-only, multi-file glob input renders without a words table."""
    from scanpath_studio import data as data_module

    _, fix_raw = data_module.load_sample_data()
    for pid, group in fix_raw.groupby("participant_id"):
        group.to_csv(tmp_path / f"{pid}.csv", index=False)

    out_file = tmp_path / "out.html"
    cli.main(
        ["render", "--fixations", str(tmp_path / "*.csv"), "-o", str(out_file)]
    )
    assert out_file.is_file()


def test_render_potec_conflicts_with_other_inputs():
    with pytest.raises(SystemExit, match="exactly one input"):
        cli.main(["render", "--potec", "d", "--sample", "-o", "out.html"])
