# Contributing

Thanks for your interest in improving Scanpath Studio! This is a small
research tool; contributions, bug reports, and feature requests are welcome via
[issues](https://github.com/lacclab/scanpath-studio/issues) and pull
requests.

## Development setup

```bash
git clone https://github.com/lacclab/scanpath-studio.git
cd scanpath-studio
pip install -e ".[test]"          # or: uv sync
streamlit run streamlit_app.py    # run the app locally
```

Tested on Python 3.11–3.14.

## Before you open a PR

```bash
pytest                            # run the test suite
ruff check --exclude other_vis .  # lint
ruff format --exclude other_vis . # auto-format
```

CI (`.github/workflows/ci.yml`) runs the same checks on every push and PR
across Python 3.11/3.12/3.13/3.14. See [AGENTS.md](AGENTS.md) and the package
[CLAUDE.md](scanpath_studio/CLAUDE.md) for an architectural overview.

## Versioning

The version lives in **one** place — `__version__` in
[`scanpath_studio/__init__.py`](scanpath_studio/__init__.py).
`pyproject.toml` reads it dynamically, so bump only that file.

## Dependencies

- `pyproject.toml` carries the **library** dependency bounds (`>=`) used when
  installing the package with pip.
- `requirements.txt` is the **deployment manifest** for the Streamlit Cloud
  demo, using compatible-release pins (`~=`) so the live app stays on a
  known-good minor while still getting patch updates. Update both when you add
  or upgrade a dependency.

## Releasing

1. Update [`CHANGELOG.md`](CHANGELOG.md) with the new version.
2. Bump `__version__` in `scanpath_studio/__init__.py`.
3. Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`.
   `.github/workflows/publish.yml` builds and publishes to PyPI via trusted
   publishing.
4. Optionally create a GitHub Release with the changelog notes.

## Regenerating the demo GIF

```bash
pip install matplotlib            # asset-generation only, not a runtime dep
python scripts/make_demo_gif.py   # writes assets/demo_dual_scanpath.gif
```

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
