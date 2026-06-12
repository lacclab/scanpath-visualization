<!--
Keep it tight — a sentence or two of what changed and why is plenty.
See CONTRIBUTING.md for the full pre-PR checklist.
-->

## Summary

<!-- What does this change and why? Link any related issue (#123). -->

## Verification

<!-- How you confirmed it works — e.g. which tests cover it, ruff clean, manual app check.
     For UI/visual changes, drop in a before/after screenshot or GIF. -->

## Checklist

- [ ] `pytest` passes
- [ ] Added/updated tests for any new behavior
- [ ] `ruff check --exclude other_vis .` and `ruff format --check --exclude other_vis .` are clean
- [ ] Added a `[Unreleased]` entry to [`CHANGELOG.md`](https://github.com/lacclab/scanpath-studio/blob/main/CHANGELOG.md) (every feature/bugfix/notable change)
- [ ] Dependency change? Updated **both** `pyproject.toml` and `requirements.txt`
