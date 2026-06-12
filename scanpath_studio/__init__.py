"""Streamlit workbench + headless API for scanpath visualization."""

from __future__ import annotations

__all__ = [
    "__version__",
    "main",
    "load_scanpath_data",
    "load_sample_data",
    "list_trials",
    "compute_word_metrics",
    "plot_scanpath",
    "animate_scanpath",
    "save_figure",
    "load_potec",
]
__version__ = "0.18.0"

# Public headless API (see api.py / datasets.py). Resolved lazily so
# `import scanpath_studio` stays cheap and doesn't pull in pandas/plotly/
# streamlit until first use.
_DATASET_EXPORTS = frozenset({"load_potec"})
_API_EXPORTS = frozenset(__all__) - {"__version__", "main"} - _DATASET_EXPORTS


def __getattr__(name: str):
    if name in _API_EXPORTS:
        from . import api

        return getattr(api, name)
    if name in _DATASET_EXPORTS:
        from . import datasets

        return getattr(datasets, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(set(globals()) | set(__all__))


def main() -> None:
    """Programmatic entry point — `from scanpath_studio import main`."""
    from .app import main as _main

    _main()
