"""Streamlit workbench for scanpath visualization."""

from __future__ import annotations

__all__ = ["__version__", "main"]
__version__ = "0.13.0"


def main() -> None:
    """Programmatic entry point — `from scanpath_visualization_app import main`."""
    from .app import main as _main

    _main()
