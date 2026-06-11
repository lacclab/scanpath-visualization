"""Streamlit workbench for scanpath visualization."""

from __future__ import annotations

__all__ = ["__version__", "main"]
__version__ = "0.16.2"


def main() -> None:
    """Programmatic entry point — `from scanpath_studio import main`."""
    from .app import main as _main

    _main()
