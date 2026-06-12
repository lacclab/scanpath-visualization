"""Keep CITATION.cff in sync with the package version.

GitHub's "Cite this repository" button and the in-app About popover both
surface a citation; this guard makes the CFF version part of the release
bump so the two never drift apart.
"""

from __future__ import annotations

import re
from pathlib import Path

from scanpath_studio import __version__

CFF_PATH = Path(__file__).resolve().parent.parent / "CITATION.cff"


def test_citation_cff_version_matches_package():
    text = CFF_PATH.read_text(encoding="utf-8")
    match = re.search(r"^version:\s*[\"']?([^\s\"']+)", text, re.MULTILINE)
    assert match, "CITATION.cff is missing a `version:` field"
    assert match.group(1) == __version__, (
        f"CITATION.cff version {match.group(1)} != package {__version__} — "
        "bump CITATION.cff (version + date-released) alongside "
        "scanpath_studio/__init__.py"
    )
