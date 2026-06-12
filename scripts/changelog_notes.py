#!/usr/bin/env python3
"""Print one release's notes from CHANGELOG.md.

Usage:
    python scripts/changelog_notes.py <version> [changelog_path] [--format FMT]

Extracts the ``## [<version>] - ...`` section (Keep a Changelog format). With
``--format slack`` (the default) Markdown is rewritten to Slack mrkdwn
(``**bold**`` -> ``*bold*``, ``### Heading`` -> ``*Heading*``); with
``--format markdown`` the section is emitted verbatim (for GitHub releases).
Prints nothing and exits 0 if the version has no section or the file is
missing, so the release workflow can fall back to a plain message.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Stop a section at the next release header or the bottom-of-file link refs.
_NEXT = re.compile(r"^## \[|^\[[^\]]+\]:\s", re.MULTILINE)


def extract(changelog: str, version: str, fmt: str = "slack") -> str:
    """Return *version*'s changelog section, or "".

    ``fmt="markdown"`` returns the section verbatim; ``fmt="slack"`` rewrites it
    to Slack mrkdwn.
    """
    header = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n", changelog, re.MULTILINE
    )
    if header is None:
        return ""
    rest = changelog[header.end() :]
    end = _NEXT.search(rest)
    notes = (rest[: end.start()] if end else rest).strip()
    if fmt == "slack":
        notes = re.sub(r"\*\*([^*]+)\*\*", r"*\1*", notes)  # **bold** -> *bold*
        notes = re.sub(
            r"^### (.*)$", r"*\1*", notes, flags=re.MULTILINE
        )  # ### H -> *H*
    return notes


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("changelog_path", nargs="?", default="CHANGELOG.md")
    parser.add_argument(
        "--format", choices=("slack", "markdown"), default="slack", dest="fmt"
    )
    ns = parser.parse_args(argv[1:])
    version = re.sub(r"^v", "", ns.version)
    path = Path(ns.changelog_path)
    if path.is_file():
        sys.stdout.write(extract(path.read_text(encoding="utf-8"), version, ns.fmt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
