#!/usr/bin/env python3
"""Print one release's notes from CHANGELOG.md, formatted for Slack mrkdwn.

Usage:
    python scripts/changelog_notes.py <version> [changelog_path]

Extracts the ``## [<version>] - ...`` section (Keep a Changelog format) and
rewrites Markdown to Slack mrkdwn (``**bold**`` -> ``*bold*``, ``### Heading``
-> ``*Heading*``). Prints nothing and exits 0 if the version has no section or
the file is missing, so the release workflow can fall back to a plain message.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Stop a section at the next release header or the bottom-of-file link refs.
_NEXT = re.compile(r"^## \[|^\[[^\]]+\]:\s", re.MULTILINE)


def extract(changelog: str, version: str) -> str:
    """Return the Slack-mrkdwn body of *version*'s changelog section, or ""."""
    header = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n", changelog, re.MULTILINE
    )
    if header is None:
        return ""
    rest = changelog[header.end() :]
    end = _NEXT.search(rest)
    notes = (rest[: end.start()] if end else rest).strip()
    notes = re.sub(r"\*\*([^*]+)\*\*", r"*\1*", notes)  # **bold** -> *bold*
    notes = re.sub(r"^### (.*)$", r"*\1*", notes, flags=re.MULTILINE)  # ### H -> *H*
    return notes


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: changelog_notes.py <version> [changelog_path]", file=sys.stderr)
        return 2
    version = re.sub(r"^v", "", argv[1])
    path = Path(argv[2]) if len(argv) > 2 else Path("CHANGELOG.md")
    if path.is_file():
        sys.stdout.write(extract(path.read_text(encoding="utf-8"), version))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
