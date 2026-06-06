#!/usr/bin/env python
"""Generate the README demo GIF: two readings of the same text, co-animated.

Renders the canonical "two readers of the same text, aligned in real time"
view from the bundled OneStop demo and writes an animated GIF to
``assets/demo_dual_scanpath.gif``. It reuses the app's own data loaders and
normalization, then draws with matplotlib (no headless Chrome required, unlike
the in-app Plotly export) so the asset can be regenerated anywhere.

Usage::

    pip install matplotlib            # asset-generation only, not a runtime dep
    python scripts/make_demo_gif.py   # writes assets/demo_dual_scanpath.gif

The visual is a teaser, not a pixel-faithful copy of the app's animation tab;
the app remains the source of truth for the real rendering.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scanpath_studio.constants import COMPARISON_PALETTE
from scanpath_studio.data import (
    infer_fix_schema,
    infer_word_schema,
    load_sample_data,
    normalize_fixations,
    normalize_words,
)

# Both bundled participants read this paragraph, so it gives a genuine
# "same text, two readers" overlay straight from the demo data.
PARAGRAPH = "2_1_1_Ele"
READER_A = "l37_1129"
READER_B = "l7_1090"
MAX_FIX = 70  # cap per reader to keep the GIF small and the loop short


def _load_pair():
    words_raw, fix_raw = load_sample_data()
    words = normalize_words(words_raw, infer_word_schema(words_raw))
    fix = normalize_fixations(fix_raw, infer_fix_schema(fix_raw))

    in_para = fix["unique_paragraph_id"] == PARAGRAPH
    words_a = words[
        (words["unique_paragraph_id"] == PARAGRAPH)
        & (words["participant_id"] == READER_A)
    ]

    def _reader(pid: str):
        sub = fix[in_para & (fix["participant_id"] == pid)]
        sub = sub.sort_values("timestamp_ms").head(MAX_FIX).reset_index(drop=True)
        # Rebase onto each reading's own clock so both start at t=0 and share
        # real reading time, exactly like make_scanpath_animation does.
        t0 = sub["timestamp_ms"].iloc[0]
        sub = sub.assign(t=sub["timestamp_ms"] - t0)
        return sub

    return words_a, _reader(READER_A), _reader(READER_B)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("assets/demo_dual_scanpath.gif"),
        help="output GIF path",
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:  # pragma: no cover - asset tooling only
        raise SystemExit(
            "matplotlib is required to build the demo GIF:\n    pip install matplotlib"
        )

    words, a, b = _load_pair()
    readers = [
        (a, COMPARISON_PALETTE[0], "Reader A"),
        (b, COMPARISON_PALETTE[1], "Reader B"),
    ]

    x0 = words["x"].min() - 30
    x1 = (words["x"] + words["width"]).max() + 30
    y0 = words["y"].min() - 30
    y1 = (words["y"] + words["height"]).max() + 30

    fig, ax = plt.subplots(figsize=(11, 11 * (y1 - y0) / (x1 - x0)), dpi=90)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)  # invert y: screen coordinates grow downward
    ax.set_aspect("equal")
    ax.axis("off")

    # Faint reading text underneath both scanpaths.
    for _, row in words.iterrows():
        ax.text(
            row["x"] + row["width"] / 2,
            row["y"] + row["height"] / 2,
            str(row["text"]),
            ha="center",
            va="center",
            fontsize=6,
            color="#b8b8b8",
            zorder=1,
        )

    artists = []
    for _, color, label in readers:
        (trail,) = ax.plot([], [], "-", color=color, lw=1.0, alpha=0.45, zorder=2)
        dots = ax.scatter(
            [],
            [],
            s=[],
            facecolor=color,
            edgecolor="white",
            linewidths=0.4,
            alpha=0.85,
            zorder=3,
        )
        (head,) = ax.plot(
            [], [], "o", mfc="none", mec=color, mew=1.6, ms=14, zorder=4, label=label
        )
        artists.append((trail, dots, head))

    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    clock = ax.text(
        0.01, 0.99, "", transform=ax.transAxes, va="top", fontsize=10, color="#444"
    )

    span = max(r["t"].iloc[-1] for r, _, _ in readers)
    times = np.linspace(0, span, args.frames)

    def update(t):
        for (sub, _color, _label), (trail, dots, head) in zip(readers, artists):
            shown = sub[sub["t"] <= t]
            trail.set_data(shown["x"], shown["y"])
            if len(shown):
                sizes = 20 + (shown["duration_ms"].to_numpy() / 460.0) * 260
                dots.set_offsets(np.column_stack([shown["x"], shown["y"]]))
                dots.set_sizes(sizes)
                head.set_data([shown["x"].iloc[-1]], [shown["y"].iloc[-1]])
            else:
                dots.set_offsets(np.empty((0, 2)))
                head.set_data([], [])
        clock.set_text(f"{t / 1000:0.1f} s  —  same paragraph, two readers")
        return []

    anim = FuncAnimation(fig, update, frames=times, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
