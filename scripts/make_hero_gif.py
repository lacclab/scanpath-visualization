#!/usr/bin/env python
"""Generate the README hero GIF: one reader's scanpath replayed fixation by fixation.

Drives the app's *own* Animated Scanpath export path — the same builders and the
same Kaleido frame rasterizer the in-app **GIF** download uses — so the asset is a
faithful capture of what the app produces, not a stand-in. Writes to
``assets/scanpath_animation.gif``.

Usage::

    plotly_get_chrome -y                  # one-time: Kaleido needs a Chrome
    python scripts/make_hero_gif.py       # writes assets/scanpath_animation.gif

The trial is the bundled OneStop demo's "Polish government / wolf" elementary
paragraph (35 fixations) — short enough for a snappy loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import scanpath_studio as sps
from scanpath_studio.animation_export import export_animation
from scanpath_studio.plots import animation_playback_ms

PARTICIPANT = "l7_1090"
PARAGRAPH = "l7_1090_2_2_4_Ele_r0"
PLAYBACK_SPEED = 3.0  # the Animated Scanpath tab's default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("assets/scanpath_animation.gif"),
        help="output GIF path",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.5,
        help="Kaleido render scale (1.0 = on-screen px; higher is crisper/larger)",
    )
    args = parser.parse_args()

    words, fixations = sps.load_sample_data()
    fig = sps.animate_scanpath(
        words, fixations, PARTICIPANT, PARAGRAPH, playback_speed=PLAYBACK_SPEED
    )

    # Match the tab's per-frame duration so the clip's runtime equals the
    # on-screen Play (see tabs.render_animation_tab / animation_playback_ms).
    _, fixs, _, _ = sps.api._select_trial(words, fixations, PARTICIPANT, PARAGRAPH)
    _span, playback_ms = animation_playback_ms([fixs], PLAYBACK_SPEED)
    n_frames = len(fig.frames or ())
    frame_ms = playback_ms / n_frames if n_frames else 16.0

    def _progress(done: int, total: int) -> None:
        print(f"\r  frame {done}/{total}", end="", flush=True)

    data = export_animation(
        fig,
        fmt="gif",
        frame_duration_ms=frame_ms,
        scale=args.scale,
        progress_callback=_progress,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    print(f"\nwrote {args.out} ({len(data) / 1024:.0f} KB, {n_frames} frames)")


if __name__ == "__main__":
    main()
