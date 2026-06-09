"""Render a Plotly scanpath animation to a shareable GIF or MP4 clip.

The *Animated Scanpath* tab builds a Plotly ``go.Figure`` with one frame per
fixation onset (see :func:`scanpath_studio.plots.make_scanpath_animation`). The
interactive **HTML** export keeps that figure verbatim — play button, slider and
all. This module is the non-interactive counterpart: it rasterizes the very same
frames and encodes them into a GIF or MP4 you can drop into a slide deck, a paper,
or a chat without needing a browser to replay it.

How it stays faithful to what the user sees on screen:

* **Same frames.** Each ``go.Frame`` is applied onto a frameless copy of the base
  figure and rendered to PNG, so word boxes, true-to-scale labels, saccades,
  order numbers and the orange current-fixation highlight all match the live view.
* **Same clock.** The on-screen Play button advances every frame at one average
  duration (``plots._anim_timeline``); we reproduce that exactly, so the clip's
  runtime equals the playback time the tab quotes (``animation_playback_ms``).
* **Same readout.** The slider's "Elapsed: X.Xs" value is re-drawn as a static
  annotation per frame, since the interactive slider can't survive rasterization.

Rendering goes through Kaleido (headless Chrome), the same engine the PNG/SVG/PDF
exports use. A *single* browser is kept warm across all frames
(``start_sync_server`` → ``calc_fig_sync`` → ``stop_sync_server``): the per-call
``fig.to_image`` cold-starts Chrome every time (~10 s/frame), whereas a warm
browser renders each frame in a fraction of a second.
"""

from __future__ import annotations

import io
from typing import Callable, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

# The interactive formats live elsewhere; these are the rasterized clip formats.
VIDEO_FORMATS: Tuple[str, ...] = ("gif", "mp4")

_MIME = {"gif": "image/gif", "mp4": "video/mp4"}

# make_scanpath_animation reserves this many px of top margin for the play/slider
# controls. With the controls stripped we reclaim it down to a slim band that
# still fits the "Elapsed" annotation.
_CONTROL_BAND_PX = 80
_STATIC_TOP_MARGIN_PX = 28

# Floor on a GIF frame delay: the format stores delays in centiseconds and many
# viewers silently promote sub-20 ms delays to ~100 ms, so clamp here to keep
# fast playback honest. MP4 has no such quirk.
_GIF_MIN_FRAME_MS = 20
# MP4 plays at one constant rate, but animation frames have durations spanning
# ~16 ms (fast/×8 playback) to several hundred ms (slow/×0.25, or downsampled long
# trials). We encode at a fixed, universally-playable rate and hold each animation
# frame for the right number of video frames (repeats compress to ~nothing in
# H.264), so the clip's runtime tracks the on-screen Play across that whole range.
_MP4_FPS = 60.0

ProgressCallback = Callable[[int, int], None]


class AnimationExportError(RuntimeError):
    """Frame rendering or encoding failed.

    The most common cause is a missing Chrome/Chromium for Kaleido; the message
    is surfaced to the user with a hint to fall back to the HTML export.
    """


def mime_for(fmt: str) -> str:
    return _MIME[fmt.lower()]


def _elapsed_labels(fig: go.Figure, n_frames: int) -> List[str]:
    """Per-frame "elapsed reading time" labels, lifted from the slider steps.

    ``_animation_time_slider`` already computes one ``"X.Xs"`` label per frame, so
    we reuse them verbatim rather than recomputing onsets. Falls back to blanks if
    the figure has no slider (e.g. an empty animation).
    """
    sliders = fig.layout.sliders
    if sliders and sliders[0].steps:
        labels = [step.label or "" for step in sliders[0].steps]
        if len(labels) >= n_frames:
            return list(labels[:n_frames])
        return list(labels) + [""] * (n_frames - len(labels))
    return [""] * n_frames


def _static_base(fig: go.Figure) -> go.Figure:
    """A frameless deep copy of ``fig`` with interactive controls removed.

    The play/pause/restart buttons and the slider are meaningless in a rasterized
    clip — and worse, they'd be burnt into every frame. ``update_layout`` can't
    clear array layout properties (passing ``None`` is a no-op and ``[]`` doesn't
    truncate the existing entries), so we assign the attributes directly. The
    reserved control band is then reclaimed so the clip isn't topped by an empty
    strip; a slim margin remains for the "Elapsed" annotation.
    """
    base = go.Figure(fig)
    base.frames = ()
    base.layout.updatemenus = []
    base.layout.sliders = []
    height = int(fig.layout.height or 600)
    base.update_layout(
        margin=dict(l=0, r=0, t=_STATIC_TOP_MARGIN_PX, b=0),
        height=max(
            height - (_CONTROL_BAND_PX - _STATIC_TOP_MARGIN_PX),
            _STATIC_TOP_MARGIN_PX + 1,
        ),
    )
    return base


def _select_frames(n: int, max_frames: Optional[int]) -> List[int]:
    """Indices of frames to render, evenly downsampled to ``max_frames``.

    Returns ``range(n)`` unchanged when no cap applies. Downsampling keeps the
    first and last frames (so the clip still starts empty and ends on the full
    scanpath) and spreads the rest evenly; callers scale the frame duration by
    ``n / len(selected)`` to preserve the overall runtime.
    """
    if max_frames is None or max_frames <= 0 or n <= max_frames:
        return list(range(n))
    return sorted(set(int(round(i)) for i in np.linspace(0, n - 1, max_frames)))


def render_png_frames(
    fig: go.Figure,
    *,
    scale: float = 1.0,
    show_elapsed: bool = True,
    frame_indices: Optional[List[int]] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[List[bytes], Tuple[int, int]]:
    """Rasterize the animation's frames to PNG bytes via one persistent Kaleido browser.

    Returns ``(png_bytes_per_frame, (width, height))``. ``frame_indices`` selects a
    subset (for downsampling); defaults to every frame. ``progress_callback`` is
    called ``(done, total)`` after each frame so the UI can drive a progress bar.

    Raises :class:`AnimationExportError` if the figure has no frames, Kaleido is
    missing, the browser won't start, or a frame fails to render (e.g. no Chrome).
    """
    try:
        import kaleido
    except Exception as exc:  # pragma: no cover - import guard
        raise AnimationExportError(
            "Kaleido is not installed, so the animation can't be rasterized to "
            "GIF/MP4. Use the HTML export instead, or `pip install kaleido`."
        ) from exc

    frames = list(fig.frames or ())
    if not frames:
        raise AnimationExportError("This animation has no frames to export.")

    indices = frame_indices if frame_indices is not None else list(range(len(frames)))
    base = _static_base(fig)
    width = int(fig.layout.width or 900)
    height = int(base.layout.height)
    elapsed = _elapsed_labels(fig, len(frames)) if show_elapsed else None

    try:
        kaleido.start_sync_server(silence_warnings=True)
    except Exception as exc:
        raise AnimationExportError(
            f"Could not start the Kaleido browser for image export: {exc}. "
            "PNG/GIF/MP4 export needs a Chrome/Chromium binary; the HTML export "
            "needs no browser."
        ) from exc

    pngs: List[bytes] = []
    try:
        for done, k in enumerate(indices, start=1):
            frame = frames[k]
            for data_obj, trace_idx in zip(frame.data, frame.traces):
                base.data[trace_idx].update(data_obj)
            if elapsed is not None:
                base.update_layout(
                    annotations=[
                        dict(
                            text=f"Elapsed: {elapsed[k]}",
                            x=0.99,
                            y=1.0,
                            xref="paper",
                            yref="paper",
                            xanchor="right",
                            yanchor="bottom",
                            showarrow=False,
                            font=dict(size=14, color="#444"),
                        )
                    ]
                )
            try:
                png = kaleido.calc_fig_sync(
                    base,
                    opts={
                        "format": "png",
                        "width": width,
                        "height": height,
                        "scale": scale,
                    },
                )
            except Exception as exc:
                raise AnimationExportError(
                    f"Rendering frame {k + 1}/{len(frames)} failed: {exc}. "
                    "Static image export needs a Chrome/Chromium browser (Kaleido)."
                ) from exc
            pngs.append(bytes(png))
            if progress_callback is not None:
                progress_callback(done, len(indices))
    finally:
        try:
            kaleido.stop_sync_server(silence_warnings=True)
        except Exception:  # pragma: no cover - best-effort teardown
            pass

    return pngs, (width, height)


def _load_rgb_frames(pngs: List[bytes]) -> List["np.ndarray"]:
    from PIL import Image

    return [np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in pngs]


def encode_gif(pngs: List[bytes], frame_duration_ms: float, *, loop: int = 0) -> bytes:
    """Encode PNG frames into an animated GIF with a uniform per-frame delay."""
    from PIL import Image

    if not pngs:
        raise AnimationExportError("No frames to encode.")
    imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in pngs]
    duration = max(int(round(frame_duration_ms)), _GIF_MIN_FRAME_MS)
    buf = io.BytesIO()
    imgs[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=imgs[1:],
        duration=duration,
        loop=loop,
        disposal=2,
        optimize=True,
    )
    return buf.getvalue()


def encode_mp4(pngs: List[bytes], frame_duration_ms: float) -> bytes:
    """Encode PNG frames into an H.264 MP4 whose runtime matches the on-screen Play.

    The on-screen Play shows every frame for ``frame_duration_ms``. An MP4 plays at
    one constant rate, so we encode at a fixed 60 fps and hold each animation frame
    for ``round(frame_duration_ms / (1000/60))`` video frames (at least one). That
    reproduces durations from ~16 ms to several hundred ms accurately — the repeated
    frames are identical, so H.264 compresses them to near-nothing. Frames stream
    through the writer one at a time (repeats reuse the same array), so memory stays
    flat regardless of clip length. H.264 ``yuv420p`` needs even dimensions, so each
    frame is edge-padded to even width/height.
    """
    import os
    import tempfile

    import imageio

    if not pngs:
        raise AnimationExportError("No frames to encode.")

    dt_ms = 1000.0 / _MP4_FPS

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    try:
        writer = imageio.get_writer(
            tmp.name,
            format="FFMPEG",
            mode="I",
            fps=_MP4_FPS,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
        )
        try:
            pad = None
            # Error-diffuse the per-frame repeat count against the cumulative
            # target time so rounding never accumulates into runtime drift: the
            # clip lands on round(n * frame_duration / dt) video frames exactly.
            emitted = 0
            for i, b in enumerate(pngs):
                arr = _load_rgb_frames([b])[0]
                if pad is None:
                    h, w = arr.shape[:2]
                    pad = (h % 2, w % 2)
                if pad[0] or pad[1]:
                    arr = np.pad(arr, ((0, pad[0]), (0, pad[1]), (0, 0)), mode="edge")
                target_total = int(round((i + 1) * frame_duration_ms / dt_ms))
                reps = max(1, target_total - emitted)
                emitted += reps
                for _ in range(reps):
                    writer.append_data(arr)
        finally:
            writer.close()
        with open(tmp.name, "rb") as fh:
            return fh.read()
    except AnimationExportError:
        raise
    except Exception as exc:
        raise AnimationExportError(
            f"MP4 encoding failed: {exc}. Try the GIF format, or check that "
            "imageio-ffmpeg is installed."
        ) from exc
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:  # pragma: no cover - best-effort cleanup
            pass


def export_animation(
    fig: go.Figure,
    *,
    fmt: str,
    frame_duration_ms: float,
    scale: float = 1.0,
    show_elapsed: bool = True,
    max_frames: Optional[int] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> bytes:
    """Render a scanpath-animation figure to GIF or MP4 bytes.

    Args:
        fig: the figure from :func:`make_scanpath_animation` (must have ``.frames``).
        fmt: ``"gif"`` or ``"mp4"``.
        frame_duration_ms: uniform per-frame duration — pass the same average the
            tab quotes (``animation_playback_ms(...) / n_frames``) so the clip's
            runtime matches the on-screen Play.
        scale: Kaleido render scale (1.0 = on-screen px; <1 is faster/smaller,
            >1 is crisper/larger).
        show_elapsed: draw the "Elapsed: X.Xs" readout in the top margin.
        max_frames: cap the number of rendered frames by even downsampling; the
            frame duration is scaled up to keep the total runtime unchanged. The
            UI uses this to bound render time on very long trials.
        progress_callback: ``(done, total)`` after each rendered frame.

    Raises:
        ValueError: unknown ``fmt``.
        AnimationExportError: rendering or encoding failed.
    """
    fmt = fmt.lower()
    if fmt not in VIDEO_FORMATS:
        raise ValueError(
            f"Unsupported format {fmt!r}; expected one of {VIDEO_FORMATS}."
        )

    n_total = len(fig.frames or ())
    indices = _select_frames(n_total, max_frames)
    # Preserve total runtime when downsampling: fewer frames, each held longer.
    effective_duration = frame_duration_ms
    if indices and len(indices) < n_total:
        effective_duration = frame_duration_ms * n_total / len(indices)

    pngs, _size = render_png_frames(
        fig,
        scale=scale,
        show_elapsed=show_elapsed,
        frame_indices=indices,
        progress_callback=progress_callback,
    )
    if fmt == "gif":
        return encode_gif(pngs, effective_duration)
    return encode_mp4(pngs, effective_duration)
