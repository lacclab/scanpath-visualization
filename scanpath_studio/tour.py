"""First-visit welcome tour.

A multi-step ``st.dialog`` that introduces the app's main surfaces (data
sources, filters, viz controls, tabs, annotations) the first time a session
opens the app. Re-playable any time via the sidebar's tutorial button.

Mechanics worth knowing before editing:

- ``st.dialog`` is a *fragment*: Back/Next clicks rerun only the dialog body,
  so navigation works without ``main()`` re-calling the dialog. A full-app
  rerun (``st.rerun()``) closes it, because nothing re-opens it.
- ``tour_seen`` is set **before** the dialog is opened, not when it's
  finished. If it were set on Done only, dismissing the dialog with X /
  Esc would re-open it on the very next widget interaction (any full rerun
  re-calls ``maybe_show_welcome_tour``), making the X appear broken.
- The tour is suppressed for embeds (``?embed=true``) and deep links
  (``?source=…&participant=…``): those sessions arrive mid-workflow from an
  external tool and shouldn't be greeted by a modal.
"""

from __future__ import annotations

import streamlit as st

# (title, markdown body) per step — keep bodies to a few lines each; the tour
# should take well under a minute.
_STEPS = [
    (
        "👀 Welcome to Scanpath Studio",
        """
Scanpath Studio is an interactive workbench for **eye movements in reading**:
fixations, saccades, and per-word reading measures drawn **true to scale**
over the stimulus text.

A demo dataset (a sample of
[OneStop Eye Movements](https://github.com/lacclab/OneStop-Eye-Movements))
is already loaded, so everything you'll see works out of the box.

This quick tour takes under a minute — reopen it anytime with
**🎓 Show tutorial** at the bottom of the sidebar.
""",
    ),
    (
        "📂 Load your data",
        """
The **Data source** panel at the top of the sidebar switches between:

- **Bundled demo** — the preloaded OneStop sample.
- **Synthetic test trial** — a tiny ground-truth trial with documented
  expected measures, handy for sanity checks.
- **Public datasets** — ready-made corpus loaders (currently PoTeC),
  downloaded on demand.
- **Upload tables** — your own words/IA and fixations tables
  (CSV / TSV / Parquet / Feather; several files per table, or either
  table alone).

Column names are auto-detected (EyeLink, Gazepoint, snake_case …), and the
**Column mapping** panels let you override any field.
""",
    ),
    (
        "🔍 Filter & pick trials",
        """
**Filter trials** (sidebar) narrows the dataset by participant or condition
before anything is drawn.

Each plotting tab then has its own **trial picker** — select a trial
directly, or pick a participant and step through their trials with a slider.
""",
    ),
    (
        "🎨 Style the visualization",
        """
The sidebar's **Visualization** section controls what's drawn: fixations
(sized by duration), saccades, a density heatmap, word boxes, and the text
itself.

**Display settings** keeps the rendering true to the experiment: enter your
monitor's resolution and the text and coordinates appear exactly as
presented. **Advanced styling** holds colorscales and marker sizing.
""",
    ),
    (
        "🗂 Five views of the data",
        """
- **Interactive Plot** — the main scanpath; hover a word for its reading
  measures, export PNG / SVG / PDF / HTML.
- **Animated Scanpath** — replay the trial fixation by fixation; export
  GIF / MP4.
- **Multiple Comparison** — a grid of scanpaths across trials or readers.
- **Raw Data** — the normalized tables, downloadable.
- **Data Statistics** — summary stats and distributions.
""",
    ),
    (
        "📝 Annotate & share",
        """
Star ⭐, tag, and write notes on trials from the Interactive Plot tab, then
filter to them or save the annotations as JSON (sidebar **Annotations**).

You can also **deep-link** into the app: URL parameters like
`?participant=…&trial=…&show_heatmap=1` open a specific trial with preset
options.

That's it — happy scanpath gazing! 👀
""",
    ),
]


def _step_back() -> None:
    st.session_state["tour_step"] = max(0, st.session_state.get("tour_step", 0) - 1)


def _step_next() -> None:
    st.session_state["tour_step"] = st.session_state.get("tour_step", 0) + 1


@st.dialog("Quick tour", width="large")
def _tour_dialog() -> None:
    """One tour step + Back / Skip / Next navigation.

    Back/Next mutate ``tour_step`` via ``on_click`` callbacks — the callback
    runs before the fragment rerun, so the body re-renders at the new step.
    Skip/Done close the dialog via a full-app ``st.rerun()``.
    """
    step = min(st.session_state.get("tour_step", 0), len(_STEPS) - 1)
    title, body = _STEPS[step]
    st.subheader(title)
    st.markdown(body)
    st.progress((step + 1) / len(_STEPS), text=f"Step {step + 1} of {len(_STEPS)}")

    back_col, skip_col, next_col = st.columns(3)
    back_col.button(
        "← Back",
        key="tour_back",
        width="stretch",
        disabled=step == 0,
        on_click=_step_back,
    )
    if step < len(_STEPS) - 1:
        if skip_col.button("Skip tour", key="tour_skip", width="stretch"):
            st.rerun()
        next_col.button(
            "Next →",
            key="tour_next",
            width="stretch",
            type="primary",
            on_click=_step_next,
        )
    else:
        if next_col.button("✓ Done", key="tour_done", width="stretch", type="primary"):
            st.rerun()


def tour_suppressed(query_params) -> bool:
    """True when the session shouldn't be greeted by the tour.

    Embeds (``?embed=true``) and deep links (``?source=…&participant=…``)
    arrive mid-workflow from an external tool. Takes the params as a mapping
    (rather than reading ``st.query_params`` itself) because AppTest can't
    inject query params — this stays unit-testable.
    """
    if (query_params.get("embed") or "").lower() in {"true", "1"}:
        return True
    return any(k in query_params for k in ("source", "participant", "trial", "tab"))


def maybe_show_welcome_tour() -> None:
    """Open the welcome tour once per session, unless this is an embed/deep link.

    Call from ``main()`` after the URL presets are read (the suppression
    checks look at ``st.query_params``) — the dialog overlays whatever
    renders after it, so data can keep loading behind the modal.
    """
    if st.session_state.get("tour_seen"):
        return
    if tour_suppressed(st.query_params):
        return
    st.session_state["tour_seen"] = True  # before opening — see module docstring
    st.session_state["tour_step"] = 0
    _tour_dialog()


def render_tour_replay_button() -> None:
    """Sidebar button that replays the tour from the first step."""
    if st.sidebar.button(
        "🎓 Show tutorial",
        key="tour_replay",
        width="stretch",
        help="Replay the quick intro tour.",
    ):
        st.session_state["tour_step"] = 0
        _tour_dialog()
