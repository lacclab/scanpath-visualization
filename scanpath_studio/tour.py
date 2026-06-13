"""First-visit welcome tour.

Two interchangeable styles, both introducing the app's main surfaces (data
sources, filters, viz controls, tabs, annotations) the first time a session
opens the app, re-playable any time via the sidebar's tutorial button:

- ``"spotlight"`` (default): a floating card that walks through the *actual*
  UI — each step scrolls the target section into view and pulses an outline
  around it. Rendered by ``render_spotlight_tour()`` at the end of ``main()``.
- ``"dialog"``: a self-contained multi-step ``st.dialog`` modal.

Switch styles with the ``TOUR_STYLE`` constant below.

Mechanics worth knowing before editing:

- Both styles run as *fragments*: Back/Next clicks rerun only the tour body,
  so navigation is instant instead of waiting for a full-app rerun (which
  re-renders the heavy plot embeds, ~10 s). The spotlight's Exit/Done just
  clear ``tour_mode`` — the fragment then renders nothing and the card +
  highlight CSS disappear with it, again with no full rerun. The dialog's
  Skip/Done close the modal client-side (``_close_dialog_clientside``).
- Spotlight targets are ``.st-key-tour_grp_*`` classes from keyed wrapper
  containers around the sidebar sections (app.py / controls.py /
  annotations.py) plus Streamlit's stable ``data-testid``/``data-baseweb``
  attributes for the tab strip. Keep ``_SPOTLIGHT_STEPS`` in sync with them.
- ``tour_seen`` is set **before** the tour is shown, not when it's finished.
  For the dialog style, setting it on Done only would make an X-dismissal
  re-open the modal on the very next widget interaction (any full rerun
  re-calls ``maybe_show_welcome_tour``), making the X appear broken.
- The tour is suppressed for embeds (``?embed=true``) and deep links
  (``?source=…&participant=…``): those sessions arrive mid-workflow from an
  external tool and shouldn't be greeted by a tutorial.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

# First-entry tutorial style: "spotlight" (floating card pointing at the real
# UI) or "dialog" (self-contained modal walkthrough). Both stay available in
# code; this only picks which one auto-opens / the replay button launches.
TOUR_STYLE = "spotlight"

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
- **Scanpath Visualization** — the main scanpath; tick **Animate** to replay it
  fixation by fixation. Hover a word for its reading measures; export.
- **Generations (WIP)** — a grid of model-generated scanpaths over the text.
- **Raw Data** — the normalized tables, downloadable.
- **Data Statistics** — summary stats and distributions.
- **Bulk Export** — bundle figures + tables across many trials into one zip.
""",
    ),
    (
        "📝 Annotate & share",
        """
Star ⭐, tag, and write notes on trials in the Scanpath Visualization tab, then
filter to them. The sidebar **💾 Save & restore** panel saves the full plot
configuration *and* your annotations to one JSON file — and reloads them.

You can also **deep-link** into the app: URL parameters like
`?participant=…&trial=…&show_heatmap=1` open a specific trial with preset
options.

That's it — happy scanpath gazing! 👀
""",
    ),
]


def _close_dialog_clientside() -> None:
    """Hide the open dialog instantly by clicking its own ✕ from a tiny script.

    The documented way to close a dialog programmatically is a full-app
    ``st.rerun()`` — but on this app a full rerun re-renders the heavy plot
    embeds, so the modal lingered ~10 s after Skip/Done. Instead, run inside
    the (fast) dialog-fragment rerun and click the dialog's close button:
    the modal hides client-side immediately and Streamlit's normal dismiss
    handling syncs state in the background. ``components.html`` iframes are
    same-origin, so the script can reach the parent document.
    """
    components.html(
        """<script>
        window.parent.document
            .querySelector('div[role="dialog"] button[aria-label="Close"]')
            ?.click();
        </script>""",
        height=0,
    )


def _step_back() -> None:
    st.session_state["tour_step"] = max(0, st.session_state.get("tour_step", 0) - 1)


def _step_next() -> None:
    st.session_state["tour_step"] = st.session_state.get("tour_step", 0) + 1


@st.dialog("Quick tour", width="large")
def _tour_dialog() -> None:
    """One tour step + Back / Skip / Next navigation.

    Back/Next mutate ``tour_step`` via ``on_click`` callbacks — the callback
    runs before the fragment rerun, so the body re-renders at the new step.
    Skip/Done close the dialog client-side (see ``_close_dialog_clientside``).
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
            _close_dialog_clientside()
        next_col.button(
            "Next →",
            key="tour_next",
            width="stretch",
            type="primary",
            on_click=_step_next,
        )
    else:
        if next_col.button("✓ Done", key="tour_done", width="stretch", type="primary"):
            _close_dialog_clientside()


# Spotlight steps: (selector, title, body). ``selector`` is what gets the
# pulsing outline + scroll-into-view; None for the selector-less welcome step.
# Bodies are markdown, kept short — the card is ~400 px wide.
_SPOTLIGHT_STEPS = [
    {
        "selector": None,
        "title": "👀 Welcome to Scanpath Studio",
        "body": "An interactive workbench for **eye movements in reading** — "
        "fixations, saccades, and per-word reading measures drawn true to "
        "scale over the stimulus text. A demo dataset is already loaded.\n\n"
        "This quick tour points at each part of the app — hit **Next**.",
    },
    {
        "selector": ".st-key-tour_grp_data_source",
        "title": "📂 Data source",
        "body": "Switch between the bundled demo, a synthetic ground-truth "
        "test trial, or your own **uploaded** words/IA and fixations tables "
        "(CSV / TSV / Parquet / Feather). Column names are auto-detected; "
        "the **Column mapping** panels let you override any field.",
    },
    {
        "selector": ".st-key-tour_grp_filter_trials",
        "title": "🔍 Filter trials",
        "body": "Narrow the trial pool by participant or condition before "
        "anything is drawn. Each plotting tab then has its own **trial "
        "picker** — by trial, text, or participant.",
    },
    {
        "selector": ".st-key-tour_grp_viz_controls",
        "title": "🎨 Visualization controls",
        "body": "Toggle what's drawn: fixations (sized by duration), "
        "saccades, the density heatmap, word boxes, and the text itself. "
        "**Display settings** (above) keeps everything true to your "
        "experiment's monitor; **Advanced styling** (below) has colorscales "
        "and marker sizing.",
    },
    {
        "selector": '[data-testid="stTabs"] [data-baseweb="tab-list"]',
        "title": "🗂 Five views of the data",
        "body": "**Scanpath Visualization** — the main scanpath; tick "
        "**Animate** to replay it. **Generations (WIP)** — a grid of "
        "model scanpaths. Plus **Raw Data**, **Data Statistics**, and "
        "**Bulk Export** for many trials at once.",
    },
    {
        "selector": ".st-key-tour_grp_save_restore",
        "title": "💾 Save & restore",
        "body": "Star ⭐, tag, and note trials in the Scanpath Visualization "
        "tab. This panel saves the full plot configuration **and** your "
        "annotations to one JSON file — and reloads them. URL parameters like "
        "`?participant=…&trial=…` deep-link straight to a trial.\n\n"
        "That's it — replay anytime via **🎓 Show tutorial** below. 👀",
    },
]

# The floating card: a keyed st.container pinned bottom-right via its
# `.st-key-tour_card` class. Plain strings (no .format) so the CSS braces
# don't need escaping.
_CARD_CSS = """
.st-key-tour_card {
    position: fixed;
    bottom: 1.25rem;
    right: 1.25rem;
    z-index: 999990;
    width: 410px;
    max-width: calc(100vw - 2.5rem);
    border-radius: 0.75rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    padding: 1rem 1.25rem;
}
"""

# Welcome step only: center the card like a modal and dim the app behind it
# (the `.tour-backdrop` div is rendered only on that step). From step 2 on,
# the card drops to the bottom-right corner so it never covers the
# highlighted section.
_WELCOME_CSS = """
.st-key-tour_card {
    top: 50%;
    left: 50%;
    right: auto;
    bottom: auto;
    transform: translate(-50%, -50%);
    width: 500px;
}
.tour-backdrop {
    position: fixed;
    inset: 0;
    z-index: 999980;
    background: rgba(0, 0, 0, 0.45);
}
"""


def _exit_spotlight() -> None:
    st.session_state["tour_mode"] = None


@st.fragment
def render_spotlight_tour() -> None:
    """Floating tour card + pulsing highlight for the current spotlight step.

    Call early in ``main()``, right after ``maybe_show_welcome_tour()``, so
    the card streams to the browser before the heavy data/plot work instead
    of seconds after the page opens. Replay clicks still activate it within
    the same run because the button arms the tour in its ``on_click``
    callback (``_arm_tour``), which runs before the rerun starts. Runs as a
    fragment: Back/Next/Exit rerun only this function, so the highlight moves
    instantly and Exit makes the card + CSS vanish without a full-app rerun
    (the fragment then renders nothing, which clears its previous elements).
    """
    if st.session_state.get("tour_mode") != "spotlight":
        return
    n = len(_SPOTLIGHT_STEPS)
    step_idx = min(st.session_state.get("tour_step", 0), n - 1)
    step = _SPOTLIGHT_STEPS[step_idx]

    accent = st.get_option("theme.primaryColor") or "#ff4b4b"
    # Card colors follow the active theme when the runtime exposes it
    # (st.context.theme, Streamlit ≥1.46); default to light otherwise.
    theme = getattr(getattr(st, "context", None), "theme", None)
    is_dark = getattr(theme, "type", "light") == "dark"
    bg, border = ("#262730", "#41434e") if is_dark else ("#ffffff", "#d5d6d9")

    highlight = ""
    if step["selector"]:
        highlight = f"""
{step["selector"]} {{
    outline: 3px solid {accent};
    outline-offset: 3px;
    border-radius: 0.5rem;
    animation: tour-pulse 1.6s ease-in-out infinite;
}}
@keyframes tour-pulse {{
    0%, 100% {{ box-shadow: 0 0 0 0 color-mix(in srgb, {accent} 50%, transparent); }}
    50% {{ box-shadow: 0 0 14px 7px color-mix(in srgb, {accent} 25%, transparent); }}
}}
"""
    st.markdown(
        "<style>"
        + _CARD_CSS
        + f".st-key-tour_card {{ background: {bg}; border: 1px solid {border}; }}"
        + (_WELCOME_CSS if step_idx == 0 else "")
        + highlight
        + "</style>",
        unsafe_allow_html=True,
    )
    if step_idx == 0:
        st.markdown('<div class="tour-backdrop"></div>', unsafe_allow_html=True)

    with st.container(key="tour_card"):
        st.markdown(f"#### {step['title']}")
        st.markdown(step["body"])
        st.progress((step_idx + 1) / n, text=f"Step {step_idx + 1} of {n}")
        back_col, exit_col, next_col = st.columns(3)
        back_col.button(
            "← Back",
            key="tour_sp_back",
            width="stretch",
            disabled=step_idx == 0,
            on_click=_step_back,
        )
        if step_idx < n - 1:
            exit_col.button(
                "Exit tour",
                key="tour_sp_exit",
                width="stretch",
                on_click=_exit_spotlight,
            )
            next_col.button(
                "Next →",
                key="tour_sp_next",
                width="stretch",
                type="primary",
                on_click=_step_next,
            )
        else:
            next_col.button(
                "✓ Done",
                key="tour_sp_done",
                width="stretch",
                type="primary",
                on_click=_exit_spotlight,
            )

        if step["selector"]:
            # Bring the highlighted section into view. Same-origin iframe
            # trick as _close_dialog_clientside; no-op if the selector is
            # gone. Subtleties, all observed live:
            # - Sidebar steps first click Streamlit's expand control, since
            #   tour sessions start with the sidebar collapsed
            #   (spotlight_tour_pending → initial_sidebar_state).
            # - The find+scroll retries until the target is visible, riding
            #   out the sidebar-expand animation.
            # - Match the first *visible* element, not the first match:
            #   Streamlit keeps inactive tab panels laid out but
            #   visibility-hidden, so a selector can hit an invisible
            #   duplicate (e.g. the Raw Data panel's inner tab strip) and
            #   scroll the page to nowhere.
            # - No scrollIntoView: smooth gets cancelled by Streamlit's
            #   re-renders, and instant also scrolls the document, moving
            #   the main column for sidebar targets. Instead, center the
            #   target within its nearest scrollable ancestor only.
            # - Skip targets that are already fully on screen.
            # - The iframe stays INSIDE the fixed-position card: when it sat
            #   at the bottom of the main column, its (re)mount could yank
            #   the main scroller to the page bottom to reveal it.
            in_sidebar = step["selector"].startswith(".st-key-tour_grp_")
            components.html(
                f"""<script>
                (function () {{
                    const doc = window.parent.document;
                    const win = doc.defaultView;
                    const findVisible = () =>
                        [...doc.querySelectorAll({step["selector"]!r})].find((e) => {{
                            const r = e.getBoundingClientRect();
                            if (r.width === 0 || r.height === 0) return false;
                            const cs = win.getComputedStyle(e);
                            return cs.visibility !== "hidden" && cs.display !== "none";
                        }});
                    let tries = 0;
                    (function attempt() {{
                        if ({str(in_sidebar).lower()}) {{
                            // The collapsed sidebar keeps its layout (nonzero
                            // rects), so gate on aria-expanded, not on
                            // findVisible(). Retries ride out hydration.
                            const sb = doc.querySelector(
                                'section[data-testid="stSidebar"]');
                            if (sb && sb.getAttribute("aria-expanded") !== "true") {{
                                doc.querySelector(
                                    'button[data-testid="stExpandSidebarButton"]'
                                )?.click();
                                if (++tries < 20) setTimeout(attempt, 150);
                                return;
                            }}
                        }}
                        const el = findVisible();
                        if (!el) {{
                            if (++tries < 20) setTimeout(attempt, 150);
                            return;
                        }}
                        for (let box = el.parentElement; box; box = box.parentElement) {{
                            const cs = win.getComputedStyle(box);
                            if (/(auto|scroll|overlay)/.test(cs.overflowY)
                                    && box.scrollHeight > box.clientHeight + 4) {{
                                const r = el.getBoundingClientRect();
                                const b = box.getBoundingClientRect();
                                const slack = 8;
                                if (r.top >= b.top - slack
                                        && r.bottom <= b.top + box.clientHeight + slack) {{
                                    return;  // already visible within its scroller
                                }}
                                box.scrollTop += r.top - b.top
                                    - Math.max(0, (box.clientHeight - r.height) / 2);
                                return;
                            }}
                        }}
                    }})();
                }})();
                </script>""",
                height=0,
            )
        else:
            # Welcome step: close the sidebar so the centered card sits over
            # a quiet page. initial_sidebar_state="collapsed" (configure_page)
            # covers fresh visitors, but the frontend's per-tab stored sidebar
            # state overrides it for returning tabs — so also click the
            # collapse control. Retries because a click during initial React
            # hydration is silently lost. The first sidebar step reopens it.
            components.html(
                """<script>
                (function () {
                    const doc = window.parent.document;
                    let tries = 0;
                    (function attempt() {
                        const sb = doc.querySelector('section[data-testid="stSidebar"]');
                        if (!sb || sb.getAttribute("aria-expanded") !== "true") return;
                        (doc.querySelector(
                            '[data-testid="stSidebarCollapseButton"] button')
                            || doc.querySelector('section[data-testid="stSidebar"]'
                                + ' [data-testid="stBaseButton-headerNoPadding"]'))
                            ?.click();
                        if (++tries < 25) setTimeout(attempt, 200);
                    })();
                })();
                </script>""",
                height=0,
            )


def spotlight_tour_pending() -> bool:
    """True when this session is about to auto-open the spotlight tour.

    Read by ``app.configure_page`` *before* ``maybe_show_welcome_tour`` sets
    ``tour_seen``: tour sessions start with the sidebar collapsed so the
    centered welcome renders over a quiet page; the first sidebar step then
    opens it (the step script clicks Streamlit's expand control).
    """
    return (
        TOUR_STYLE == "spotlight"
        and not st.session_state.get("tour_seen")
        and not tour_suppressed(st.query_params)
    )


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


def _start_tour() -> None:
    """Kick off the configured tour style from step 0."""
    st.session_state["tour_step"] = 0
    if TOUR_STYLE == "spotlight":
        st.session_state["tour_mode"] = "spotlight"
    else:
        _tour_dialog()


def _arm_tour() -> None:
    """``on_click`` callback for the replay button: arm the tour from step 0.

    Callbacks run *before* the rerun, so the tour's render call early in
    ``main()`` — which executes long before the sidebar button — picks the
    request up within the same run. Dialogs can't be opened from callbacks,
    so the dialog style sets a request flag that ``maybe_show_welcome_tour``
    (the early call site) serves.
    """
    st.session_state["tour_step"] = 0
    if TOUR_STYLE == "spotlight":
        st.session_state["tour_mode"] = "spotlight"
    else:
        st.session_state["_tour_dialog_requested"] = True


def maybe_show_welcome_tour() -> None:
    """Start the welcome tour once per session, unless this is an embed/deep link.

    Call from ``main()`` after the URL presets are read (the suppression
    checks look at ``st.query_params``) but BEFORE the heavy data/plot work,
    immediately followed by ``render_spotlight_tour()`` — Streamlit streams
    elements in run order, so anything rendered after the data load appears
    seconds late. The dialog style opens here and overlays whatever renders
    after it; the spotlight style just arms ``tour_mode``.
    """
    if st.session_state.pop("_tour_dialog_requested", False):
        # Replay request from the sidebar button's on_click callback.
        _tour_dialog()
        return
    if st.session_state.get("tour_seen"):
        return
    if tour_suppressed(st.query_params):
        return
    st.session_state["tour_seen"] = True  # before opening — see module docstring
    _start_tour()


def render_tour_replay_button() -> None:
    """Sidebar button that replays the tour from the first step."""
    st.sidebar.button(
        "🎓 Show tutorial",
        key="tour_replay",
        width="stretch",
        help="Replay the quick intro tour.",
        on_click=_arm_tour,
    )
