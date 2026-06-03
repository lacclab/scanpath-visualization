"""Per-trial researcher annotations: favorites (stars), tags, and free notes.

Annotations are keyed by ``(participant_id, trial_id)`` and live in Streamlit
session state, so they persist across reruns within a session. There is no
backend; to keep annotations across sessions or share them, the sidebar offers
a JSON **download** (a portable sidecar) and **restore** (re-upload).

The module is split into a *pure* core (``records_to_store`` /
``store_to_records`` / ``serialize`` / ``deserialize`` — no Streamlit, unit
tested) and a thin session-backed layer plus the small render helpers used by
``tabs.py`` and ``app.py``.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Set, Tuple

import streamlit as st

ANNOTATIONS_STATE_KEY = "trial_annotations"
SCHEMA_VERSION = 1

# Per-trial annotation widgets use this prefix so they can be cleared on import
# (forcing a re-seed from the freshly loaded store). Kept distinct from the
# sidebar control keys ("anno_download" / "anno_upload").
_WIDGET_PREFIX = "annotrial_"
_LAST_IMPORT_KEY = "_anno_last_import"

# Always-available tag suggestions (users can add their own on top).
PRESET_TAGS = ["To exclude", "Review", "Good example", "Check alignment"]

# (participant_id, trial_id) -> {"star": bool, "tags": list[str], "note": str}
Key = Tuple[str, str]
Entry = Dict[str, object]


# ---------------------------------------------------------------------------
# Pure core (no Streamlit) — unit tested in tests/test_annotations.py
# ---------------------------------------------------------------------------


def default_entry() -> Entry:
    return {"star": False, "tags": [], "note": ""}


def _normalize_entry(star: object, tags: object, note: object) -> Entry:
    clean_tags = sorted({str(t).strip() for t in (tags or []) if str(t).strip()})
    return {"star": bool(star), "tags": clean_tags, "note": str(note or "").strip()}


def is_empty_entry(entry: Entry) -> bool:
    """True when an entry carries no information (and can be dropped)."""
    return (
        not entry.get("star")
        and not entry.get("tags")
        and not str(entry.get("note") or "").strip()
    )


def records_to_store(records: List[dict]) -> Dict[Key, Entry]:
    """Build a ``{(pid, tid): entry}`` store from a list of flat records."""
    store: Dict[Key, Entry] = {}
    for rec in records or []:
        pid = rec.get("participant_id")
        tid = rec.get("trial_id")
        if pid is None or tid is None:
            continue
        entry = _normalize_entry(
            rec.get("star", False), rec.get("tags", []), rec.get("note", "")
        )
        if not is_empty_entry(entry):
            store[(str(pid), str(tid))] = entry
    return store


def store_to_records(store: Dict[Key, Entry]) -> List[dict]:
    """Flatten a store into a sorted list of records for JSON export."""
    records = []
    for (pid, tid), entry in sorted(store.items()):
        records.append(
            {
                "participant_id": pid,
                "trial_id": tid,
                "star": bool(entry.get("star", False)),
                "tags": list(entry.get("tags", [])),
                "note": str(entry.get("note", "")),
            }
        )
    return records


def serialize(store: Dict[Key, Entry]) -> str:
    """Serialize a store to a JSON document string."""
    return json.dumps(
        {"schema": SCHEMA_VERSION, "annotations": store_to_records(store)}, indent=2
    )


def deserialize(text: str) -> Dict[Key, Entry]:
    """Parse a JSON document (object with ``annotations`` or a bare list)."""
    data = json.loads(text)
    if isinstance(data, dict):
        records = data.get("annotations", [])
    elif isinstance(data, list):
        records = data
    else:
        records = []
    return records_to_store(records)


# ---------------------------------------------------------------------------
# Session-backed layer
# ---------------------------------------------------------------------------


def _store() -> Dict[Key, Entry]:
    return st.session_state.setdefault(ANNOTATIONS_STATE_KEY, {})


def get_entry(participant_id: str, trial_id: str) -> Entry:
    return _store().get((str(participant_id), str(trial_id)), default_entry())


def set_entry(
    participant_id: str, trial_id: str, *, star: bool, tags: List[str], note: str
) -> None:
    """Upsert an annotation; empty entries are pruned to keep the store small."""
    key = (str(participant_id), str(trial_id))
    entry = _normalize_entry(star, tags, note)
    store = _store()
    if is_empty_entry(entry):
        store.pop(key, None)
    else:
        store[key] = entry


def known_tags() -> List[str]:
    """Preset tags plus any tag used anywhere in the store, sorted."""
    tags: Set[str] = set(PRESET_TAGS)
    for entry in _store().values():
        tags.update(entry.get("tags", []))
    return sorted(tags)


def starred_keys() -> Set[Key]:
    return {k for k, v in _store().items() if v.get("star")}


def keys_with_tag(tag: str) -> Set[Key]:
    return {k for k, v in _store().items() if tag in v.get("tags", [])}


def annotated_count() -> int:
    return len(_store())


# ---------------------------------------------------------------------------
# UI render helpers
# ---------------------------------------------------------------------------


def _add_tag_callback(tags_key: str, newtag_key: str) -> None:
    """on_change for the 'add tag' input: append to the multiselect's state.

    Runs before the next rerun, so writing the multiselect's session_state here
    is allowed (the widget hasn't been instantiated yet that run)."""
    new_tag = str(st.session_state.get(newtag_key, "")).strip()
    if not new_tag:
        return
    current = list(st.session_state.get(tags_key, []))
    if new_tag not in current:
        st.session_state[tags_key] = current + [new_tag]
    st.session_state[newtag_key] = ""


def render_trial_annotations(participant_id: str, trial_id: str) -> None:
    """Render the per-trial annotations expander (star / tags / notes)."""
    entry = get_entry(participant_id, trial_id)
    slug = f"{participant_id}__{trial_id}"
    star_key = f"{_WIDGET_PREFIX}star_{slug}"
    tags_key = f"{_WIDGET_PREFIX}tags_{slug}"
    note_key = f"{_WIDGET_PREFIX}note_{slug}"
    newtag_key = f"{_WIDGET_PREFIX}newtag_{slug}"

    # Seed widget state once from the store (re-seeds after a JSON import, which
    # clears these keys).
    st.session_state.setdefault(star_key, entry["star"])
    st.session_state.setdefault(tags_key, list(entry["tags"]))
    st.session_state.setdefault(note_key, entry["note"])

    label = "📝 Annotations" + (" ⭐" if entry["star"] else "")
    if entry["tags"]:
        label += f" · {', '.join(entry['tags'])}"
    with st.expander(label, expanded=False):
        star = st.checkbox("⭐ Favorite (star this trial)", key=star_key)
        # Options must include every currently-selected tag (incl. ones added
        # via the input) or st.multiselect raises.
        options = sorted(
            set(known_tags())
            | set(entry["tags"])
            | set(st.session_state.get(tags_key, []))
        )
        tags = st.multiselect(
            "Tags",
            options=options,
            key=tags_key,
            help="Pick presets (e.g. 'To exclude') or add your own below.",
        )
        st.text_input(
            "Add a tag",
            key=newtag_key,
            placeholder="type a tag, press Enter",
            on_change=_add_tag_callback,
            args=(tags_key, newtag_key),
        )
        note = st.text_area(
            "Notes",
            key=note_key,
            placeholder="Researcher notes for this trial…",
            height=100,
        )
        set_entry(participant_id, trial_id, star=star, tags=tags, note=note)
        st.caption(
            "Saved for this session. Use the sidebar **Annotations** panel to "
            "download a JSON copy or restore one."
        )


def render_annotations_sidebar() -> None:
    """Render the sidebar Annotations panel: count + JSON download/restore."""
    st.sidebar.header("Annotations")
    count = annotated_count()
    st.sidebar.caption(
        f"{count} trial(s) annotated this session."
        if count
        else "No annotations yet — star, tag, or note trials in the Interactive Plot tab."
    )
    st.sidebar.download_button(
        "⬇ Download annotations (JSON)",
        data=serialize(_store()),
        file_name="scanpath_annotations.json",
        mime="application/json",
        disabled=(count == 0),
        key="anno_download",
        help="A portable sidecar of all stars / tags / notes from this session.",
    )
    uploaded = st.sidebar.file_uploader(
        "Restore annotations (JSON)",
        type=["json"],
        key="anno_upload",
        help="Re-load a previously downloaded annotations file. Replaces the current set.",
    )
    if uploaded is not None:
        signature = (uploaded.name, uploaded.size)
        if st.session_state.get(_LAST_IMPORT_KEY) != signature:
            try:
                store = deserialize(uploaded.getvalue().decode("utf-8"))
            except Exception as exc:  # malformed file
                st.sidebar.error(f"Could not load annotations: {exc}")
            else:
                st.session_state[ANNOTATIONS_STATE_KEY] = store
                st.session_state[_LAST_IMPORT_KEY] = signature
                # Drop per-trial widget state so it re-seeds from the new store.
                for key in [
                    k
                    for k in list(st.session_state.keys())
                    if k.startswith(_WIDGET_PREFIX)
                ]:
                    del st.session_state[key]
                st.sidebar.success(f"Loaded {len(store)} annotation(s).")
                st.rerun()


def select_keys(
    store: Dict[Key, Entry],
    keys: List[Key],
    *,
    favorites_only: bool = False,
    required_tags: Optional[List[str]] = None,
    excluded_tags: Optional[List[str]] = None,
) -> List[Key]:
    """Pure core of :func:`filter_keys` — filter ``keys`` against ``store``.

    - ``favorites_only``: keep only starred trials.
    - ``required_tags``: keep trials carrying *any* of these tags.
    - ``excluded_tags``: drop trials carrying *any* of these tags.
    """
    required = set(required_tags or [])
    excluded = set(excluded_tags or [])
    out: List[Key] = []
    for key in keys:
        entry = store.get(key)
        tags = set(entry.get("tags", [])) if entry else set()
        starred = bool(entry.get("star")) if entry else False
        if favorites_only and not starred:
            continue
        if required and not (tags & required):
            continue
        if excluded and (tags & excluded):
            continue
        out.append(key)
    return out


def filter_keys(
    keys: List[Key],
    *,
    favorites_only: bool = False,
    required_tags: Optional[List[str]] = None,
    excluded_tags: Optional[List[str]] = None,
) -> List[Key]:
    """Session-backed wrapper around :func:`select_keys`."""
    return select_keys(
        _store(),
        keys,
        favorites_only=favorites_only,
        required_tags=required_tags,
        excluded_tags=excluded_tags,
    )
