"""Manual error overrides set by the user in the Viewer.

Overrides are stored in ``st.session_state`` per inference id as an explicit
verdict ``{fiber_id: is_error}``, so they survive page changes and are applied
identically by the Viewer and the Analysis pages. They never mutate cached
``Fibers`` objects.
"""

import attrs
import streamlit as st

from dnafiber.postprocess.fiber import Fibers

_OVERRIDES_KEY = "error_overrides"
_NONCES_KEY = "error_overrides_nonces"
_ENTRIES_KEY = "error_overrides_entries"  # inference id -> entry id


def get_overrides(inference_id: str) -> dict[int, bool]:
    return st.session_state.get(_OVERRIDES_KEY, {}).get(inference_id, {})


def overridden_ids(inference_id: str) -> list[int]:
    return sorted(get_overrides(inference_id))


def all_overrides_signature() -> str:
    """Changes whenever any override of any image changes."""
    return repr(sorted(
        (iid, sorted(o.items()))
        for iid, o in st.session_state.get(_OVERRIDES_KEY, {}).items()
    ))


def overrides_with_other_settings(entry_id: str, inference_id: str) -> int:
    """Number of overrides made on the same image but under other settings
    (model, TTA, pixel size, clarity...), which therefore do not apply here."""
    entries = st.session_state.get(_ENTRIES_KEY, {})
    return sum(
        len(get_overrides(iid))
        for iid, eid in entries.items()
        if eid == entry_id and iid != inference_id
    )


def overrides_signature(inference_id: str) -> str:
    """A string that changes whenever the overrides of `inference_id` change (for cache keys)."""
    return ",".join(
        f"{fid}:{int(err)}" for fid, err in sorted(get_overrides(inference_id).items())
    )


def apply_overrides(fibers: Fibers, inference_id: str) -> Fibers:
    """Return a copy of `fibers` with the user's verdicts applied (proba_error set to 0 or 1)."""
    overrides = get_overrides(inference_id)
    if not overrides:
        return fibers
    return Fibers(
        [
            attrs.evolve(f, proba_error=1.0 if overrides[f.fiber_id] else 0.0)
            if f.fiber_id in overrides
            else f
            for f in fibers
        ],
        path=fibers.path,
    )


def update_overrides_from_component(
    component_value,
    component_key: str,
    inference_id: str,
    model_fibers: Fibers,
    threshold: float,
    entry_id: str | None = None,
) -> bool:
    """Commit a selection sent by the fiber_ui component.

    The selection is the full set of fibers whose model verdict the user wants
    flipped. Fibers already overridden keep their stored verdict, new ones get
    the opposite of the model verdict, and deselected ones are dropped.

    Returns True if the overrides were updated (the caller should rerun).
    """
    if not isinstance(component_value, dict) or "nonce" not in component_value:
        return False
    nonces = st.session_state.setdefault(_NONCES_KEY, {})
    if nonces.get(component_key) == component_value["nonce"]:
        return False
    nonces[component_key] = component_value["nonce"]

    previous = get_overrides(inference_id)
    model_verdicts = {f.fiber_id: f.proba_error >= threshold for f in model_fibers}
    new = {}
    for fid in component_value.get("selected", []):
        fid = int(fid)
        if fid in previous:
            new[fid] = previous[fid]
        elif fid in model_verdicts:
            new[fid] = not model_verdicts[fid]

    all_overrides = st.session_state.setdefault(_OVERRIDES_KEY, {})
    if new:
        all_overrides[inference_id] = new
    else:
        all_overrides.pop(inference_id, None)
    if entry_id is not None:
        st.session_state.setdefault(_ENTRIES_KEY, {})[inference_id] = entry_id
    return True
