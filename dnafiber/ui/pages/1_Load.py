"""DN-AI loading page.

Channel role model:
  - Two required roles: First analog (analog_1) and Second analog (analog_2).
    These are the only roles the model consumes today.
  - One optional named role: ssDNA. Stored if assigned; not consumed yet.
  - Free-form "Other" label per channel. Stored as display metadata; not
    consumed yet. Not reused across files within a session.
  - Anything else: Ignore.

Queue entry shape (consumed by the viewer page and downstream code):
    {
        "id": str,                   # stable, mapping-aware identifier
        "display_name": str,
        "mode": "multichannel" | "per_role",
        "sources": {role_id: (Path, channel_index)},   # analog_1, analog_2, ssdna
        "extra_labels": {channel_index: str},          # Other-named channels
    }
"""

from pathlib import Path
from typing import Optional

import streamlit as st

from dnafiber.ui.consts import DefaultValues as DV
from dnafiber.ui.components import performance_button
from dnafiber.ui.utils import init_session_states, retain_session_state


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCEPTED_EXTENSIONS = [
    ".czi",
    ".tif",
    ".tiff",
    ".dv",
    ".nd2",
    ".png",
    ".jpg",
    ".jpeg",
]

ROLE_ANALOG_1 = "analog_1"
ROLE_ANALOG_2 = "analog_2"
ROLE_SSDNA = "ssdna"
ROLE_OTHER = "__other__"
ROLE_IGNORE = "__ignore__"

ROLE_LABELS = {
    ROLE_ANALOG_1: "First analog",
    ROLE_ANALOG_2: "Second analog",
    ROLE_SSDNA: "ssDNA",
    ROLE_OTHER: "Other…",
    ROLE_IGNORE: "— Ignore —",
}

DEFAULT_ROLE_COLORS = {
    ROLE_ANALOG_1: "#FF0000",
    ROLE_ANALOG_2: "#00FF00",
    ROLE_SSDNA: "#0000FF",
    ROLE_OTHER: "#888888",
    ROLE_IGNORE: "#444444",
}


def role_color(role_id: str) -> str:
    """Live color lookup. Analog colors are user-overridable; the rest are fixed."""
    if role_id == ROLE_ANALOG_1:
        return st.session_state.get(
            "color_analog_1", DEFAULT_ROLE_COLORS[ROLE_ANALOG_1]
        )
    if role_id == ROLE_ANALOG_2:
        return st.session_state.get(
            "color_analog_2", DEFAULT_ROLE_COLORS[ROLE_ANALOG_2]
        )
    return DEFAULT_ROLE_COLORS.get(role_id, "#888888")


ROLE_OPTIONS = [
    ROLE_ANALOG_1,
    ROLE_ANALOG_2,
    ROLE_SSDNA,
    ROLE_OTHER,
    ROLE_IGNORE,
]


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

retain_session_state(st.session_state)
init_session_states()

st.set_page_config(page_title="DN-AI", page_icon="🔬", layout="wide")


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("n_channels", 2)
    ss.setdefault("channel_mapping", {0: ROLE_ANALOG_1, 1: ROLE_ANALOG_2})
    ss.setdefault("channel_other_names", {})  # int → str
    ss.setdefault("files_uploaded", [])
    ss.setdefault("pixel_size", DV.PIXEL_SIZE)
    ss.setdefault("color_analog_1", DEFAULT_ROLE_COLORS[ROLE_ANALOG_1])
    ss.setdefault("color_analog_2", DEFAULT_ROLE_COLORS[ROLE_ANALOG_2])
    ss.setdefault("multitile_strategy", "compact")


_init_state()


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _case_insensitive_glob(ext: str) -> str:
    return "*" + "".join(f"[{c.lower()}{c.upper()}]" if c.isalpha() else c for c in ext)


def _resolve_path(text: str, extensions: list[str]) -> tuple[list[Path], Optional[str]]:
    if not text:
        return [], None
    if text.startswith("file://"):
        text = text[7:]
    p = Path(text.strip())
    if p.is_dir():
        found: list[Path] = []
        for ext in extensions:
            found.extend(p.rglob(_case_insensitive_glob(ext)))
        return sorted(found, key=lambda x: x.name), None
    if p.is_file():
        if p.suffix.lower() in extensions:
            return [p], None
        return [], f"Extension `{p.suffix}` is not in the accepted formats."
    return [], f"`{p}` is not a valid file or folder."


# ---------------------------------------------------------------------------
# Setup section
# ---------------------------------------------------------------------------


def render_setup() -> None:
    st.subheader("Experiment setup")
    cols = st.columns([1, 1, 1, 2])
    with cols[0]:
        st.number_input(
            "Pixel size (µm)",
            min_value=0.01,
            max_value=10.0,
            step=0.01,
            key="pixel_size",
            help=f"Default: {DV.PIXEL_SIZE} µm. The model is trained at "
            "0.26 µm/px; images are resampled to match.",
        )
    with cols[1]:
        st.color_picker(
            f"{ROLE_LABELS[ROLE_ANALOG_1]} color",
            key="color_analog_1",
            help="Used to render this analog's segments in the viewer.",
        )
    with cols[2]:
        st.color_picker(
            f"{ROLE_LABELS[ROLE_ANALOG_2]} color",
            key="color_analog_2",
            help="Used to render this analog's segments in the viewer.",
        )
    with cols[3]:
        st.selectbox(
            "Multi-tiles strategy (ND2 only)",
            options=["compact", "spatially preserving"],
            key="multitile_strategy",
            help="Defines how multiple tiles are combined into a single image (if applicable). Compact mode creates a square collage of tiles, more efficiently using pixels but losing spatial relationships. Spatially preserving mode arranges tiles according to their recorded positions (read from metadata). Note that even in the latter, we try to minimize empty space.",
        )


# ---------------------------------------------------------------------------
# Channel mapping (multichannel tab)
# ---------------------------------------------------------------------------


def _validate_mapping(mapping: dict[int, str]) -> list[str]:
    issues: list[str] = []
    counts: dict[str, int] = {}
    for v in mapping.values():
        counts[v] = counts.get(v, 0) + 1

    for required in (ROLE_ANALOG_1, ROLE_ANALOG_2):
        c = counts.get(required, 0)
        label = ROLE_LABELS[required]
        if c == 0:
            issues.append(f"**{label}** must be assigned to a channel.")
        elif c > 1:
            issues.append(f"**{label}** is assigned to multiple channels.")

    if counts.get(ROLE_SSDNA, 0) > 1:
        issues.append(
            f"**{ROLE_LABELS[ROLE_SSDNA]}** is assigned to multiple channels."
        )
    return issues


def _render_channel_row(channel_idx: int) -> None:
    """One row: channel index header, role selectbox, optional Other-name input."""
    mapping = st.session_state["channel_mapping"]
    others = st.session_state["channel_other_names"]

    current = mapping.get(channel_idx, ROLE_IGNORE)
    if current not in ROLE_OPTIONS:
        current = ROLE_IGNORE

    color = role_color(current)
    st.markdown(
        f"<div style='border-left: 4px solid {color}; padding-left: 8px; "
        f"margin-bottom: 4px;'><strong>Channel {channel_idx}</strong></div>",
        unsafe_allow_html=True,
    )
    mapping[channel_idx] = st.selectbox(
        f"Role for channel {channel_idx}",
        options=ROLE_OPTIONS,
        format_func=lambda r: ROLE_LABELS[r],
        index=ROLE_OPTIONS.index(current),
        key=f"map_ch_{channel_idx}",
        label_visibility="collapsed",
    )
    if mapping[channel_idx] == ROLE_OTHER:
        others[channel_idx] = st.text_input(
            f"Name for channel {channel_idx}",
            value=others.get(channel_idx, ""),
            key=f"other_name_ch_{channel_idx}",
            placeholder="e.g. DAPI",
            label_visibility="collapsed",
        )
    else:
        others.pop(channel_idx, None)


def render_multichannel_tab(extensions: list[str]) -> None:
    st.markdown("**1. Channel layout**")
    st.caption(
        "Set how many channels your files contain and what each one represents. "
        "This mapping is applied to every file you queue from this tab. "
        "Different file types put channels in different orders — when in doubt, "
        "open one file in ImageJ to confirm."
    )

    n = int(
        st.number_input(
            "Number of channels per file",
            min_value=1,
            max_value=8,
            step=1,
            key="n_channels",
            format="%d",
        )
    )

    mapping = st.session_state["channel_mapping"]
    # Drop entries beyond the current channel count; leave existing assignments
    # in place so resizing 3→2→3 keeps the user's earlier choices.
    for k in list(mapping.keys()):
        if k >= n:
            mapping.pop(k)
    for i in range(n):
        mapping.setdefault(i, ROLE_IGNORE)

    per_row = min(n, 4)
    for row_start in range(0, n, per_row):
        cols = st.columns(per_row)
        for j, col in enumerate(cols):
            i = row_start + j
            if i >= n:
                break
            with col:
                _render_channel_row(i)

    issues = _validate_mapping(mapping)
    for issue in issues:
        st.error(issue)

    st.markdown("**2. Add files**")
    path_text = st.text_input(
        "Folder or file path",
        placeholder="/path/to/folder  or  /path/to/file.tif",
    )
    files, error = _resolve_path(path_text, extensions)
    if error:
        st.error(error)

    cols = st.columns([1, 3])
    with cols[0]:
        add_clicked = st.button(
            "Add to queue",
            disabled=bool(issues) or not files,
            type="primary",
        )
    with cols[1]:
        if files:
            st.caption(f"{len(files)} file(s) ready to add.")

    if add_clicked and files and not issues:
        entries = [
            _make_multichannel_entry(
                f, mapping, st.session_state["channel_other_names"]
            )
            for f in files
        ]
        added = _add_entries(entries)
        skipped = len(entries) - added
        msg = f"Added {added} file(s)."
        if skipped:
            msg += f" Skipped {skipped} duplicate(s)."
        st.success(msg)


def render_per_role_tab(extensions: list[str]) -> None:
    st.caption(
        "Provide one folder (or file) per analog. Files are paired across "
        "analogs by sorted filename — make sure both folders contain the same "
        "number of files in matching order."
    )

    cols = st.columns(2)
    role_files: dict[str, list[Path]] = {}
    for col, role_id in zip(cols, (ROLE_ANALOG_1, ROLE_ANALOG_2)):
        with col:
            color = role_color(role_id)
            st.markdown(
                f"<h4 style='color: {color}; margin-bottom: 0.2em;'>"
                f"{ROLE_LABELS[role_id]}</h4>",
                unsafe_allow_html=True,
            )
            txt = st.text_input(
                f"Path for {ROLE_LABELS[role_id]}",
                key=f"pr_path_{role_id}",
                placeholder="folder or file",
                label_visibility="collapsed",
            )
            files, err = _resolve_path(txt, extensions)
            if err:
                st.error(err)
            elif files:
                st.caption(f"{len(files)} file(s)")
            role_files[role_id] = files

    issues: list[str] = []
    if not role_files[ROLE_ANALOG_1]:
        issues.append(f"**{ROLE_LABELS[ROLE_ANALOG_1]}** has no files.")
    if not role_files[ROLE_ANALOG_2]:
        issues.append(f"**{ROLE_LABELS[ROLE_ANALOG_2]}** has no files.")
    if not issues:
        n1, n2 = len(role_files[ROLE_ANALOG_1]), len(role_files[ROLE_ANALOG_2])
        if n1 != n2:
            issues.append(
                f"Mismatched file counts: {ROLE_LABELS[ROLE_ANALOG_1]} has "
                f"{n1}, {ROLE_LABELS[ROLE_ANALOG_2]} has {n2}. Files are paired "
                "by sorted filename and must have identical counts."
            )

    for issue in issues:
        st.error(issue)

    if not issues:
        sorted_a1 = sorted(role_files[ROLE_ANALOG_1], key=lambda p: p.name)
        sorted_a2 = sorted(role_files[ROLE_ANALOG_2], key=lambda p: p.name)
        with st.expander("Preview pairing", expanded=False):
            st.dataframe(
                [
                    {
                        ROLE_LABELS[ROLE_ANALOG_1]: a.name,
                        ROLE_LABELS[ROLE_ANALOG_2]: b.name,
                    }
                    for a, b in zip(sorted_a1, sorted_a2)
                ],
                use_container_width=True,
                hide_index=True,
            )

        if st.button("Add bundles to queue", type="primary"):
            entries = [_make_per_role_entry(a, b) for a, b in zip(sorted_a1, sorted_a2)]
            added = _add_entries(entries)
            skipped = len(entries) - added
            msg = f"Added {added} bundle(s)."
            if skipped:
                msg += f" Skipped {skipped} duplicate(s)."
            st.success(msg)


# ---------------------------------------------------------------------------
# Entry construction & queue management
# ---------------------------------------------------------------------------


def _make_multichannel_entry(
    path: Path,
    mapping: dict[int, str],
    others: dict[int, str],
) -> dict:
    sources: dict[str, tuple[Path, int]] = {}
    extra_labels: dict[int, str] = {}
    for ch_idx, role_id in mapping.items():
        if role_id in (ROLE_ANALOG_1, ROLE_ANALOG_2, ROLE_SSDNA):
            sources[role_id] = (path, ch_idx)
        elif role_id == ROLE_OTHER:
            name = others.get(ch_idx, "").strip()
            if name:
                extra_labels[ch_idx] = name
    # Mapping is part of the id so re-adding the same file with a different
    # mapping creates a distinct entry (and invalidates downstream caches).
    mapping_sig = ",".join(f"{i}:{r}" for i, r in sorted(mapping.items()))
    return {
        "id": f"{path}::{mapping_sig}",
        "display_name": path.name,
        "mode": "multichannel",
        "sources": sources,
        "extra_labels": extra_labels,
    }


def _make_per_role_entry(path_a1: Path, path_a2: Path) -> dict:
    return {
        "id": f"a1={path_a1}::a2={path_a2}",
        "display_name": f"{path_a1.name} | {path_a2.name}",
        "mode": "per_role",
        "sources": {
            ROLE_ANALOG_1: (path_a1, 0),
            ROLE_ANALOG_2: (path_a2, 0),
        },
        "extra_labels": {},
    }


def _add_entries(new_entries: list[dict]) -> int:
    existing = {e["id"] for e in st.session_state["files_uploaded"]}
    added = 0
    for e in new_entries:
        if e["id"] not in existing:
            st.session_state["files_uploaded"].append(e)
            existing.add(e["id"])
            added += 1
    return added


# ---------------------------------------------------------------------------
# Queue display
# ---------------------------------------------------------------------------


def render_queue() -> None:
    queue = st.session_state["files_uploaded"]
    st.subheader(f"Queue ({len(queue)})")
    if not queue:
        st.caption("No files queued yet.")
        return

    rows = []
    for entry in queue:
        sources = entry["sources"]
        if entry["mode"] == "multichannel":
            mapping_summary = ", ".join(
                f"{ROLE_LABELS[role]}: ch{ch}"
                for role, (_, ch) in sorted(sources.items())
            )
        else:
            mapping_summary = ", ".join(
                f"{ROLE_LABELS[role]}: {Path(p).name}"
                for role, (p, _) in sorted(sources.items())
            )
        if entry.get("extra_labels"):
            extras = ", ".join(
                f"ch{ch}: {name}" for ch, name in sorted(entry["extra_labels"].items())
            )
            mapping_summary += f"  •  Other: {extras}"
        rows.append(
            {
                "File": entry["display_name"],
                "Mode": entry["mode"],
                "Mapping": mapping_summary,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
    with st.sidebar:
        performance_button()
        if st.button("Clear queue"):
            st.session_state["files_uploaded"] = []
            st.rerun()
        st.caption(f"{len(st.session_state['files_uploaded'])} entries queued.")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

render_setup()
st.divider()

st.subheader("Add files")
extensions = st.segmented_control(
    "Accepted formats",
    options=ACCEPTED_EXTENSIONS,
    default=ACCEPTED_EXTENSIONS,
    format_func=lambda x: x.upper().lstrip("."),
    selection_mode="multi",
)

tab_mc, tab_pr = st.tabs(["Multichannel files", "One file per analog"])
with tab_mc:
    render_multichannel_tab(extensions)
with tab_pr:
    render_per_role_tab(extensions)

st.divider()
with st.sidebar:
    render_queue()
render_sidebar()
