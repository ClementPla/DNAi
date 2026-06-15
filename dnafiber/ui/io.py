import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st

from dnafiber.data.utils import preprocess
from dnafiber.data.readers import (
    read_img,
)

# ---------------------------------------------------------------------------
# Output channel layout
# ---------------------------------------------------------------------------

# Maps a role id to the output RGB channel it occupies.
# Roles not in this map are ignored at assembly time (e.g. custom-schema roles
# the segmentation model doesn't know about).
ROLE_TO_OUTPUT_CHANNEL: dict[str, int] = {
    "analog_1": 0,
    "analog_2": 1,
    "ssdna": 2,
}


def load_image_from_entry(
    entry: dict[str, Any],
    *,
    pixel_size: float = 0.13,
    clarity: float = 1.0,
    multitile_strategy: str = "compact",
) -> np.ndarray:
    """Assemble a model-ready RGB image from a queue entry.

    Returns HxWx3 uint8. Roles in `ROLE_TO_OUTPUT_CHANNEL` are placed in their
    designated output channel; unknown roles are ignored. If only one analog
    is present, it is duplicated into the missing analog's channel — matching
    the legacy fallback in `load_multifile_image`.
    """
    sources = entry["sources"]
    mode = entry["mode"]
    if not sources:
        raise ValueError(f"Entry {entry.get('id')!r} has no sources.")

    # Read each unique path once even if multiple roles point to it.
    read_cache: dict[Path, np.ndarray] = {}

    def _read(path: Path) -> np.ndarray:
        if path not in read_cache:
            read_cache[path] = read_img(path, multitile_strategy=multitile_strategy)
        return read_cache[path]

    role_planes: dict[str, np.ndarray] = {}
    for role_id, (path, channel_index) in sources.items():
        arr = _read(Path(path))
        role_planes[role_id] = _plane_for_role(arr, channel_index, mode)

    # Canvas size from any available plane (they should all match).
    h, w = next(iter(role_planes.values())).shape[:2]
    result = np.zeros((h, w, 3), dtype=np.float32)

    for role_id, plane in role_planes.items():
        out_idx = ROLE_TO_OUTPUT_CHANNEL.get(role_id)
        if out_idx is None:
            continue
        # Defensive shape check — surfaces dimension mismatches early instead
        # of silently broadcasting wrong.
        if plane.shape[:2] != (h, w):
            raise ValueError(
                f"Role {role_id!r} plane has shape {plane.shape}, "
                f"expected ({h}, {w}). Source files have inconsistent sizes."
            )
        result[:, :, out_idx] = plane

    # Legacy backfill: missing analog inherits the other.
    has_a1 = "analog_1" in role_planes
    has_a2 = "analog_2" in role_planes
    if has_a1 and not has_a2:
        result[:, :, 1] = result[:, :, 0]
    elif has_a2 and not has_a1:
        result[:, :, 0] = result[:, :, 1]
    result = preprocess(result, pixel_size=pixel_size, clarity=clarity)
    return result


def _plane_for_role(arr: np.ndarray, channel_index: int, mode: str) -> np.ndarray:
    """Extract a 2D plane from a read image for a given role.

    Behavior matches the originals:
      - per_role mode with an RGB read: convert RGB→gray (matches
        `load_multifile_image`'s `cv2.cvtColor(..., COLOR_RGB2GRAY)`).
      - per_role mode with native single-channel: take that plane.
      - multichannel mode: take the user-selected channel index.
    """
    if mode == "per_role":
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            if arr.shape[2] >= 3:
                # Treat the file as a single-channel image stored as RGB and
                # collapse to grayscale. This matches the legacy assembly.
                return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
            if arr.shape[2] == 1:
                return arr[:, :, 0]
        raise ValueError(f"Cannot handle per-role array of shape {arr.shape}.")

    # multichannel
    if arr.ndim == 2:
        if channel_index != 0:
            raise ValueError(
                f"Channel index {channel_index} requested but image is 2D."
            )
        return arr
    if arr.ndim == 3:
        if channel_index >= arr.shape[2]:
            raise ValueError(
                f"Channel index {channel_index} out of range "
                f"(image has {arr.shape[2]} channels)."
            )
        return arr[:, :, channel_index]
    raise ValueError(f"Cannot handle array of shape {arr.shape}.")


# ---------------------------------------------------------------------------
# Cached wrapper
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=8)
def get_image_from_entry(
    _entry: dict[str, Any],
    *,
    pixel_size: float,
    clarity: float,
    id: str,
    multitile_strategy: str = "compact",
) -> np.ndarray:
    """Cached wrapper. The leading underscore on `_entry` tells Streamlit not
    to hash the dict — caching is keyed entirely by `id` (and the scalar
    pixel_size / clarity, which Streamlit hashes natively).
    """
    return load_image_from_entry(
        _entry,
        pixel_size=pixel_size,
        clarity=clarity,
        multitile_strategy=multitile_strategy,
    )


# ---------------------------------------------------------------------------
# Entry id helper
# ---------------------------------------------------------------------------


def build_entry_id(
    entry: dict[str, Any],
    *,
    pixel_size: float,
    clarity: float,
    multitile_strategy: str = "compact",
) -> str:
    """Stable id capturing the entry, its mapping, and the load-time params.

    Replaces `build_file_id`. Includes the full mapping so changing how a
    channel is interpreted invalidates the cache, even when the underlying
    file path is the same.
    """
    mapping_sig = "|".join(
        f"{role}={path}#{ch}" for role, (path, ch) in sorted(entry["sources"].items())
    )
    payload = f"{entry['id']}::{entry['mode']}::{mapping_sig}::ps={pixel_size:.6f}::cl={clarity:.6f}::mts={multitile_strategy}::sources={str(entry['sources'])}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
