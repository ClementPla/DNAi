from typing import Tuple, Optional, List
from dataclasses import dataclass
import numpy as np
import cv2
from dnafiber.postprocess.fiber import Fibers, FiberProps, Bbox

from typing import Tuple, Optional, List
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class Rect:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


def _intersects(a: Rect, b: Rect) -> bool:
    """Check if two rectangles intersect."""
    return a.x < b.right and a.right > b.x and a.y < b.bottom and a.bottom > b.y


def _split_rect_around_placed(free_rect: Rect, placed: Rect) -> List[Rect]:
    """
    Split a free rectangle around a placed rectangle.
    Returns list of new free rectangles (may be empty if fully covered).
    """
    if not _intersects(free_rect, placed):
        return [free_rect]

    new_rects = []

    # Left remainder
    if placed.x > free_rect.x:
        new_rects.append(
            Rect(
                x=free_rect.x,
                y=free_rect.y,
                width=placed.x - free_rect.x,
                height=free_rect.height,
            )
        )

    # Right remainder
    if placed.right < free_rect.right:
        new_rects.append(
            Rect(
                x=placed.right,
                y=free_rect.y,
                width=free_rect.right - placed.right,
                height=free_rect.height,
            )
        )

    # Bottom remainder
    if placed.y > free_rect.y:
        new_rects.append(
            Rect(
                x=free_rect.x,
                y=free_rect.y,
                width=free_rect.width,
                height=placed.y - free_rect.y,
            )
        )

    # Top remainder
    if placed.bottom < free_rect.bottom:
        new_rects.append(
            Rect(
                x=free_rect.x,
                y=placed.bottom,
                width=free_rect.width,
                height=free_rect.bottom - placed.bottom,
            )
        )

    return new_rects


def _is_contained(inner: Rect, outer: Rect) -> bool:
    """Check if inner is fully contained within outer."""
    return (
        inner.x >= outer.x
        and inner.y >= outer.y
        and inner.right <= outer.right
        and inner.bottom <= outer.bottom
    )


def _prune_free_rects(free_rects: List[Rect]) -> List[Rect]:
    """Remove rectangles that are fully contained within others."""
    pruned = []
    for i, r1 in enumerate(free_rects):
        contained = False
        for j, r2 in enumerate(free_rects):
            if i != j and _is_contained(r1, r2):
                contained = True
                break
        if not contained:
            pruned.append(r1)
    return pruned


def _find_best_fit(
    free_rects: List[Rect], w: int, h: int, allow_rotation: bool
) -> Tuple[Optional[int], bool]:
    """
    Find the free rectangle that best fits the given dimensions.
    Uses "Best Short Side Fit" heuristic.
    Returns (index, rotated) or (None, False) if no fit.
    """
    best_idx = None
    best_short_side = float("inf")
    best_rotated = False

    for i, rect in enumerate(free_rects):
        # Try normal orientation
        if w <= rect.width and h <= rect.height:
            short_side = min(rect.width - w, rect.height - h)
            if short_side < best_short_side:
                best_short_side = short_side
                best_idx = i
                best_rotated = False

        # Try rotated (swap w and h)
        if allow_rotation and h <= rect.width and w <= rect.height:
            short_side = min(rect.width - h, rect.height - w)
            if short_side < best_short_side:
                best_short_side = short_side
                best_idx = i
                best_rotated = True

    return best_idx, best_rotated


def mosaic(
    fibers: "Fibers",
    original_image: np.ndarray,
    downsample: int = 2,
    padding: int = 4,
    allow_rotation: bool = True,
    context_margin: float = 0.0,
) -> Tuple["Fibers", np.ndarray]:
    """
    Create a tightly-packed mosaic of all fibers using MaxRects bin packing.

    Args:
        fibers: Fibers collection to mosaic
        original_image: Source image to crop fibers from
        downsample: Factor to downsample crops (1 = no downsampling)
        padding: Pixels between fibers in the mosaic
        allow_rotation: Whether to allow 90° rotation for better packing
        context_margin: Fraction to expand bounding box (0.2 = 20% larger on each side)

    Returns:
        Tuple of (new Fibers with mosaic coordinates, mosaic image)
    """
    if len(fibers) == 0:
        empty_img = np.zeros((1, 1, 3), dtype=original_image.dtype)
        return Fibers([]), empty_img

    h_img, w_img = original_image.shape[:2]

    # Extract and downsample crops
    items = []
    for fiber in fibers.fibers:
        x, y, fw, fh = fiber.bbox

        # Expand bounding box by context_margin
        margin_x = int(fw * context_margin)
        margin_y = int(fh * context_margin)

        # Expanded crop region (clipped to image bounds)
        crop_x0 = max(0, x - margin_x)
        crop_y0 = max(0, y - margin_y)
        crop_x1 = min(w_img, x + fw + margin_x)
        crop_y1 = min(h_img, y + fh + margin_y)

        if crop_x0 >= crop_x1 or crop_y0 >= crop_y1:
            continue

        crop = original_image[crop_y0:crop_y1, crop_x0:crop_x1].copy()

        # Create expanded mask with zeros, place original mask inside
        crop_h, crop_w = crop.shape[:2]
        expanded_mask = np.zeros((crop_h, crop_w), dtype=fiber.data.dtype)

        # Calculate where the original fiber mask goes within the expanded crop
        # Original bbox started at (x, y), crop starts at (crop_x0, crop_y0)
        mask_offset_x = x - crop_x0
        mask_offset_y = y - crop_y0

        # Handle case where original fiber bbox was partially outside image
        orig_x0, orig_y0 = max(0, x), max(0, y)
        orig_x1, orig_y1 = min(w_img, x + fw), min(h_img, y + fh)

        # Slice of fiber.data that's within image bounds
        data_x0, data_y0 = orig_x0 - x, orig_y0 - y
        data_x1, data_y1 = data_x0 + (orig_x1 - orig_x0), data_y0 + (orig_y1 - orig_y0)
        fiber_data_clipped = fiber.data[data_y0:data_y1, data_x0:data_x1]

        # Where to place it in expanded mask
        place_x0 = orig_x0 - crop_x0
        place_y0 = orig_y0 - crop_y0
        place_x1 = place_x0 + fiber_data_clipped.shape[1]
        place_y1 = place_y0 + fiber_data_clipped.shape[0]

        expanded_mask[place_y0:place_y1, place_x0:place_x1] = fiber_data_clipped

        # Downsample
        if downsample > 1:
            new_w = max(1, crop.shape[1] // downsample)
            new_h = max(1, crop.shape[0] // downsample)
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            expanded_mask = cv2.resize(
                expanded_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST
            )

        items.append(
            {
                "fiber": fiber,
                "crop": crop,
                "mask": expanded_mask,
                "width": crop.shape[1] + padding,
                "height": crop.shape[0] + padding,
            }
        )

    if not items:
        empty_img = np.zeros((1, 1, 3), dtype=original_image.dtype)
        return Fibers([]), empty_img

    # Sort by area (largest first) for better packing
    items.sort(key=lambda it: it["width"] * it["height"], reverse=True)

    # Estimate initial canvas size
    total_area = sum(it["width"] * it["height"] for it in items)
    canvas_size = int(np.sqrt(total_area) * 1.2)

    # Try packing, expand canvas if needed
    for attempt in range(10):
        free_rects = [Rect(0, 0, canvas_size, canvas_size)]
        placements = []
        success = True

        for item in items:
            w, h = item["width"], item["height"]

            idx, rotated = _find_best_fit(free_rects, w, h, allow_rotation)

            if idx is None:
                success = False
                break

            rect = free_rects[idx]

            if rotated:
                w, h = h, w

            item["placed_x"] = rect.x
            item["placed_y"] = rect.y
            item["placed_w"] = w
            item["placed_h"] = h
            item["rotated"] = rotated
            placements.append(item)

            placed = Rect(rect.x, rect.y, w, h)

            new_free_rects = []
            for free_rect in free_rects:
                new_free_rects.extend(_split_rect_around_placed(free_rect, placed))

            free_rects = _prune_free_rects(new_free_rects)

        if success:
            break
        canvas_size = int(canvas_size * 1.3)

    if not success:
        raise RuntimeError("Failed to pack all fibers into mosaic")

    # Compute tight canvas bounds
    max_x = max(it["placed_x"] + it["placed_w"] for it in placements) - padding
    max_y = max(it["placed_y"] + it["placed_h"] for it in placements) - padding
    canvas_h, canvas_w = max_y, max_x

    # Build mosaic image
    if original_image.ndim == 3:
        mosaic_img = np.ones(
            (canvas_h, canvas_w, original_image.shape[2]),
            dtype=original_image.dtype,
        ) * np.asarray([30, 30, 46], dtype=original_image.dtype).reshape((1, 1, 3))
    else:
        mosaic_img = np.ones((canvas_h, canvas_w), dtype=original_image.dtype) * 255

    new_fiber_list = []

    for item in placements:
        fiber = item["fiber"]
        crop = item["crop"]
        mask = item["mask"]
        px, py = item["placed_x"], item["placed_y"]
        rotated = item["rotated"]

        if rotated:
            crop = np.rot90(crop, k=-1)
            mask = np.rot90(mask, k=-1)

        ch, cw = crop.shape[:2]

        mosaic_img[py : py + ch, px : px + cw] = crop

        new_bbox = Bbox(x=px, y=py, width=cw, height=ch)

        new_fiber = FiberProps(
            bbox=new_bbox,
            data=mask,
            fiber_id=fiber.fiber_id,
            red_pixels=None,
            green_pixels=None,
            category=fiber.category,
            is_an_error=fiber.is_an_error,
            svg_rep=None,
            trace=None,
            endpoint_correction=fiber.endpoint_correction,
        )
        new_fiber_list.append(new_fiber)

    return Fibers(new_fiber_list), mosaic_img
