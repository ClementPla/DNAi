from dnafiber.postprocess.fiber import Fibers
from typing import Dict, Tuple, Union
import numpy as np
import cv2
from tqdm.auto import tqdm
from skimage.segmentation import expand_labels


def get_crops(
    image: np.ndarray,
    fibers: Fibers,
    bbox_inflate: float = 1.0,
    resize: Union[int, Tuple[int, int], None] = None,
    return_masks: bool = False,
) -> Dict[int, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    crops = {}

    if isinstance(resize, int):
        resize = (resize, resize)

    for fiber in tqdm(fibers, desc="Cropping fibers"):
        fiber_id = fiber.fiber_id
        ox, oy, ow, oh = fiber.bbox
        cx, cy = ox + ow / 2, oy + oh / 2

        # 1. Determine the square side length based on the largest dimension
        side_length = max(ow, oh) * bbox_inflate
        half_side = side_length / 2

        # 2. Calculate ideal boundaries
        ix1, iy1 = cx - half_side, cy - half_side
        ix2, iy2 = cx + half_side, cy + half_side

        # 3. Clip boundaries to image dimensions for the actual extraction
        x1, y1 = int(max(0, ix1)), int(max(0, iy1))
        x2, y2 = int(min(image.shape[1], ix2)), int(min(image.shape[0], iy2))

        # 4. Extract raw crop (might not be square if clipped by image edges)
        raw_crop = image[y1:y2, x1:x2]

        # 5. Pad to square if we hit an image edge
        # We calculate padding based on how much was clipped from the ideal square
        pad_left = int(max(0, -ix1))
        pad_top = int(max(0, -iy1))
        pad_right = int(max(0, ix2 - image.shape[1]))
        pad_bottom = int(max(0, iy2 - image.shape[0]))

        if any([pad_left, pad_top, pad_right, pad_bottom]):
            # Apply padding to the image crop
            crop = np.pad(
                raw_crop,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
                if raw_crop.ndim == 3
                else ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
            )
        else:
            crop = raw_crop

        if return_masks:
            # Create a blank mask matching the *padded* crop size
            mask_full = np.zeros(crop.shape[:2], dtype=np.uint8)

            # Calculate relative position of the fiber mask
            # Local offset = (original top-left) - (padded crop top-left)
            mx1 = int(ox - (x1 - pad_left))
            my1 = int(oy - (y1 - pad_top))

            fiber_mask = fiber.data
            mh, mw = fiber_mask.shape

            # Place original mask into the square canvas
            mask_full[my1 : my1 + mh, mx1 : mx1 + mw] = fiber_mask
            mask_full = expand_labels(mask_full, distance=3)

            if resize is not None:
                crop = cv2.resize(crop, resize, interpolation=cv2.INTER_AREA)
                mask_full = cv2.resize(
                    mask_full, resize, interpolation=cv2.INTER_NEAREST
                )

            crops[fiber_id] = (crop, mask_full)
        else:
            if resize is not None:
                crop = cv2.resize(crop, resize, interpolation=cv2.INTER_AREA)
            crops[fiber_id] = crop

    return crops
