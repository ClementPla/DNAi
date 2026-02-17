import numpy as np
from numba import njit
from skimage.morphology import skeletonize
from dnafiber.postprocess.skan import (
    find_endpoints,
    compute_points_angle,
)
import cv2


@njit
def _extend_endpoint_along_ray(mask, start_y, start_x, angle, max_length=50):
    H, W = mask.shape
    last_point = (start_y, start_x)
    dir_y = np.sin(angle)
    dir_x = np.cos(angle)
    for i in range(1, max_length + 1):
        y = int(round(start_y + dir_y * i))
        x = int(round(start_x + dir_x * i))
        if y < 0 or y >= H or x < 0 or x >= W:
            break
        if mask[y, x] == 0:
            break
        last_point = (y, x)
    return last_point


def skeletonize_preserve_endpoints(mask):
    """Skeletonize, then extend endpoints along their direction to the mask boundary."""
    mask_bool = np.ascontiguousarray(mask.astype(np.bool_))
    skel = skeletonize(mask_bool).astype(np.uint8)

    endpoints = find_endpoints(skel)
    if len(endpoints) == 0:
        return skel

    endpoints_arr = np.asarray(endpoints, dtype=np.int64)
    angles = compute_points_angle(skel, endpoints_arr, steps=5, oriented=True)

    for (y, x), angle in zip(endpoints, angles):
        last_point = _extend_endpoint_along_ray(
            mask_bool, y, x, float(angle), max_length=50
        )
        # Draw a line from (y, x) to last_point
        cv2.line(skel, (x, y), (last_point[1], last_point[0]), color=1, thickness=1)

    return skel
