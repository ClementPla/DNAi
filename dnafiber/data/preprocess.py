import cv2
import numpy as np

from time import time

REF_PIXEL_SIZE = 0.26
REF_DOWNSAMPLE = 8
REF_KERNEL = 5


def preprocess(img, pixel_size=0.13, verbose=True):
    start_total = time()

    # Handle channel conversion FIRST (on original dtype)
    start = time()
    if img.shape[2] == 4:
        img = img[:, :, :3]
    elif img.shape[2] == 2:
        h, w = img.shape[:2]
        converted = np.zeros((h, w, 3), dtype=img.dtype)
        converted[:, :, 0] = img[:, :, 0]
        converted[:, :, 1] = img[:, :, 1]
        img = converted
    if verbose:
        print(f"Channel handling done in {time() - start:.2f}s")

    # Compute target downsample/kernel
    target_product = (REF_DOWNSAMPLE * REF_KERNEL * REF_PIXEL_SIZE) / pixel_size
    candidates = []
    for kernel in [3, 5]:
        downsample = max(1, round(target_product / kernel))
        error = abs(kernel * downsample - target_product)
        candidates.append((error, downsample, kernel))
    _, best_downsample, best_kernel = min(candidates, key=lambda x: x[0])

    # Pad image to avoid edge artifacts
    pad = best_kernel * best_downsample + 10  # extra margin
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    # Downsample BEFORE float conversion
    start = time()
    fx = fy = 1 / best_downsample
    small = cv2.resize(img_padded, None, fx=fx, fy=fy, interpolation=cv2.INTER_LANCZOS4)
    small = small.astype(np.float32)
    if verbose:
        print(f"Downsampling done in {time() - start:.2f}s")

    # Normalize small for median computation
    start = time()
    pmin = small.min(axis=(0, 1), keepdims=True)
    pmax = small.max(axis=(0, 1), keepdims=True)
    norm_scale = 1.0 / (pmax - pmin + 1e-6)
    np.subtract(small, pmin, out=small)
    np.multiply(small, norm_scale, out=small)
    if verbose:
        print(f"Stats computation done in {time() - start:.2f}s")

    # Median blur on small image
    start = time()
    small_median = cv2.medianBlur(small, ksize=best_kernel)
    del small
    if verbose:
        print(f"Median blur done in {time() - start:.2f}s")

    # Upsample median to full resolution
    start = time()
    local_median = cv2.resize(
        small_median,
        (img_padded.shape[1], img_padded.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    del small_median
    if verbose:
        print(f"Upsampling done in {time() - start:.2f}s")

    # Convert full image to float and normalize same way
    start = time()
    img_float = img_padded.astype(np.float32)
    img_float = (img_float - pmin) * norm_scale

    # Background subtraction at full resolution
    img_float -= local_median
    del local_median

    # Clip negatives to zero (background becomes black)
    np.maximum(img_float, 0, out=img_float)
    if verbose:
        print(f"Background subtraction done in {time() - start:.2f}s")

    # === JOINT NORMALIZATION (preserves R/G ratio) ===
    start = time()

    # Use global 99.5th percentile across R and G combined for upper bound
    # === PER-CHANNEL SCALING WITH SHARED ZERO FLOOR ===
    # Clip negatives to zero (background)
    np.maximum(img_float, 0, out=img_float)

    # Apply a small noise floor to eliminate glow/haze
    # Pixels below this threshold become black
    noise_floor = 0.01  # adjust if needed (in normalized units)
    for c in range(2):
        ch = img_float[:, :, c]
        ch[ch < noise_floor] = 0

    # Joint scaling - use max of both channels' 99th percentile
    # p99_r = np.percentile(img_float[:, :, 0], 99)
    # p99_g = np.percentile(img_float[:, :, 1], 80)
    # p99 = max(p99_r, p99_g)

    # scale = 255.0 / (p99 + 1e-8)

    h, w = img_padded.shape[:2]
    result = np.zeros((h, w, 3), dtype=np.uint8)

    for c in range(2):
        p99 = np.percentile(img_float[:, :, c], 99)
        scale = 255.0 / (p99 + 1e-8)
        ch = img_float[:, :, c] * scale
        np.clip(ch, 0, 255, out=ch)
        result[:, :, c] = ch.astype(np.uint8)

    del img_float
    if verbose:
        print(f"Joint scaling done in {time() - start:.2f}s")

    # Remove padding
    result = result[pad:-pad, pad:-pad]

    if verbose:
        print(f"Total preprocessing: {time() - start_total:.2f}s")

    return result
