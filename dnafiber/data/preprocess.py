import cv2
import numpy as np
import torch
import kornia
from time import time

REF_PIXEL_SIZE = 0.26
REF_DOWNSAMPLE = 8
REF_KERNEL = 5


def preprocess(img, pixel_size=0.13, device="cuda", verbose=False):
    start_total = time()

    h, w = img.shape[:2]
    if max(h, w) < 4096:
        return preprocess_cpu(img, pixel_size=pixel_size, verbose=verbose)
    if device == "cpu":
        return preprocess_cpu(img, pixel_size=pixel_size, verbose=verbose)

    # Handle channels on CPU before transfer (avoid transferring alpha channel)
    start = time()
    if img.shape[2] == 4:
        img = img[:, :, :3]
    elif img.shape[2] == 2:
        converted = np.zeros((h, w, 3), dtype=img.dtype)
        converted[:, :, 0] = img[:, :, 0]
        converted[:, :, 1] = img[:, :, 1]
        img = converted
    if verbose:
        print(f"Channel handling done in {time() - start:.2f}s")

    # Compute downsample/kernel params
    target_product = (REF_DOWNSAMPLE * REF_KERNEL * REF_PIXEL_SIZE) / pixel_size
    candidates = []
    for kernel in [3, 5]:
        downsample = max(1, round(target_product / kernel))
        error = abs(kernel * downsample - target_product)
        candidates.append((error, downsample, kernel))
    _, best_downsample, best_kernel = min(candidates, key=lambda x: x[0])

    # === Downsample on CPU before float conversion (much faster) ===
    start = time()
    fx = fy = 1 / best_downsample
    small_np = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
    if verbose:
        print(f"Downsampling done in {time() - start:.2f}s")

    # Transfer small to GPU and compute all stats there
    start = time()
    small = torch.from_numpy(small_np).to(device=device, dtype=torch.float32)
    small = small.permute(2, 0, 1).unsqueeze(0)  # HWC -> 1CHW

    # Initial normalization stats from small
    pmin = small.amin(dim=(2, 3), keepdim=True)
    pmax = small.amax(dim=(2, 3), keepdim=True)
    norm_scale = 1.0 / (pmax - pmin + 1e-8)

    # Normalize small in-place
    small = (small - pmin) * norm_scale
    if verbose:
        print(f"Stats computation done in {time() - start:.2f}s")

    # Median blur on GPU
    start = time()
    small_median = kornia.filters.median_blur(
        small, kernel_size=(best_kernel, best_kernel)
    )
    if verbose:
        print(f"Median blur done in {time() - start:.2f}s")

    # Compute percentiles on small (after background subtraction)
    start = time()
    small_bg_sub = small - small_median
    small_flat = small_bg_sub.reshape(small_bg_sub.shape[1], -1)  # C x N
    p99 = torch.quantile(small_flat, 0.99, dim=1).view(1, -1, 1, 1)
    p35 = torch.quantile(small_flat, 0.35, dim=1).view(1, -1, 1, 1)

    # Compute final normalization params from clipped small
    small_clipped = torch.clamp(torch.clamp(small_bg_sub, max=p99), min=p35)
    final_min = small_clipped.amin(dim=(2, 3), keepdim=True)
    final_max = small_clipped.amax(dim=(2, 3), keepdim=True)
    final_scale = 255.0 / (final_max - final_min + 1e-8)

    del small, small_bg_sub, small_clipped, small_flat
    if verbose:
        print(f"Percentile computation done in {time() - start:.2f}s")

    # Upsample median
    start = time()
    local_median = torch.nn.functional.interpolate(
        small_median,
        size=(h, w),
        mode="nearest",
    )
    del small_median
    if verbose:
        print(f"Upsampling done in {time() - start:.2f}s")

    # === Fused processing of full image ===
    start = time()

    # Precompute combined transformation (same algebra as CPU)
    # Pipeline: ((img - pmin) * norm_scale - median - final_min) * final_scale
    # = img * (norm_scale * final_scale) + ((-pmin * norm_scale - final_min) * final_scale) - median * final_scale
    combined_scale = norm_scale * final_scale  # 1CHW shape
    combined_offset = (-pmin * norm_scale - final_min) * final_scale
    median_scale = final_scale

    # Compute output clipping bounds
    out_p1 = (p35 - final_min) * final_scale
    out_p99 = (p99 - final_min) * final_scale
    clip_min = torch.clamp(out_p1, min=0)
    clip_max = torch.clamp(out_p99, max=255)

    # Transfer full image to GPU and process
    t_img = torch.from_numpy(img).to(device=device, dtype=torch.float32)
    t_img = t_img.permute(2, 0, 1).unsqueeze(0)  # HWC -> 1CHW

    # Fused transform
    t_img = t_img * combined_scale + combined_offset - local_median * median_scale

    # Clip and convert
    t_img = torch.clamp(t_img, min=clip_min, max=clip_max)
    t_img = torch.clamp(t_img, min=0.0, max=255.0)  # Safety clamp
    t_img = t_img.to(torch.uint8)

    # Zero blue channel
    t_img[:, 2, :, :] = 0

    # Convert back to HWC numpy
    result = t_img.squeeze(0).permute(1, 2, 0).cpu().numpy()

    del t_img, local_median
    if verbose:
        print(f"Fused processing done in {time() - start:.2f}s")
        print(f"Total preprocessing: {time() - start_total:.2f}s")

    return result


def preprocess_cpu(img, pixel_size=0.13, verbose=True):
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

    # Downsample BEFORE float conversion (much faster on uint8/uint16)
    start = time()
    fx = fy = 1 / best_downsample
    small = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
    small = small.astype(np.float32)
    if verbose:
        print(f"Downsampling done in {time() - start:.2f}s")

    # Compute all normalization parameters from small
    start = time()
    pmin = small.min(axis=(0, 1), keepdims=True)
    pmax = small.max(axis=(0, 1), keepdims=True)
    norm_scale = 1.0 / (pmax - pmin + 1e-8)

    # Normalize small in-place
    np.subtract(small, pmin, out=small)
    np.multiply(small, norm_scale, out=small)
    if verbose:
        print(f"Stats computation done in {time() - start:.2f}s")

    # Median blur on small image
    start = time()
    small_median = cv2.medianBlur(small, ksize=best_kernel)
    if verbose:
        print(f"Median blur done in {time() - start:.2f}s")

    # Compute percentiles on small
    start = time()
    np.subtract(small, small_median, out=small)
    p99 = np.percentile(small, 99, axis=(0, 1))
    p1 = np.percentile(small, 35, axis=(0, 1))

    # Compute final normalization params from small
    small_clipped = np.clip(small, p1, p99)
    final_min = small_clipped.min(axis=(0, 1))
    final_max = small_clipped.max(axis=(0, 1))
    final_scale = 255.0 / (final_max - final_min + 1e-8)
    del small, small_clipped
    if verbose:
        print(f"Percentile computation done in {time() - start:.2f}s")

    # Upsample median
    start = time()
    local_median = cv2.resize(
        small_median,
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    del small_median
    if verbose:
        print(f"Upsampling done in {time() - start:.2f}s")

    # === SINGLE FUSED PASS with delayed float conversion ===
    start = time()
    h, w = img.shape[:2]
    result = np.zeros((h, w, 3), dtype=np.uint8)

    # Precompute combined transformation parameters per channel
    # Original pipeline: ((img - pmin) * norm_scale - median_norm - final_min) * final_scale
    # Where median_norm is the upsampled median (already in normalized space)
    # Simplify: (img * norm_scale - pmin * norm_scale - median_norm - final_min) * final_scale
    #         = img * (norm_scale * final_scale) + ((-pmin * norm_scale - final_min) * final_scale) - median_norm * final_scale

    combined_scale = (norm_scale.flatten() * final_scale).astype(np.float32)
    combined_offset = (
        (-pmin.flatten() * norm_scale.flatten() - final_min) * final_scale
    ).astype(np.float32)
    median_scale = final_scale.astype(np.float32)

    for c in range(2):  # Skip channel 2
        # Convert chunk to float and apply fused transform
        img_ch = img[:, :, c].astype(np.float32)
        median_ch = local_median[:, :, c]

        # Fused: img * combined_scale + combined_offset - median * median_scale
        temp = (
            img_ch * combined_scale[c]
            + combined_offset[c]
            - median_ch * median_scale[c]
        )

        # Clip to valid range accounting for p1/p99 bounds
        # The p1/p99 clipping in normalized space translates to output bounds
        out_p1 = (p1[c] - final_min[c]) * final_scale[c]
        out_p99 = (p99[c] - final_min[c]) * final_scale[c]
        np.clip(temp, max(0, out_p1), min(255, out_p99), out=temp)

        result[:, :, c] = temp.astype(np.uint8)
        del img_ch, temp

    del local_median
    if verbose:
        print(f"Fused processing done in {time() - start:.2f}s")
        print(f"Total preprocessing: {time() - start_total:.2f}s")

    return result
