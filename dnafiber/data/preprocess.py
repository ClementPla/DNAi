import cv2
import numpy as np

from time import time

REF_PIXEL_SIZE = 0.26
REF_DOWNSAMPLE = 8
REF_KERNEL = 5

import cv2
import numpy as np
from time import time

def preprocess(img, pixel_size=0.13, verbose=True, clarity=1.0):
    start_total = time()
    
    # 1. CHANNEL TRIMMING (In-place/View-based)
    # We take a view instead of a copy if possible
    if img.shape[2] > 3:
        img = img[:, :, :3]
    
    # 2. CALC PARAMETERS (Same logic, zero memory cost)
    REF_PIXEL_SIZE, REF_DOWNSAMPLE, REF_KERNEL = 0.26, 8, 5
    target_product = (REF_DOWNSAMPLE * REF_KERNEL * REF_PIXEL_SIZE) / pixel_size
    candidates = [(abs(k * max(1, round(target_product / k)) - target_product), max(1, round(target_product / k)), k) 
                  for k in [3, 5]]
    _, best_downsample, best_kernel = min(candidates, key=lambda x: x[0])

    # 3. STATS GATHERING ON DOWNSAMPLED PROXY
    # We do NOT pad the full image yet. We downsample first to save RAM.
    start = time()
    small = cv2.resize(img, None, fx=1/best_downsample, fy=1/best_downsample, interpolation=cv2.INTER_AREA)
    
    # Compute stats on the small proxy to avoid full-scale float conversion
    pmin = small.min(axis=(0, 1), keepdims=True).astype(np.float32)
    pmax = small.max(axis=(0, 1), keepdims=True).astype(np.float32)
    norm_scale = 1.0 / (pmax - pmin + 1e-6)
    
    # Create the background model while still small
    small_f = (small.astype(np.float32) - pmin) * norm_scale
    small_median = cv2.medianBlur(small_f, ksize=best_kernel)
    del small, small_f # Free RAM immediately
    if verbose: print(f"Background model built in {time() - start:.2f}s")

    # 4. TILE-BASED PROCESSING
    # This is the "Big Image" secret: process the full-res subtraction in chunks
    h, w = img.shape[:2]
    result = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Upsample the median model once (this is the only large float array we keep briefly)
    # If the image is truly massive, you would upsample this tile-by-tile as well.
    local_median_full = cv2.resize(small_median, (w, h), interpolation=cv2.INTER_LINEAR)
    del small_median

    # 5. FINAL PASS: SCALE AND SUBTRACT
    for c in range(min(3, img.shape[2])):
        # Work on one channel at a time
        ch_float = img[:, :, c].astype(np.float32)
        
        # Background subtraction
        ch_float = (ch_float - pmin[0,0,c]) * norm_scale[0,0,c]
        ch_float -= (local_median_full[:, :, c]) * clarity
        

        stride = max(1, ch_float.size // 1_000_000) 
        sample = ch_float.reshape(-1)[::stride] 
        
        p99 = np.percentile(sample, 99.0 * clarity)
        
        # Apply Scaling
        scale = 255.0 / (max(p99, 1e-6))
        ch_float *= scale
        
        # Clip and Cast
        np.clip(ch_float, 0, 255, out=ch_float)
        result[:, :, c] = ch_float.astype(np.uint8)
    return result
