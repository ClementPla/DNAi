import math
import numpy as np
import torch
import torch.nn.functional as F


def compute_padding(H, W, kernel_size):
    """
    Compute how much padding is needed to make (H, W) divisible by kernel_size.
    """
    H_pad = math.ceil(H / kernel_size) * kernel_size - H
    W_pad = math.ceil(W / kernel_size) * kernel_size - W
    return H_pad, W_pad


def extract_patches(image, kernel_size):
    """
    Extract non-overlapping patches of size kernel_size x kernel_size from an image.

    Args:
        image (Tensor): Input tensor of shape (C, H, W)
        kernel_size (int): Size of the square patches

    Returns:
        patches (Tensor): Tensor of shape (num_patches, C, kernel_size, kernel_size)
    """
    C, H, W = image.shape
    H_pad, W_pad = compute_padding(H, W, kernel_size)

    # Pad image symmetrically
    pad_top = H_pad // 2
    pad_bottom = H_pad - pad_top
    pad_left = W_pad // 2
    pad_right = W_pad - pad_left

    # Pad using F.pad; input needs to be (N, C, H, W)
    image = image.unsqueeze(0)  # Add batch dimension
    padded_image = F.pad(image, (0, 2 * pad_right, 0, 2 * pad_bottom))

    # Use Unfold
    unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=kernel_size)
    patches = unfold(padded_image)  # (1, C * kernel_size * kernel_size, num_patches)

    # Reshape to (num_patches, C, kernel_size, kernel_size)
    patches = patches.squeeze(0).transpose(0, 1)  # (num_patches, C * K * K)
    patches = patches.view(-1, C, kernel_size, kernel_size)

    return patches


def get_tiles(img, tile_size, resize=None, interpolation="bilinear"):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    img = torch.from_numpy(img).permute(2, 0, 1).cuda().float()
    # Pad the image to make sure it is divisible by tile_size
    tiles = extract_patches(img, tile_size)
    if resize is not None:
        tiles = F.interpolate(tiles, scale_factor=resize, mode=interpolation).byte()
    tiles = tiles.permute(0, 2, 3, 1).cpu().squeeze(-1).numpy()

    return tiles


def filter_tiles(tiles):
    # If the tile contains more than 50% of black pixels, we remove it

    filtered_tiles = []
    for tile in tiles:
        c1 = np.sum(tile[:, :, 0] == 0)
        h, w = tile.shape[0:2]
        if c1 < (h * w) / 2:
            filtered_tiles.append(tile)
    return filtered_tiles
