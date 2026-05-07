import numpy as np
import nd2
from pathlib import Path


def collapse_large_gaps(coords, tile_size, max_gap_factor=1.5):
    """
    Collapse only excessively large gaps.
    """

    coords = np.array(coords, dtype=float)

    order = np.argsort(coords)

    sorted_coords = coords[order]

    adjusted = sorted_coords.copy()

    threshold = tile_size * max_gap_factor

    cumulative_shift = 0

    for i in range(1, len(sorted_coords)):
        gap = sorted_coords[i] - sorted_coords[i - 1]

        if gap > threshold:
            excess = gap - threshold

            cumulative_shift += excess

        adjusted[i] -= cumulative_shift

    result = np.empty_like(adjusted)

    result[order] = adjusted

    return result


def make_feather_mask(h, w):
    yy = np.linspace(-1, 1, h)
    xx = np.linspace(-1, 1, w)

    X, Y = np.meshgrid(xx, yy)

    dist = np.sqrt(X**2 + Y**2)

    mask = 1 - np.clip(dist, 0, 1)

    mask += 1e-3

    return mask.astype(np.float32)


def spatial_arrangement_to_image(
    tiles: np.ndarray,
    events: list[dict],
    pixel_size_um: float,
    max_gap_factor=1.5,
    global_scale=0.5,
    flip_x=True,
    flip_y=True,
    feather=True,
) -> np.ndarray:
    # ---------------------------------------------------------
    # tiles
    # ---------------------------------------------------------

    n_tiles, tile_c, tile_h, tile_w = tiles.shape

    # ---------------------------------------------------------
    # stage coordinates
    # ---------------------------------------------------------
    x_um = np.array([e["X Coord [µm]"] for e in events])
    y_um = np.array([e["Y Coord [µm]"] for e in events])

    x_um -= x_um.min()
    y_um -= y_um.min()

    # µm -> pixel coordinates
    x_pix = x_um / pixel_size_um
    y_pix = y_um / pixel_size_um

    # ---------------------------------------------------------
    # collapse gigantic gaps
    # ---------------------------------------------------------
    x_pix = collapse_large_gaps(
        x_pix,
        tile_w,
        max_gap_factor=max_gap_factor,
    )

    y_pix = collapse_large_gaps(
        y_pix,
        tile_h,
        max_gap_factor=max_gap_factor,
    )

    # ---------------------------------------------------------
    # mild global compression
    # ---------------------------------------------------------
    x_pix *= global_scale
    y_pix *= global_scale

    x_pix = np.round(x_pix).astype(int)
    y_pix = np.round(y_pix).astype(int)

    # Nikon orientation correction
    y_pix = y_pix.max() - y_pix

    # ---------------------------------------------------------
    # canvas
    # ---------------------------------------------------------
    canvas_h = y_pix.max() + tile_h
    canvas_w = x_pix.max() + tile_w

    mosaic = np.zeros((tile_c, canvas_h, canvas_w), dtype=np.float32)
    weights = np.zeros((tile_c, canvas_h, canvas_w), dtype=np.float32)

    # ---------------------------------------------------------
    # blending weights
    # ---------------------------------------------------------
    if feather:
        weight_mask = make_feather_mask(tile_h, tile_w)
    else:
        weight_mask = np.ones((tile_h, tile_w), dtype=np.float32)

    # ---------------------------------------------------------
    # paste tiles
    # ---------------------------------------------------------
    for tile, x, y in zip(tiles, x_pix, y_pix):
        if flip_x:
            tile = np.fliplr(tile)

        if flip_y:
            tile = np.flipud(tile)

        ys = slice(y, y + tile_h)
        xs = slice(x, x + tile_w)

        mosaic[:, ys, xs] += tile * weight_mask
        weights[:, ys, xs] += weight_mask

    # normalize overlaps
    weights[weights == 0] = 1

    mosaic /= weights

    return mosaic


def compact_arrangement_to_image(tiles: np.ndarray) -> np.ndarray:
    n_tiles, tile_c, tile_h, tile_w = tiles.shape

    n_cols = int(np.ceil(np.sqrt(n_tiles)))
    n_rows = int(np.ceil(n_tiles / n_cols))

    canvas_h = n_rows * tile_h
    canvas_w = n_cols * tile_w

    mosaic = np.zeros((tile_c, canvas_h, canvas_w), dtype=tiles.dtype)

    for idx, tile in enumerate(tiles):
        row = idx // n_cols
        col = idx % n_cols

        y = row * tile_h
        x = col * tile_w

        ys = slice(y, y + tile_h)
        xs = slice(x, x + tile_w)

        mosaic[:, ys, xs] = tile

    return mosaic


def stitch_nd2_combined(
    nd2_path,
    global_scale=0.5,
    max_gap_factor=1.5,
    flip_x=True,
    flip_y=True,
    multitile_strategy: str = "compact",
    feather=True,
):
    """
    Combined geometry-preserving ND2 montage:
      - preserve local geometry
      - collapse huge empty gaps
      - apply mild global compression
      - smooth overlap blending
    """

    with nd2.ND2File(Path(nd2_path)) as f:
        tiles = f.asarray().squeeze()
        events = f.events()
        pixel_size_um = f.metadata.channels[0].volume.axesCalibration[0]
        if tiles.ndim == 3:
            return tiles

        if multitile_strategy == "compact":
            return compact_arrangement_to_image(tiles)
        else:
            return spatial_arrangement_to_image(
                tiles,
                events,
                pixel_size_um,
                max_gap_factor=max_gap_factor,
                global_scale=global_scale,
                flip_x=flip_x,
                flip_y=flip_y,
                feather=feather,
            )
