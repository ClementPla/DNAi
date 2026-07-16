from pylibCZIrw import czi as pyczi
from tifffile import imread
import cv2
import numpy as np

from dnafiber.data.nd2_format import stitch_nd2_combined


def format_raw_image(image):
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    if image.shape[0] == 2:
        # Add a zero channel
        zeros = np.zeros((1, image.shape[1], image.shape[2]), dtype=image.dtype)
        image = np.concatenate([image, zeros], axis=0)
    # From CHW -> HWC
    if image.ndim == 3 and image.shape[0] in [2, 3, 4]:
        image = np.moveaxis(image, 0, -1)
    # Map to uint8 (full range)
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        image = 255.0 * (image / np.max(image))
    return image


def read_czi(filepath):
    with pyczi.open_czi(str(filepath)) as czidoc:
        c_start, c_end = czidoc.total_bounding_box["C"]
        # read() composes the full mosaic for each channel; drops the
        # trailing singleton channel axis to get (H, W) planes.
        planes = [
            czidoc.read(plane={"C": c})[..., 0] for c in range(c_start, c_end)
        ]
    data = np.stack(planes, axis=0).squeeze()  # (C, H, W)
    data = format_raw_image(data)
    return data


def read_tiff(filepath):
    data = imread(filepath).squeeze()
    data = format_raw_image(data)
    return data


def read_dv(filepath):
    from mrc import DVFile

    with DVFile(filepath) as dv:
        data = dv.asarray().squeeze()[:2]
    data = format_raw_image(data)
    return data


def read_nd2(
    filepath,
    max_gap_factor=1.0,
    global_scale=1.0,
    flip_x=True,
    flip_y=True,
    multitile_strategy: str = "compact",
):
    data = stitch_nd2_combined(
        filepath,
        max_gap_factor=max_gap_factor,
        global_scale=global_scale,
        flip_x=flip_x,
        flip_y=flip_y,
        multitile_strategy=multitile_strategy,
    )
    data = format_raw_image(data)
    return data


def read_img(filepath, multitile_strategy: str = "compact") -> np.ndarray:
    filename = str(filepath.name)
    if filename.endswith(".czi"):
        return read_czi(filepath)
    elif filename.endswith(".tif") or filename.endswith(".tiff"):
        return read_tiff(filepath)
    elif filename.endswith(".dv"):
        return read_dv(filepath)
    elif filename.endswith(".nd2"):
        return read_nd2(filepath, multitile_strategy=multitile_strategy)
    elif (
        filename.endswith(".png")
        or filename.endswith(".jpg")
        or filename.endswith(".jpeg")
        or filename.endswith(".bmp")
    ):
        image: np.ndarray = cv2.imread(str(filepath), cv2.IMREAD_COLOR_RGB)
        return image
    else:
        raise NotImplementedError(f"File type {filename} is not supported yet")
