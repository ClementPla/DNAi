from czifile import CziFile
from tifffile import imread
import cv2
import numpy as np


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
    with CziFile(filepath) as czi:
        data = czi.asarray().squeeze()
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


def read_img(filepath) -> np.ndarray:
    filename = str(filepath.name)
    if filename.endswith(".czi"):
        return read_czi(filepath)
    elif filename.endswith(".tif") or filename.endswith(".tiff"):
        return read_tiff(filepath)
    elif filename.endswith(".dv"):
        return read_dv(filepath)
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
