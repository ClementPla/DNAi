from PIL import Image
import io
import base64
from xml.dom import minidom
import cv2
import numpy as np
import math
import streamlit as st
from dnafiber.data.readers import read_img, format_raw_image
from dnafiber.data.preprocess import preprocess
from dnafiber.postprocess.core import extract_fibers
from time import time
from skimage.morphology import skeletonize
from skimage.segmentation import expand_labels


def extract_bboxes(mask):
    mask = np.array(mask)
    mask = mask.astype(np.uint8)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    bboxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        bboxes.append([x, y, x + w, y + h])
    return bboxes


def convert_rgb_to_mask(image, threshold=200):
    output = np.zeros(image.shape[:2], dtype=np.uint8)
    output[image[:, :, 0] > 150] = 1
    output[image[:, :, 1] > 150] = 2
    binary_mask = output > 0
    skeleton = skeletonize(binary_mask) * output
    output = expand_labels(skeleton, 2)
    output = np.clip(output, 0, 2)
    output = output.astype(np.uint8)
    return {"mask": output}


def numpy_to_base64_png(image_array):
    """
    Encodes a NumPy image array to a base64 string (PNG format).

    Args:
        image_array: A NumPy array representing the image.

    Returns:
        A base64 string representing the PNG image.
    """
    # Convert NumPy array to PIL Image
    image = Image.fromarray(image_array)

    # Create an in-memory binary stream
    buffer = io.BytesIO()

    # Save the image to the buffer in PNG format
    image.save(buffer, format="png")

    # Get the byte data from the buffer
    png_data = buffer.getvalue()

    # Encode the byte data to base64
    base64_encoded = base64.b64encode(png_data).decode()

    return f"data:image/png;base64,{base64_encoded}"


def numpy_to_base64_jpeg(image_array, quality=85):
    """
    Encodes a NumPy image array to a base64 string (JPEG format).

    Args:
        image_array: A NumPy array representing the image.
        quality: Quality of the JPEG encoding (1-100).

    Returns:
        A base64 string representing the JPEG image.
    """
    # Convert NumPy array to PIL Image
    image = Image.fromarray(image_array)

    # Create an in-memory binary stream
    buffer = io.BytesIO()

    # Save the image to the buffer in JPEG format
    image.save(buffer, format="JPEG", quality=quality)

    # Get the byte data from the buffer
    jpeg_data = buffer.getvalue()

    # Encode the byte data to base64
    base64_encoded = base64.b64encode(jpeg_data).decode()

    return f"data:image/jpeg;base64,{base64_encoded}"


@st.cache_data
def pad_image_to_croppable(_image, bx, by, uid=None):
    # Pad the image to be divisible by bx and by
    h, w = _image.shape[:2]
    if h % bx != 0:
        pad_h = bx - (h % bx)
    else:
        pad_h = 0
    if w % by != 0:
        pad_w = by - (w % by)
    else:
        pad_w = 0
    _image = cv2.copyMakeBorder(
        _image,
        math.ceil(pad_h / 2),
        math.floor(pad_h / 2),
        math.ceil(pad_w / 2),
        math.floor(pad_w / 2),
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return _image


def load_image(filepath, reverse_channel, pixel_size=0.13, device="cpu", verbose=False):
    """
    A cacheless version of the get_image function.
    This function does not use caching and is intended for use in scenarios where caching is not desired.
    """
    start = time()
    image = read_img(filepath)
    if verbose:
        print(f"Image read in {time() - start:.2f}s")
    start = time()
    image = preprocess(image, pixel_size=pixel_size, verbose=verbose)
    if verbose:
        print(f"Image preprocessed in {time() - start:.2f}s")
    if reverse_channel:
        # RGB->GRB
        image = image[:, :, [1, 0, 2]]

    return image


def load_multifile_image(_filepaths, pixel_size=0.13, device="cpu"):
    result = None

    if _filepaths[0] is not None:
        chan1 = read_img(_filepaths[0])
        chan1 = cv2.cvtColor(chan1, cv2.COLOR_RGB2GRAY)
        h, w = chan1.shape[:2]
    else:
        chan1 = None
    if _filepaths[1] is not None:
        chan2 = read_img(
            _filepaths[1], False, _filepaths[1].file_id, pixel_size=pixel_size
        )
        chan2 = cv2.cvtColor(chan2, cv2.COLOR_RGB2GRAY)
        h, w = chan2.shape[:2]
    else:
        chan2 = None

    result = np.zeros((h, w, 3), dtype=np.uint8)

    if chan1 is not None:
        result[:, :, 0] = chan1
    else:
        result[:, :, 0] = chan2

    if chan2 is not None:
        result[:, :, 1] = chan2
    else:
        result[:, :, 1] = chan1

    result = format_raw_image(result)
    result = preprocess(result, pixel_size=pixel_size, device=device)
    return result


def mask_filepath_to_fibers(filepath, RGB2GRB=False):
    mask = cv2.imread(str(filepath), cv2.IMREAD_COLOR_RGB)
    if RGB2GRB:
        mask = mask[:, :, [1, 0, 2]]
    mask = convert_rgb_to_mask(mask=mask)["mask"]
    fibers = extract_fibers(mask)
    return fibers
