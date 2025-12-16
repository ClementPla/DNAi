import streamlit as st
import cv2
from dnafiber.ui.utils import create_display_files
from dnafiber.data.readers import read_czi
from dnafiber.data.preprocess import preprocess
import numpy as np
from streamlit_image_zoom import image_zoom

input_image = "/home/clement/Documents/data/DNAFiber/train/images/5/tile_2.jpeg"
ref_image = cv2.imread(input_image, cv2.IMREAD_COLOR_RGB) / 255.0
files = st.session_state.files_uploaded
displayed_names = create_display_files(files)
with st.sidebar:
    selected_file = st.selectbox(
        "Pick an image",
        displayed_names,
        index=0,
        help="Select an image to view and analyze.",
    )

# Find index of the selected file
index = displayed_names.index(selected_file)
file = files[index]

if file.name.endswith(".czi"):
    img = read_czi(file)[::-1]
    c, h, w = img.shape
    zeros = np.zeros((3 - c, h, w), dtype=img.dtype)
    img = np.concatenate([img, zeros], axis=0).transpose(1, 2, 0)
else:
    img = cv2.imread(str(file), cv2.IMREAD_COLOR_RGB)

img = img.astype(np.float32)
img -= img.min()
img /= img.max()


col1, col2 = st.columns([1, 1])
with col1:
    st.header("Original Image")
    image_zoom(
        (img * 255).astype(np.uint8),
        mode="both",
        size=1000,
        zoom_factor=25,
    )
with col2:
    st.header("Preprocessed Image")
    img = (preprocess(img, 0.26) * 255).astype(np.uint8)
    image_zoom(
        img,
        mode="both",
        size=1000,
        zoom_factor=25,
    )
