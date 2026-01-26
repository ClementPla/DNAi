import streamlit as st
from dnafiber.inference import run_model, probas_to_segmentation
from dnafiber.postprocess.core import refine_segmentation
import numpy as np
from dnafiber.ui.utils import _get_model
import torch
from dnafiber.postprocess.fiber import Fibers
import time


@st.cache_data(show_spinner="Running predictions...")
def ui_inference(
    _model,
    _image,
    _device,
    use_tta=True,
    pixel_size=0.13,
    prediction_threshold=1 / 3,
    low_end_hardware=False,
    verbose=True,
    key="default",
) -> np.ndarray | Fibers:
    start = time.time()
    probas = ui_exec_model(
        _model,
        _image,
        _device,
        pixel_size=pixel_size,
        use_tta=use_tta,
        low_end_hardware=low_end_hardware,
        key=key,
    )

    if verbose:
        print("Segmentation time:", time.time() - start)
    prediction = probas_to_segmentation(
        probas, prediction_threshold=prediction_threshold
    )
    start = time.time()
    if verbose:
        print("Post-processing segmentation...")
    with st.spinner("Post-processing segmentation..."):
        prediction = refine_segmentation(_image, prediction, device=_device)
    if verbose:
        print("Post-processing time:", time.time() - start)
    return prediction


@st.cache_data(show_spinner="Get probability maps...", max_entries=1)
def ui_exec_model(
    _model,
    _image,
    _device,
    use_tta=True,
    pixel_size=0.13,
    low_end_hardware=False,
    key="default",
):
    return run_model(
        _model,
        image=_image,
        device=_device,
        scale=pixel_size,
        use_tta=use_tta,
        verbose=True,
        low_end_hardware=low_end_hardware,
    )


@st.cache_resource
def get_model(model_name):
    model = _get_model(
        revision=model_name,
    )
    return model
