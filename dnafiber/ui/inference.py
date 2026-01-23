import streamlit as st
from dnafiber.inference import run_model
from dnafiber.postprocess.core import refine_segmentation
import numpy as np
from dnafiber.ui.utils import _get_model
import torch
from dnafiber.postprocess.error_detection import load_model
from dnafiber.postprocess.fiber import Fibers
import time


@st.cache_data(show_spinner="Running predictions...")
def ui_inference(
    _model,
    _image,
    _device,
    use_tta=True,
    use_correction=True,
    pixel_size=0.13,
    prediction_threshold=1 / 3,
    low_end_hardware=False,
    key="default",
) -> np.ndarray | Fibers:
    return inference(
        _model,
        _image,
        _device,
        pixel_size=pixel_size,
        use_tta=use_tta,
        prediction_threshold=prediction_threshold,
        use_correction=use_correction,
        low_end_hardware=low_end_hardware,
    )


@st.cache_resource
def get_model(model_name):
    model = _get_model(
        device="cuda" if torch.cuda.is_available() else "cpu",
        revision=model_name,
    )
    return model


def inference(
    model,
    image,
    device,
    pixel_size,
    use_tta=True,
    only_segmentation=False,
    use_correction=None,
    prediction_threshold=1 / 3,
    low_end_hardware=False,
    verbose=True,
) -> np.ndarray | Fibers:
    """
    A cacheless version of the ui_inference function.
    This function does not use caching and is intended for use in scenarios where caching is not desired.
    """

    if use_correction:
        correction_model = load_model()
    else:
        correction_model = None
    h, w = image.shape[:2]

    formatted_model = []
    if isinstance(model, list):
        for m in model:
            if isinstance(m, str):
                formatted_model.append(get_model(m))
            else:
                formatted_model.append(m)
    else:
        if isinstance(model, str):
            formatted_model = [get_model(model)]
        else:
            formatted_model = [model]
    start = time.time()
    with torch.inference_mode():
        output = run_model(
            formatted_model,
            image=image,
            device=device,
            scale=pixel_size,
            use_tta=use_tta,
            verbose=verbose,
            prediction_threshold=prediction_threshold,
            low_end_hardware=low_end_hardware,
        ).argmax(dim=1).byte().squeeze(0).cpu().numpy()


    if verbose:
        print("Segmentation time:", time.time() - start)
    if only_segmentation:
        return output
    with st.spinner("Post-processing segmentation..."):
        start = time.time()
        output = refine_segmentation(
            image, output, correction_model=correction_model, device=device
        )
        if verbose:
            print("Post-processing time:", time.time() - start)
    return output
