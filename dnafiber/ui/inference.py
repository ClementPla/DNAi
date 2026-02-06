import streamlit as st
from dnafiber.error_detection.inference import detect_error
from dnafiber.inference import run_model, probas_to_segmentation
from dnafiber.model.utils import get_error_detection_model
from dnafiber.postprocess.core import refine_segmentation
import numpy as np
from dnafiber.ui.utils import _get_model

from dnafiber.postprocess.fiber import Fibers
import time
import monai.inferers.utils as monai_utils

_original_monai_tqdm = monai_utils.tqdm


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
    """Public API - handles UI, delegates to cached computation."""

    # Check if already cached
    cache_key = f"_inference_result_{key}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    # Not cached - show progress UI
    start = time.time()
    progress_bar = st.progress(0, text="Starting inference...")

    class StreamlitTqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable
            self.total = total or (
                len(iterable) if hasattr(iterable, "__len__") else None
            )
            self.desc = desc or "Processing"
            self.n = 0

        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update(1)
            self.close()

        def update(self, n=1):
            self.n += n
            if self.total:
                progress_bar.progress(
                    self.n / self.total, text=f"{self.desc}: {self.n}/{self.total}"
                )

        def close(self):
            progress_bar.empty()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monai_utils.tqdm = StreamlitTqdm
    try:
        probas = run_model(
            _model,
            image=_image,
            device=_device,
            scale=pixel_size,
            use_tta=use_tta,
            verbose=verbose,
            low_end_hardware=low_end_hardware,
        )
    finally:
        monai_utils.tqdm = _original_monai_tqdm

    if verbose:
        print("Segmentation time:", time.time() - start)
    st.toast(f"Inference completed in {time.time() - start:.2f} seconds.")

    prediction = probas_to_segmentation(
        probas, prediction_threshold=prediction_threshold
    )

    start = time.time()
    if verbose:
        print("Post-processing segmentation...")

    with st.spinner("Post-processing segmentation..."):
        prediction = refine_segmentation(prediction)

    if verbose:
        print("Post-processing time:", time.time() - start)
    st.toast(f"Post-processing completed in {time.time() - start:.2f} seconds.")

    # Cache the small prediction result
    st.session_state[cache_key] = prediction
    return prediction


@st.cache_resource
def get_model(model_name):
    return _get_model(revision=model_name)


@st.cache_resource
def get_postprocess_model():
    return get_error_detection_model()


def clear_inference_cache(key=None):
    """Clear cached inference results."""
    if key:
        cache_key = f"_inference_result_{key}"
        st.session_state.pop(cache_key, None)
    else:
        keys = [k for k in st.session_state if k.startswith("_inference_result_")]
        for k in keys:
            del st.session_state[k]


@st.cache_data
def detect_error_with_cache(
    _image: np.ndarray,
    _fibers: Fibers,
    _correction_model,
    _device,
    _pixel_size,
    _batch_size=32,
    key="default",
):
    return detect_error(
        _fibers.deepcopy(),
        _image,
        _correction_model,
        device=_device,
        pixel_size=_pixel_size,
        batch_size=_batch_size,
    )
