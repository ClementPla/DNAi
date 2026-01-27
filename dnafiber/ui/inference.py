import streamlit as st
from dnafiber.inference import run_model, probas_to_segmentation
from dnafiber.postprocess.core import refine_segmentation
import numpy as np
from dnafiber.ui.utils import _get_model
from dnafiber.postprocess.fiber import Fibers
import time
from contextlib import contextmanager
from unittest.mock import patch
import monai.inferers.utils as monai_utils
import tqdm





import monai.inferers.utils as monai_utils

# Store original at module load
_original_monai_tqdm = monai_utils.tqdm




@st.cache_data(show_spinner="Running predictions...")
def ui_inference(
    _model,
    _image,
    _device,
    _progress_bar=None,
    use_tta=True,
    pixel_size=0.13,
    prediction_threshold=1 / 3,
    low_end_hardware=False,
    verbose=True,
    key="default",
) -> np.ndarray | Fibers:
    start = time.time()
    class StreamlitTqdm:
        """A tqdm-like class that updates a Streamlit progress bar."""
        
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable
            self.total = total or (len(iterable) if hasattr(iterable, '__len__') else None)
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
                _progress_bar.progress(self.n / self.total, text=f"{self.desc}: {self.n}/{self.total}")
        
        def close(self):
            _progress_bar.empty()
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            self.close()
    
    @contextmanager
    def streamlit_tqdm():
        """Redirect MONAI's tqdm to Streamlit progress bar."""
        monai_utils.tqdm = StreamlitTqdm
        try:
            yield
        finally:
            monai_utils.tqdm = _original_monai_tqdm
            
    with streamlit_tqdm():
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
