import os
import streamlit.components.v1 as components
from dnafiber.data.utils import numpy_to_base64_jpeg
import time

_RELEASE = True


if not _RELEASE:
    _component_func = components.declare_component(
        "fiber_ui",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("fiber_ui", path=build_dir)


def fiber_ui(
    image,
    fibers,
    pixel_size,
    error_threshold=0.5,
    first_analog_color="#FF0000",
    second_analog_color="#00FF00",
    selected_ids=None,
    key=None,
):
    """Create a new instance of "fiber_ui".

    Parameters
    ----------
    selected_ids: list[int], optional
        Fiber ids currently committed as selected (manual error overrides).
        The frontend initialises and resyncs its selection from it.

    Returns
    -------
    None until the user presses "Send", then a dict
    ``{"selected": list[int], "nonce": float}``. The nonce is unique per send.

    """

    start = time.time()
    data_uri = numpy_to_base64_jpeg(image)
    print("Image encoding time:", time.time() - start)
    start = time.time()
    component_value = _component_func(
        image=data_uri,
        elements=fibers,
        image_w=image.shape[1],
        image_h=image.shape[0],
        pixel_size=pixel_size,
        error_threshold=error_threshold,
        key=key,
        first_analog_color=first_analog_color,
        second_analog_color=second_analog_color,
        selected_ids=[int(i) for i in (selected_ids or [])],
        default=None,
    )
    print("Component call time:", time.time() - start)
    return component_value
