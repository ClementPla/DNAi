import os
import streamlit.components.v1 as components
from dnafiber.data.utils import numpy_to_base64_png, numpy_to_base64_jpeg
import time

_RELEASE = False


if not _RELEASE:
    _component_func = components.declare_component(
        "fiber_ui",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("fiber_ui", path=build_dir)


def fiber_ui(image, fibers, pixel_size, key=None):
    """Create a new instance of "fiber_ui".

    Parameters
    ----------

    Returns
    -------

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
        key=key,
        default=[],
    )
    print("Component call time:", time.time() - start)
    return component_value
