import cv2
import streamlit as st
import torch
from dnafiber.data.consts import CMAP
from PIL import Image
import io
import time
from dnafiber.model.models_zoo import ENSEMBLE, Models
from dnafiber.ui.components import (
    get_mosaic,
    model_configuration_inputs,
    performance_button,
    pixel_size_input,
    reverse_channels_input,
    show_fibers,
    table_components,
    distribution_analysis,
    rescale_with_cache,
)
from dnafiber.ui.custom.fiber_ui import fiber_ui
from dnafiber.ui.inference import (
    detect_error_with_cache,
    get_model,
    get_postprocess_model,
    ui_inference,
)
from dnafiber.ui.utils import (
    build_file_id,
    build_inference_id,
    get_image,
    get_multifile_image,
    get_resized_image,
)
from dnafiber.ui.consts import DefaultValues as DV
from dnafiber.ui.utils import retain_session_state, create_display_files
from dnafiber.ui.hardware import sidebar_diagnostics

retain_session_state(st.session_state)
st.set_page_config(
    layout="wide",
    page_icon=":microscope:",
)

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 4rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)


def on_session_start():
    can_start = (
        st.session_state.get("files_uploaded", None) is not None
        and len(st.session_state.files_uploaded) > 0
    )

    if can_start:
        return can_start

    cldu_exists = (
        st.session_state.get("analog_2_files", None) is not None
        and len(st.session_state.analog_2_files) > 0
    )
    idu_exists = (
        st.session_state.get("analog_1_files", None) is not None
        and len(st.session_state.analog_1_files) > 0
    )

    if cldu_exists and idu_exists:
        if len(st.session_state.get("analog_2_files")) != len(
            st.session_state.get("analog_1_files")
        ):
            st.error("Please upload the same number of CldU and IdU files.")
            return False


def start_inference(
    image,
    image_name,
    model_name,
    use_tta=DV.USE_TTA,
    prediction_threshold=DV.PREDICTION_THRESHOLD,
    inference_id=None,
    detect_errors=DV.DETECT_ERRORS,
):
    org_h, org_w = image.shape[:2]
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if model_name == Models.ENSEMBLE:
        model = []
        for _ in range(len(ENSEMBLE)):
            with st.spinner(f"Loading model {_ + 1}/{len(ENSEMBLE)}..."):
                model.append(get_model(ENSEMBLE[_]))
    else:
        with st.spinner("Loading model..."):
            model = get_model(model_name)

    prediction = ui_inference(
        _model=model,
        _image=image,
        _device="cuda" if torch.cuda.is_available() else "cpu",
        use_tta=use_tta,
        pixel_size=st.session_state.get("pixel_size", DV.PIXEL_SIZE),
        low_end_hardware=st.session_state.get("low_end_hardware", DV.LOW_END_HARDWARE),
        key=inference_id,
    )
    prediction = prediction.valid_copy()

    if detect_errors:
        with st.spinner("Detecting errors in fibers..."):
            correction_model = get_postprocess_model().to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            prediction = detect_error_with_cache(
                _fibers=prediction,
                _image=image,
                _correction_model=correction_model,
                _device="cuda" if torch.cuda.is_available() else "cpu",
                _pixel_size=st.session_state.get("pixel_size", DV.PIXEL_SIZE),
                _batch_size=32
                if st.session_state.get("low_end_hardware", DV.LOW_END_HARDWARE)
                else 64,
                key=inference_id,
            )
    if detect_errors:
        hide_error = st.checkbox(
            "Hide detected errors",
            help="Toggle the visibility of fibers detected as errors in the viewer and table.",
        )
        if hide_error:
            prediction = prediction.filter_errors()
    tab_viewer, tab_mosaic, tab_table, tab_distributions = st.tabs(
        ["Viewer", "Mosaic", "Table", "Distribution"]
    )

    with tab_viewer:
        max_dim = max(org_h, org_w)
        max_size = 10000
        if max_dim > max_size:
            st.toast(
                f"Images are displayed at a lower resolution of {max_size} pixel wide"
            )

        rescaled_image, scale = rescale_with_cache(image, inference_id)
        start = time.time()
        selected_fibers_img = fiber_ui(
            rescaled_image,
            prediction.svgs(
                scale=scale,
                color1=st.session_state.get("color1", "red"),
                color2=st.session_state.get("color2", "green"),
            ),
            pixel_size=st.session_state.get("pixel_size", DV.PIXEL_SIZE),
            error_threshold=prediction_threshold,
            key=inference_id,
        )
    for fiber in prediction:
        if fiber.fiber_id in selected_fibers_img:
            fiber.proba_error = 1.0 - fiber.proba_error
    with tab_mosaic:
        prediction_mosaic, image_mosaic = get_mosaic(image, prediction, inference_id)
        selected_fibers = fiber_ui(
            image_mosaic,
            prediction_mosaic.svgs(
                scale=1,
                color1=st.session_state.get("color1", "red"),
                color2=st.session_state.get("color2", "green"),
            ),
            pixel_size=st.session_state.get("pixel_size", DV.PIXEL_SIZE),
            error_threshold=prediction_threshold,
            key=inference_id + "_mosaic",
        )
    for fiber in prediction:
        if fiber.fiber_id in selected_fibers:
            fiber.proba_error = 1.0 - fiber.proba_error

    st.download_button(
        label="Download Fibers object",
        data=prediction.to_pickle(),
        file_name=f"fibers_{image_name}.pkl",
        mime="application/octet-stream",
    )

    with tab_table:
        if len(prediction) == 0:
            st.warning("No fibers detected in this image.")
        else:
            df = show_fibers(
                _prediction=prediction,
                inference_id=inference_id,
            )
            table_components(df, error_threshold=prediction_threshold)
    with tab_distributions:
        if len(prediction) == 0:
            st.warning("No fibers detected in this image.")
        else:
            distribution_analysis(prediction)

    with st.sidebar:
        with st.expander("Download results", expanded=False):
            width = st.slider(
                "Set fiber width for visualization",
                min_value=1,
                max_value=50,
                value=3,
                step=1,
            )

            prepare_download = st.button(
                "Prepare download", help="Prepare all results for download."
            )
            if prepare_download:
                with st.spinner("Preparing files..."):
                    start = time.time()
                    labelmap = prediction.filter_errors(
                        prediction_threshold
                    ).get_labelmap(org_h, org_w, width)
                    labelmap = CMAP(labelmap, bytes=True)[:, :, :3]
                    labelmap = Image.fromarray(labelmap)

                    # Create an in-memory binary stream
                    buffer = io.BytesIO()

                    # Save the image to the buffer in PNG format
                    labelmap.save(buffer, format="jpeg")

                    # Get the byte data from the buffer
                    labelmap_jpeg_data = buffer.getvalue()

                    st.success(
                        f"Segmentation map is ready! ({time.time() - start:.2f}s)"
                    )
                    start = time.time()
                    bbox_map = prediction.filter_errors(
                        prediction_threshold
                    ).get_bounding_boxes_map(org_h, org_w, width=width, image=image)
                    bbox_map = Image.fromarray(bbox_map)
                    buffer = io.BytesIO()
                    bbox_map.save(buffer, format="jpeg")
                    bbox_jpeg_data = buffer.getvalue()
                    st.success(
                        f"Bounding boxes map is ready! ({time.time() - start:.2f}s)"
                    )
                st.download_button(
                    "Segmentation map",
                    data=labelmap_jpeg_data,
                    file_name="segmentation_map.jpg",
                    mime="image/jpeg",
                )

                st.download_button(
                    "Bounding boxes map",
                    data=bbox_jpeg_data,
                    file_name="bounding_boxes_map.jpg",
                    mime="image/jpeg",
                )


if on_session_start():
    with st.sidebar:
        pixel_size_input()
        reverse_channels_input()
    files = st.session_state.files_uploaded
    displayed_names = create_display_files(files)
    with st.sidebar:
        selected_file = st.selectbox(
            "Pick an image",
            displayed_names,
            index=0,
            help="Select an image to view and analyze.",
        )
        st.slider(
            "Clarity adjustment",
            min_value=0.95,
            max_value=1.0,
            step=0.001,
            key="clarity",
            help="Adjust the clarity of the image preprocessing. Lower values may lead to noisier images.",
        )

    # Find index of the selected file
    index = displayed_names.index(selected_file)
    file = files[index]

    file_id = build_file_id(
        file,
        pixel_size=st.session_state.get("pixel_size", DV.PIXEL_SIZE),
        reverse_channels=st.session_state.get("reverse_channels", DV.REVERSE_CHANNELS),
        clarity=st.session_state.get("clarity", DV.CLARITY),
    )
    if isinstance(file, tuple):
        if file[0] is None or file[1] is None:
            missing = "First analog" if file[0] is None else "Second analog"
            st.warning(
                f"In this image, {missing} channel is missing. We assume the intended goal is to segment the DNA fibers without differentiation. \
                       Note the model may still predict two classes and try to compute a ratio; these informations can be ignored."
            )
        image = get_multifile_image(
            file,
            clarity=st.session_state.get("clarity", DV.CLARITY),
            pixel_size=st.session_state.get("pixel_size", DV.PIXEL_SIZE),
        )
    else:
        image = get_image(
            file,
            reverse_channel=st.session_state.get(
                "reverse_channels", DV.REVERSE_CHANNELS
            ),
            id=file_id,
            pixel_size=st.session_state.get("pixel_size", DV.PIXEL_SIZE),
            clarity=st.session_state.get("clarity", DV.CLARITY),
        )

    h, w = image.shape[:2]

    thumbnail = get_resized_image(image, file_id)

    with st.sidebar:
        performance_button()
        model_name = model_configuration_inputs()

    with st.sidebar:
        st.image(image, caption=f"Current image {w}x{h}", width="stretch")

    inference_id = build_inference_id(
        file_id,
        str(model_name),
        st.session_state.get("use_tta", DV.USE_TTA),
        st.session_state.get("low_end_hardware", DV.LOW_END_HARDWARE),
    )

    start_inference(
        image=image,
        image_name=selected_file,
        model_name=model_name,
        use_tta=st.session_state.get("use_tta", DV.USE_TTA),
        prediction_threshold=st.session_state.get(
            "prediction_threshold", DV.PREDICTION_THRESHOLD
        ),
        inference_id=inference_id,
        detect_errors=st.session_state.get(
            "use_error_detection_model", DV.DETECT_ERRORS
        ),
    )

    sidebar_diagnostics()
else:
    st.switch_page("pages/1_Load.py")

# Add a callback to mouse move event
