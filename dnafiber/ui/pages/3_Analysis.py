import streamlit as st
import torch
import pandas as pd
import plotly.express as px
from catppuccin import PALETTE

from dnafiber.ui.hardware import create_diagnostics_container, update_diagnostics
from dnafiber.ui.utils import (
    retain_session_state,
)
from dnafiber.ui.io import (
    get_image_from_entry,
    build_entry_id,
)
from dnafiber.model.models_zoo import ENSEMBLE, Models
from dnafiber.postprocess.types import FiberType
from dnafiber.ui.inference import (
    detect_error_with_cache,
    get_postprocess_model,
    ui_inference,
    get_model,
)
from dnafiber.ui.components import (
    model_configuration_inputs,
    performance_button,
    pixel_size_input,
    show_fibers_cacheless,
    table_components,
)
from dnafiber.ui.utils import build_inference_id
from dnafiber.ui.consts import DefaultValues as DV


retain_session_state(st.session_state)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
cols = st.columns(2)
with cols[0]:
    replicate_separator = st.pills(
        "Replicate separator",
        options=["-", "_"],
        selection_mode="single",
        default="-",
        format_func=lambda x: f"Condition{x}Replicate.ext",
        help="Select the separator used in image names to separate category from replicate (e.g. 'category-replicate1')",
    )
    if replicate_separator is None:
        replicate_separator = "-"
with cols[1]:
    with st.expander("What is the replicate separator?", expanded=False):
        st.write(
            "If your images are named like `condition1-replicate1`, `condition1-replicate2`, `condition2-replicate1`, select `-` as the separator. If they are named like `condition1_replicate1`, `condition1_replicate2`,  select `_`. Note that the text preceeding the separator will be considered the category (or condition) and the text following it will be considered the replicate. This is used to group images in the charts. You can have multiple `-` or `_` in your image names, but only the last one will be considered the separator. For example, `condition1-subconditionA-replicate1` with `-` as separator will be parsed as category `condition1-subconditionA` and replicate `replicate1`."
        )


def image_name_to_category(image_name: str) -> str:
    """Convert image name to category. Assumes 'category-image_name' format."""
    return replicate_separator.join(image_name.split(replicate_separator)[:-1])


def plot_result(selected_category):
    if st.session_state.get("results", None) is None or selected_category is None:
        return
    only_bilateral = st.checkbox("Show only bicolor fibers", value=True)
    remove_outliers = st.checkbox(
        "Remove outliers", value=True, help="Remove outliers from the data"
    )
    reorder = st.checkbox("Reorder groups by median ratio", value=True)
    show_points = st.checkbox(
        "Show points",
        value=True,
        help="Show a swarm plot next to the distribution plot",
    )
    normalize = st.checkbox(
        "Normalize ratios",
        value=False,
        help="Normalize ratios to a baseline",
    )

    baseline = None
    if normalize:
        baseline = st.selectbox(
            "Select baseline",
            options=st.session_state.results["image_name"]
            .apply(image_name_to_category)
            .unique(),
            help="Select the baseline to normalize ratios",
        )

    if remove_outliers:
        min_ratio, max_ratio = st.slider(
            "Ratio range",
            min_value=0.0,
            max_value=10.0,
            value=(0.0, 10.0),
            step=0.1,
            help="Select the ratio range to display",
        )

    df = st.session_state.results.copy()
    clean_df = df[["Ratio", "image_name", "Fiber type"]].copy()
    clean_df["Ratio"] = clean_df["Ratio"].astype(float)
    clean_df["Image"] = clean_df["image_name"].apply(image_name_to_category)

    if only_bilateral:
        clean_df = clean_df[clean_df["Fiber type"] == FiberType.TWO_SEGMENTS.value]
    if remove_outliers:
        clean_df = clean_df[
            (clean_df["Ratio"] >= min_ratio) & (clean_df["Ratio"] <= max_ratio)
        ]
    if baseline:
        mean_value_baseline = clean_df[clean_df["Image"] == baseline]["Ratio"].median()
        clean_df["Ratio"] = clean_df["Ratio"] / mean_value_baseline

    if selected_category:
        clean_df = clean_df[clean_df["Image"].isin(selected_category)]
        if not reorder:
            clean_df["Image"] = pd.Categorical(
                clean_df["Image"], categories=selected_category, ordered=True
            )
            clean_df.sort_values("Image", inplace=True)

    if reorder:
        image_order = (
            clean_df.groupby("Image")["Ratio"]
            .median()
            .sort_values(ascending=True)
            .index
        )
        clean_df["Image"] = pd.Categorical(
            clean_df["Image"], categories=image_order, ordered=True
        )
        clean_df.sort_values("Image", inplace=True)

    palette = [c.hex for c in PALETTE.latte.colors]
    fig = px.box(
        clean_df,
        y="Ratio",
        x="Image",
        color="Image",
        points="all" if show_points else "outliers",
        color_discrete_sequence=palette,
        log_y=True,
        range_y=[0.125 / 2, 16],
    )
    fig.update_yaxes(
        tickvals=[0.25, 0.5, 1, 2, 4, 8],
        ticktext=["0.25", "0.5", "1", "2", "4", "8"],
        type="log",
    )
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def infer(
    entry,
    model,
    use_tta=DV.USE_TTA,
    predict_error=DV.USE_CORRECTION,
    prediction_threshold=DV.PREDICTION_THRESHOLD,
    inference_id="",
    low_end_hardware=DV.LOW_END_HARDWARE,
):
    file_id = build_entry_id(
        entry,
        pixel_size=st.session_state.get("pixel_size", DV.PIXEL_SIZE),
        clarity=st.session_state.get("clarity", DV.CLARITY),
        multitile_strategy=st.session_state.get("multitile_strategy", "compact"),
    )
    image = get_image_from_entry(
        entry,
        pixel_size=st.session_state.get("pixel_size", DV.PIXEL_SIZE),
        clarity=st.session_state.get("clarity", DV.CLARITY),
        id=file_id,
        multitile_strategy=st.session_state.get("multitile_strategy", "compact"),
    )
    results = ui_inference(
        model,
        image,
        _device="cuda" if torch.cuda.is_available() else "cpu",
        use_tta=use_tta,
        pixel_size=st.session_state.get("pixel_size", DV.PIXEL_SIZE),
        key=inference_id,
        verbose=True,
        low_end_hardware=low_end_hardware,
    )
    if predict_error:
        print("Detecting errors in fibers...")
        results = detect_error_with_cache(
            _fibers=results,
            _image=image,
            _correction_model=get_postprocess_model(),
            _device="cuda" if torch.cuda.is_available() else "cpu",
            _pixel_size=st.session_state.get("pixel_size", DV.PIXEL_SIZE),
            _batch_size=32 if low_end_hardware else 64,
            key=inference_id,
        )
        results = results.filter_errors(prediction_threshold)
    df = show_fibers_cacheless(results)
    df["image_name"] = entry["display_name"]
    return df


def run_inference(model_name, use_tta=DV.USE_TTA, use_correction=DV.USE_CORRECTION):
    if model_name == Models.ENSEMBLE:
        model = []
        for i in range(len(ENSEMBLE)):
            with st.spinner(f"Loading model {i + 1}/{len(ENSEMBLE)}..."):
                model.append(get_model(ENSEMBLE[i]))
    else:
        with st.spinner("Loading model..."):
            model = get_model(model_name)

    my_bar = st.progress(0, text="Running segmentation...")
    diag_container = create_diagnostics_container()
    all_entries = st.session_state.files_uploaded
    all_results = []

    for i, entry in enumerate(all_entries):
        update_diagnostics(diag_container)
        filename = entry["display_name"]

        file_id = build_entry_id(
            entry,
            pixel_size=st.session_state.get("pixel_size", DV.PIXEL_SIZE),
            clarity=st.session_state.get("clarity", DV.CLARITY),
            multitile_strategy=st.session_state.get("multitile_strategy", "compact"),
        )
        prediction_threshold = st.session_state.get(
            "prediction_threshold", DV.PREDICTION_THRESHOLD
        )
        inference_id = build_inference_id(
            file_id,
            model_name,
            use_tta=use_tta,
            low_end_hardware=st.session_state.get(
                "low_end_hardware", DV.LOW_END_HARDWARE
            ),
        )

        try:
            df = infer(
                entry,
                model,
                use_tta=use_tta,
                inference_id=inference_id,
                predict_error=use_correction,
                prediction_threshold=prediction_threshold,
                low_end_hardware=st.session_state.get(
                    "low_end_hardware", DV.LOW_END_HARDWARE
                ),
            )
        except Exception as e:
            st.error(f"Error processing {filename}: {e}")
            continue
        all_results.append(df)
        my_bar.progress(i / len(all_entries), text=f"{filename} done")

    if not all_results:
        my_bar.empty()
        st.warning("No results were produced. Check the errors above.")
        return

    results_dict = {k: [] for k in all_results[0].keys()}
    for result in all_results:
        for k, v in result.items():
            results_dict[k].extend(v)

    my_bar.empty()
    st.session_state.results = pd.DataFrame(results_dict)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

if st.session_state.get("files_uploaded", None):
    run_segmentation = st.button("Run Segmentation", width="stretch")

    with st.sidebar:
        performance_button()
        pixel_size_input()
        model_name = model_configuration_inputs()

    tab_segmentation, tab_charts = st.tabs(["Segmentation", "Charts"])

    with tab_segmentation:
        st.subheader("Segmentation")
        if run_segmentation:
            run_inference(
                model_name=model_name,
                use_tta=st.session_state.get("use_tta", DV.USE_TTA),
                use_correction=st.session_state.get(
                    "use_error_detection_model", DV.USE_CORRECTION
                ),
            )
            st.balloons()
        if st.session_state.get("results", None) is not None:
            table_components(
                st.session_state.results,
                error_threshold=st.session_state.get(
                    "prediction_threshold", DV.PREDICTION_THRESHOLD
                ),
            )

    with tab_charts:
        if st.session_state.get("results", None) is not None:
            results = st.session_state.results
            categories = (
                results["image_name"].apply(image_name_to_category).unique().tolist()
            )
            selected_category = st.multiselect(
                "Select a category", categories, default=categories
            )
            plot_result(selected_category)

else:
    st.switch_page("pages/1_Load.py")
