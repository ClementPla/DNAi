import math

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dnafiber.postprocess.fiber import Fibers


import plotly.express as px

from dnafiber.ui.mosaic import mosaic


@st.cache_data
def show_fibers(
    _prediction, _image, inference_id=None, resolution=400, show_errors=True
):
    return show_fibers_cacheless(
        _prediction,
        _image,
        resolution=resolution,
        show_errors=show_errors,
    )


def show_fibers_cacheless(_prediction, _image, resolution=400, show_errors=True):
    data = dict(
        fiber_id=[],
        firstAnalog=[],
        secondAnalog=[],
        ratio=[],
        fiber_type=[],
        # visualization=[],
        is_valid=[],
    )

    for fiber in _prediction:
        data["fiber_id"].append(fiber.fiber_id)
        r, g = fiber.counts
        red_length = st.session_state["pixel_size"] * r
        green_length = st.session_state["pixel_size"] * g
        data["firstAnalog"].append(f"{red_length:.3f} ")
        data["secondAnalog"].append(f"{green_length:.3f} ")
        data["ratio"].append(
            f"{green_length / red_length if red_length > 0 else 0:.3f}"
        )
        data["fiber_type"].append(fiber.fiber_type)
        data["is_valid"].append(fiber.is_acceptable)

    df = pd.DataFrame(data)
    df = df.rename(
        columns={
            "firstAnalog": "First analog (µm)",
            "secondAnalog": "Second analog (µm)",
            "ratio": "Ratio",
            "fiber_type": "Fiber type",
            "fiber_id": "Fiber ID",
        }
    )
    return df


def table_components(df):
    event = st.dataframe(
        df,
        on_select="rerun",
        selection_mode="multi-row",
        width="stretch",
    )

    rows = event["selection"]["rows"]
    columns = [c for c in df.columns if c != "Visualization"]
    selected_df = df.iloc[rows][columns]

    cols = st.columns(3)
    with cols[0]:
        copy_to_clipboard = st.button(
            "Copy selected fibers to clipboard",
            help="Copy the selected fibers to clipboard in CSV format.",
        )
        if copy_to_clipboard:
            selected_df.to_clipboard(index=False)
    with cols[1]:
        st.download_button(
            "Download valid fibers",
            data=df[df["is_valid"]][columns].to_csv(index=False).encode("utf-8"),
            file_name=f"fibers_valid.csv",
            mime="text/csv",
        )
    with cols[2]:
        st.download_button(
            "Download selected fibers",
            data=selected_df.to_csv(index=False).encode("utf-8"),
            file_name="fibers_segment.csv",
            mime="text/csv",
        )


def distribution_analysis(predictions: Fibers):
    df = predictions.to_df()
    df = df[df["Fiber type"] == "double"]
    df = df[(df.Ratio > 0.125) & (df.Ratio < 8)]
    df["Length"] = df["First analog (µm)"] + df["Second analog (µm)"]
    cap = st.checkbox(
        "Cap number of fibers",
        value=False,
        help="If checked, we only keep the N fibers closest to the barycenter of the distribution.",
    )
    if cap:
        N = st.slider(
            "Number of fibers",
            min_value=1,
            max_value=len(df),
            value=50,
            step=1,
        )
    cap_values = ["Length"]
    mean_points = df[cap_values].median().values

    df["Distance"] = np.linalg.norm(
        df[cap_values].values - mean_points,
        axis=1,
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            df.nsmallest(N if cap else len(df), "Distance"),
            y="Ratio",
            points="all",
            title="Distribution of the Ratio",
            labels={"Ratio": "Ratio (second analog / first analog)"},
        )
        fig.update_layout(height=500, margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = go.Figure(
            data=go.Scatter3d(
                x=df["First analog (µm)"],
                y=df["Second analog (µm)"],
                z=df["Length"],
                mode="markers",
                marker=dict(size=1, color=df["Distance"]),
            )
        )

        fig.update_layout(
            scene=dict(
                xaxis_title="First analog (µm)",
                yaxis_title="Second analog (µm)",
                zaxis_title="Length (µm)",
            ),
            height=700,
        )
        st.plotly_chart(fig, width="stretch")


@st.cache_data(max_entries=5)
def viewer_components(_image, _prediction, inference_id):
    image = _image
    if image.max() > 25:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        max_size = 10000
        h, w = image.shape[:2]
        size = max(h, w)
        scale = 1.0
        if size > max_size:
            scale = max_size / size
            image = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LINEAR,
            )

        return image, scale


@st.cache_data(show_spinner="Setting up mosaic...", max_entries=5)
def get_mosaic(_image, _prediction, inference_id):
    prediction_mosaic, image_mosaic = mosaic(
        _prediction,
        _image,
        downsample=1,
        padding=50,
        allow_rotation=False,
        context_margin=0.5,
    )
    return prediction_mosaic, image_mosaic
