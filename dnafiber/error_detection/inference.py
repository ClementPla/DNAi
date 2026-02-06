from dnafiber.error_detection.const import MEAN_FEATURES, STD_FEATURES
from dnafiber.images.crop import get_crops
import numpy as np
import torch
from tqdm.auto import tqdm
from typing import TYPE_CHECKING
import streamlit as st

if TYPE_CHECKING:
    from dnafiber.postprocess.fiber import Fibers, FiberProps


def detect_error(
    fibers: "Fibers",
    image: np.ndarray,
    correction_model,
    device,
    pixel_size=0.13,  # microns per pixel
    batch_size=128,
) -> "Fibers":
    """Detect errors in the fibers using the correction model."""

    crops = get_crops(image, fibers, bbox_inflate=4.0, resize=224, return_masks=True)
    crop_images = (
        np.asarray([crop[0] for crop in crops.values()]).astype(np.float32) / 255.0
    )
    crop_masks = np.asarray([crop[1] for crop in crops.values()]).astype(np.float32)

    lengths = np.array([fiber.length for fiber in fibers]) * pixel_size
    intensities = (
        np.array(
            [
                fiber.get_mean_intensity(
                    image[
                        fiber.bbox.y : fiber.bbox.y + fiber.bbox.height,
                        fiber.bbox.x : fiber.bbox.x + fiber.bbox.width,
                    ]
                )
                for fiber in fibers
            ]
        )
        / 255.0
    )
    tortuosities = np.array([fiber.tortuosity for fiber in fibers])
    curvatures = np.array([fiber.get_curvature() for fiber in fibers])

    features = np.stack(
        [lengths, intensities[:, 0], intensities[:, 1], tortuosities, curvatures],
        axis=1,
    )
    features = (features - MEAN_FEATURES[np.newaxis, :]) / STD_FEATURES[np.newaxis, :]
    # Combine images and masks into 4-channel input

    crop_inputs = np.concatenate(
        [crop_images, crop_masks[:, :, :, np.newaxis]], axis=-1
    )  # Shape: (N, H, W, 4)
    crop_inputs = torch.from_numpy(crop_inputs).permute(0, 3, 1, 2).to(device)
    features = torch.from_numpy(features).float().to(device)
    correction_model.eval()
    progress_bar = st.progress(0, text="Detecting errors...")
    with torch.no_grad():
        predictions = []
        for i in tqdm(range(0, len(crop_inputs), batch_size)):
            progress_bar.progress(
                i / len(crop_inputs),
                text=f"Detecting errors... ({i}/{len(crop_inputs)})",
            )
            batch_inputs = crop_inputs[i : i + batch_size]
            batch_features = features[i : i + batch_size]
            batch_preds = (
                torch.sigmoid(correction_model(batch_inputs, batch_features)) > 0.5
            )

            predictions.extend(batch_preds.cpu().numpy())
    for fiber, pred in zip(fibers, predictions):
        fiber.is_an_error = bool(pred)
    progress_bar.empty()
    return fibers
