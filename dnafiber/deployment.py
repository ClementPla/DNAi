import math
import time
import warnings
from pathlib import Path
from dnafiber.error_detection.inference import detect_error
from dnafiber.inference import run_model, probas_to_segmentation
import cv2
import pandas as pd
import torch

from dnafiber.postprocess import refine_segmentation

from dnafiber.postprocess.fiber import Fibers
from dnafiber.data.utils import numpy_to_base64_png
from dnafiber.data.utils import load_image, load_multifile_image
import numpy as np

from dnafiber.ui.io import load_image_from_entry


def run_one_file(
    file,
    model,
    reverse_channels=False, 
    channel_mapping: dict[str, int] | None = None, # Should contain at least "analog_1" and "analog_2" keys if provided.
    pixel_size=0.13,
    use_tta=True,
    verbose=True,
    low_end_hardware=False, 
    clarity=1.0,
    error_detection_model=None,
    device=None,
    multitile_strategy: str = "compact", 
) -> Fibers:
    start = time.time()
    if reverse_channels:
        warnings.warn("reverse_channels is deprecated and will be removed in a future version. Please use channel_mapping instead.")
    if channel_mapping is not None and reverse_channels:
        warnings.warn("Both channel_mapping and reverse_channels are specified. channel_mapping will be used.")
    if channel_mapping is not None and not all(role in channel_mapping for role in ["analog_1", "analog_2"]):
        raise ValueError("channel_mapping must contain at least 'analog_1' and 'analog_2' keys.")
    is_cuda_available = torch.cuda.is_available()
    if isinstance(file, np.ndarray):
            image = file
            filename = "Provided Image"
    elif isinstance(file, tuple):
        # File-per-analog mode (legacy path).
        filename = (file[0] or file[1]).name
        image = load_multifile_image(file, pixel_size=pixel_size, clarity=clarity)
    elif channel_mapping is not None:
        # New path: explicit mapping. Build an ad-hoc entry and reuse the
        # GUI loader so script and GUI go through the same code path.
        entry = {
            "id": str(file),
            "display_name": Path(file).name,
            "mode": "multichannel",
            "sources": {role: (Path(file), idx) for role, idx in channel_mapping.items()},
            "extra_labels": {},
        }
        filename = entry["display_name"]
        image = load_image_from_entry(
            entry,
            pixel_size=pixel_size,
            clarity=clarity,
            multitile_strategy=multitile_strategy,
        )
    else:
        # Legacy single-file path with reverse_channels toggle.
        filename = file.name
        image = load_image(
            file, reverse_channels,
            pixel_size=pixel_size, clarity=clarity,
        )
    if verbose:
        print(f"Image loading time: {time.time() - start:.2f} seconds for {filename}")
    start = time.time()
    if device is None:
        device = "cuda" if is_cuda_available else "cpu"

    results = inference(
        model=model,
        image=image,
        pixel_size=pixel_size,
        device=device,
        use_tta=use_tta,
        low_end_hardware=low_end_hardware,
        verbose=verbose,
        error_detection_model=error_detection_model,
    )

    return results


def inference(
    model,
    image,
    device,
    pixel_size,
    use_tta=True,
    low_end_hardware=False,
    verbose=True,
    error_detection_model=None,
) -> np.ndarray | Fibers:
    start = time.time()
    output = run_model(
        model,
        image=image,
        device=device,
        scale=pixel_size,
        use_tta=use_tta,
        verbose=verbose,
        low_end_hardware=low_end_hardware,
    )
    with torch.no_grad():
        output = probas_to_segmentation(output)

    if verbose:
        print("Segmentation time:", time.time() - start)

    start = time.time()
    output = refine_segmentation(output)
    if verbose:
        print("Post-processing time:", time.time() - start)
    if error_detection_model is not None:
        output = detect_error(
            output,
            image,
            error_detection_model,
            device=device,
            pixel_size=pixel_size,
            batch_size=128 if not low_end_hardware else 32,
            verbose=verbose,
        )
    return output


def format_results(results: Fibers, pixel_size: float) -> pd.DataFrame:
    """
    Format the results for display in the UI.
    """
    results = [fiber for fiber in results if fiber.is_valid]
    all_results = dict(
        FirstAnalog=[], SecondAnalog=[], length=[], ratio=[], fiber_type=[]
    )
    all_results["FirstAnalog"].extend([fiber.red * pixel_size for fiber in results])
    all_results["SecondAnalog"].extend([fiber.green * pixel_size for fiber in results])
    all_results["length"].extend(
        [fiber.red * pixel_size + fiber.green * pixel_size for fiber in results]
    )
    all_results["ratio"].extend([fiber.ratio for fiber in results])
    all_results["fiber_type"].extend([fiber.fiber_type for fiber in results])

    return pd.DataFrame.from_dict(all_results)


def format_results_to_dataframe(
    _prediction,
    _image,
    resolution=400,
    include_thumbnails=True,
    pixel_size=0.13,
    include_bbox=False,
    include_segmentation=False,
):
    data = dict(
        fiber_id=[],
        firstAnalog=[],
        secondAnalog=[],
        ratio=[],
        fiber_type=[],
    )
    if include_thumbnails:
        data["Visualization"] = []
        data["Segmentation"] = []
    if include_bbox:
        data["bbox"] = []
    if include_segmentation:
        data["segmentation"] = []
    for fiber in _prediction:
        data["fiber_id"].append(fiber.fiber_id)
        r, g = fiber.counts
        red_length = pixel_size * r
        green_length = pixel_size * g
        data["firstAnalog"].append(f"{red_length:.3f} ")
        data["secondAnalog"].append(f"{green_length:.3f} ")
        data["ratio"].append(f"{green_length / red_length:.3f}")
        data["fiber_type"].append(fiber.fiber_type)
        if include_segmentation:
            data["segmentation"].append(fiber.data)
        if include_bbox:
            data["bbox"].append(fiber.bbox)

        if not include_thumbnails:
            continue

        x, y, w, h = fiber.bbox

        # Extract a region twice as large as the bbox from the image
        offsetX = math.floor(w / 2)
        offsetY = math.floor(h / 2)
        visu = _image[
            max(0, y - offsetY) : min(_image.shape[0], y + h + offsetY),
            max(0, x - offsetX) : min(_image.shape[1], x + w + offsetX),
        ]

        # Express the bbox in the same coordinate system as the visualization
        x = max(0, offsetX)
        y = max(0, offsetY)

        # Draw the bbox on the visualization
        cv2.rectangle(visu, (x, y), (x + w, y + h), (0, 0, 255), 3)
        segmentation = fiber.data
        # Scale the visualization to a minimum width of 256 pixels

        if visu.shape[1] != resolution:
            scale = resolution / visu.shape[1]
            visu = cv2.resize(
                visu,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LINEAR,
            )
            segmentation = cv2.resize(
                segmentation,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST_EXACT,
            )
            offsetX = math.floor(offsetX * scale)
            offsetY = math.floor(offsetY * scale)

        red_mask = segmentation == 1
        green_mask = segmentation == 2
        # Convert the segmentation to a 3-channel image
        segmentation = cv2.cvtColor(segmentation, cv2.COLOR_GRAY2BGR)
        # segmentation== 1 is red, segmentation==2 is green
        segmentation[red_mask] = np.array([255, 0, 0])
        segmentation[green_mask] = np.array([0, 255, 0])
        # Make sure the
        data["Visualization"].append(visu)
        data["Segmentation"].append(segmentation)
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
    if include_thumbnails:
        df["Visualization"] = df["Visualization"].apply(
            lambda x: numpy_to_base64_png(x)
        )
        df["Segmentation"] = df["Segmentation"].apply(lambda x: numpy_to_base64_png(x))
    return df
