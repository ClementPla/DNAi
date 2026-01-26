import math
import time
from dnafiber.inference import run_model, probas_to_segmentation
import cv2
import pandas as pd
import torch

from dnafiber.postprocess import refine_segmentation

from dnafiber.postprocess.fiber import Fibers
from dnafiber.data.utils import numpy_to_base64_png
from dnafiber.data.utils import load_image, load_multifile_image
import numpy as np


def run_one_file(
    file,
    model,
    reverse_channels=False,
    pixel_size=0.13,
    prediction_threshold=1 / 3,
    use_tta=True,
    verbose=True,
    low_end_hardware=False,
    clarity=1.0,
) -> Fibers:
    start = time.time()

    is_cuda_available = torch.cuda.is_available()
    if isinstance(file, np.ndarray):
        # If the file is already an image array, we don't need to load it
        image = file
        filename = "Provided Image"
    elif isinstance(file, tuple):
        if file[0] is None:
            filename = file[1].name
        if file[1] is None:
            filename = file[0].name
        image = load_multifile_image(
            file,
            pixel_size=pixel_size,
            clarity=clarity,
        )
    else:
        filename = file.name
        image = load_image(
            file,
            reverse_channels,
            pixel_size=pixel_size,
            clarity=clarity,
        )
    if verbose:
        print(f"Image loading time: {time.time() - start:.2f} seconds for {filename}")
    start = time.time()
    results = inference(
        model=model,
        image=image,
        pixel_size=pixel_size,
        device="cuda" if is_cuda_available else "cpu",
        use_tta=use_tta,
        low_end_hardware=low_end_hardware,
        prediction_threshold=prediction_threshold,
        verbose=verbose,
    )

    return results


def inference(
    model,
    image,
    device,
    pixel_size,
    use_tta=True,
    prediction_threshold=1 / 3,
    low_end_hardware=False,
    verbose=True,
) -> np.ndarray | Fibers:
    start = time.time()
    with torch.inference_mode():
        output = run_model(
            model,
            image=image,
            device=device,
            scale=pixel_size,
            use_tta=use_tta,
            verbose=verbose,
            low_end_hardware=low_end_hardware,
        )
        output = probas_to_segmentation(
            output, prediction_threshold=prediction_threshold
        )

    if verbose:
        print("Segmentation time:", time.time() - start)

    start = time.time()
    output = refine_segmentation(image, output, device=device)
    if verbose:
        print("Post-processing time:", time.time() - start)
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
