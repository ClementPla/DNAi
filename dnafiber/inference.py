import torch.nn.functional as F
import numpy as np
import torch
from torchvision.transforms._functional_tensor import normalize
import pandas as pd
from skimage.segmentation import expand_labels
import albumentations as A
from monai.inferers import SlidingWindowInferer
from dnafiber.ui.utils import _get_model
from dnafiber.model.autopadDPT import AutoPad
import kornia as K
import torch.nn as nn
import ttach as tta

transform = A.Compose(
    [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2(),
    ]
)


def preprocess_image(image):
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    image = normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return image


def convert_to_dataset(counts):
    data = {"index": [], "red": [], "green": [], "ratio": []}
    for k, v in counts.items():
        data["index"].append(k)
        data["green"].append(v["green"])
        data["red"].append(v["red"])
        if v["red"] == 0:
            data["ratio"].append(np.nan)
        else:
            data["ratio"].append(v["green"] / (v["red"]))
    df = pd.DataFrame(data)
    return df


def convert_mask_to_image(mask, expand=False):
    if expand:
        mask = expand_labels(mask, distance=expand)
    h, w = mask.shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    GREEN = np.array([0, 255, 0])
    RED = np.array([255, 0, 0])

    image[mask == 1] = RED
    image[mask == 2] = GREEN

    return image


class BridgeGap(nn.Module):
    def __init__(self, predictive_threshold=1 / 3):
        super().__init__()

        self.register_buffer("kernel", torch.ones((3, 3)))
        self.predictive_threshold = predictive_threshold

    def forward(self, probabilities):
        # Assume channel 0 is background
        background = probabilities[:, 0:1, :, :]
        foreground = 1.0 - background

        # Dilation to bridge gaps
        bridged_foreground = K.morphology.dilation(foreground, self.kernel)

        # Thresholding
        mask = (bridged_foreground > self.predictive_threshold).float()

        # Re-assign back to background (inverted)
        probabilities[:, 0:1, :, :] = 1.0 - mask
        return probabilities


class EnsembleModel(nn.Module):
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        outputs = None
        for i, model in enumerate(self.models):
            out = self.softmax(model(x))
            if outputs is None:
                outputs = out
            else:
                outputs += out * self.weights[i]
        return outputs


class SafeTTAWrapper(nn.Module):
    def __init__(self, model, transforms, merge_mode="mean"):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        output_sum = None
        for transformer in self.transforms:
            augmented = transformer.augment_image(x)
            pred = self.model(augmented)
            deaugmented = transformer.deaugment_mask(pred)

            # Force consistent size after de-augmentation
            if deaugmented.shape[2] != h or deaugmented.shape[3] != w:
                deaugmented = F.interpolate(deaugmented, size=(h, w), mode="bilinear")

            if self.merge_mode == "tsharpen":
                deaugmented = deaugmented**2
            if output_sum is None:
                output_sum = deaugmented
            else:
                output_sum += deaugmented

        output_avg = output_sum / len(self.transforms)
        return output_avg


class Inferer(nn.Module):
    def __init__(
        self,
        model,
        sliding_window_inferer=None,
        use_tta=False,
        prediction_threshold=1 / 3,
    ):
        super().__init__()

        self.model = AutoPad(
            nn.Sequential(
                EnsembleModel(models=model),
                BridgeGap(prediction_threshold),
            ),
            32,
        )
        self.model.eval()

        self.sliding_window_inferer = sliding_window_inferer

        if use_tta:
            transforms = tta.Compose(
                [
                    tta.Rotate90(angles=[0, 90]),
                    tta.Scale(scales=[1, 0.75, 1.25]),
                ]
            )
            self.model = SafeTTAWrapper(self.model, transforms, merge_mode="tsharpen")

    def forward(self, image):
        if self.sliding_window_inferer is not None:
            output = self.sliding_window_inferer(image, self.model)
        else:
            output = self.model(image)
        return output


@torch.inference_mode()
def run_model(
    model_input,  # Renamed to avoid confusion with model instances
    image,
    device,
    scale=0.13,
    use_tta=False,
    prediction_threshold=1 / 3,
    verbose=False,
    low_end_hardware=False,
):
    device = torch.device(device)

    # 1. Load Model (only if string)
    if isinstance(model_input, str):
        model_instance = _get_model(device=device, revision=model_input)
    else:
        model_instance = model_input

    # 2. Setup Scaling
    model_pixel_size = 0.26
    # If image is 0.13 and model wants 0.26, scale factor is 0.5
    rescale_factor = scale / model_pixel_size

    # 3. Preprocess Tensor
    # transform already handles Normalize + ToTensorV2
    tensor = transform(image=image)["image"].unsqueeze(0)
    h, w = tensor.shape[2], tensor.shape[3]

    # Move to device EARLY to speed up interpolation
    if not low_end_hardware:
        tensor = tensor.to(device)

    # 4. Sliding Window Configuration
    sliding_window = None
    if int(h * rescale_factor) > 1024 or int(w * rescale_factor) > 1024:
        sliding_window = SlidingWindowInferer(
            roi_size=(512, 512) if low_end_hardware else (1024, 1024),
            sw_batch_size=1 if low_end_hardware else 4,  # Reduced for stability
            overlap=0.25,
            mode="gaussian",
            sw_device=device,
            device=torch.device("cpu") if low_end_hardware else device,
            progress=verbose,
        )

    # 5. Build Inference Graph
    inferer = Inferer(
        model=model_instance,
        sliding_window_inferer=sliding_window,
        use_tta=use_tta,
        prediction_threshold=prediction_threshold,
    ).to(device)

    # 6. Execute with Mixed Precision
    with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        # Resize to model's expected physical scale
        input_tensor = F.interpolate(
            tensor,
            size=(int(h * rescale_factor), int(w * rescale_factor)),
            mode="bilinear",
            align_corners=False,
        )
        if low_end_hardware:
            input_tensor = input_tensor.to(device=device)
        probs = inferer(input_tensor)

        # Resize back to original pixel dimensions
        probabilities = F.interpolate(
            probs, size=(h, w), mode="bilinear", align_corners=False
        )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return probabilities
