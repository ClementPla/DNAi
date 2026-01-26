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
import torch.nn as nn

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
    """Performs Morphological Closing to bridge gaps without thickening fibers."""

    def __init__(self, predictive_threshold=1 / 3):
        super().__init__()
        self.register_buffer("kernel", torch.ones((3, 3)))
        self.predictive_threshold = predictive_threshold

    def forward(self, probabilities):
        # 0: BG, 1: Red, 2: Green.
        # We bridge the foreground (1.0 - background)
        background = probabilities[:, 0:1, :, :]
        foreground = 1.0 - background

        # Closing = Dilation -> Erosion
        dilated = K.morphology.dilation(foreground, self.kernel)
        bridged_foreground = K.morphology.erosion(dilated, self.kernel)

        # Apply threshold and update background channel
        mask = (bridged_foreground > self.predictive_threshold).float()
        probabilities[:, 0:1, :, :] = 1.0 - mask
        # Note: Channels 1 and 2 retain their original relative probability
        # but are now masked by the bridged foreground.
        return probabilities * mask


class SafeTTAWrapper(nn.Module):
    def __init__(self, model, transforms, merge_mode="tsharpen"):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode

    def forward(self, x):
        h, w = x.shape[2:]
        output_sum = 0
        for transformer in self.transforms:
            augmented = transformer.augment_image(x)
            pred = self.model(augmented)
            deaugmented = transformer.deaugment_mask(pred)

            if deaugmented.shape[2:] != (h, w):
                deaugmented = F.interpolate(deaugmented, size=(h, w), mode="bilinear")

            output_sum += (
                deaugmented**2 if self.merge_mode == "tsharpen" else deaugmented
            )

        return output_sum / len(self.transforms)


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        accumulated_output = 0
        for model in self.models:
            accumulated_output += model(x)
        return accumulated_output / len(self.models)


# --- Main Inference Logic ---


@torch.inference_mode()
def run_model(
    model_input,
    image,
    device,
    scale=0.13,
    use_tta=False,
    low_end_hardware=False,
    verbose=False,
):
    device = torch.device(device)
    model_list = model_input if isinstance(model_input, list) else [model_input]

    # 1. Preprocessing & Scaling
    tensor = transform(image=image)["image"].unsqueeze(0)
    h_orig, w_orig = tensor.shape[2:]
    rescale_factor = scale / 0.26
    input_tensor = F.interpolate(
        tensor,
        size=(int(h_orig * rescale_factor), int(w_orig * rescale_factor)),
        mode="bilinear",
    )

    # 2. Setup Sliding Window
    sw_inferer = SlidingWindowInferer(
        roi_size=(1024, 1024),
        sw_batch_size=2 if low_end_hardware else 4,
        overlap=0.1,
        mode="gaussian",
        sw_device=device,
        device="cpu" if low_end_hardware else device,
        progress=verbose,
    )

    # 3. Execution Strategy
    if not low_end_hardware:
        # --- SPEED MODE: Parallel Ensemble ---
        # Load everything once, run SW once
        loaded_models = []
        for m_ref in model_list:
            m = _get_model(revision=m_ref) if isinstance(m_ref, str) else m_ref
            m = nn.Sequential(m, nn.Softmax(dim=1))
            loaded_models.append(m)

        # Build the graph
        ensemble = EnsembleModel(models=loaded_models).to(device)
        exec_unit = AutoPad(ensemble, 32)

        if use_tta:
            tta_transforms = tta.Compose([tta.Scale(scales=[0.75, 1, 1.25])])
            exec_unit = SafeTTAWrapper(exec_unit, tta_transforms)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            accumulated_probs = sw_inferer(input_tensor, exec_unit)

    else:
        # --- MEMORY MODE: Sequential Ensemble ---
        # (Same as the previous step: load one, run SW, unload, repeat)
        accumulated_probs = None
        weight = 1.0 / len(model_list)

        for m_ref in model_list:
            model = _get_model(revision=m_ref) if isinstance(m_ref, str) else m_ref

            model = nn.Sequential(model, nn.Softmax(dim=1)).to(device).eval()
            exec_unit = AutoPad(model, 32)

            if use_tta:
                exec_unit = SafeTTAWrapper(
                    exec_unit, tta.Compose([tta.Rotate90(angles=[0, 90])])
                )

            with torch.autocast(
                device_type=device.type, enabled=(device.type == "cuda")
            ):
                current_output = sw_inferer(input_tensor, exec_unit).to(device)
                if accumulated_probs is None:
                    accumulated_probs = current_output * weight
                else:
                    accumulated_probs += current_output * weight
            del model, exec_unit
            torch.cuda.empty_cache()

    # # 4. Final Post-Processing
    # post_processor = BridgeGap(prediction_threshold).to(device)
    # final_probs = post_processor(accumulated_probs)

    return F.interpolate(accumulated_probs, size=(h_orig, w_orig), mode="bilinear")


def probas_to_segmentation(probas, prediction_threshold=1 / 3) -> np.ndarray:
    bg_probs = probas[:, 0:1, :, :]
    fg_probs = 1.0 - bg_probs

    # Create a mask where fibers are detected
    # A higher threshold means fewer fibers are detected (higher precision)
    # A lower threshold means more fibers are detected (higher recall)
    fiber_mask = (fg_probs >= prediction_threshold).float()

    # Apply mask:
    # If below threshold, background becomes 1.0, colors become 0.0
    final_probs = probas.clone()
    final_probs[:, 0:1, :, :] = torch.where(
        fiber_mask == 1, bg_probs, torch.ones_like(bg_probs)
    )
    final_probs[:, 1:2, :, :] = torch.where(
        fiber_mask == 1, probas[:, 1:2, :, :], torch.zeros_like(bg_probs)
    )
    final_probs[:, 2:3, :, :] = torch.where(
        fiber_mask == 1, probas[:, 2:3, :, :], torch.zeros_like(bg_probs)
    )

    # 4. Final Resize back to original pixel dimensions
    return final_probs.argmax(dim=1).byte().squeeze(0).cpu().numpy()
