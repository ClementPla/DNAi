import numpy as np


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def recolor_for_display(
    image: np.ndarray,
    color1: str,
    color2: str,
    ssdna_color: str = "#0000FF",
) -> np.ndarray:
    """Remap the model-order RGB image to the user's display colors.

    `image` is the image fed to the model: channel 0 is the first analog,
    channel 1 the second analog, channel 2 the ssDNA signal. Each plane is
    tinted with its display color and the contributions are summed. This is a
    display-only transform and must never feed inference or error detection.
    With the default red/green/blue colors it is the identity.
    """
    planes = (
        (image[:, :, 0], color1),
        (image[:, :, 1], color2),
        (image[:, :, 2], ssdna_color),
    )
    out = np.zeros((*image.shape[:2], 3), dtype=np.float32)
    for plane, color in planes:
        r, g, b = _hex_to_rgb(color)
        intensity = plane.astype(np.float32) / 255.0
        out[:, :, 0] += intensity * r
        out[:, :, 1] += intensity * g
        out[:, :, 2] += intensity * b
    return np.clip(out, 0, 255).astype(np.uint8)
