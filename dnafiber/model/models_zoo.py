from enum import Enum


class Models(str, Enum):
    UNET_SE_RESNET50 = "unet_se_resnet50"
    UNET_SE_RESNET152 = "unet_se_resnet152"

    UNET_EFFICIENTNET_B2 = "unet_timm-efficientnet-b2"
    UNET_EFFICIENTNET_B5 = "unet_timm-efficientnet-b5"
    UNET_CONVNEXT_BASE = "unet_tu-convnextv2_base"

    SEGFORMER_MIT_B0 = "segformer_mit_b0"
    SEGFORMER_MIT_B1 = "segformer_mit_b1"
    SEGFORMER_MIT_B2 = "segformer_mit_b2"

    ENSEMBLE = "ensemble"


MODELS_ZOO = {
    "UNet SE ResNet50": Models.UNET_SE_RESNET50,
    "UNet SE ResNet152": Models.UNET_SE_RESNET152,
    "UNet EfficientNet B2": Models.UNET_EFFICIENTNET_B2,
    "UNet EfficientNet B5": Models.UNET_EFFICIENTNET_B5,
    "UNet ConvNeXt Base": Models.UNET_CONVNEXT_BASE,
    "Segformer MIT B0": Models.SEGFORMER_MIT_B0,
    "Segformer MIT B1": Models.SEGFORMER_MIT_B1,
    "Segformer MIT B2": Models.SEGFORMER_MIT_B2,
}
MODELS_ZOO_R = {v: k for k, v in MODELS_ZOO.items()}

ENSEMBLE = [
    Models.UNET_SE_RESNET50,
    Models.SEGFORMER_MIT_B2,
    Models.UNET_SE_RESNET152,
    Models.UNET_CONVNEXT_BASE,
]
