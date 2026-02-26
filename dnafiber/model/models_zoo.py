from enum import Enum


class Models(str, Enum):
    UNET_MOBILEONE_S0 = "unet_mobileone_s0"
    UNET_MOBILEONE_S1 = "unet_mobileone_s1"
    UNET_MOBILEONE_S2 = "unet_mobileone_s2"
    UNET_MOBILEONE_S3 = "unet_mobileone_s3"
    UNET_RESNET18 = "unet_resnet18"
    UNET_RESNET34 = "unet_resnet34"
    UNET_SE_RESNET50 = "unet_se_resnet50"
    UNET_SE_RESNET152 = "unet_se_resnet152"

    UNET_EFFICIENTNET_B2 = "unet_timm-efficientnet-b2"
    UNET_EFFICIENTNET_B5 = "unet_timm-efficientnet-b5"
    UNET_CONVNEXT_BASE = "unet_tu-convnextv2_base"

    SEGFORMER_MIT_B0 = "segformer_mit_b0"
    SEGFORMER_MIT_B1 = "segformer_mit_b1"
    SEGFORMER_MIT_B2 = "segformer_mit_b2"
    SEGFORMER_MIT_B3 = "segformer_mit_b3"
    ENSEMBLE = "ensemble"


MODELS_ZOO = {
    "UNet MobileOne S0": Models.UNET_MOBILEONE_S0,
    "UNet MobileOne S1": Models.UNET_MOBILEONE_S1,
    "UNet MobileOne S2": Models.UNET_MOBILEONE_S2,
    "UNet MobileOne S3": Models.UNET_MOBILEONE_S3,
    "UNet ResNet18": Models.UNET_RESNET18,
    "UNet ResNet34": Models.UNET_RESNET34,
    "UNet SE ResNet50": Models.UNET_SE_RESNET50,
    "UNet SE ResNet152": Models.UNET_SE_RESNET152,
    "UNet ConvNeXt Base": Models.UNET_CONVNEXT_BASE,
    "Segformer MIT B0": Models.SEGFORMER_MIT_B0,
    "Segformer MIT B1": Models.SEGFORMER_MIT_B1,
    "Segformer MIT B2": Models.SEGFORMER_MIT_B2,
    "Segformer MIT B3": Models.SEGFORMER_MIT_B3,
}
MODELS_ZOO_R = {v: k for k, v in MODELS_ZOO.items()}

ENSEMBLE = [
    Models.UNET_SE_RESNET50,
    Models.SEGFORMER_MIT_B2,
    Models.UNET_SE_RESNET152,
    Models.UNET_CONVNEXT_BASE,
]
