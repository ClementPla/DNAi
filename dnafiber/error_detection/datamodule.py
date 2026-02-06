from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dnafiber.data.consts import CMAP, CMAP_NO_ALPHA
import albumentations as A

import torch
from dnafiber.error_detection.const import MEAN_FEATURES, STD_FEATURES


class ErrorDetectionDataset(Dataset):
    def __init__(self, root_path: Path | str, transform=None):
        self.root_path = Path(root_path)
        img_paths = list((self.root_path / "images").glob("*.png"))

        # 1. Load and convert to float32
        loaded_imgs = [cv2.imread(str(p))[:, :, ::-1] for p in img_paths]
        self.imgs = np.array(loaded_imgs).astype(np.float32)

        self.imgs_names = np.array([p.stem for p in img_paths])
        self.labels = pd.read_csv(self.root_path / "labels.csv")
        self.transform = transform

    def __getitem__(self, index):
        img = self.imgs[index]

        # Extract mask (channel 2) before potential transforms
        mask = img[:, :, 2].copy()
        # Create a 3-channel version where mask is zeroed out for standard RGB processing
        img_rgb = img.copy() / 255.0  # Normalize to [0, 1]
        img_rgb[:, :, 2] = 0

        if self.transform:
            augmented = self.transform(image=img_rgb, mask=mask)
            img_rgb = augmented["image"]
            mask = augmented["mask"]

        # Convert image to [3, H, W]
        img_t = A.pytorch.ToTensorV2()(image=img_rgb)["image"]
        # Convert mask to [1, H, W]
        mask_t = (
            torch.from_numpy(mask).unsqueeze(0)
            if isinstance(mask, np.ndarray)
            else mask.unsqueeze(0)
        )

        # Concatenate into 4 channels
        combined_img = torch.cat([img_t, mask_t], dim=0)

        return (
            combined_img,
            torch.from_numpy(self.get_features(index)),
            self.get_gt(index),
        )

    def get_gt(self, index):
        img_name = self.imgs_names[index]
        label = self.labels[self.labels["name"] == img_name]["is_error"].values[0]
        return int(label)

    def get_features(self, index):
        features = self.labels[self.labels["name"] == self.imgs_names[index]][
            ["length", "R_mean", "G_mean", "tortuosity", "curvature"]
        ].values[0]
        return (features.astype(np.float32) - MEAN_FEATURES) / STD_FEATURES

    @property
    def features(self):
        return (
            self.labels[
                ["length", "R_mean", "G_mean", "tortuosity", "curvature"]
            ].values.astype(np.float32)
            - MEAN_FEATURES
        ) / STD_FEATURES

    @property
    def y(self):
        return self.labels["is_error"].values.astype(np.int64)

    def __len__(self):
        return len(self.imgs)

    def plot(self, index=None, name=None):
        if index is None and name is None:
            index = np.random.randint(0, len(self))
        elif name is not None:
            index = np.where(self.imgs_names == name)[0][0]

        img_to_show = self.imgs[index].copy()
        mask = img_to_show[:, :, 2]
        vis_img = img_to_show.copy()
        vis_img[:, :, 2] = 0

        label = self.get_gt(index)
        features = self.get_features(index)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(vis_img)
        axes[0].set_title("Image")
        axes[1].imshow(mask, cmap=CMAP_NO_ALPHA, interpolation="nearest")
        axes[1].set_title("Mask")
        axes[2].imshow(vis_img * 0.5)
        axes[2].imshow(mask, cmap=CMAP, interpolation="nearest")
        axes[2].set_title(f"Overlay - Label: {'Error' if label == 1 else 'No Error'}")
        for ax in axes:
            ax.axis("off")
        # Change figure background color based on label
        fig.patch.set_facecolor("lightcoral" if label == 1 else "lightgreen")
        title = f"Normalized Length: {features[0]:.2f}, R: {features[1]:.2f}, G: {features[2]:.2f}, Tortuosity: {features[3]:.2f}, Curvature: {features[4]:.2f}"
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()


class ErrorDetectionDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path | str,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = ErrorDetectionDataset(
            self.data_dir / "train",
            transform=A.Compose(
                [
                    A.Compose(
                        [
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            A.RandomRotate90(p=0.5),
                            A.GaussianBlur(p=0.2),
                        ]
                    ),
                ],
            ),
        )

        self.val_dataset = ErrorDetectionDataset(self.data_dir / "val")
        self.test_dataset = ErrorDetectionDataset(
            self.data_dir / "test",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
