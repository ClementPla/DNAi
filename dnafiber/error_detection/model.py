from pytorch_lightning import LightningModule
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
    BinaryAUROC,
    BinaryCohenKappa,
    BinaryAveragePrecision,
)
from torchmetrics import MetricCollection
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from huggingface_hub import PyTorchModelHubMixin


class FiberErrorDetector(nn.Module):
    def __init__(self, model_name="fastvit_sa12"):
        super().__init__()
        # 1. Image Encoder: Global Pool ensures we get a flat vector
        self.encoder = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=4,
            num_classes=0,  # Remove the classification head
            global_pool="avg",
        )

        # Determine the size of the image feature vector
        num_features = self.encoder.num_features

        # 2. Combined Head: Image Features + 1 (Length)
        self.head = nn.Sequential(
            nn.Linear(num_features + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),  # Final Binary Output
        )
        self.metadata_mlp = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU()
        )

    def forward(self, x, features):
        # x: Image [Batch, 3, 224, 224]
        # length: [Batch, 1]

        img_features = self.encoder(x)  # Output: [Batch, num_features]
        features = self.metadata_mlp(features)  # [Batch, 32]

        combined = torch.cat((img_features, features), dim=1)

        logits = self.head(combined)
        return logits


class ErrorDetectionModule(
    LightningModule,
    PyTorchModelHubMixin,
):
    def __init__(self, model_name: str = "fastvit_sa12", lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = FiberErrorDetector(model_name=model_name)
        self.val_metrics = MetricCollection(
            {
                "Accuracy": BinaryAccuracy(),
                "Precision": BinaryPrecision(),
                "Recall": BinaryRecall(),
                "Specificity": BinarySpecificity(),
                "AUROC": BinaryAUROC(),
                "CohenKappa": BinaryCohenKappa(),
                "AveragePrecision": BinaryAveragePrecision(),
            }
        )
        self.test_metrics = self.val_metrics.clone(prefix="test/")
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10))
        self.lr = lr

    def forward(self, x, features):
        return self.model(x, features)

    def training_step(self, batch, batch_idx):
        image, features, y = batch
        y = y.unsqueeze(1)  # Ensure y is [Batch, 1] for BCEWithLogitsLoss
        logits = self(image, features)
        loss = self.criterion(logits, y.float())
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, features, y = batch
        y = y.unsqueeze(1)  # Ensure y is [Batch, 1] for BCEWithLogitsLoss
        logits = self(image, features)
        loss = self.criterion(logits, y.float())
        self.log("val/loss", loss)
        self.log_dict(
            self.val_metrics(torch.sigmoid(logits), y),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

    def test_step(self, batch, batch_idx):
        image, features, y = batch
        y = y.unsqueeze(1)  # Ensure y is [Batch, 1] for BCEWithLogitsLoss

        logits = self(image, features)
        self.log_dict(
            self.test_metrics(torch.sigmoid(logits), y),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)

        # 1. Define Warmup (e.g., first 5 epochs)
        # Ramps from lr * 0.1 to lr * 1.0
        warmup_epochs = 10
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )

        # 2. Define Cosine Annealing (e.g., remaining 45 epochs)
        # Decays from lr to eta_min
        total_epochs = self.trainer.max_epochs
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=(total_epochs - warmup_epochs), eta_min=1e-6
        )

        # 3. Combine them
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # or 'step' if you want per-batch updates
            },
        }
