from lightning import LightningModule
import segmentation_models_pytorch as smp
from monai.losses.dice import DiceCELoss
from torchmetrics.classification import JaccardIndex

try:
    from torchmetrics.classification import Dice
except ImportError:
    from torchmetrics.segmentation import DiceScore as Dice


from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection
from huggingface_hub import PyTorchModelHubMixin
import torch
import torchvision
from dnafiber.metric import DNAFIBERMetric
from skimage.measure import label
from torch.optim.lr_scheduler import SequentialLR, LinearLR


def _convert_activations(module, from_activation, to_activation):
    """Recursively convert activation functions in a module"""
    for name, child in module.named_children():
        if isinstance(child, from_activation):
            setattr(module, name, to_activation)
        else:
            _convert_activations(child, from_activation, to_activation)


class Trainee(LightningModule, PyTorchModelHubMixin):
    def __init__(
        self, learning_rate=0.001, weight_decay=0.0002, num_classes=3, **model_config
    ):
        super().__init__()
        self.model_config = model_config
        if (
            self.model_config.get("arch", None) is None
            or self.model_config["arch"] == "maskrcnn"
        ):
            self.model = None
        elif self.model_config["arch"] == "steered_cnn":
            from dnafiber.model.steered_cnn import DNAFiberSteeredCNN

            self.model = DNAFiberSteeredCNN(**self.model_config)
        elif "patch14" in self.model_config.get("encoder_name", ""):
            from dnafiber.model.autopadDPT import AutoPad

            # Pad input to be divisible by 14. Input size are usually 512x512, 1024x1024 or 2048x2048
            self.model = AutoPad(
                smp.create_model(
                    classes=3,
                    drop_path_rate=0.0,
                    **self.model_config,
                )
            )

        else:
            self.model = smp.create_model(
                classes=3,
                drop_path_rate=0.0,
                **self.model_config,
            )

        # self.model.compile()
        self.loss = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            lambda_ce=1.0,
            lambda_dice=1.0,
            label_smoothing=0.1,
        )
        try:
            self.metric = MetricCollection(
                {
                    "dice": Dice(num_classes=num_classes, ignore_index=0),
                    "jaccard": JaccardIndex(
                        num_classes=num_classes,
                        task="multiclass" if num_classes > 2 else "binary",
                        ignore_index=0,
                    ),
                    "detection": DNAFIBERMetric(),
                }
            )
        except ValueError:
            self.metric = MetricCollection(
                {
                    "dice": Dice(num_classes=num_classes),
                    "jaccard": JaccardIndex(
                        num_classes=num_classes,
                        task="multiclass" if num_classes > 2 else "binary",
                    ),
                    "detection": DNAFIBERMetric(),
                }
            )

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y = y.clamp(0, 2)
        y_hat = self(x)
        loss = self.get_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def get_loss(self, y_hat, y):
        y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=0.0, neginf=0.0)
        loss = self.loss(y_hat, y.unsqueeze(1))
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y = y.clamp(0, 2)
        y_hat = self(x)
        loss = self.get_loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.metric.update(y_hat.argmax(dim=1), y)
        return y_hat

    def on_validation_epoch_end(self):
        scores = self.metric.compute()
        self.log_dict(scores, sync_dist=True)
        self.metric.reset()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=500)
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=self.learning_rate / 3,
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[500]
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def get_fiber_probability(self, probas: torch.Tensor):
        pos_pred = 1 - probas[:, 0, :, :]
        preds = pos_pred > 1 / 2
        binary = preds.long().detach().cpu()
        labelmap = torch.zeros_like(binary, dtype=torch.long)
        for i, p in enumerate(binary):
            labelmap[p] = torch.from_numpy(label(p.numpy(), connectivity=2))
        labelmap = labelmap.to(probas.device)

        return labelmap, pos_pred

