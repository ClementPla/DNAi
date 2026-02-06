from dnafiber.error_detection.datamodule import ErrorDetectionDataModule
from dnafiber.error_detection.model import ErrorDetectionModule
from dnafiber.model.utils import upload_error_detection_model
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import torch

seed_everything(1234, workers=True)
torch.set_float32_matmul_precision("high")


def train():
    datamodule = ErrorDetectionDataModule(
        "/home/clement/Documents/data/DNAFiber/error detection/",
        batch_size=64,
        num_workers=8,
    )
    wandb_logger = WandbLogger(project="dnafiber-error-detection", log_model="all")
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints/error_detection",
        monitor="CohenKappa",
        mode="max",
        save_top_k=1,
    )
    model = ErrorDetectionModule(model_name="convnext_tiny", lr=1e-4)
    trainer = Trainer(
        max_epochs=250,
        accelerator="cuda",
        check_val_every_n_epoch=5,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, LearningRateMonitor()],
    )
    # trainer.fit(model, datamodule=datamodule)
    model = ErrorDetectionModule.load_from_checkpoint(
        "/home/clement/Documents/Projets/DNAFiber/checkpoints/error_detection/epoch=24-step=600.ckpt"
    )
    trainer.test(model=model, datamodule=datamodule)

    # Get the best model path from the checkpoint callback
    # best_model_path = checkpoint_callback.best_model_path
    model = ErrorDetectionModule.load_from_checkpoint(
        "/home/clement/Documents/Projets/DNAFiber/checkpoints/error_detection/epoch=24-step=600.ckpt"
    )
    upload_to_hub(model, model_name="convnext_tiny")


def upload_to_hub(model: ErrorDetectionModule, model_name: str):
    upload_error_detection_model(model, revision=f"{model_name}")


if __name__ == "__main__":
    train()
