from dotenv import load_dotenv

load_dotenv()
import os

from huggingface_hub import HfApi
from lightning.pytorch.utilities import rank_zero_only
from dnafiber.trainee import Trainee
from dnafiber.model.models_zoo import ENSEMBLE, Models

HF_TOKEN = os.environ.get("HF_TOKEN")


@rank_zero_only
def upload_to_hub(model, arch, encoder):
    hfapi = HfApi()
    branch_name = f"{arch}_{encoder}"
    hfapi.create_repo(
        "ClementP/DeepFiberQV3",
        token=HF_TOKEN,
        exist_ok=True,
        repo_type="model",
    )
    hfapi.create_branch(
        "ClementP/DeepFiberQV3",
        branch=branch_name,
        token=HF_TOKEN,
        exist_ok=True,
    )

    model.push_to_hub(
        "ClementP/DeepFiberQV3",
        branch=branch_name,
        token=HF_TOKEN,
    )


def _get_model(revision: Models):
    if revision is None:
        model = Trainee.from_pretrained(
            "ClementP/DeepFiberQV3",
            arch="unet",
            encoder_name="se_resnet50",
            encoder_weights=None,
        )
    else:
        model = Trainee.from_pretrained(
            "ClementP/DeepFiberQV3",
            revision=revision,
            force_download=False,
            encoder_weights=None,
        )
    return model.eval().to("cpu")


def get_ensemble_models(compile=False):
    models = []
    for rev in ENSEMBLE:
        model = _get_model(revision=rev)
        if compile:
            model.compile(dynamic=True)
        models.append(model)

    return models
