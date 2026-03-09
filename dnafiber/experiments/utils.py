from pathlib import Path

from dnafiber.deployment import run_one_file
from dnafiber.model.utils import get_ensemble_models, get_error_detection_model
from tqdm.auto import tqdm


def lisa_images_infer_if_needed(images_folder: Path, output_folder: Path):
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
    list_images = list(images_folder.glob("*.czi"))
    # Run a first loop over the images to check if the predictions is already done

    predictions_done = False
    for img_filepath in list_images:
        output_path = output_folder / img_filepath.with_suffix(".pkl").name
        if not output_path.exists():
            predictions_done = False
            break
        else:
            predictions_done = True

    if predictions_done:
        print("Predictions already done, skipping inference.")
        return
    models = [m.cuda() for m in get_ensemble_models(compile=False)]
    detection_model = get_error_detection_model(compile=False).cuda()
    for img_filepath in tqdm(list_images):
        output_path = output_folder / img_filepath.with_suffix(".pkl").name
        if output_path.exists():
            print(f"Prediction for {img_filepath.name} already exists, skipping.")
            continue

        prediction = run_one_file(
            img_filepath,
            model=models,
            error_detection_model=detection_model,
            verbose=False,
            use_tta=True,
            pixel_size=0.1441270,
            reverse_channels=True,
            # pixel_size=0.1441270,
        ).valid_copy()
        prediction.to_pickle(output_path)


def leica_infer_if_needed(images_folder: Path, output_folder: Path):
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
    list_images = list(images_folder.glob("*.tif"))
    # Run a first loop over the images to check if the predictions is already done

    predictions_done = False
    for img_filepath in list_images:
        output_path = output_folder / img_filepath.with_suffix(".pkl").name
        if not output_path.exists():
            predictions_done = False
            break
        else:
            predictions_done = True

    if predictions_done:
        print("Predictions already done, skipping inference.")
        return

    models = [m.cuda() for m in get_ensemble_models(compile=False)]
    detection_model = get_error_detection_model(compile=False).cuda()
    for img_filepath in tqdm(list_images):
        output_path = output_folder / img_filepath.with_suffix(".pkl").name
        if output_path.exists():
            print(f"Prediction for {img_filepath.name} already exists, skipping.")
            continue

        prediction = run_one_file(
            img_filepath,
            model=models,
            error_detection_model=detection_model,
            verbose=False,
            use_tta=True,
            reverse_channels=False,
        ).valid_copy()
        prediction.to_pickle(output_path)
