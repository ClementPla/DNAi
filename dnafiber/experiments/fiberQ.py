from pathlib import Path
from dnafiber.data.utils import mask_filepath_to_fibers


def load_fibers_from_pred_folder(pred_folder_path: str | Path):
    pred_folder_path = Path(pred_folder_path)
    all_files = list(pred_folder_path.rglob("*Segm_Clean.png"))
    fibers_dict = dict()
    for file in all_files:
        fibers_dict[file.parent] = mask_filepath_to_fibers(file)
    return fibers_dict
