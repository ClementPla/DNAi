from pathlib import Path
from typing import Optional
import numpy as np
import cv2

from dnafiber.postprocess.core import extract_fibers, Fibers
from dnafiber.data.utils import mask_filepath_to_fibers


def load_gt(root: Path):
    root = Path(root)
    list_annotators = [f.stem for f in root.glob("*")]
    results = dict()
    input_path = root / list_annotators[0]
    all_files = list(input_path.rglob("*.png"))

    for file in all_files:
        input_file_path = file.relative_to(input_path)
        results[str(input_file_path)] = dict()
        for annotator in list_annotators:
            img_path = root / annotator / input_file_path
            results[str(input_file_path)][annotator] = mask_filepath_to_fibers(img_path)

    return results


def count_commons_fibers(
    annotators_fibers: list[Fibers], indices_gt: Optional[slice] = None, ratio=0.8
) -> np.array:
    """
    For each of the N annotators, count the number of fibers found by 0, 1, ..., N-1 annotators.
    :param: annotators_fibers: A list of Fibers objects from different annotators.
    """
    N = len(annotators_fibers)
    if indices_gt is not None:
        common_map = build_union_map(list(annotators_fibers)[indices_gt], ratio)
    else:
        common_map = build_union_map(annotators_fibers, ratio)
    results = np.zeros((len(common_map), N), dtype=bool)
    for i, fiber in enumerate(common_map):
        for j, annotator in enumerate(annotators_fibers):
            if annotator.contains(fiber, ratio):
                results[i, j] = True

    return results


def build_union_map(annotators_fibers: list[Fibers], ratio=0.8) -> np.array:
    """
    Build a union map of fibers found by different annotators.
    :param: annotators_fibers: A list of Fibers objects from different annotators.
    """
    union_map = Fibers([])
    for annotator in annotators_fibers:
        for fiber in annotator:
            union_map.append_if_not_exists(fiber, ratio)
    return union_map


def get_fibers_found_only_per_ai(
    human_fibers: list[Fibers], ai_fibers: Fibers, ratio=0.8
):
    """
    Get the fibers found only by the AI and not by any human annotator.
    :param: human_fibers: A list of Fibers objects from different human annotators.
    :param: ai_fibers: A Fibers object from the AI annotator.
    """
    union_human = build_union_map(human_fibers, ratio)
    only_ai = ai_fibers.difference(union_human, ratio)
    return only_ai


def get_fibers_found_per_ai_and_humans(
    human_fibers: list[Fibers], ai_fibers: Fibers, ratio=0.8
):
    """
    Get the fibers found by both the AI and at least one human annotator.
    :param: human_fibers: A list of Fibers objects from different human annotators.
    :param: ai_fibers: A Fibers object from the AI annotator.
    """
    union_human = build_union_map(human_fibers, ratio)
    common_ai_humans = ai_fibers.intersection(union_human, ratio)
    return common_ai_humans
