from pathlib import Path
from typing import Optional
import numpy as np

from dnafiber.postprocess.core import Fibers
from dnafiber.data.utils import mask_filepath_to_fibers
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_array


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
        common_map = build_union_map_clustered(annotators_fibers[indices_gt], ratio)
    else:
        common_map = build_union_map_clustered(annotators_fibers, ratio)
    results = np.zeros((len(common_map), N), dtype=bool)
    for i, fiber in enumerate(common_map):
        for j, annotator in enumerate(annotators_fibers):
            if annotator.contains(fiber, ratio):
                results[i, j] = True

    return results


def build_union_map(annotators_fibers: list[Fibers], ratio=0.8) -> Fibers:
    """
    Build a union map of fibers found by different annotators.
    :param: annotators_fibers: A list of Fibers objects from different annotators.
    """
    union_map = Fibers([])
    for annotator in annotators_fibers:
        for fiber in annotator:
            union_map.append_if_not_exists(fiber, ratio)
    return union_map


def build_union_map_clustered(annotators_fibers, ratio=0.5, max_distance=5.0):
    all_fibers = []
    for annotator in annotators_fibers:
        for fiber in annotator:
            all_fibers.append(fiber)

    n = len(all_fibers)
    rows, cols = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if all_fibers[i].overlaps(all_fibers[j], ratio):
                rows.extend([i, j])
                cols.extend([j, i])

    adj = csr_array((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_components, labels = connected_components(adj, directed=False)

    # Pick one representative per cluster (e.g., first fiber)
    union = Fibers([])
    for c in range(n_components):
        idx = np.where(labels == c)[0][0]
        union.append(all_fibers[idx])

    return union


def build_union_map_clustered_v2(annotators_fibers, max_distance=5.0, ratio=0.5):
    """Build a union map using connected components to avoid order-dependent merging."""
    all_fibers = []
    for annotator in annotators_fibers:
        for fiber in annotator:
            all_fibers.append(fiber)

    n = len(all_fibers)
    if n == 0:
        return Fibers([]), np.array([], dtype=int), []

    rows, cols = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if all_fibers[i].skeleton_match(
                all_fibers[j], max_distance=max_distance, ratio=ratio
            ):
                rows.extend([i, j])
                cols.extend([j, i])

    adj = csr_array((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_components, labels = connected_components(adj, directed=False)

    # Pick one representative per cluster (first fiber)
    union = Fibers([])
    for c in range(n_components):
        idx = np.where(labels == c)[0][0]
        union.append(all_fibers[idx])

    return union, labels, all_fibers


def get_fibers_found_only_per_ai(
    human_fibers: list[Fibers], ai_fibers: Fibers, ratio=0.8
) -> Fibers:
    """
    Get the fibers found only by the AI and not by any human annotator.
    :param: human_fibers: A list of Fibers objects from different human annotators.
    :param: ai_fibers: A Fibers object from the AI annotator.
    """
    union_human = build_union_map_clustered(human_fibers, ratio)
    only_ai = ai_fibers.difference(union_human, ratio)
    return only_ai


def get_fibers_found_per_ai_and_humans(
    human_fibers: list[Fibers], ai_fibers: Fibers, ratio=0.8
) -> Fibers:
    """
    Get the fibers found by both the AI and at least one human annotator.
    :param: human_fibers: A list of Fibers objects from different human annotators.
    :param: ai_fibers: A Fibers object from the AI annotator.
    """
    union_human = build_union_map_clustered(human_fibers, ratio)
    common_ai_humans = ai_fibers.intersection(union_human, ratio)
    return common_ai_humans


def compute_agreement_stats(
    gts, human_keys=["H1", "H2", "H3", "H4"], max_distance=5.0, ratio=0.5
):
    """
    Compute inter-annotator agreement statistics.

    Returns:
        total_distinct: total number of distinct fibers in the union
        found_by_all: number of fibers found by all annotators
        pct_found_by_all: percentage found by all
        pct_unique_per_annotator: average % of fibers unique to one annotator
    """
    total_distinct = 0
    found_by_all = 0
    unique_per_annotator = {k: 0 for k in human_keys}
    total_per_annotator = {k: 0 for k in human_keys}

    for img, all_types_annotations in gts.items():
        humans_annotations = [all_types_annotations[h] for h in human_keys]

        # Build clustered union
        union, labels, all_fibers = build_union_map_clustered_v2(
            humans_annotations, max_distance=max_distance, ratio=ratio
        )

        n_clusters = len(union)
        total_distinct += n_clusters

        # Track which annotator contributed to each cluster
        # Build annotator index for each fiber
        annotator_ids = []
        for ann_idx, annotator in enumerate(humans_annotations):
            for _ in annotator:
                annotator_ids.append(ann_idx)

        # For each cluster, find which annotators are present
        cluster_annotators = {}
        for fiber_idx, cluster_id in enumerate(labels):
            if cluster_id not in cluster_annotators:
                cluster_annotators[cluster_id] = set()
            cluster_annotators[cluster_id].add(annotator_ids[fiber_idx])

        # Count fibers found by all annotators
        for cluster_id, annotators in cluster_annotators.items():
            if len(annotators) == len(human_keys):
                found_by_all += 1

        # Count fibers unique to each annotator
        for cluster_id, annotators in cluster_annotators.items():
            if len(annotators) == 1:
                ann_idx = list(annotators)[0]
                unique_per_annotator[human_keys[ann_idx]] += 1

        for ann_idx, key in enumerate(human_keys):
            total_per_annotator[key] += sum(
                1 for cid, anns in cluster_annotators.items() if ann_idx in anns
            )

    pct_found_by_all = found_by_all / total_distinct * 100 if total_distinct > 0 else 0

    # Average % unique per annotator (unique / total for that annotator)
    pct_unique = []
    for key in human_keys:
        if total_per_annotator[key] > 0:
            pct_unique.append(
                unique_per_annotator[key] / total_per_annotator[key] * 100
            )
        else:
            pct_unique.append(0)

    avg_pct_unique = np.mean(pct_unique)

    print(f"Total distinct fibers: {total_distinct}")
    print(
        f"Found by all {len(human_keys)} annotators: {found_by_all} ({pct_found_by_all:.1f}%)"
    )
    print(f"Average % unique to one annotator: {avg_pct_unique:.1f}%")
    print()
    for key in human_keys:
        print(
            f"  {key}: {total_per_annotator[key]} total, "
            f"{unique_per_annotator[key]} unique "
            f"({unique_per_annotator[key] / total_per_annotator[key] * 100:.1f}%)"
        )

    return {
        "total_distinct": total_distinct,
        "found_by_all": found_by_all,
        "pct_found_by_all": pct_found_by_all,
        "avg_pct_unique": avg_pct_unique,
        "unique_per_annotator": unique_per_annotator,
        "total_per_annotator": total_per_annotator,
    }
