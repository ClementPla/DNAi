import numpy as np
import cv2
from typing import List, Tuple
from dnafiber.postprocess.skan import find_endpoints, compute_points_angle
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_array
from skimage.morphology import skeletonize
from dnafiber.postprocess.skan import find_line_intersection, prolongate_endpoints
from dnafiber.postprocess.fiber import FiberProps, Bbox, Fibers
from itertools import compress
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from dnafiber.postprocess.error_detection import correct_fibers
from scipy.optimize import linear_sum_assignment

cmlabel = ListedColormap(["black", "red", "green"])

MIN_ANGLE = 20
MIN_BRANCH_LENGTH = 5
MIN_BRANCH_DISTANCE = 30


def handle_multiple_fiber_in_cc(
    fiber,
    junctions_fiber,
    coordinates,
    fiber_width=3,
    # Cost weights
    weight_distance=1.0,
    weight_angle=1.0,
    weight_color=0.3,
    # Normalization factors
    max_distance=None,
    max_angle_deviation=90.0,
    # Hard cutoff
    impossible_cost=1e6,
    max_allowed_distance=None,
):
    if max_distance is None:
        max_distance = fiber_width * 2
    if max_allowed_distance is None:
        max_allowed_distance = fiber_width * 2

    # Erase junctions - vectorized
    for y, x in junctions_fiber:
        fiber[
            y - fiber_width : y + fiber_width + 1,
            x - fiber_width : x + fiber_width + 1,
        ] = 0

    endpoints = find_endpoints(fiber > 0)
    endpoints = np.asarray(endpoints)

    if len(endpoints) == 0:
        return []

    # Filter endpoints close to junctions
    junctions_arr = np.asarray(junctions_fiber)
    dist_to_junctions = cdist(endpoints, junctions_arr, metric="euclidean")
    close_to_junction = np.any(dist_to_junctions < fiber_width * 2, axis=1)
    endpoints = endpoints[close_to_junction]

    if len(endpoints) == 0:
        return []

    retval, branches, branches_stats, _ = cv2.connectedComponentsWithStatsWithAlgorithm(
        fiber, connectivity=8, ccltype=cv2.CCL_BOLELLI, ltype=cv2.CV_16U
    )
    branches_bboxes = branches_stats[
        :,
        [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT],
    ]
    branch_areas = branches_stats[:, cv2.CC_STAT_AREA]

    # Vectorized endpoint property lookup
    endpoint_rows = endpoints[:, 0]
    endpoint_cols = endpoints[:, 1]
    endpoints_ids = branches[endpoint_rows, endpoint_cols]
    endpoints_color = fiber[endpoint_rows, endpoint_cols]

    # Filter small branches - vectorized
    kept_branches_mask = branch_areas >= MIN_BRANCH_LENGTH
    kept_branches_mask[0] = False  # Background is never kept

    # Filter endpoints from kept branches
    remaining_mask = kept_branches_mask[endpoints_ids]
    if not np.any(remaining_mask):
        return []

    endpoints = endpoints[remaining_mask]
    endpoints_color = endpoints_color[remaining_mask]
    endpoints_ids = endpoints_ids[remaining_mask]

    N = len(endpoints)
    if N < 2:
        return []

    # Zero out removed branches in one operation
    branches[~kept_branches_mask[branches]] = 0

    # === Compute cost components (all vectorized) ===

    # 1. Distance cost
    dist_matrix = cdist(endpoints, endpoints, metric="euclidean")
    dist_cost = np.minimum(dist_matrix / max_distance, 1.0)

    # 2. Angle cost
    angles = compute_points_angle(fiber, endpoints, steps=15, oriented=True)
    angles_deg = np.rad2deg(angles)

    # Vectorized angle difference
    diff = np.abs(angles_deg[:, None] - angles_deg[None, :])
    diff = np.minimum(diff, 360.0 - diff)
    angle_deviation = np.abs(diff - 180.0)
    angle_cost = np.minimum(angle_deviation / max_angle_deviation, 1.0)

    # 3. Color cost - vectorized
    color_cost = (endpoints_color[:, None] != endpoints_color[None, :]).astype(
        np.float64
    )

    # === Combined cost matrix ===
    total_cost = (
        weight_distance * dist_cost
        + weight_angle * angle_cost
        + weight_color * color_cost
    )

    # Mask invalid pairs (diagonal + same branch + too far)
    invalid_mask = (
        (np.arange(N)[:, None] == np.arange(N)[None, :])  # diagonal
        | (endpoints_ids[:, None] == endpoints_ids[None, :])  # same branch
        | (dist_matrix > max_allowed_distance)  # too far
    )
    total_cost[invalid_mask] = impossible_cost

    # === Hungarian algorithm ===
    row_ind, col_ind = linear_sum_assignment(total_cost)

    # Filter invalid matches
    valid = total_cost[row_ind, col_ind] < impossible_cost
    row_ind = row_ind[valid]
    col_ind = col_ind[valid]

    # === Build adjacency with sparse matrix directly ===
    # Edges: diagonal + same-branch + matched pairs
    rows = []
    cols = []

    # Diagonal (self-loops)
    rows.extend(range(N))
    cols.extend(range(N))

    # Same branch connections
    for branch_id in np.unique(endpoints_ids):
        idxs = np.where(endpoints_ids == branch_id)[0]
        if len(idxs) > 1:
            for i in idxs:
                for j in idxs:
                    rows.append(i)
                    cols.append(j)

    # Matched pairs
    for i, j in zip(row_ind, col_ind):
        rows.extend([i, j])
        cols.extend([j, i])

    A = csr_array((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(N, N))

    n_components, ccs = connected_components(A, directed=False, return_labels=True)

    # Build edge lookup for drawing
    added_edges = {}
    for i, j in zip(row_ind, col_ind):
        added_edges[i] = j
        added_edges[j] = i

    # === Build results - optimized ===
    results = []
    unique_clusters = np.unique(ccs)

    for c in unique_clusters:
        idx = np.where(ccs == c)[0]
        branches_ids_cluster = np.unique(endpoints_ids[idx])

        # Compute bounding box
        bboxes = branches_bboxes[branches_ids_cluster]
        min_x = np.min(bboxes[:, 0])
        min_y = np.min(bboxes[:, 1])
        max_x = np.max(bboxes[:, 0] + bboxes[:, 2])
        max_y = np.max(bboxes[:, 1] + bboxes[:, 3])

        # Create branch mask efficiently
        branch_mask = np.isin(branches[min_y:max_y, min_x:max_x], branches_ids_cluster)
        new_fiber = branch_mask * fiber[min_y:max_y, min_x:max_x]

        # Draw connecting lines
        for cidx in idx:
            if cidx not in added_edges:
                continue
            other = added_edges[cidx]
            if cidx > other:  # Draw each edge only once
                continue

            pointA = (endpoints[cidx, 1] - min_x, endpoints[cidx, 0] - min_y)
            pointB = (endpoints[other, 1] - min_x, endpoints[other, 0] - min_y)
            colA, colB = endpoints_color[cidx], endpoints_color[other]

            new_fiber = cv2.line(
                new_fiber,
                pointA,
                pointB,
                color=2 if colA != colB else int(colA),
                thickness=1,
            )

        bbox = Bbox(
            x=coordinates[0] + min_x,
            y=coordinates[1] + min_y,
            width=max_x - min_x,
            height=max_y - min_y,
        )
        results.append(FiberProps(bbox=bbox, data=new_fiber))

    return results


def handle_ccs_with_junctions(
    ccs: List[np.ndarray],
    junctions: List[List[Tuple[int, int]]],
    coordinates: List[Tuple[int, int]],
    fiber_width=3,
):
    """
    Handle the connected components with junctions.
    The function takes a list of connected components, a list of list of junctions and a list of coordinates.
    The junctions
    The coordinates corresponds to the top left corner of the connected component.
    """
    jncts_fibers = []
    for fiber, junction, coordinate in zip(ccs, junctions, coordinates):
        jncts_fibers += handle_multiple_fiber_in_cc(
            fiber, junction, coordinate, fiber_width=fiber_width
        )

    return jncts_fibers


def extract_fibers(
    mask,
    post_process=True,
    x_offset: int = 0,
    y_offset: int = 0,
    endpoint_correction: bool = True,
) -> Fibers:
    retval, labels, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(
        (mask > 0).astype(np.uint8),
        connectivity=8,
        ccltype=cv2.CCL_BOLELLI,
        ltype=cv2.CV_16U,
    )

    bboxes = stats[
        :,
        [
            cv2.CC_STAT_LEFT,
            cv2.CC_STAT_TOP,
            cv2.CC_STAT_WIDTH,
            cv2.CC_STAT_HEIGHT,
        ],
    ]

    local_fibers = []
    coordinates = []
    junctions = []
    endpoint_corrections = []  # Store correction values
    for i in range(1, retval):
        bbox = bboxes[i]
        x1, y1, w, h = bbox
        local_mask = mask[y1 : y1 + h, x1 : x1 + w]
        local_label = (labels[y1 : y1 + h, x1 : x1 + w] == i).astype(np.uint8)
        local_fiber = local_mask * local_label
        local_fiber_binary = local_fiber > 0
        dist_transform = distance_transform_edt(local_fiber_binary)
        local_binary_fiber = skeletonize(local_fiber_binary).astype(np.uint8)
        local_fiber = local_binary_fiber * local_fiber
        local_fibers.append(local_fiber)
        coordinates.append(np.asarray([x1, y1, w, h]))

        # Calculate endpoint correction
        correction = 0
        if local_label.any():
            # Find endpoints in local coordinates
            local_skel = (local_binary_fiber > 0).astype(np.uint8)
            endpoints = find_endpoints(local_skel)

            if len(endpoints) > 0 and endpoint_correction:
                # Compute the average distance at endpoints
                distances = []
                for y, x in endpoints:
                    distances.append(2 * dist_transform[y, x])

                if len(distances) > 0:
                    correction = int(np.mean(distances))

        endpoint_corrections.append(correction)

        local_junctions = find_line_intersection(local_binary_fiber)
        local_junctions = np.where(local_junctions)
        local_junctions = np.array(local_junctions).transpose()
        junctions.append(local_junctions)

    fiberprops = []
    if post_process:
        has_junctions = [len(j) > 0 for j in junctions]

        for i, (fiber, coordinate, correction) in enumerate(
            zip(
                compress(local_fibers, np.logical_not(has_junctions)),
                compress(coordinates, np.logical_not(has_junctions)),
                compress(endpoint_corrections, np.logical_not(has_junctions)),
            )
        ):
            bbox = Bbox(
                x=coordinate[0],
                y=coordinate[1],
                width=coordinate[2],
                height=coordinate[3],
            )
            fiberprops.append(
                FiberProps(
                    bbox=bbox,
                    data=fiber,
                    fiber_id=i,
                    endpoint_correction=correction,
                )
            )
        # Handle fibers with junctions
        try:
            fiberprops += handle_ccs_with_junctions(
                compress(local_fibers, has_junctions),
                compress(junctions, has_junctions),
                compress(coordinates, has_junctions),
                fiber_width=3 if not endpoint_correction else correction,
            )
        except (IndexError, ValueError):
            # If there is an IndexError, it means that there are no fibers with junctions
            pass
    else:
        for i, (fiber, coordinate, correction) in enumerate(
            zip(local_fibers, coordinates, endpoint_corrections)
        ):
            bbox = Bbox(
                x=coordinate[0],
                y=coordinate[1],
                width=coordinate[2],
                height=coordinate[3],
            )
            fiberprops.append(
                FiberProps(bbox=bbox, data=fiber, endpoint_correction=correction)
            )

    for fiber in fiberprops:
        fiber.bbox.x += x_offset
        fiber.bbox.y += y_offset

    return Fibers(fiberprops)


def refine_segmentation(
    image,
    segmentation,
    x_offset=0,
    y_offset=0,
    correction_model=None,
    device=None,
    verbose=False,
):
    fibers = extract_fibers(
        segmentation,
        post_process=True,
        x_offset=x_offset,
        y_offset=y_offset,
    )
    if correction_model is not None:
        fibers = correct_fibers(
            fibers,
            image,
            correction_model=correction_model,
            device=device,
            verbose=verbose,
        )

    fibers = Fibers(fibers=fibers)
    # We set an id to each fiber
    for i, fiber in enumerate(fibers.fibers):
        fiber.fiber_id = i + 1
    return fibers
