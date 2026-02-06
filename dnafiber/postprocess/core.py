import numpy as np
import cv2
from typing import List, Tuple
from dnafiber.error_detection.inference import detect_error
from dnafiber.postprocess.skan import find_endpoints, compute_points_angle
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_array
from skimage.morphology import skeletonize
from dnafiber.postprocess.skan import find_line_intersection
from dnafiber.postprocess.fiber import FiberProps, Bbox, Fibers
from itertools import compress
from scipy.cluster.hierarchy import fcluster, linkage


MIN_ANGLE = 45
MIN_BRANCH_LENGTH = 1
MIN_BRANCH_DISTANCE = 100

MIN_BRANCH_KEEP = 15


def handle_multiple_fiber_in_cc(
    fiber,
    junctions_fiber,
    coordinates,
    fiber_width=3,
    # Cost weights
    weight_distance=0.5,
    weight_angle=1.0,
    weight_color=1.0,
    # Normalization factors
    max_distance=None,
    max_angle_deviation=90.0,
    junction_cluster_distance=10,
    debug=True,
):
    if max_distance is None:
        max_distance = fiber_width * 2.5

    # 1. Cluster nearby junctions
    junctions_fiber = cluster_junctions(
        junctions_fiber, min_distance=junction_cluster_distance
    )

    if len(junctions_fiber) == 0:
        return []

    # 2. Erase all junctions
    working = fiber.copy()
    for y, x in junctions_fiber:
        cv2.circle(working, (x, y), fiber_width, color=0, thickness=-1)

    # 3. Label branches
    num_branches, branches, stats, _ = cv2.connectedComponentsWithStatsWithAlgorithm(
        (working > 0).astype(np.uint8),
        connectivity=8,
        ccltype=cv2.CCL_BOLELLI,
        ltype=cv2.CV_16U,
    )

    if num_branches <= 1:
        return []

    branch_bboxes = stats[
        :, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]
    ]
    branch_areas = stats[:, cv2.CC_STAT_AREA]

    valid_branches = np.where(branch_areas >= MIN_BRANCH_LENGTH)[0]
    valid_branches = valid_branches[valid_branches != 0]

    if len(valid_branches) == 0:
        return []

    # 4. Find endpoints
    endpoints = find_endpoints(working > 0)
    endpoints = np.asarray(endpoints)

    if len(endpoints) == 0:
        return []

    junctions_arr = np.asarray(junctions_fiber)
    dist_to_junctions = cdist(endpoints, junctions_arr, metric="euclidean")

    endpoint_branch = branches[endpoints[:, 0], endpoints[:, 1]]
    endpoint_color = working[endpoints[:, 0], endpoints[:, 1]]
    endpoint_angle = compute_points_angle(working, endpoints, steps=15, oriented=True)

    # 5. Build branch adjacency by processing each junction locally
    adjacency_pairs = []
    reconnection_lines = []  # (endpoint_i, endpoint_j) for matched pairs
    endpoint_to_junction = []  # (endpoint_i, junction_idx) for T-junction tips

    for junc_idx in range(len(junctions_fiber)):
        distances = dist_to_junctions[:, junc_idx]
        close_mask = distances < fiber_width * 3
        valid_mask = close_mask & np.isin(endpoint_branch, valid_branches)

        candidate_indices = np.where(valid_mask)[0]

        if len(candidate_indices) < 2:
            # Single endpoint: it's a T-junction tip, connect to junction
            if len(candidate_indices) == 1:
                endpoint_to_junction.append((candidate_indices[0], junc_idx))
            continue

        # Keep only closest endpoint per branch
        branches_seen = {}
        for idx in candidate_indices:
            br = endpoint_branch[idx]
            dist = distances[idx]
            if br not in branches_seen or dist < branches_seen[br][1]:
                branches_seen[br] = (idx, dist)

        local_indices = np.array([v[0] for v in branches_seen.values()])
        n = len(local_indices)

        if n < 2 or n > 4:
            continue

        # Find best pairing
        pairs = match_junction_endpoints(
            local_indices,
            endpoints,
            endpoint_angle,
            endpoint_color,
            weight_distance,
            weight_angle,
            weight_color,
            max_distance,
            max_angle_deviation,
        )

        # Track matched endpoints
        matched = set()
        for i, j in pairs:
            bi, bj = endpoint_branch[i], endpoint_branch[j]
            if bi != bj:
                adjacency_pairs.append((bi, bj))
            reconnection_lines.append((i, j))
            matched.add(i)
            matched.add(j)

        # T-junction: unmatched endpoints connect to junction center
        for idx in local_indices:
            if idx not in matched:
                endpoint_to_junction.append((idx, junc_idx))

    # 6. Build graph and find connected components
    branch_to_idx = {b: i for i, b in enumerate(valid_branches)}
    num_valid = len(valid_branches)

    rows, cols = [], []
    for i in range(num_valid):
        rows.append(i)
        cols.append(i)

    for bi, bj in adjacency_pairs:
        if bi in branch_to_idx and bj in branch_to_idx:
            i, j = branch_to_idx[bi], branch_to_idx[bj]
            rows.extend([i, j])
            cols.extend([j, i])

    adj_matrix = csr_array(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(num_valid, num_valid)
    )

    n_fibers, fiber_labels = connected_components(
        adj_matrix, directed=False, return_labels=True
    )

    # 7. Build FiberProps for each fiber
    results = []

    for fiber_id in range(n_fibers):
        branch_indices = np.where(fiber_labels == fiber_id)[0]
        fiber_branch_ids = valid_branches[branch_indices]

        # Start with branch bounding boxes
        bboxes = branch_bboxes[fiber_branch_ids]
        min_x = np.min(bboxes[:, 0])
        min_y = np.min(bboxes[:, 1])
        max_x = np.max(bboxes[:, 0] + bboxes[:, 2])
        max_y = np.max(bboxes[:, 1] + bboxes[:, 3])

        # Expand bbox to include junction centers for T-junctions in this fiber
        for ep_idx, junc_idx in endpoint_to_junction:
            if endpoint_branch[ep_idx] in fiber_branch_ids:
                jy, jx = junctions_arr[junc_idx]
                min_x = min(min_x, jx)
                min_y = min(min_y, jy)
                max_x = max(max_x, jx + 1)
                max_y = max(max_y, jy + 1)

        branch_mask = np.isin(branches[min_y:max_y, min_x:max_x], fiber_branch_ids)
        new_fiber = branch_mask * working[min_y:max_y, min_x:max_x]

        # Draw lines between matched endpoints
        for i, j in reconnection_lines:
            bi, bj = endpoint_branch[i], endpoint_branch[j]
            if bi in fiber_branch_ids or bj in fiber_branch_ids:
                pt1 = (endpoints[i, 1] - min_x, endpoints[i, 0] - min_y)
                pt2 = (endpoints[j, 1] - min_x, endpoints[j, 0] - min_y)
                col_i, col_j = endpoint_color[i], endpoint_color[j]
                color = 2 if col_i != col_j else int(col_i)
                new_fiber = cv2.line(new_fiber, pt1, pt2, color=color, thickness=1)

        # Draw lines from T-junction tips to junction center
        for ep_idx, junc_idx in endpoint_to_junction:
            if endpoint_branch[ep_idx] in fiber_branch_ids:
                pt_ep = (endpoints[ep_idx, 1] - min_x, endpoints[ep_idx, 0] - min_y)
                pt_junc = (
                    junctions_arr[junc_idx, 1] - min_x,
                    junctions_arr[junc_idx, 0] - min_y,
                )
                color = int(endpoint_color[ep_idx])
                new_fiber = cv2.line(
                    new_fiber, pt_ep, pt_junc, color=color, thickness=1
                )

        bbox = Bbox(
            x=coordinates[0] + min_x,
            y=coordinates[1] + min_y,
            width=max_x - min_x,
            height=max_y - min_y,
        )
        new_fiber_props = FiberProps(bbox=bbox, data=new_fiber)
        # if the new fiber is not single color and longer than MIN_BRANCH_LENGTH, keep it
        if (
            new_fiber_props.category != "single"
            and new_fiber_props.length >= MIN_BRANCH_KEEP
        ):
            results.append(new_fiber_props)

    return results


def cluster_junctions(junctions, min_distance=10):
    """Merge junctions that are too close together."""
    if len(junctions) <= 1:
        return np.asarray(junctions)

    junctions = np.asarray(junctions)

    Z = linkage(junctions, method="single")
    labels = fcluster(Z, t=min_distance, criterion="distance")

    clustered = []
    for label in np.unique(labels):
        mask = labels == label
        centroid = junctions[mask].mean(axis=0).astype(int)
        clustered.append(centroid)

    return np.array(clustered)


def match_junction_endpoints(
    local_indices,
    endpoints,
    angles,
    colors,
    weight_distance,
    weight_angle,
    weight_color,
    max_distance,
    max_angle_deviation,
):
    """Match endpoints at a single junction. Returns list of (global_idx_i, global_idx_j) pairs."""
    n = len(local_indices)

    def pair_cost(i, j):
        gi, gj = local_indices[i], local_indices[j]

        # Distance
        dist = np.linalg.norm(endpoints[gi] - endpoints[gj])
        dist_cost = min(dist / max_distance, 1.0)

        # Angle: good match = ~180° apart
        angle_diff = np.abs(np.rad2deg(angles[gi] - angles[gj]))
        angle_diff = min(angle_diff, 360 - angle_diff)
        angle_deviation = np.abs(angle_diff - 180)
        angle_cost = min(angle_deviation / max_angle_deviation, 1.0)

        # Color
        color_cost = 0.0 if colors[gi] == colors[gj] else 1.0

        return (
            weight_distance * dist_cost
            + weight_angle * angle_cost
            + weight_color * color_cost
        )

    if n == 2:
        return [(local_indices[0], local_indices[1])]

    elif n == 3:
        # T-junction: find best single pair
        candidates = [(0, 1), (0, 2), (1, 2)]
        best = min(candidates, key=lambda p: pair_cost(p[0], p[1]))
        return [(local_indices[best[0]], local_indices[best[1]])]

    elif n == 4:
        # Crossing: find best pairing (2 pairs)
        candidates = [
            [(0, 1), (2, 3)],
            [(0, 2), (1, 3)],
            [(0, 3), (1, 2)],
        ]
        best = min(candidates, key=lambda ps: pair_cost(*ps[0]) + pair_cost(*ps[1]))
        return [
            (local_indices[best[0][0]], local_indices[best[0][1]]),
            (local_indices[best[1][0]], local_indices[best[1][1]]),
        ]

    return []


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
        # Find endpoints in local coordinates
        endpoints = find_endpoints(local_binary_fiber)
        if endpoint_correction:
            # Compute the average distance at endpoints
            distances = []
            for y, x in endpoints:
                distances.append(2 * dist_transform[y, x])

            if distances:
                correction = int(np.mean(distances))

        endpoint_corrections.append(correction)

        local_junctions = find_line_intersection(local_binary_fiber)
        local_junctions = np.where(local_junctions)
        local_junctions = np.array(local_junctions).transpose()
        junctions.append(local_junctions)

    fiberprops = []
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

    if endpoint_corrections:
        fiber_width = int(np.mean(endpoint_corrections))
    else:
        fiber_width = 3
    # Handle fibers with junctions
    try:
        fiberprops += handle_ccs_with_junctions(
            compress(local_fibers, has_junctions),
            compress(junctions, has_junctions),
            compress(coordinates, has_junctions),
            fiber_width=fiber_width,
        )
    except (IndexError, ValueError):
        # If there is an IndexError, it means that there are no fibers with junctions
        pass

    for fiber in fiberprops:
        fiber.bbox.x += x_offset
        fiber.bbox.y += y_offset

    return Fibers(fiberprops)


def refine_segmentation(
    segmentation,
    x_offset=0,
    y_offset=0,
) -> Fibers:
    fibers = extract_fibers(
        segmentation,
        x_offset=x_offset,
        y_offset=y_offset,
    )
    # We set an id to each fiber
    for i, fiber in enumerate(fibers):
        fiber.fiber_id = i + 1

    return fibers
