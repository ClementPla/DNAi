# Functions to generate kernels of curve intersection
import numpy as np
import cv2
import itertools
from numba import njit, int64
from numba.typed import List
from numba.types import Tuple

import numba

from numba import njit, prange
from numba.typed import List

# Define tuple type once at module level
tuple_type = numba.types.UniTuple(numba.types.int64, 2)
# Define the element type: a tuple of two int64
tuple_type = Tuple((int64, int64))


def find_neighbours(fibers_map, point):
    """
    Find the next point in the fiber starting from the given point.
    The function returns None if the point is not in the fiber.
    """
    # Get the fiber id
    neighbors = []
    h, w = fibers_map.shape
    for i in range(-1, 2):
        for j in range(-1, 2):
            # Skip the center point
            if i == 0 and j == 0:
                continue
            # Get the next point
            nextpoint = (point[0] + i, point[1] + j)
            # Check if the next point is in the image
            if (
                nextpoint[0] < 0
                or nextpoint[0] >= h
                or nextpoint[1] < 0
                or nextpoint[1] >= w
            ):
                continue

            # Check if the next point is in the fiber
            if fibers_map[nextpoint]:
                neighbors.append(nextpoint)
    return neighbors


@njit(inline="always")
def get_neighbors_8_inline(y, x, shape_y, shape_x):
    """Inlined neighbor generation to avoid function call overhead."""
    neighbors = List.empty_list(tuple_type)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < shape_y and 0 <= nx < shape_x:
                neighbors.append((ny, nx))
    return neighbors


@njit
def trace_from_point_optimized(skel, point, max_length=25):
    """Optimized tracing with pre-allocated structures."""
    y, x = point
    shape_y, shape_x = skel.shape

    # Early exit
    if y < 0 or y >= shape_y or x < 0 or x >= shape_x or skel[y, x] != 1:
        return List.empty_list(tuple_type)

    visited = np.zeros((shape_y, shape_x), dtype=np.uint8)
    path = List.empty_list(tuple_type)

    # Use a simple array-based stack instead of List for better performance
    stack_y = np.empty(max_length * 8, dtype=np.int64)
    stack_x = np.empty(max_length * 8, dtype=np.int64)
    stack_ptr = 0

    stack_y[0] = y
    stack_x[0] = x
    stack_ptr = 1

    while stack_ptr > 0 and len(path) < max_length:
        stack_ptr -= 1
        cy = stack_y[stack_ptr]
        cx = stack_x[stack_ptr]

        if visited[cy, cx]:
            continue
        visited[cy, cx] = 1
        path.append((cy, cx))

        # Inline neighbor iteration
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < shape_y and 0 <= nx < shape_x:
                    if skel[ny, nx] == 1 and visited[ny, nx] == 0:
                        stack_y[stack_ptr] = ny
                        stack_x[stack_ptr] = nx
                        stack_ptr += 1

    return path


@njit
def fit_line_simple(points_y, points_x):
    """
    Simple linear regression to find line direction.
    Returns (vx, vy) normalized direction vector.
    """
    n = len(points_y)
    if n < 2:
        return 1.0, 0.0

    # Compute means
    mean_x = 0.0
    mean_y = 0.0
    for i in range(n):
        mean_x += points_x[i]
        mean_y += points_y[i]
    mean_x /= n
    mean_y /= n

    # Compute covariance matrix elements
    cxx = 0.0
    cyy = 0.0
    cxy = 0.0
    for i in range(n):
        dx = points_x[i] - mean_x
        dy = points_y[i] - mean_y
        cxx += dx * dx
        cyy += dy * dy
        cxy += dx * dy

    # Principal direction via eigenvalue analysis of 2x2 covariance matrix
    # For 2x2 symmetric matrix [[cxx, cxy], [cxy, cyy]]
    # Eigenvector for larger eigenvalue gives principal direction

    diff = cxx - cyy
    trace = cxx + cyy

    if trace < 1e-10:
        return 1.0, 0.0

    discriminant = np.sqrt(diff * diff + 4 * cxy * cxy)

    # Eigenvector for larger eigenvalue
    if abs(cxy) > 1e-10:
        lambda1 = (trace + discriminant) / 2
        vx = lambda1 - cyy
        vy = cxy
    elif cxx >= cyy:
        vx = 1.0
        vy = 0.0
    else:
        vx = 0.0
        vy = 1.0

    # Normalize
    norm = np.sqrt(vx * vx + vy * vy)
    if norm > 1e-10:
        vx /= norm
        vy /= norm

    return vx, vy


@njit(parallel=False)  # Set parallel=True if you have many points
def compute_points_angle_numba(binary_map, points, steps=25, oriented=False):
    """
    Fully JIT-compiled angle computation.

    Args:
        binary_map: boolean or uint8 array where 1 = fiber
        points: (N, 2) array of (y, x) coordinates
        steps: number of pixels to trace
        oriented: whether to compute oriented angles

    Returns:
        (N,) array of angles in radians
    """
    n_points = len(points)
    angles = np.zeros(n_points, dtype=np.float32)

    for i in range(n_points):
        point = (points[i, 0], points[i, 1])
        path = trace_from_point_optimized(binary_map, point, steps)

        if len(path) < 2:
            angles[i] = 0.0
            continue

        # Extract path coordinates
        path_y = np.empty(len(path), dtype=np.float64)
        path_x = np.empty(len(path), dtype=np.float64)
        for j in range(len(path)):
            path_y[j] = path[j][0]
            path_x[j] = path[j][1]

        # Fit line
        vx, vy = fit_line_simple(path_y, path_x)

        if oriented:
            # Compute mean position
            mean_x = 0.0
            for j in range(len(path)):
                mean_x += path_x[j]
            mean_x /= len(path)

            angle = np.arctan2(vy, vx)
            if mean_x > points[i, 1]:
                angle -= np.pi
            angles[i] = angle
        else:
            angles[i] = np.arctan2(vy, vx)

    return angles


# Wrapper to handle input conversion
def compute_points_angle(fibers_map, points, steps=25, oriented=False):
    """
    Optimized angle computation using Numba.
    """
    binary_map = (fibers_map > 0).astype(np.uint8)
    points_arr = np.asarray(points, dtype=np.int64)

    if points_arr.ndim == 1:
        points_arr = points_arr.reshape(1, -1)

    return compute_points_angle_numba(binary_map, points_arr, steps, oriented)


def generate_nonadjacent_combination(input_list, take_n):
    all_comb = []
    for comb in itertools.combinations(input_list, take_n):
        comb = np.array(comb)
        d = np.diff(comb)
        # Check no adjacent elements AND no wraparound adjacency (0 and 7)
        if np.all(d != 1) and not (0 in comb and 7 in comb):
            all_comb.append(comb)
    return all_comb


def populate_intersection_kernel(combinations):
    template = np.array(([0, 0, 0], [0, 1, 0], [0, 0, 0]), dtype="int")
    match = [(0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0)]
    kernels = []

    for comb in combinations:
        tmp = np.copy(template)
        for m in comb:
            tmp[match[m][0], match[m][1]] = 1
        kernels.append(tmp)

    return kernels


def give_intersection_kernels():
    """
    Generates all the intersection kernels in a 9x9 matrix.
    INPUT:
        None
    OUTPUT:
        kernels =      (List) list of 9 x 9 kernels/masks. each element is a mask.
    """
    input_list = np.arange(8)
    taken_n = [4, 3]
    kernels = []
    for taken in taken_n:
        comb = generate_nonadjacent_combination(input_list, taken)
        tmp_ker = populate_intersection_kernel(comb)
        kernels.extend(tmp_ker)
    return kernels


def find_line_intersection(input_image, show=0):
    """
    Applies morphologyEx with parameter HitsMiss to look for all the curve
    intersection kernels generated with give_intersection_kernels() function.
    INPUT:
        input_image =  (np.array dtype=np.uint8) binarized m x n image matrix
    OUTPUT:
        output_image = (np.array dtype=np.uint8) image where the nonzero pixels
                        are the line intersection.
    """
    input_image = input_image.astype(np.uint8)
    kernel = np.array(give_intersection_kernels())
    output_image = np.zeros(input_image.shape)
    for i in np.arange(len(kernel)):
        out = cv2.morphologyEx(
            input_image,
            cv2.MORPH_HITMISS,
            kernel[i, :, :],
            borderValue=0,
            borderType=cv2.BORDER_CONSTANT,
        )
        output_image = output_image + out

    return output_image


@njit
def get_neighbors_8(y, x, shape):
    neighbors = List.empty_list(tuple_type)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
                neighbors.append((ny, nx))
    return neighbors


@njit
def find_endpoints(skel):
    endpoints = List.empty_list(tuple_type)
    for y in range(skel.shape[0]):
        for x in range(skel.shape[1]):
            if skel[y, x] == 1:
                count = 0
                neighbors = get_neighbors_8(y, x, skel.shape)
                for ny, nx in neighbors:
                    if skel[ny, nx] == 1:
                        count += 1
                if count == 1:
                    endpoints.append((y, x))

    return endpoints


@njit
def trace_skeleton(skel):
    endpoints = find_endpoints(skel)
    if len(endpoints) < 1:
        return List.empty_list(tuple_type)  # Return empty list with proper type

    return trace_from_point_smooth(skel, endpoints[0], max_length=skel.sum())


@njit
def trace_from_point_smooth(skel, point, max_length=25):
    """Trace a continuous path, preferring straight-line continuation."""
    path = List.empty_list(tuple_type)

    y, x = point
    shape_y, shape_x = skel.shape

    if y < 0 or y >= shape_y or x < 0 or x >= shape_x or skel[y, x] != 1:
        return path

    visited = np.zeros((shape_y, shape_x), dtype=np.uint8)

    cy, cx = y, x
    prev_dy, prev_dx = 0, 0  # No previous direction initially

    while len(path) < max_length:
        visited[cy, cx] = 1
        path.append((cy, cx))

        # Collect all valid neighbors
        neighbors_y = np.empty(8, dtype=np.int64)
        neighbors_x = np.empty(8, dtype=np.int64)
        neighbors_dy = np.empty(8, dtype=np.int64)
        neighbors_dx = np.empty(8, dtype=np.int64)
        n_neighbors = 0

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < shape_y and 0 <= nx < shape_x:
                    if skel[ny, nx] == 1 and visited[ny, nx] == 0:
                        neighbors_y[n_neighbors] = ny
                        neighbors_x[n_neighbors] = nx
                        neighbors_dy[n_neighbors] = dy
                        neighbors_dx[n_neighbors] = dx
                        n_neighbors += 1

        if n_neighbors == 0:
            break

        # Pick neighbor that best continues the previous direction
        best_idx = 0
        if n_neighbors > 1 and (prev_dy != 0 or prev_dx != 0):
            best_score = -999.0
            for i in range(n_neighbors):
                # Dot product with previous direction (higher = more aligned)
                score = prev_dy * neighbors_dy[i] + prev_dx * neighbors_dx[i]
                if score > best_score:
                    best_score = score
                    best_idx = i

        # Move to best neighbor
        prev_dy = neighbors_dy[best_idx]
        prev_dx = neighbors_dx[best_idx]
        cy = neighbors_y[best_idx]
        cx = neighbors_x[best_idx]

    return path


@njit
def trace_from_point(skel, point, max_length=25):
    visited = np.zeros_like(skel, dtype=np.uint8)
    path = List.empty_list(tuple_type)

    # Check if the starting point is on the skeleton
    y, x = point
    if y < 0 or y >= skel.shape[0] or x < 0 or x >= skel.shape[1] or skel[y, x] != 1:
        return path

    stack = List.empty_list(tuple_type)
    stack.append(point)

    while len(stack) > 0 and len(path) < max_length:
        y, x = stack.pop()
        if visited[y, x]:
            continue
        visited[y, x] = 1
        path.append((y, x))
        neighbors = get_neighbors_8(y, x, skel.shape)
        for ny, nx in neighbors:
            if skel[ny, nx] == 1 and not visited[ny, nx]:
                stack.append((ny, nx))
    return path


@njit(locals={"difference": numba.float32})
def follow_along_direction_until_change(
    start_point, start_color, angle, image, threshold, max_length=25
):
    """
    Follow the fiber along the direction of the start point until the color changes significantly.
    Returns the maximum step.
    """
    # Convert start_point to a tuple of integers to ensure type compatibility
    start_point = (int(start_point[0]), int(start_point[1]))
    y, x = start_point
    # Explore the image in the direction of the with a cone

    cone_angle = np.deg2rad(5)  # Angle of the cone in radians

    path = List.empty_list(tuple_type)

    offset_angles = np.linspace(0, cone_angle, num=10)
    all_angles = np.concatenate(
        (angle + offset_angles, angle - offset_angles[1:])
    )  # Add negative angles for symmetry
    for step in range(1, max_length):
        found_continuity = False
        for alpha in all_angles:
            new_y = int(start_point[0] + step * np.sin(alpha))
            new_x = int(start_point[1] + step * np.cos(alpha))

            while abs(new_y - y) > 1:
                new_y += -1 if new_y > y else 1
            while abs(new_x - x) > 1:
                new_x += -1 if new_x > x else 1

            # Check if the point is out of bounds
            if (
                new_y < 0
                or new_y >= image.shape[0]
                or new_x < 0
                or new_x >= image.shape[1]
            ):
                return path
            # Look up the color at a cone

            current_color = image[new_y, new_x].astype(np.float32)

            if current_color.any() == 0:
                continue

            difference = np.sqrt(np.sum((current_color - start_color) ** 2)) / np.sqrt(
                np.sum(start_color**2)
            )
            if difference < threshold:
                found_continuity = True
                path.append((new_y, new_x))
                y, x = new_y, new_x

                break

        if not found_continuity:
            return path

    return path


@njit
def fill_path(image, path, value):
    for point in path:
        y, x = point
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            image[y, x] = value


def prolongate_endpoints(image, skeleton, segmentation, max_search=75, threshold=0.1):
    """
    Estimate the orientation of the fibers and prolongate the endpoints
    based on the skeleton if the difference in color in the image is not significant.
    This is to avoid a segmentation too short.
    """

    endpoints = np.asarray(find_endpoints(skeleton))
    if len(endpoints) == 0:
        return segmentation, skeleton

    points_angle = compute_points_angle(skeleton, endpoints, steps=200, oriented=True)

    for i, (point, angle) in enumerate(zip(endpoints, points_angle)):
        # Prolongate the endpoint in the direction of the angle
        y, x = point
        label = int(segmentation[y, x])

        # Extract the bounding box of the image (max_search pixels in each direction)
        y_min = max(0, y - max_search)
        y_max = min(image.shape[0], y + max_search)
        x_min = max(0, x - max_search)
        x_max = min(image.shape[1], x + max_search)

        bbox = image[y_min:y_max, x_min:x_max]

        # Local thresholding
        if bbox.size == 0:
            continue
        bbox = cv2.GaussianBlur(bbox, None, sigmaX=1.5, sigmaY=1.5)
        # threshold_value = threshold_otsu(bbox)
        # # Apply thresholding to the bounding box
        # bbox[bbox < threshold_value] = 0

        # Express the start point in the local coordinate system of the bounding box
        start_color = bbox[y - y_min, x - x_min]

        start_point = (y - y_min, x - x_min)

        path = follow_along_direction_until_change(
            start_point,
            start_color,
            angle,
            bbox.astype(np.float32),
            threshold=threshold,
            max_length=max_search,
        )

        if len(path) > 0:
            # Express the path in the global coordinate system
            path = [(y + y_min, x + x_min) for (y, x) in path]
            fill_path(segmentation, path, label)

    return segmentation, (segmentation > 0).astype(np.uint8)
