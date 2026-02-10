from __future__ import annotations

import json
import numpy as np
from typing import TYPE_CHECKING
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from dnafiber.postprocess.fiber import FiberProps, Fibers


def generate_svg(fiber: FiberProps, scale=1.0, color1="red", color2="green") -> str:
    bbox_data = fiber.bbox.to_dict()
    trace_data = fiber.get_trace()
    offset_x, offset_y = bbox_data["x"], bbox_data["y"]
    data = fiber.data[trace_data[:, 0], trace_data[:, 1]]

    traces_polylines = []
    colors = []

    if len(data) == 0:
        return None

    current_color = data[0]
    current_line = [(trace_data[0, 1], trace_data[0, 0])]

    for j in range(1, len(data)):  # Start from 1 since we already added point 0
        color = data[j]
        x, y = trace_data[j, 1], trace_data[j, 0]

        if color != current_color:
            # Close the previous path with the last point before color change
            # Finalize current polyline
            traces_polylines.append(
                " ".join(
                    f"{int((px + offset_x) * scale)},{int((py + offset_y) * scale)}"
                    for px, py in current_line
                )
            )
            colors.append(color1 if current_color == 1 else color2)

            # Start new line
            current_color = color
            current_line = [(x, y)]
        else:
            # if dist > 5 or j == len(data) - 1:
            current_line.append((x, y))

    # Don't forget the last segment
    if current_line:
        traces_polylines.append(
            " ".join(
                f"{int((px + offset_x) * scale)},{int((py + offset_y) * scale)}"
                for px, py in current_line
            )
        )
        colors.append(color1 if current_color == 1 else color2)

    bbox_data["points"] = traces_polylines
    bbox_data["colors"] = colors
    bbox_data["x"] = int(bbox_data["x"] * scale)
    bbox_data["y"] = int(bbox_data["y"] * scale)
    bbox_data["width"] = int(bbox_data["width"] * scale)
    bbox_data["height"] = int(bbox_data["height"] * scale)
    bbox_data["fiber_id"] = fiber.fiber_id
    bbox_data["type"] = fiber.fiber_type.value
    bbox_data["ratio"] = fiber.ratio if not np.isnan(fiber.ratio) else -1
    bbox_data["proba_error"] = (
        fiber.proba_error if not np.isnan(fiber.proba_error) else -1
    )

    return json.dumps(bbox_data)


def match_fibers_pairs(
    fibers1: Fibers, fibers2: Fibers, overlap_ratio=10
) -> list[tuple[FiberProps, FiberProps]]:
    n, m = len(fibers1), len(fibers2)
    if n == 0 or m == 0:
        return []

    # Build IoU matrix (or use negative for cost)
    iou_matrix = np.zeros((n, m))
    for i, fiber in enumerate(fibers1):
        for j, other_fiber in enumerate(fibers2):
            iou_matrix[i, j] = fiber.bbox_iou(other_fiber)

    # Hungarian on negative IoU (minimize cost = maximize IoU)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    pairs = []
    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] >= overlap_ratio:  # Only keep pairs above threshold
            pairs.append((fibers1[i], fibers2[j]))

    return pairs
