from __future__ import annotations

import json
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dnafiber.postprocess.fiber import FiberProps


def generate_svg(fiber: FiberProps, scale=1.0) -> str:
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
            colors.append("red" if current_color == 1 else "green")

            # Start new line
            current_color = color
            current_line = [(x, y)]
        else:
            # Same color - add point if far enough from last point (for simplification)
            # Or always add if you want full resolution
            dist = (
                (x - current_line[-1][0]) ** 2 + (y - current_line[-1][1]) ** 2
            ) ** 0.5
            if dist > 5 or j == len(data) - 1:
                current_line.append((x, y))

    # Don't forget the last segment
    if current_line:
        traces_polylines.append(
            " ".join(
                f"{int((px + offset_x) * scale)},{int((py + offset_y) * scale)}"
                for px, py in current_line
            )
        )
        colors.append("red" if current_color == 1 else "green")

    bbox_data["points"] = traces_polylines
    bbox_data["colors"] = colors
    bbox_data["x"] = int(bbox_data["x"] * scale)
    bbox_data["y"] = int(bbox_data["y"] * scale)
    bbox_data["width"] = int(bbox_data["width"] * scale)
    bbox_data["height"] = int(bbox_data["height"] * scale)
    bbox_data["fiber_id"] = fiber.fiber_id
    bbox_data["type"] = fiber.fiber_type
    bbox_data["ratio"] = fiber.ratio if not np.isnan(fiber.ratio) else -1
    bbox_data["is_error"] = bool(fiber.is_an_error)

    return json.dumps(bbox_data)
