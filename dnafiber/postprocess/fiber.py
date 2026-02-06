import attrs
from matplotlib.collections import LineCollection
import numpy as np
from typing import Optional, Tuple
from dnafiber.postprocess.skan import trace_skeleton, compute_trace_curvature
from skimage.segmentation import expand_labels
from dnafiber.postprocess.utils import generate_svg
import pickle
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from dnafiber.data.consts import CMAP
from scipy.optimize import linear_sum_assignment


@attrs.define
class Bbox:
    x: int
    y: int
    width: int
    height: int

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    @bbox.setter
    def bbox(self, value: Tuple[int, int, int, int]):
        self.x, self.y, self.width, self.height = value

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def to_dict(self):
        return {
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
        }

    def __getitem__(self, index):
        return self.bbox[index]


@attrs.define
class FiberProps:
    bbox: Bbox
    data: (
        np.ndarray
    )  # 2D array representing the fiber pixels (0: background, 1: red, 2: green)
    fiber_id: int = -1
    red_pixels: int = None
    green_pixels: int = None
    category: str = None
    is_an_error: bool = False
    svg_rep: str = None  # SVG representation of the fiber, for visualization purposes
    trace: np.ndarray = None  # Coordinates of the skeletons of the fiber
    endpoint_correction: float = 0.0  # NEW: pixels to add for endpoint caps
    curvature: float = None  # Cached curvature value

    @property
    def bbox(self):
        return self.bbox.bbox

    @bbox.setter
    def bbox(self, value):
        self.bbox = value

    @property
    def data(self):
        return self.data

    @data.setter
    def data(self, value):
        self.data = value

    @property
    def red(self):
        if self.red_pixels is None:
            self.red_pixels, self.green_pixels = self.counts
        return self.red_pixels

    @property
    def green(self):
        if self.green_pixels is None:
            self.red_pixels, self.green_pixels = self.counts
        return self.green_pixels

    @property
    def length(self):
        return sum(self.counts)

    @property
    def counts(self):
        if self.red_pixels is None or self.green_pixels is None:
            red_raw = np.sum(self.data == 1)
            green_raw = np.sum(self.data == 2)
            self.red_pixels = red_raw
            self.green_pixels = green_raw

        return self.red_pixels, self.green_pixels

    @property
    def fiber_type(self):
        if self.category is not None:
            return self.category
        red_pixels, green_pixels = self.counts
        if red_pixels == 0 or green_pixels == 0:
            self.category = "single"
        else:
            self.category = estimate_fiber_category(self.get_trace(), self.data)
        return self.category

    def get_curvature(self):
        if self.curvature is not None:
            return self.curvature

        trace = self.get_trace()
        if trace is None or len(trace) < 3:
            self.curvature = 0.0
        else:
            self.curvature = float(compute_trace_curvature(np.ascontiguousarray(trace)))
        return self.curvature

    @property
    def tortuosity(self) -> float:
        """
        Arc-length tortuosity: (actual_length / euclidean_distance) - 1
        """
        trace = self.get_trace()
        if trace is None or len(trace) < 2:
            return 0.0

        actual_len = len(trace)
        # Distance from first point to last point
        dy = trace[-1][0] - trace[0][0]
        dx = trace[-1][1] - trace[0][1]
        chord_len = np.sqrt(dy**2 + dx**2)

        if chord_len == 0:
            return 0.0

        return float((actual_len / chord_len) - 1.0)

    def get_mean_intensity(self, image):
        """
        Compute the mean intensity of the fiber in the given image.
        Assumes image is a 2D or 3D numpy array.
        """

        if image.ndim == 3:
            R_intensity = np.mean(image[self.data == 1, 0])
            G_intensity = np.mean(image[self.data == 2, 1])
            return np.asarray([R_intensity, G_intensity], dtype=np.float32)
        else:
            mask = self.data > 0
            intensities = image[mask]
            mean_intensity = np.mean(intensities)
        return mean_intensity

    def get_trace(self):
        if self.trace is not None:
            return self.trace
        # Generate trace if not provided
        self.trace = np.ascontiguousarray(trace_skeleton(self.data > 0))

        if not self.trace.size:
            self.trace = np.empty((0, 2), dtype=int)
        return self.trace

    @property
    def ratio(self):
        if self.red == 0:
            return np.nan
        return self.green / self.red

    @property
    def is_valid(self):
        try:
            _ = self.fiber_type
        except IndexError:
            # Happens if there is no pixel remaining for this fiber, which indicates it is invalid.
            return False

        return self.is_double or self.is_triple or self.is_more_than_triple

    @property
    def is_acceptable(self):
        return not self.is_an_error

    @property
    def is_double(self):
        return self.fiber_type == "double"

    @property
    def is_triple(self):
        return self.fiber_type in ["one-two-one", "two-one-two"]

    @property
    def is_more_than_triple(self):
        return self.fiber_type == "multiple"

    def scaled_coordinates(self, scale: float) -> Tuple[int, int]:
        """
        Scale down the coordinates of the fiber's bounding box.
        """
        x, y, width, height = self.bbox
        return (
            int(x * scale),
            int(y * scale),
            int(width * scale),
            int(height * scale),
        )

    def bbox_intersect(self, other, ratio=0.25) -> bool:
        """
        Check if the bounding boxes of two fibers intersect by at least a certain ratio.
        """
        x1, y1, w1, h1 = self.bbox
        x2, y2, w2, h2 = other.bbox

        intersection_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(
            0, min(y1 + h1, y2 + h2) - max(y1, y2)
        )
        self_area = w1 * h1
        other_area = w2 * h2
        return (
            intersection_area / float(self_area + other_area - intersection_area)
            >= ratio
        )

    def bbox_iou(self, other) -> float:
        """
        Compute the Intersection over Union (IoU) of the bounding boxes of two fibers.
        """
        x1, y1, w1, h1 = self.bbox
        x2, y2, w2, h2 = other.bbox

        intersection_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(
            0, min(y1 + h1, y2 + h2) - max(y1, y2)
        )
        self_area = w1 * h1
        other_area = w2 * h2
        union_area = self_area + other_area - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / float(union_area)

    def svg_representation(self, scale=1.0, color1="red", color2="green"):
        try:
            svg_representation = generate_svg(
                self, scale=scale, color1=color1, color2=color2
            )
        except Exception as e:
            print(f"Error generating SVG representation: {e}")
            return None
        return svg_representation

    def debug(self, img=None, scale=1.0, linewidth=2, with_features=False):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(np.zeros_like(self.data), cmap="gray")
        if img is not None:
            x, y, w, h = self.bbox
            ax.imshow(img[y : y + h, x : x + w])

        ax.imshow(self.data, cmap=CMAP, vmin=0, vmax=2)

        # Draw the skeleton
        trace = self.get_trace()
        if trace.size > 0:
            # Start color
            blue = np.array([0, 0, 1.0], dtype=np.float32)
            # End color
            white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            interpolation = np.linspace(0, 1, len(trace))
            colors = np.outer(1 - interpolation, blue) + np.outer(interpolation, white)
            # Build segments: each segment connects consecutive points
            points = np.column_stack([trace[:, 1] * scale, trace[:, 0] * scale])
            segments = np.stack([points[:-1], points[1:]], axis=1)  # (N-1, 2, 2)

            lc = LineCollection(segments, colors=colors[:-1], linewidth=linewidth)
            ax.add_collection(lc)
        ax.set_title(
            f"Fiber ID: {self.fiber_id}, Type: {self.fiber_type}, Red: {self.red}, Green: {self.green}, Ratio: {self.ratio:.2f}, Is error: {self.is_an_error}"
        )
        if with_features:
            curvature = self.get_curvature()
            tortuosity = self.tortuosity
            ax.set_xlabel(f"Curvature: {curvature:.4f}, Tortuosity: {tortuosity:.4f}")
        plt.show()


def filter_invalid_bbox(fibers: list[FiberProps]) -> list[FiberProps]:
    valid_fibers = []
    for fiber in fibers:
        x, y, w, h = fiber.bbox
        if w > 0 and h > 0 and fiber.data.size > 0 and x >= 0 and y >= 0:
            valid_fibers.append(fiber)
    return valid_fibers


@attrs.define
class Fibers:
    fibers: list[FiberProps] = attrs.field(factory=list, converter=filter_invalid_bbox)
    path: Optional[str | Path] = None

    def __iter__(self):
        return iter(self.fibers)

    def __getitem__(self, index):
        return self.fibers[index]

    def __len__(self):
        return len(self.fibers)

    @property
    def ratios(self):
        return [fiber.ratio for fiber in self.fibers]

    @property
    def lengths(self):
        return [fiber.length for fiber in self.fibers]

    @property
    def reds(self):
        return [fiber.red for fiber in self.fibers]

    @property
    def greens(self):
        return [fiber.green for fiber in self.fibers]

    def get_labelmap(self, h, w, fiber_width=1):
        labelmap = np.zeros((h, w), dtype=np.uint8)
        for fiber in self.fibers:
            x, y, fw, fh = fiber.bbox
            # Clip coordinates to image boundaries
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(w, x + fw), min(h, y + fh)

            if x0 >= x1 or y0 >= y1:
                continue  # fiber bbox completely outside

            roi = labelmap[y0:y1, x0:x1]
            fiber_data = fiber.data[: y1 - y0, : x1 - x0]

            binary = fiber_data > 0
            roi[binary] = fiber_data[binary]
        if fiber_width > 1:
            labelmap = expand_labels(labelmap, fiber_width)
        return labelmap

    def get_bounding_boxes_map(self, h, w, width=1, image=None):
        if image is None:
            bbox_map = np.zeros((h, w), dtype=np.uint8)
        else:
            bbox_map = image.copy()
        for i, fiber in enumerate(self.fibers, start=1):
            try:
                x, y, bw, bh = fiber.bbox

                # Draw a square around the bbox
                x1 = max(0, x - width)
                y1 = max(0, y - width)
                x2 = min(bbox_map.shape[1], x + bw + width)
                y2 = min(bbox_map.shape[0], y + bh + width)

                cv2.rectangle(
                    bbox_map,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    thickness=width,
                )
            except IndexError as e:
                print(f"Error processing fiber {fiber.fiber_id}: {e}")
        return bbox_map

    def contains(self, fiber: FiberProps, ratio=0.5):
        for existing_fiber in self.fibers:
            if existing_fiber.bbox_intersect(fiber, ratio):
                return True
        return False

    def append(self, fiber: FiberProps):
        self.fibers.append(fiber)

    def append_if_not_exists(self, fiber: FiberProps, ratio=0.5):
        """
        Append a fiber to the list if it does not already exist.
        """
        if not self.contains(fiber, ratio):
            self.append(fiber)

    def valid_copy(self):
        return Fibers([fiber for fiber in self.fibers if fiber.is_valid])

    def filtered_copy(self):
        return Fibers(
            [fiber for fiber in self.fibers if (fiber.is_acceptable and fiber.is_valid)]
        )

    def filter_border(self, h, w, border=1):
        return Fibers(
            [
                fiber
                for fiber in self.fibers
                if not (
                    fiber.bbox[0] < border
                    or fiber.bbox[1] < border
                    or fiber.bbox[0] + fiber.bbox[2] > w - border
                    or fiber.bbox[1] + fiber.bbox[3] > h - border
                )
            ]
        )

    def only_double_copy(self):
        return Fibers([fiber for fiber in self.fibers if fiber.is_double])

    def only_triple_copy(self):
        return Fibers([fiber for fiber in self.fibers if fiber.is_triple])

    def union(self, other, ratio=0.5):
        union = Fibers(self.fibers)
        for fiber in other:
            union.append_if_not_exists(fiber, ratio)
        return union

    def difference(self, other, ratio):
        substract = Fibers([])
        intersection = self.intersection(other, ratio)
        for fiber in self.fibers:
            if not intersection.contains(fiber, ratio):
                substract.append_if_not_exists(fiber, ratio)
        return substract

    def intersection(self, other, ratio=0.5):
        intersection = Fibers([])
        for fiber in self.fibers:
            for other_fiber in other:
                if fiber.bbox_intersect(other_fiber, ratio):
                    intersection.append_if_not_exists(fiber, ratio)
        return intersection

    def joined_intersection(self, other, ratio=0.5):
        intersection = []
        for fiber in self.fibers:
            for other_fiber in other.fibers:
                if fiber.bbox_intersect(other_fiber, ratio):
                    (intersection.append((fiber, other_fiber)))
        return intersection

    def order_as(self, other, ratio=0.5):
        result = Fibers([])
        for i, other_fiber in enumerate(other.fibers):
            for fiber in self.fibers:
                if fiber.bbox_intersect(other_fiber, ratio=ratio):
                    result.append_if_not_exists(fiber, ratio=ratio)
        return result

    def matched_pairs(self, other, ratio=0.5) -> list[Tuple[FiberProps, FiberProps]]:
        n, m = len(self.fibers), len(other.fibers)
        if n == 0 or m == 0:
            return []

        # Build IoU matrix (or use negative for cost)
        iou_matrix = np.zeros((n, m))
        for i, fiber in enumerate(self.fibers):
            for j, other_fiber in enumerate(other.fibers):
                iou_matrix[i, j] = fiber.bbox_iou(other_fiber)

        # Hungarian on negative IoU (minimize cost = maximize IoU)
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        pairs = []
        for i, j in zip(row_ind, col_ind):
            if iou_matrix[i, j] >= ratio:  # Only keep pairs above threshold
                pairs.append((self.fibers[i], other.fibers[j]))

        return pairs

    def to_df(
        self, pixel_size=0.13, img_name: Optional[str] = None, filter_invalid=True
    ):
        import pandas as pd

        data = {
            "Fiber ID": [],
            "First analog (µm)": [],
            "Second analog (µm)": [],
            "Ratio": [],
            "Fiber type": [],
            "Valid": [],
        }

        for i, fiber in enumerate(self.fibers):
            if filter_invalid and not fiber.is_valid:
                continue
            data["Fiber ID"].append(i)
            r, g = fiber.counts
            red_length = pixel_size * r
            green_length = pixel_size * g
            data["First analog (µm)"].append(red_length)
            data["Second analog (µm)"].append(green_length)
            data["Ratio"].append(fiber.ratio)
            data["Fiber type"].append(fiber.fiber_type)
            data["Valid"].append(fiber.is_valid)
        df = pd.DataFrame.from_dict(data)
        if img_name:
            df["Image Name"] = img_name
        return df

    def svgs(self, scale=1.0, color1="red", color2="green"):
        svgs = [
            fiber.svg_representation(scale, color1=color1, color2=color2)
            for fiber in self.fibers
        ]
        return [svg for svg in svgs if svg is not None]

    def debug(self, h=1024, w=1024, img=None, fiber_width=1):
        if img is not None:
            h, w = img.shape[:2]
        labelmap = self.get_labelmap(h, w, fiber_width=fiber_width)

        ratio = np.nanmean(self.ratios)
        if img is not None:
            plt.imshow(img)

        plt.imshow(labelmap, cmap=CMAP)
        plt.title(f"Fiber Labelmap, {len(self.fibers)} fibers, mean ratio: {ratio:.2f}")

        plt.axis("off")
        plt.show()

    def get_fiber_by_id(self, fiber_id: int) -> Optional[FiberProps]:
        for fiber in self.fibers:
            if fiber.fiber_id == fiber_id:
                return fiber
        return None

    def to_pickle(self, path: Optional[str] = None) -> Optional[bytes]:
        """
        Serialize the Fibers object to pickle format.

        Args:
            path: If provided, write to file. If None, return bytes.

        Returns:
            Pickle bytes if path is None, otherwise None.
        """
        if path is None:
            return pickle.dumps(self)
        else:
            with open(path, "wb") as f:
                pickle.dump(self, f)
            return None

    @staticmethod
    def from_pickle(path: str) -> "Fibers":
        with open(path, "rb") as f:
            fibers = pickle.load(f)
        return fibers

    def __add__(self, other: "Fibers") -> "Fibers":
        combined_fibers = self.fibers + other.fibers
        return Fibers(combined_fibers)

    def __reduce__(self):
        return (self.__class__, (self.fibers, self.path))

    def deepcopy(self) -> "Fibers":
        copied_fibers = [
            FiberProps(
                bbox=fiber.bbox,
                data=fiber.data,
                fiber_id=fiber.fiber_id,
                is_an_error=fiber.is_an_error,
            )
            for fiber in self.fibers
        ]
        return Fibers(copied_fibers, path=self.path)


def estimate_fiber_category(fiber_trace: np.ndarray, fiber_data: np.ndarray) -> str:
    """
    Estimate the fiber category based on the number of red and green pixels.
    """
    coordinates = fiber_trace
    coordinates = np.asarray(coordinates)
    try:
        values = fiber_data[coordinates[:, 0], coordinates[:, 1]]
    except IndexError:
        return "unknown"
    diff = np.diff(values)
    jump = np.sum(diff != 0)
    n_ccs = jump + 1
    if n_ccs == 2:
        return "double"
    elif n_ccs == 3:
        if values[0] == 1:
            return "one-two-one"
        else:
            return "two-one-two"
    else:
        return "multiple"
