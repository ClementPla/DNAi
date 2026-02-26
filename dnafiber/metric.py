import torch
import torchmetrics.functional as F
from skimage.measure import label
from torchmetrics import Metric
from typing import Dict
import numpy as np


class DNAFIBERMetric(Metric):
    """Instance-level detection and per-fiber segmentation metric for DNA fibers.

    Evaluates two aspects:
      1. **Detection**: Whether predicted connected components match ground-truth
         fibers, using greedy max-overlap assignment (one-to-one).
      2. **Segmentation quality**: For each matched pair, measures per-class
         (red=1, green=2) Dice, recall, and precision on the union region.

    Matching strategy:
      - For each image, an overlap matrix is built between predicted and GT blobs.
      - Greedy assignment selects the best (pred, gt) pair by overlap, removes both
        from further consideration, and repeats. This prevents many-to-one matches.
      - Unmatched predictions count as false positives.
      - Unmatched GT fibers count as false negatives (captured via N - detection_tp).

    Args:
        iou_threshold: Minimum IoU between a predicted blob and GT blob to allow
            a match. Set to 0.0 to match any overlap (original behavior). Default: 0.1.
        **kwargs: Additional arguments passed to ``torchmetrics.Metric``.

    States:
        detection_tp: Number of true positive detections (matched pred-GT pairs).
        detection_fp: Number of false positive detections (unmatched predictions).
        N: Total number of ground-truth fibers.
        N_predicted: Total number of predicted fibers.
        fiber_{red,green}_{dice,recall,precision}: Accumulated per-fiber segmentation
            scores, summed over all matched pairs. Divide by detection_tp to get means.
    """

    def __init__(self, iou_threshold: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.iou_threshold = iou_threshold

        # Detection counts
        self.add_state(
            "detection_tp",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "detection_fp",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "N",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "N_predicted",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
        )

        # Per-fiber segmentation quality (accumulated, divide by detection_tp for mean)
        for color in ("red", "green"):
            for metric_name in ("dice", "recall", "precision"):
                self.add_state(
                    f"fiber_{color}_{metric_name}",
                    default=torch.tensor(0.0, dtype=torch.float32),
                    dist_reduce_fx="sum",
                )

    def _compute_overlap_matrix(
        self,
        pred_labels: np.ndarray,
        gt_labels: np.ndarray,
        n_pred: int,
        n_gt: int,
    ) -> np.ndarray:
        """Compute IoU overlap matrix between predicted and GT connected components.

        Args:
            pred_labels: (H, W) integer label map for predictions (0=background).
            gt_labels: (H, W) integer label map for ground truth (0=background).
            n_pred: Number of predicted blobs (max label value).
            n_gt: Number of GT blobs (max label value).

        Returns:
            (n_pred, n_gt) IoU matrix.
        """
        iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float64)

        # Vectorized: for each pixel, record which (pred_label, gt_label) pair it belongs to
        pred_flat = pred_labels.ravel()
        gt_flat = gt_labels.ravel()

        # Only consider pixels where both are nonzero
        mask = (pred_flat > 0) & (gt_flat > 0)
        if not mask.any():
            return iou_matrix

        p_ids = pred_flat[mask] - 1  # 0-indexed
        g_ids = gt_flat[mask] - 1

        # Vectorized intersection counting via linear index
        linear_idx = p_ids * n_gt + g_ids
        intersection_counts = np.bincount(linear_idx, minlength=n_pred * n_gt)
        intersection_matrix = intersection_counts.reshape(n_pred, n_gt).astype(
            np.float64
        )

        # Compute areas
        pred_areas = np.bincount(pred_flat, minlength=n_pred + 1)[1:].astype(np.float64)
        gt_areas = np.bincount(gt_flat, minlength=n_gt + 1)[1:].astype(np.float64)

        # IoU = intersection / (area_pred + area_gt - intersection), vectorized
        union_matrix = pred_areas[:, None] + gt_areas[None, :] - intersection_matrix
        with np.errstate(divide="ignore", invalid="ignore"):
            iou_matrix = np.where(
                union_matrix > 0, intersection_matrix / union_matrix, 0.0
            )

        return iou_matrix

    def _greedy_match(self, iou_matrix: np.ndarray) -> list[tuple[int, int]]:
        """Greedy one-to-one matching based on IoU overlap matrix.

        Iteratively selects the (pred, gt) pair with highest IoU, removes both
        from further consideration, and repeats until no valid pairs remain.

        Args:
            iou_matrix: (n_pred, n_gt) IoU matrix.

        Returns:
            List of (pred_label_1indexed, gt_label_1indexed) matched pairs.
        """
        matches = []
        matrix = iou_matrix.copy()

        while True:
            if matrix.size == 0:
                break
            best_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
            best_iou = matrix[best_idx]
            if best_iou < self.iou_threshold:
                break

            pred_idx, gt_idx = best_idx
            matches.append((pred_idx + 1, gt_idx + 1))  # 1-indexed labels

            # Remove matched pred and gt from further consideration
            matrix[pred_idx, :] = -1
            matrix[:, gt_idx] = -1

        return matches

    def _compute_fiber_metrics(
        self,
        pred_fiber: torch.Tensor,
        gt_fiber: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-class Dice, recall, and precision for a matched fiber pair.

        Only evaluates on pixels where at least one of (pred, gt) is nonzero,
        effectively ignoring mutual-background pixels in the union region.

        Args:
            pred_fiber: 1D tensor of predicted class labels on the union mask.
            gt_fiber: 1D tensor of GT class labels on the union mask.

        Returns:
            Dict with keys {red,green}_{dice,recall,precision}.
        """
        # Exclude pixels where both pred and gt are background
        nonzero_mask = (pred_fiber > 0) | (gt_fiber > 0)
        pred_fiber = pred_fiber[nonzero_mask]
        gt_fiber = gt_fiber[nonzero_mask]

        if pred_fiber.numel() == 0:
            zeros = torch.tensor(0.0, device=pred_fiber.device)
            return {
                k: zeros
                for k in [
                    "red_dice",
                    "green_dice",
                    "red_recall",
                    "green_recall",
                    "red_precision",
                    "green_precision",
                ]
            }

        # Dice via Jaccard: dice = 2*iou / (1+iou)
        jaccard = F.jaccard_index(
            pred_fiber,
            gt_fiber,
            num_classes=3,
            ignore_index=0,
            average=None,
            task="multiclass",
        )
        jaccard = torch.nan_to_num(jaccard, nan=0.0)
        dices = 2.0 * jaccard / (1.0 + jaccard)
        dices = torch.nan_to_num(dices, nan=0.0)

        recall = F.recall(
            pred_fiber,
            gt_fiber,
            num_classes=3,
            ignore_index=0,
            task="multiclass",
            average=None,
        )
        recall = torch.nan_to_num(recall, nan=0.0)

        precision = F.precision(
            pred_fiber,
            gt_fiber,
            num_classes=3,
            ignore_index=0,
            task="multiclass",
            average=None,
        )
        precision = torch.nan_to_num(precision, nan=0.0)

        return {
            "red_dice": dices[1],
            "green_dice": dices[2],
            "red_recall": recall[1],
            "green_recall": recall[2],
            "red_precision": precision[1],
            "green_precision": precision[2],
        }

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with a batch of predictions and targets.

        Args:
            preds: (B, C, H, W) logits or (B, H, W) class predictions.
                Classes: 0=background, 1=red, 2=green.
            target: (B, 1, H, W) or (B, H, W) ground-truth class labels.
        """
        if preds.ndim == 4:
            preds = preds.argmax(dim=1)
        if target.ndim == 4:
            target = target.squeeze(1)

        B = preds.shape[0]
        binary_preds = (preds > 0).detach().cpu().numpy()
        binary_target = (target > 0).detach().cpu().numpy()

        for i in range(B):
            # Connected component labeling on CPU
            pred_labels_np = label(binary_preds[i], connectivity=2)
            gt_labels_np = label(binary_target[i], connectivity=2)
            n_pred = int(pred_labels_np.max())
            n_gt = int(gt_labels_np.max())

            self.N += n_gt
            self.N_predicted += n_pred

            if n_pred == 0 or n_gt == 0:
                # All predictions are FP if no GT, no matches possible
                self.detection_fp += n_pred
                continue

            # Build IoU overlap matrix and perform greedy 1-to-1 matching
            iou_matrix = self._compute_overlap_matrix(
                pred_labels_np, gt_labels_np, n_pred, n_gt
            )
            matches = self._greedy_match(iou_matrix)

            n_matched = len(matches)
            self.detection_tp += n_matched
            self.detection_fp += n_pred - n_matched

            # Move label maps to device for segmentation metric computation
            pred_labels_t = torch.from_numpy(pred_labels_np).to(preds.device)
            gt_labels_t = torch.from_numpy(gt_labels_np).to(preds.device)

            # Compute per-fiber segmentation metrics for matched pairs
            for pred_lbl, gt_lbl in matches:
                pred_mask = pred_labels_t == pred_lbl
                gt_mask = gt_labels_t == gt_lbl
                union_mask = pred_mask | gt_mask

                pred_fiber = preds[i][union_mask]
                gt_fiber = target[i][union_mask]

                metrics = self._compute_fiber_metrics(pred_fiber, gt_fiber)
                for color in ("red", "green"):
                    for metric_name in ("dice", "recall", "precision"):
                        key = f"fiber_{color}_{metric_name}"
                        setattr(
                            self,
                            key,
                            getattr(self, key) + metrics[f"{color}_{metric_name}"],
                        )

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute final aggregated metrics.

        Returns:
            Dictionary containing:
                - detection_precision: TP / (TP + FP)
                - detection_recall: TP / N_gt
                - detection_f1: Harmonic mean of detection precision and recall
                - fiber_{red,green}_{dice,recall,precision}: Mean per-fiber scores
                - total_real_fibers: Total GT fiber count
                - total_predicted_fibers: Total predicted fiber count
        """
        _zero = torch.zeros(
            1, dtype=torch.float32, device=self.detection_tp.device
        ).squeeze()

        tp = self.detection_tp.float()
        fp = self.detection_fp.float()
        n = self.N.float()

        det_precision = tp / (tp + fp) if (tp + fp) > 0 else _zero
        det_recall = tp / n if n > 0 else _zero
        det_f1 = (
            2 * det_precision * det_recall / (det_precision + det_recall)
            if (det_precision + det_recall) > 0
            else _zero
        )

        result = {
            "detection_precision": det_precision,
            "detection_recall": det_recall,
            "detection_f1": det_f1,
            "total_real_fibers": self.N.float(),
            "total_predicted_fibers": self.N_predicted.float(),
        }

        for color in ("red", "green"):
            for metric_name in ("dice", "recall", "precision"):
                key = f"fiber_{color}_{metric_name}"
                value = getattr(self, key)
                result[key] = value / tp if tp > 0 else _zero

        return result
