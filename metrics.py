"""
metrics.py
----------
Evaluation metrics for binary segmentation.
All metrics computed on numpy arrays (after thresholding predictions).

Metrics:
  - Dice coefficient (F1)
  - IoU / Jaccard Index
  - Pixel Accuracy
  - Precision
  - Recall
  - Hausdorff Distance (95th percentile)
"""

import numpy as np
from scipy.spatial.distance import directed_hausdorff


def threshold(pred: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    return (pred >= thresh).astype(np.uint8)


def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Dice = 2*TP / (2*TP + FP + FN)
    Equivalent to F1 score. Primary metric for segmentation tasks.
    """
    pred   = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    IoU / Jaccard Index = TP / (TP + FP + FN)
    """
    pred   = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Fraction of correctly classified pixels."""
    return (pred == target).mean()


def precision_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Precision = TP / (TP + FP)"""
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return (tp + smooth) / (tp + fp + smooth)


def recall_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Recall / Sensitivity = TP / (TP + FN)"""
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return (tp + smooth) / (tp + fn + smooth)


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray) -> float:
    """
    95th percentile Hausdorff Distance — measures boundary accuracy.
    Widely used in MICCAI segmentation challenges.
    Returns inf if either mask is empty.
    """
    pred_pts   = np.argwhere(pred > 0)
    target_pts = np.argwhere(target > 0)

    if len(pred_pts) == 0 or len(target_pts) == 0:
        return float("inf")

    d1 = directed_hausdorff(pred_pts, target_pts)[0]
    d2 = directed_hausdorff(target_pts, pred_pts)[0]
    return max(d1, d2)


def compute_all_metrics(
    pred_prob: np.ndarray,
    target: np.ndarray,
    thresh: float = 0.5
) -> dict:
    """
    Compute all metrics at once.

    Parameters
    ----------
    pred_prob : float array ∈ [0,1] (H, W) — sigmoid output
    target    : binary array ∈ {0,1} (H, W) — ground truth
    thresh    : binarization threshold

    Returns
    -------
    dict with keys: dice, iou, accuracy, precision, recall, hausdorff95
    """
    pred_bin = threshold(pred_prob, thresh)

    return {
        "dice":        round(dice_coefficient(pred_bin, target), 4),
        "iou":         round(iou_score(pred_bin, target), 4),
        "accuracy":    round(pixel_accuracy(pred_bin, target), 4),
        "precision":   round(precision_score(pred_bin, target), 4),
        "recall":      round(recall_score(pred_bin, target), 4),
        "hausdorff95": round(hausdorff_distance_95(pred_bin, target), 2),
    }
