# ======================================================================
# File: utils/metrics.py
# Description: Common evaluation metrics for segmentation.
# ======================================================================

import numpy as np

def dice_score(pred, gt, eps=1e-6):
    """Computes the Dice coefficient between two binary masks."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    return 2 * intersection / (pred.sum() + gt.sum() + eps)

def iou_score(pred, gt, eps=1e-6):
    """Computes the Intersection over Union (Jaccard) between two binary masks."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / (union + eps)

def volume_similarity_index_scalar(vol1, vol2):
    if (vol1 + vol2) == 0:
        return 1.0  # Both empty, perfect similarity
    return 1.0 - abs(vol1 - vol2) / (vol1 + vol2)
