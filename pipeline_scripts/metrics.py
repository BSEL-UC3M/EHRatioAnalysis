# ======================================================================
# File: utils/metrics.py
# Description: Common evaluation metrics for segmentation.
# ======================================================================

import numpy as np

def volume_similarity_index_scalar(vol1, vol2):
    if (vol1 + vol2) == 0:
        return 1.0  # Both empty, perfect similarity
    return 1.0 - abs(vol1 - vol2) / (vol1 + vol2)
