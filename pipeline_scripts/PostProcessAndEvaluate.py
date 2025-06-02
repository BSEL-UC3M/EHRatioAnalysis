# ======================================================================
# File: pipeline_scripts/PostProcessAndEvaluate.py
# Description: ONLY postprocesses predicted and GT masks (no volume calc).
# Author: @cfusterbarcelo
# Created: 2025-05-30
# ======================================================================

import os
from pipeline_scripts.PostProcess3D import postprocess_all_patients_ears

def postprocess_pred_and_gt(
    orig_folder,
    pred_mask_folder,
    out_pred_folder,
    overlay_pred_folder,
    gt_mask_folder=None,
    out_gt_folder=None,
    overlay_gt_folder=None
):
    """
    Postprocesses predicted and GT masks (performs 3D filling and largest CC).
    - orig_folder: folder with original images for overlays.
    - pred_mask_folder: folder with predicted mask slices (input).
    - out_pred_folder: output folder for postprocessed predicted masks.
    - overlay_pred_folder: output for overlays of predicted.
    - gt_mask_folder: folder with GT mask slices (input).
    - out_gt_folder: output folder for postprocessed GT masks.
    - overlay_gt_folder: output for overlays of GT.
    """
    print(f"\n🧼 Postprocessing predicted masks: {pred_mask_folder}")
    postprocess_all_patients_ears(
        orig_folder=orig_folder,
        mask_folder=pred_mask_folder,
        out_folder=out_pred_folder,
        overlay_folder=overlay_pred_folder,
        has_masks=False
    )

    print(f"\n🧼 Postprocessing GT masks: {gt_mask_folder}")
    postprocess_all_patients_ears(
        orig_folder=orig_folder,
        mask_folder=gt_mask_folder,
        out_folder=out_gt_folder,
        overlay_folder=overlay_gt_folder,
        has_masks=True
    )

    print("\n✅ Postprocessing of predicted and GT masks complete!")

