# ======================================================================
# File: pipeline_scripts/PostProcessAndEvaluate.py
# Description: Postprocesses predicted and GT masks, computes volume-level metrics.
# Author: @cfusterbarcelo
# Created: 2025-05-30
# ======================================================================

import os
import pandas as pd
from PIL import Image
import numpy as np
import re
from collections import defaultdict
from pipeline_scripts.PostProcess3D import postprocess_all_patients_ears, report_mask_volumes
from pipeline_scripts.metrics import volume_similarity_index_scalar
from pipeline_scripts.plots import plot_scatter_pred_vs_gt_volume, plot_vsi_bar, plot_volume_difference_bar

def extract_patient_and_ear(filename):
    """
    Example input: 'PEI_99_62915675_crop0_mask.png' or 'MRC_100_63727456_crop1_mask.png'
    Returns: patient_id (e.g. 'PEI_99_62915675'), ear ('left' or 'right')
    """
    m = re.match(r"(.+)_crop([01])_mask", filename)
    if m:
        patient_id, crop_idx = m.groups()
        ear = 'left' if crop_idx == '0' else 'right'
        return patient_id, ear
    else:
        # fallback, not matching expected pattern
        return None, None
    
def postprocess_and_evaluate_volumes(
    orig_folder,
    pred_mask_folder,
    out_pred_folder,
    overlay_pred_folder,
    results_csv,
    gt_mask_folder=None,
    out_gt_folder=None,
    overlay_gt_folder=None,
    metrics_csv=None
):
    """
    Postprocesses predicted masks (and GT masks if provided), computes volume-level metrics,
    and saves the results.
    """

    # ---- Step 1: Postprocess predicted masks ----
    print(f"\n🧼 Postprocessing predicted masks: {pred_mask_folder}")
    postprocess_all_patients_ears(
        orig_folder=orig_folder,
        mask_folder=pred_mask_folder,
        out_folder=out_pred_folder,
        overlay_folder=overlay_pred_folder,
        has_masks=False
    )

    # ---- Step 2: Postprocess GT masks if available ----
    if gt_mask_folder and out_gt_folder and overlay_gt_folder:
        print(f"\n🧼 Postprocessing GT masks: {gt_mask_folder}")
        postprocess_all_patients_ears(
            orig_folder=orig_folder,         # or could use the GT orig folder
            mask_folder=gt_mask_folder,
            out_folder=out_gt_folder,
            overlay_folder=overlay_gt_folder,
            has_masks=True
        )

    # ---- Step 3: Volume comparison report for predictions ----
    print("\n📊 Reporting predicted mask volumes (before/after postprocess)...")
    report_mask_volumes(
        before_folder=pred_mask_folder,
        after_folder=out_pred_folder,
        output_csv=results_csv
    )

    # ---- Step 4: Compute metrics if GT available ----
    if gt_mask_folder and out_gt_folder and metrics_csv:
        print("\n📏 Calculating VOLUME ONLY (no slice alignment, just per ear)...")

        pred_vols = defaultdict(list)
        gt_vols = defaultdict(list)

        # Group predicted masks
        for pred_fname in sorted(os.listdir(out_pred_folder)):
            patient_id, ear = extract_patient_and_ear(pred_fname)
            if patient_id and ear:
                pred_vols[(patient_id, ear)].append(os.path.join(out_pred_folder, pred_fname))

        # Group GT masks
        for gt_fname in sorted(os.listdir(out_gt_folder)):
            m = re.match(r"(.+?)_(left|right)\.[a-zA-Z0-9]+$", gt_fname)
            if m:
                patient_id, ear = m.groups()
                gt_vols[(patient_id, ear)].append(os.path.join(out_gt_folder, gt_fname))
            else:
                print(f"GT file not matched by regex: {gt_fname}")

        metric_rows = []
        all_keys = set(list(pred_vols.keys()) + list(gt_vols.keys()))
        for key in all_keys:
            patient_id, ear = key
            pred_slice_files = pred_vols.get(key, [])
            gt_slice_files   = gt_vols.get(key, [])

            pred_volume = sum((np.array(Image.open(f)) > 0).sum() for f in pred_slice_files) if pred_slice_files else 0
            gt_volume   = sum((np.array(Image.open(f)) > 0).sum() for f in gt_slice_files) if gt_slice_files else 0

            metric_rows.append({
                "patient_id": patient_id,
                "ear": ear,
                "pred_volume_voxels": int(pred_volume),
                "gt_volume_voxels": int(gt_volume),
                "num_pred_slices": len(pred_slice_files),
                "num_gt_slices": len(gt_slice_files),
                "vsi": volume_similarity_index_scalar(pred_volume, gt_volume)
            })

        # Save metrics
        df = pd.DataFrame(metric_rows)
        df.to_csv(metrics_csv, index=False)
        print(f"✅ Volume table saved to {metrics_csv}")

        if metrics_csv and os.path.exists(metrics_csv):
            df = pd.read_csv(metrics_csv)
            plot_scatter_pred_vs_gt_volume(df, save_path=metrics_csv.replace(".csv", "_scatter.png"))
            plot_vsi_bar(df, save_path=metrics_csv.replace(".csv", "_vsi.png"))
            plot_volume_difference_bar(df, save_path=metrics_csv.replace(".csv", "_diff.png"))
        else:
            print("⚠️ metrics_csv not found, skipping plots.")

    print("\n✅ Postprocessing & evaluation complete!")
