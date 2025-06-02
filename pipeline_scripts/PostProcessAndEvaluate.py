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
from pipeline_scripts.PostProcess3D import postprocess_all_patients_ears, extract_patient_and_ear_gt, extract_patient_and_ear_pred
from pipeline_scripts.metrics import volume_similarity_index_scalar
from pipeline_scripts.plots import plot_scatter_pred_vs_gt_volume, plot_vsi_bar, plot_volume_difference_bar

def get_voxel_volume(dataset_type):
    if dataset_type.upper() == "MRC":
        return 0.5 * 0.5 * 0.5  # mm^3
    elif dataset_type.upper() == "PEI":
        return 0.5 * 0.5 * 0.8  # mm^3
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

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
    gt_mask_folder,
    out_gt_folder,
    overlay_gt_folder,
    metrics_csv,
    dataset_type="MRC"
):
    """
    Postprocesses predicted masks (and GT masks), computes volume-level metrics per patient/ear,
    and saves the results and plots in the specified folder.
    """

    # --- Step 1: Postprocess predicted masks ---
    print(f"\n🧼 Postprocessing predicted masks: {pred_mask_folder}")
    postprocess_all_patients_ears(
        orig_folder=orig_folder,
        mask_folder=pred_mask_folder,
        out_folder=out_pred_folder,
        overlay_folder=overlay_pred_folder,
        has_masks=False
    )

    # --- Step 2: Postprocess GT masks the same way ---
    print(f"\n🧼 Postprocessing GT masks: {gt_mask_folder}")
    postprocess_all_patients_ears(
        orig_folder=orig_folder,
        mask_folder=gt_mask_folder,
        out_folder=out_gt_folder,
        overlay_folder=overlay_gt_folder,
        has_masks=True
    )

    # --- Step 3: Aggregate per patient/ear and calculate volumes ---
    voxel_volume = get_voxel_volume(dataset_type)
    pred_groups = defaultdict(list)
    gt_groups = defaultdict(list)

    for pred_fname in sorted(os.listdir(out_pred_folder)):
        patient_id, ear = extract_patient_and_ear_pred(pred_fname)
        if patient_id and ear:
            pred_groups[(patient_id, ear)].append(os.path.join(out_pred_folder, pred_fname))

    for gt_fname in sorted(os.listdir(out_gt_folder)):
        patient_id, ear = extract_patient_and_ear_gt(gt_fname)
        if patient_id and ear:
            gt_groups[(patient_id, ear)].append(os.path.join(out_gt_folder, gt_fname))

    metric_rows = []
    all_keys = sorted(set(pred_groups) | set(gt_groups))
    for (patient_id, ear) in all_keys:
        pred_files = pred_groups.get((patient_id, ear), [])
        gt_files   = gt_groups.get((patient_id, ear), [])
        # Aggregate all mask slices (pred or GT) into one volume for each
        pred_volume = sum((np.array(Image.open(f)) > 0).sum() for f in pred_files) if pred_files else 0
        gt_volume   = sum((np.array(Image.open(f)) > 0).sum() for f in gt_files) if gt_files else 0

        metric_rows.append({
            "patient_id": patient_id,
            "ear": ear,
            "pred_volume_voxels": int(pred_volume),
            "gt_volume_voxels": int(gt_volume),
            "pred_volume_mm3": round(pred_volume * voxel_volume, 3),
            "gt_volume_mm3": round(gt_volume * voxel_volume, 3),
            "num_pred_slices": len(pred_files),
            "num_gt_slices": len(gt_files),
            "vsi": volume_similarity_index_scalar(pred_volume, gt_volume)
        })

    # --- Save metrics ---
    df = pd.DataFrame(metric_rows)
    df.to_csv(metrics_csv, index=False)
    print(f"✅ Volume table saved to {metrics_csv}")

    # --- Save plots in same folder ---
    plot_base = os.path.dirname(metrics_csv)
    if metrics_csv and os.path.exists(metrics_csv):
        df = pd.read_csv(metrics_csv)
        plot_scatter_pred_vs_gt_volume(df, save_path=os.path.join(plot_base, "volume_scatter.png"))
        plot_vsi_bar(df, save_path=os.path.join(plot_base, "vsi_bar.png"))
        plot_volume_difference_bar(df, save_path=os.path.join(plot_base, "volume_difference.png"))
    else:
        print("⚠️ metrics_csv not found, skipping plots.")

    print("\n✅ Postprocessing & evaluation complete!")

