# ======================================================================
# File: pipeline_setup/RatioCalculator.py
# Description: Calculates EH Ratio using predicted segmentation masks.
# Author: @cfusterbarcelo
# Created: 09/04/2025
# ======================================================================

import os
import re
import csv
import numpy as np
from glob import glob
from collections import defaultdict
from PIL import Image

from pipeline_scripts.plots import plot_eh_ratio_and_vsi

def parse_patient_and_ear_pred(filename):
    # For predicted masks: MRC_97_58881260_crop0_mask.png
    match = re.match(r"(MRC|PEI)_(\d+)_\d+_crop([01])_mask\.png", os.path.basename(filename))
    if not match:
        return None, None
    _, patient_id, crop_idx = match.groups()
    ear = "right" if crop_idx == "0" else "left"
    return patient_id, ear

def parse_patient_and_ear_gt(filename):
    # For GT masks: MRC_98_61925610_left.tif
    match = re.match(r"(MRC|PEI)_(\d+)_\d+_(left|right)\.tif", os.path.basename(filename))
    if not match:
        return None, None
    _, patient_id, ear = match.groups()
    return patient_id, ear


def compute_mask_volume(mask, voxel_volume_mm3):
    return float(np.sum(mask > 0) * voxel_volume_mm3)

def collect_volumes_from_folder(folder_path, voxel_volume_mm3, parse_func):
    volume_dict = defaultdict(float)
    mask_paths = sorted(glob(os.path.join(folder_path, "*")))
    for mask_path in mask_paths:
        patient_id, ear = parse_func(mask_path)
        if patient_id is None or ear is None:
            print(f"⚠️ Skipping unrecognized mask filename: {mask_path}")
            continue
        # Use TIFF or PNG loader
        mask = np.array(Image.open(mask_path))
        volume = compute_mask_volume(mask, voxel_volume_mm3)
        volume_dict[(patient_id, ear)] += volume
    return volume_dict


def save_volume_ratios_csv(mrc_volumes, pei_volumes, output_csv):
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["patient_id", "ear", "mrc_volume_mm3", "pei_volume_mm3", "eh_ratio"])
        all_keys = set(mrc_volumes.keys()) | set(pei_volumes.keys())
        for (pid, ear) in sorted(all_keys):
            mrc_vol = mrc_volumes.get((pid, ear), 0.0)
            pei_vol = pei_volumes.get((pid, ear), 0.0)
            ratio = pei_vol / mrc_vol if mrc_vol > 0 else ""
            writer.writerow([pid, ear, f"{mrc_vol:.2f}", f"{pei_vol:.2f}", f"{ratio:.3f}" if ratio != "" else ""])

def save_vsi_csv(pred_vols, gt_vols, output_csv):
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["patient_id", "ear", "pred_volume_mm3", "gt_volume_mm3", "vsi"])
        keys = set(pred_vols.keys()) & set(gt_vols.keys())
        for (pid, ear) in sorted(keys):
            pred_vol = pred_vols[(pid, ear)]
            gt_vol = gt_vols[(pid, ear)]
            if pred_vol + gt_vol > 0:
                vsi = 1.0 - abs(pred_vol - gt_vol) / (pred_vol + gt_vol)
            else:
                vsi = ""
            writer.writerow([pid, ear, f"{pred_vol:.2f}", f"{gt_vol:.2f}", f"{vsi:.3f}" if vsi != "" else ""])

def compute_eh_ratios(
    mrc_mask_folder,
    pei_mask_folder,
    output_path,
    mrc_voxel_size=(0.5, 0.5, 0.5),
    pei_voxel_size=(0.5, 0.5, 0.8),
    mrc_gt_mask_folder=None,
    pei_gt_mask_folder=None,
):
    print("\n📐 Calculating EH volume ratios per patient & ear...")

    mrc_voxel_volume = np.prod(mrc_voxel_size)
    pei_voxel_volume = np.prod(pei_voxel_size)
    output_csv_path = output_path + ".csv"

    # -------- Predicted -------------
    mrc_volumes = collect_volumes_from_folder(mrc_mask_folder, mrc_voxel_volume, parse_patient_and_ear_pred)
    pei_volumes = collect_volumes_from_folder(pei_mask_folder, pei_voxel_volume, parse_patient_and_ear_pred)
    save_volume_ratios_csv(mrc_volumes, pei_volumes, output_csv_path)

    # -------- GT (if provided) -------
    if mrc_gt_mask_folder and pei_gt_mask_folder:
        print("\n🟩 Calculating for ground truth postprocessed masks...")
        gt_csv_path = os.path.splitext(output_path)[0] + "_gt.csv"
        mrc_gt_volumes = collect_volumes_from_folder(mrc_gt_mask_folder, mrc_voxel_volume, parse_patient_and_ear_gt)
        pei_gt_volumes = collect_volumes_from_folder(pei_gt_mask_folder, pei_voxel_volume, parse_patient_and_ear_gt)
        save_volume_ratios_csv(mrc_gt_volumes, pei_gt_volumes, gt_csv_path)

        # -------- VSI -----------
        vsi_mrc_csv = os.path.splitext(output_path)[0] + "_vsi_mrc.csv"
        vsi_pei_csv = os.path.splitext(output_path)[0] + "_vsi_pei.csv"
        save_vsi_csv(mrc_volumes, mrc_gt_volumes, vsi_mrc_csv)
        save_vsi_csv(pei_volumes, pei_gt_volumes, vsi_pei_csv)

    results_folder = os.path.dirname(output_path)
    plot_eh_ratio_and_vsi(
        pred_csv=output_csv_path,
        gt_csv=gt_csv_path,
        vsi_csv=vsi_mrc_csv,
        results_folder=results_folder,
        modality="MRC"
    )
    plot_eh_ratio_and_vsi(
        pred_csv=output_csv_path,
        gt_csv=gt_csv_path,
        vsi_csv=vsi_pei_csv,
        results_folder=results_folder,
        modality="PEI"
    )

