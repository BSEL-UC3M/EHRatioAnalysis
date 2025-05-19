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

def parse_patient_and_ear(filename):
    """
    Extract patient ID and ear (left or right) from filename.
    Assumes format like MRC_12_80711463_crop0_mask.png or PEI_12_80711463_crop1_mask.png
    Returns: patient_id (e.g. 12), ear ("left"/"right")
    """
    match = re.match(r"(MRC|PEI)_(\d+)_\d+_crop([01])_mask\.png", os.path.basename(filename))
    if not match:
        return None, None
    _, patient_id, crop_idx = match.groups()
    ear = "right" if crop_idx == "0" else "left"
    return patient_id, ear

def compute_mask_volume(mask, voxel_volume_mm3):
    return float(np.sum(mask > 0) * voxel_volume_mm3)

def collect_volumes_from_folder(folder_path, voxel_volume_mm3):
    """
    Parses all mask images in the folder and computes volumes per patient and ear.
    Returns a dict: {(patient_id, ear): volume_mm3}
    """
    volume_dict = defaultdict(float)
    mask_paths = sorted(glob(os.path.join(folder_path, "*_mask.png")))

    for mask_path in mask_paths:
        patient_id, ear = parse_patient_and_ear(mask_path)
        if patient_id is None or ear is None:
            print(f"⚠️ Skipping unrecognized mask filename: {mask_path}")
            continue
        # TODO: check if it's calculating with png instead of on tiff
        mask = np.array(Image.open(mask_path))
        volume = compute_mask_volume(mask, voxel_volume_mm3)
        volume_dict[(patient_id, ear)] += volume

    return volume_dict

def save_volume_ratios_csv(mrc_volumes, pei_volumes, output_csv):
    """
    Saves the merged MRC/PEI volume table with EH ratio to a CSV.
    """
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["patient_id", "ear", "mrc_volume_mm3", "pei_volume_mm3", "eh_ratio"])

        all_keys = set(mrc_volumes.keys()) | set(pei_volumes.keys())
        for (pid, ear) in sorted(all_keys):
            mrc_vol = mrc_volumes.get((pid, ear), 0.0)
            pei_vol = pei_volumes.get((pid, ear), 0.0)
            ratio = pei_vol / mrc_vol if mrc_vol > 0 else ""
            writer.writerow([pid, ear, f"{mrc_vol:.2f}", f"{pei_vol:.2f}", f"{ratio:.3f}" if ratio != "" else ""])

    print(f"\n📊 EH ratio table saved to: {output_csv}")

def compute_eh_ratios(
    mrc_mask_folder,
    pei_mask_folder,
    output_csv_path,
    mrc_voxel_size=(0.5, 0.5, 0.5),
    pei_voxel_size=(0.5, 0.5, 0.8),
):
    """
    Computes EH ratio using segmentation masks in MRC and PEI folders.
    """
    print("\n📐 Calculating EH volume ratios per patient & ear...")

    mrc_voxel_volume = np.prod(mrc_voxel_size)
    pei_voxel_volume = np.prod(pei_voxel_size)

    mrc_volumes = collect_volumes_from_folder(mrc_mask_folder, voxel_volume_mm3=mrc_voxel_volume)
    pei_volumes = collect_volumes_from_folder(pei_mask_folder, voxel_volume_mm3=pei_voxel_volume)

    save_volume_ratios_csv(mrc_volumes, pei_volumes, output_csv_path)
