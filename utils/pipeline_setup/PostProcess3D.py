# =============================================================================
# File: pipeline_setup/PostProcess3D.py
# Description: Postprocessing of the segmentation masks with fill holes and 
#  connected components.
# Author: @cfusterbarcelo
# Created: 08/04/2025
# ==============================================================================

import os
import re
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
import cc3d

def parse_patient_and_ear(filename):
    match = re.match(r"(MRC|PEI)_(\d+)_\d+_crop([01])_mask\.png", os.path.basename(filename))
    if not match:
        return None, None
    _, patient_id, crop_idx = match.groups()
    ear = "right" if crop_idx == "0" else "left"
    return patient_id, ear

def extract_slice_index(filename):
    match = re.match(r"(MRC|PEI)_(\d+)_(\d+)_crop[01]_mask\.png", filename)
    if not match:
        return 0
    return int(match.group(3))

def postprocess_3d_mask(stack, dust_threshold=500, connectivity=26, fill_3d_holes=True):
    """
    stack: [num_slices, H, W] binary mask (bool or 0/1)
    Steps:
        1. Remove small components (3D) with cc3d.dust.
        2. Keep only the largest 3D connected component.
        3. Fill holes (optionally in 3D or 2D).
    Returns:
        Cleaned stack (bool)
    """
    # Remove small components
    cleaned = cc3d.dust(stack, threshold=dust_threshold, connectivity=connectivity, in_place=False)

    # Label components
    labels_out = cc3d.connected_components(cleaned, connectivity=connectivity)
    if labels_out.max() == 0:
        return cleaned  # nothing left
    # Get the largest 3D component
    biggest = (labels_out == np.argmax(np.bincount(labels_out.flat)[1:]) + 1)
    
    # Fill holes
    if fill_3d_holes:
        biggest = binary_fill_holes(biggest)  # 3D fill
    else:
        # Or fill holes per slice if preferred
        for i in range(biggest.shape[0]):
            biggest[i] = binary_fill_holes(biggest[i])

    return biggest.astype(np.uint8)

def postprocess_all_patients_ears(mask_folder, out_folder, dust_threshold=500):
    os.makedirs(out_folder, exist_ok=True)
    groups = {}
    for fname in sorted(os.listdir(mask_folder)):
        if not fname.endswith("_mask.png"):
            continue
        patient_id, ear = parse_patient_and_ear(fname)
        if patient_id is None or ear is None:
            print(f"⚠️ Skipping unrecognized mask filename: {fname}")
            continue
        key = (patient_id, ear)
        if key not in groups:
            groups[key] = []
        groups[key].append(fname)
    for (patient_id, ear), mask_list in groups.items():
        mask_list_sorted = sorted(mask_list, key=extract_slice_index)
        stack = []
        for fname in mask_list_sorted:
            mask = np.array(Image.open(os.path.join(mask_folder, fname)).convert("L"))
            stack.append(mask > 127)
        stack = np.stack(stack, axis=0)
        cleaned = postprocess_3d_mask(stack, dust_threshold=dust_threshold)
        for idx, fname in enumerate(mask_list_sorted):
            out_path = os.path.join(out_folder, fname)
            Image.fromarray((cleaned[idx] * 255).astype(np.uint8)).save(out_path)

# -- keep only report_mask_volumes and remove volume_difference_report

def report_mask_volumes(
    before_folder,
    after_folder,
    output_csv="mask_volume_comparison.csv"
):
    """
    Save a CSV comparing mask voxel volumes before/after postprocessing for each patient/ear.
    """
    import pandas as pd
    def collect_masks(folder):
        mapping = {}
        for fname in os.listdir(folder):
            if not fname.endswith("_mask.png"):
                continue
            parts = fname.split("_")
            pid = "_".join(parts[:2])
            crop_i = int(parts[-2].replace("crop", ""))
            ear = "left" if crop_i == 0 else "right"
            key = (pid, ear)
            if key not in mapping:
                mapping[key] = []
            mapping[key].append(os.path.join(folder, fname))
        return mapping

    before_map = collect_masks(before_folder)
    after_map = collect_masks(after_folder)
    results = []

    for key in before_map:
        pid, ear = key
        before_masks = before_map[key]
        after_masks = after_map.get(key, [])
        before_volume = sum((np.array(Image.open(f)) > 127).sum() for f in before_masks)
        after_volume = sum((np.array(Image.open(f)) > 127).sum() for f in after_masks)

        results.append({
            "patient_id": pid,
            "ear": ear,
            "volume_before": before_volume,
            "volume_after": after_volume,
            "delta": after_volume - before_volume,
            "percent_change": 100 * (after_volume - before_volume) / (before_volume+1e-6)
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Volume comparison report saved to {output_csv}")
    return df
