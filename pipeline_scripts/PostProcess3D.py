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
from pipeline_scripts.plots import save_segmentation_overlay

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

def postprocess_3d_mask(stack,  connectivity=26, fill_3d_holes=True):
    """
    stack: [num_slices, H, W] binary mask (bool or 0/1)
    Steps:
        1. Fill holes (in 3D).
        2. Keep only the largest 3D connected component.
    Returns:
        Cleaned stack (bool)
    """

    cleaned = stack
    # Fill holes
    if fill_3d_holes:
        cleaned = binary_fill_holes(cleaned)  # 3D fill
    else:
        # Or fill holes per slice if preferred
        for i in range(cleaned.shape[0]):
            cleaned[i] = binary_fill_holes(cleaned[i])

    # Label comp4onents
    labels_out = cc3d.connected_components(cleaned, connectivity=connectivity)
    if labels_out.max() == 0:
        return cleaned  # nothing left
    # Get the largest 3D component
    biggest = (labels_out == np.argmax(np.bincount(labels_out.flat)[1:]) + 1)

    return biggest.astype(np.uint8)

def parse_gt_patient_ear_slice(filename):
    # Matches PEI_100_63728457_left.tif
    m = re.match(r"(MRC|PEI)_(\d+)_(\d+)_(left|right)\.[a-zA-Z0-9]+$", filename)
    if not m:
        return None, None, None
    prefix, patient_id, slice_idx, ear = m.groups()
    return f"{prefix}_{patient_id}", ear, int(slice_idx)


def postprocess_all_patients_ears(orig_folder, mask_folder, out_folder, overlay_folder, has_masks=False):
    os.makedirs(out_folder, exist_ok=True)
    groups = {}

    if not has_masks:
        # Crop-style masks
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
    else:
        # GT mask style
        for fname in sorted(os.listdir(mask_folder)):
            if not (fname.lower().endswith(".tif") or fname.lower().endswith(".tiff") or fname.lower().endswith(".png")):
                continue
            patient_id, ear, slice_idx = parse_gt_patient_ear_slice(fname)
            if patient_id is None or ear is None:
                print(f"⚠️ Skipping unrecognized GT mask filename: {fname}")
                continue
            key = (patient_id, ear)
            if key not in groups:
                groups[key] = []
            groups[key].append((fname, slice_idx))

    # For GT, sort by slice index; for pred, sort by filename extractor
    for (patient_id, ear), mask_list in groups.items():
        if has_masks:
            mask_list_sorted = [fname for fname, _ in sorted(mask_list, key=lambda x: x[1])]
        else:
            mask_list_sorted = sorted(mask_list, key=extract_slice_index)
        stack = []
        for fname in mask_list_sorted:
            mask = np.array(Image.open(os.path.join(mask_folder, fname)).convert("L")) > 0
            stack.append(mask)
        stack = np.stack(stack, axis=0)
        cleaned = postprocess_3d_mask(stack)
        for idx, fname in enumerate(mask_list_sorted):
            out_path = os.path.join(out_folder, fname)
            mask_save = (cleaned[idx] * 255).astype(np.uint8)
            Image.fromarray(mask_save).save(out_path)

            # Get info for original image filename
            match = re.match(r"(MRC|PEI)_(\d+)_(\d+)_crop([01])_mask\.png", fname)
            if match:
                prefix, pid, slice_idx, crop_i = match.groups()
                # Rebuild the corresponding input filename
                orig_basename = f"{prefix}_{pid}_{slice_idx}_crop{crop_i}_input.png"
                orig_path = os.path.join(orig_folder, orig_basename)
                if not os.path.exists(orig_path):
                    # Try with .tif extension if .png doesn't exist
                    orig_basename = f"{prefix}_{pid}_{slice_idx}_crop{crop_i}_input.tif"
                    orig_path = os.path.join(orig_folder, orig_basename)
                if not os.path.exists(orig_path):
                    print(f"⚠️ Original image not found for overlay: {orig_path}")
                    continue
            else:
                if not has_masks:  # Only warn for predicted/crop-style, not GT
                    print(f"⚠️ Could not parse filename for overlay: {fname}")
                continue

            # Read original image (grayscale or RGB)
            orig_image = np.array(Image.open(orig_path))
            if orig_image.ndim == 2:
                orig_image = np.stack([orig_image]*3, axis=-1)
            elif orig_image.ndim == 3 and orig_image.shape[2] == 1:
                orig_image = np.repeat(orig_image, 3, axis=-1)
            # The binary mask for overlay (bool or 0/1)
            binary_mask = cleaned[idx]

            overlay_name = fname.replace("_mask.png", "_overlay.png")
            overlay_path = os.path.join(overlay_folder, overlay_name)

            # Call your overlay utility
            os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
            save_segmentation_overlay(image_np=orig_image, mask_np=binary_mask, save_path=overlay_path)
            
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
