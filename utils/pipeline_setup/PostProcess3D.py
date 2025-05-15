import os
import numpy as np
import pandas as pd
from PIL import Image
from utils.pipeline_setup.utils import postprocess_3d_mask

def postprocess_all_patients_ears(mask_folder, out_folder):
    """
    Groups all masks by patient+ear, applies 3D postprocessing, saves 2D cleaned masks.
    Assumes mask filenames like {pid}_{idx}_crop{i}_mask.png.
    """
    os.makedirs(out_folder, exist_ok=True)
    # Build dict: {(pid, ear): [(slice_idx, mask_path), ...]}
    groups = {}

    for fname in sorted(os.listdir(mask_folder)):
        if not fname.endswith("_mask.png"):
            continue
        base = fname.replace("_mask.png", "")
        parts = base.split("_")
        pid = "_".join(parts[:2])
        crop_i = int(parts[-1].replace("crop", ""))
        ear = "left" if crop_i == 0 else "right"  # adjust if needed

        key = (pid, ear)
        if key not in groups:
            groups[key] = []
        # You might want to parse idx for slice sorting, here we just use mask list order
        groups[key].append((crop_i, fname))  # If you have true slice order, use it

    # For each (patient, ear), stack, postprocess, and save back
    for (pid, ear), mask_list in groups.items():
        # sort by slice index (here by crop_i, if needed you can parse further)
        mask_list_sorted = sorted(mask_list, key=lambda x: x[0])
        stack = []
        for crop_i, fname in mask_list_sorted:
            mask = np.array(Image.open(os.path.join(mask_folder, fname)).convert("L"))
            stack.append(mask > 127)
        stack = np.stack(stack, axis=0)  # [num_slices, H, W]
        cleaned = postprocess_3d_mask(stack)
        # Save each cleaned mask slice
        for idx, (crop_i, fname) in enumerate(mask_list_sorted):
            out_path = os.path.join(out_folder, fname)
            Image.fromarray((cleaned[idx] * 255).astype(np.uint8)).save(out_path)

def report_mask_volumes(
    before_folder,
    after_folder,
    output_csv="mask_volume_comparison.csv"
):
    """
    Save a CSV comparing mask voxel volumes before/after postprocessing for each patient/ear.
    """
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

def volume_difference_report(before_folder, after_folder, output_csv="mask_volume_comparison.csv"):
    # Map: (pid, ear) -> list of mask file paths
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
        before_volume = 0
        after_volume = 0

        for f in before_masks:
            before_volume += (np.array(Image.open(f)) > 127).sum()
        for f in after_masks:
            after_volume += (np.array(Image.open(f)) > 127).sum()

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
    print(f"Report saved to {output_csv}")
    return df