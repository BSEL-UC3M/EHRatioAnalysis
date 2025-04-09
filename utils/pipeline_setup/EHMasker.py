# ======================================================================
# File: pipeline_setup/EHMasker.py
# Description: Segmentation step for MRC or PEI cropped ear regions.
# Author: @cfusterbarcelo
# Created: 09/04/2025
# ======================================================================

import os
import torch
import numpy as np
import cv2
import tifffile as tiff
from PIL import Image
from tqdm import tqdm
from models.segmentator.segmentator import UNetOptimizedDO  # or change to your model
from utils.classification_postprocess import extract_patient_and_index
from utils.pipeline_setup.plots import save_segmentation_overlay

def run_ehmasker_inference(
    image_folder,
    detections,
    model_path,
    device,
    result_folder,
    dataset_type="MRC"
):
    """
    Run segmentation on dynamically cropped ear regions from detected bounding boxes.
    Saves binary masks and returns dict with {filename: [mask_left, mask_right]}
    """
    os.makedirs(result_folder, exist_ok=True)
    masks_folder = os.path.join(result_folder, "masks")
    overlays_folder = os.path.join(result_folder, "overlays")
    input_folder = os.path.join(result_folder, "input")
    os.makedirs(masks_folder, exist_ok=True)
    os.makedirs(overlays_folder, exist_ok=True)
    os.makedirs(input_folder, exist_ok=True)

    # Load model
    model = UNetOptimizedDO().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    masks = {}

    for fname, dets in tqdm(detections.items(), desc=f"{dataset_type} Segmentation"):
        image_path = os.path.join(image_folder, fname)

        try:
            img = tiff.imread(image_path)
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            elif img.shape[2] == 1:
                img = np.concatenate([img]*3, axis=-1)
        except Exception as e:
            print(f"⚠️ Failed to load {fname}: {e}")
            continue

        mask_list = []
        for i, det in enumerate(dets):
            pid, idx = extract_patient_and_index(fname)
            x1, y1, x2, y2 = map(int, det["bbox"])
            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_LINEAR)
            crop = crop.astype(np.float32)
            crop_min = crop.min()
            crop_max = crop.max()
            image = (crop - crop_min) / (crop_max - crop_min + 1e-8)
            input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
            input_vis_path = os.path.join(input_folder, f"{pid}_{idx}_crop{i}_input.png")
            input_vis = (image * 255).astype(np.uint8)
            Image.fromarray(input_vis).save(input_vis_path)

            with torch.no_grad():
                output = model(input_tensor)
                mask = torch.sigmoid(output).squeeze().cpu().numpy()
                binary_mask = (mask > 0.5).astype(np.uint8)

            outname = f"{pid}_{idx}_crop{i}_mask.png"
            mask_path = os.path.join(masks_folder, outname)
            Image.fromarray(binary_mask * 255).save(mask_path)

            # Save overlay
            overlay_path = os.path.join(overlays_folder, f"{pid}_{idx}_crop{i}_overlay.png")
            save_segmentation_overlay(image_np=image, mask_np=binary_mask, save_path=overlay_path)


            mask_list.append(binary_mask)

        masks[fname] = mask_list

    print(f"✅ Saved segmentation masks to: {result_folder}")
    return masks
