# ======================================================================
# File: pipeline_setup/plots.py
# Description: Plotting utilities for YOLO detection outputs.
# Author: @cfusterbarcelo
# Created: 09/04/2025
# ======================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import cv2

def plot_confidence_distribution(detections_dict, save_path):
    """
    Plot histogram of detection confidences per class.
    Args:
        detections_dict: {filename: [(bbox, conf, class_id), ...]}
        save_path: folder to save the plot
    """
    os.makedirs(save_path, exist_ok=True)
    class_confidences = defaultdict(list)

    for fname, entries in detections_dict.items():
        for entry in entries:
            try:
                cls = int(entry["class"])
                conf = float(entry["conf"])
                class_confidences[cls].append(conf)
            except Exception as e:
                print(f"⚠️ Skipping invalid entry in {fname}: {e}")

    plt.figure(figsize=(8, 4))

    for cls, confs in class_confidences.items():
        plt.hist(confs, bins=20, alpha=0.6, label=f"Class {cls}")

    plt.title("Detection Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "confidence_distribution.png"))
    plt.close()

def plot_detection_heatmap(detections_dict, image_shape, save_path):
    """
    Create a heatmap of detection centers.
    Args:
        detections_dict: {filename: [(bbox, conf, class_id), ...]}
        image_shape: (H, W) to size the heatmap
        save_path: folder to save the plot
    """
    os.makedirs(save_path, exist_ok=True)
    H, W = image_shape
    heatmap = np.zeros((H, W))

    for entries in detections_dict.values():
        for entry in entries:
            try:
                x1, y1, x2, y2 = entry["bbox"]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                if 0 <= cy < H and 0 <= cx < W:
                    heatmap[cy, cx] += 1
            except Exception as e:
                print(f"⚠️ Error parsing bbox: {e}")

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Detection Density")
    plt.title("Detection Heatmap")
    plt.savefig(os.path.join(save_path, "detection_heatmap.png"))
    plt.close()

def save_segmentation_overlay(image_np, mask_np, save_path, title=None):
    """
    Save a visualization of the segmentation mask overlaid on the original image.
    Args:
        image_np (H, W, 3): Input image in [0,1] or [0,255] range
        mask_np (H, W): Binary mask (0 or 1)
        save_path (str): Output path to save the figure
        title (str): Optional title for the plot
    """
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)

    # Create red mask with alpha
    overlay = np.zeros((*mask_np.shape, 4))
    overlay[..., 0] = 1.0  # Red
    overlay[..., 3] = 0.4 * mask_np  # Alpha only where mask is 1

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(image_np)
    ax.imshow(overlay)
    ax.axis("off")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_postprocessing_comparison(
    before_folder,
    after_folder,
    patient_id,
    ear,
    num_slices=5
):
    """
    Plots before/after 3D postprocessing for a patient/ear.
    - before_folder: Path to folder with original mask PNGs
    - after_folder: Path to folder with postprocessed mask PNGs
    - patient_id: Patient identifier prefix
    - ear: 'left' or 'right'
    - num_slices: How many slices to show
    """
    # Detect crop index for ear (crop0 = left, crop1 = right)
    crop_idx = 0 if ear.lower() == "left" else 1

    # Get mask files for this patient/ear
    before_files = sorted([
        f for f in os.listdir(before_folder)
        if f.startswith(patient_id) and f"_crop{crop_idx}_" in f and f.endswith("_mask.png")
    ])
    after_files = sorted([
        f for f in os.listdir(after_folder)
        if f.startswith(patient_id) and f"_crop{crop_idx}_" in f and f.endswith("_mask.png")
    ])

    n_show = min(num_slices, len(before_files), len(after_files))
    if n_show == 0:
        print(f"No slices found for {patient_id} ({ear})")
        return

    plt.figure(figsize=(n_show*3, 6))
    for i in range(n_show):
        mask_before = np.array(Image.open(os.path.join(before_folder, before_files[i])))
        mask_after = np.array(Image.open(os.path.join(after_folder, after_files[i])))

        plt.subplot(2, n_show, i+1)
        plt.imshow(mask_before, cmap="gray")
        plt.title(f"Before\nSlice {i}")
        plt.axis("off")
        plt.subplot(2, n_show, n_show+i+1)
        plt.imshow(mask_after, cmap="gray")
        plt.title(f"After\nSlice {i}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_scatter_pred_vs_gt_volume(df, save_path=None, title="Predicted vs GT Volume (per ear)", use_mm3=True):
    """
    Scatter plot comparing predicted and GT volumes per ear (in mm³ or voxels).
    """
    x_col = 'gt_volume_mm3' if use_mm3 else 'gt_volume_voxels'
    y_col = 'pred_volume_mm3' if use_mm3 else 'pred_volume_voxels'
    unit = "mm³" if use_mm3 else "voxels"

    plt.figure(figsize=(7,7))
    colors = df['ear'].map({'left': 'royalblue', 'right': 'darkorange'})
    plt.scatter(df[x_col], df[y_col], s=80, alpha=0.85, edgecolor='k', c=colors)
    min_v = min(df[x_col].min(), df[y_col].min())
    max_v = max(df[x_col].max(), df[y_col].max())
    plt.plot([min_v, max_v], [min_v, max_v], 'k--', label='Perfect agreement')
    plt.xlabel(f'GT Volume ({unit})')
    plt.ylabel(f'Predicted Volume ({unit})')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

def plot_vsi_bar(df, save_path=None, title="Volume Similarity Index (VSI) per ear"):
    """
    Bar plot of VSI per patient/ear, colored by ear.
    """
    df_sorted = df.sort_values("vsi", ascending=False)
    labels = df_sorted['patient_id'] + "_" + df_sorted['ear']
    colors = df_sorted['ear'].map({'left': 'royalblue', 'right': 'darkorange'})
    plt.figure(figsize=(0.4*len(labels)+2, 4))
    plt.bar(labels, df_sorted['vsi'], color=colors)
    plt.axhline(1, ls='--', c='green', label='Perfect (1.0)')
    plt.ylim(0, 1.05)
    plt.ylabel('VSI')
    plt.xticks(rotation=90, ha='right')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

def plot_volume_difference_bar(df, save_path=None, title="Predicted - GT Volume per ear", use_mm3=True):
    """
    Bar plot of (Predicted - GT) volume per patient/ear.
    """
    df = df.copy()
    if use_mm3:
        df['vol_diff'] = df['pred_volume_mm3'] - df['gt_volume_mm3']
        unit = "mm³"
    else:
        df['vol_diff'] = df['pred_volume_voxels'] - df['gt_volume_voxels']
        unit = "voxels"

    labels = df['patient_id'] + "_" + df['ear']
    colors = df['ear'].map({'left': 'royalblue', 'right': 'darkorange'})
    plt.figure(figsize=(0.4*len(labels)+2, 4))
    plt.bar(labels, df['vol_diff'], color=colors)
    plt.axhline(0, ls='--', c='black')
    plt.ylabel(f'Predicted - GT Volume ({unit})')
    plt.xticks(rotation=90, ha='right')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

def plot_eh_ratio_and_vsi(pred_csv, gt_csv, vsi_csv, results_folder, modality="MRC"):
    """
    Plots EH Ratio (Pred vs GT), their difference, and VSI per ear/patient.
    Saves plots to results_folder and shows them interactively.
    
    Parameters:
        pred_csv: str, path to predicted EH ratio csv
        gt_csv: str, path to GT EH ratio csv
        vsi_csv: str, path to VSI csv
        results_folder: str, folder to save figures
        modality: str, e.g. "MRC" or "PEI" (for filenames/titles)
    """
    # Load data
    pred_df = pd.read_csv(pred_csv)
    gt_df = pd.read_csv(gt_csv)
    vsi_df = pd.read_csv(vsi_csv)

    # Merge EH ratios
    merged = pd.merge(
        pred_df[["patient_id", "ear", "eh_ratio"]].rename(columns={"eh_ratio":"eh_ratio_pred"}),
        gt_df[["patient_id", "ear", "eh_ratio"]].rename(columns={"eh_ratio":"eh_ratio_gt"}),
        on=["patient_id", "ear"], how="inner"
    )
    merged["eh_ratio_pred"] = pd.to_numeric(merged["eh_ratio_pred"], errors="coerce")
    merged["eh_ratio_gt"] = pd.to_numeric(merged["eh_ratio_gt"], errors="coerce")
    merged["diff"] = merged["eh_ratio_pred"] - merged["eh_ratio_gt"]

    labels = merged["patient_id"].astype(str) + " " + merged["ear"].astype(str)
    x = np.arange(len(labels))
    bar_width = 0.35

    # Plot 1: Grouped bar plot of EH Ratio (Pred vs GT)
    plt.figure(figsize=(10,6))
    plt.bar(x - bar_width/2, merged["eh_ratio_pred"], bar_width, label="Predicted", color="#4e79a7")
    plt.bar(x + bar_width/2, merged["eh_ratio_gt"], bar_width, label="Ground Truth", color="#f28e2c")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Patient - Ear")
    plt.ylabel("EH Ratio")
    plt.title(f"EH Ratio: Prediction vs Ground Truth per Patient/Ear ({modality})")
    plt.legend()
    plt.tight_layout()
    eh_ratio_fig_path = os.path.join(results_folder, f"EH_Ratio_{modality}.png")
    plt.savefig(eh_ratio_fig_path)

    # Plot 2: VSI per Ear/Patient
    vsi_df["vsi"] = pd.to_numeric(vsi_df["vsi"], errors="coerce")
    vsi_labels = vsi_df["patient_id"].astype(str) + " " + vsi_df["ear"].astype(str)

    plt.figure(figsize=(10,6))
    plt.bar(vsi_labels, vsi_df["vsi"], color="#76b7b2")
    plt.ylim(0,1)
    plt.xlabel("Patient - Ear")
    plt.ylabel("Volume Similarity Index (VSI)")
    plt.title(f"VSI per Patient/Ear ({modality})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    vsi_fig_path = os.path.join(results_folder, f"VSI_{modality}.png")
    plt.savefig(vsi_fig_path)

    # Plot 3: Difference in EH Ratio as a bar
    plt.figure(figsize=(10,4))
    plt.bar(labels, merged["diff"], color="#e15759")
    plt.axhline(0, color="k", linestyle="--", lw=1)
    plt.ylabel("EH Ratio Difference (Pred - GT)")
    plt.title(f"Difference in EH Ratio (Prediction - Ground Truth) per Patient/Ear ({modality})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    diff_fig_path = os.path.join(results_folder, f"EH_Ratio_Diff_{modality}.png")
    plt.savefig(diff_fig_path)

    print(f"EH Ratio and VSI plots saved to {results_folder}")

