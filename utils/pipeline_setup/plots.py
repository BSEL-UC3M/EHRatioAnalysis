# ======================================================================
# File: pipeline_setup/plots.py
# Description: Plotting utilities for YOLO detection outputs.
# Author: @cfusterbarcelo
# Created: 09/04/2025
# ======================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

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
