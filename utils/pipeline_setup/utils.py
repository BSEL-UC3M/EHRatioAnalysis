# ======================================================================
# File: pipeline_setup/utils.py
# Description: Utils file for inference pipeline
# Author: @cfusterbarcelo
# Created: 08/04/2025
# ==============================================================================

import os
import numpy as np
from scipy.ndimage import gaussian_filter
from collections import defaultdict

def find_model_by_keywords(root_folder, required_keywords, extension=".pt"):
    """
    Search recursively in `root_folder` for a model file (.pt) that contains all `required_keywords`.
    Returns the path of the first match found, or None.
    """
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith(extension) and all(k.lower() in file.lower() for k in required_keywords):
                return os.path.join(dirpath, file)
    return None

def histogram_adjustment(image, lower_threshold_factor=2.4, upper_threshold_factor=2.2):
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)

    lower_threshold = mean_intensity - lower_threshold_factor * std_intensity
    upper_threshold = mean_intensity + upper_threshold_factor * std_intensity

    adjusted_image = np.clip(image, lower_threshold, upper_threshold)
    adjusted_image = (adjusted_image - adjusted_image.min()) / (adjusted_image.max() - adjusted_image.min())

    return adjusted_image

def invert_image(image):
    return 1.0 - image

def preprocess_pei_image(img_array):
    """
    Apply PEI preprocessing steps to a single image (as numpy array scaled 0-1).
    """
    img_adjusted = 2 * img_array + gaussian_filter(img_array, sigma=6)
    img_adjusted = histogram_adjustment(img_adjusted)
    img_inverted = invert_image(img_adjusted)
    return img_inverted

def setup_pipeline_folders(base_results_folder, timestamp):
    """
    Creates a clean folder structure for classification, detection, and segmentation,
    separated by MRC and PEI under a timestamped run.

    Returns a dict with paths to each key output directory.
    """
    tasks = ["classification", "detection", "segmentation"]
    datasets = ["mrc", "pei"]
    folder_paths = defaultdict(dict)

    for task in tasks:
        for ds in datasets:
            base = os.path.join(base_results_folder, task, ds, timestamp)
            os.makedirs(base, exist_ok=True)

            # Extra subfolders for classification
            if task == "classification":
                plots_path = os.path.join(base, "plots")
                labels_path = os.path.join(base, "plots_with_labels")
                os.makedirs(plots_path, exist_ok=True)
                os.makedirs(labels_path, exist_ok=True)

                folder_paths[task][ds] = {
                    "base": base,
                    "plots": plots_path,
                    "plots_with_labels": labels_path
                }
            else:
                folder_paths[task][ds] = {
                    "base": base
                }

    return folder_paths