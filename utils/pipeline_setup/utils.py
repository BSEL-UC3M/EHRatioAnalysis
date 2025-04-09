# ======================================================================
# File: pipeline_setup/utils.py
# Description: Utils file for inference pipeline
# Author: @cfusterbarcelo
# Created: 08/04/2025
# ==============================================================================

import os
import numpy as np
from pathlib import Path
from tifffile import imread, imwrite
from datetime import datetime
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

def setup_pipeline_folders(base_results_folder):
    """
    Creates a clean folder structure for classification, detection, and segmentation,
    separated by MRC and PEI.

    Returns a dict with paths to each key output directory.
    """
    tasks = ["classification", "detection", "segmentation"]
    datasets = ["mrc", "pei"]
    folder_paths = defaultdict(dict)

    for task in tasks:
        for ds in datasets:
            base = os.path.join(base_results_folder, task, ds)
            os.makedirs(base, exist_ok=True)

            # Initialize with base path
            folder_paths[task][ds] = {"base": base}

            # Extra folders for classification
            if task == "classification":
                plots_path = os.path.join(base, "plots")
                labels_path = os.path.join(base, "plots_with_labels")
                os.makedirs(plots_path, exist_ok=True)
                os.makedirs(labels_path, exist_ok=True)
                folder_paths[task][ds].update({
                    "plots": plots_path,
                    "plots_with_labels": labels_path
                })

            # Extra folders for detection
            if task == "detection":
                crops_path = os.path.join(base, "crops")
                vis_path = os.path.join(base, "visuals")
                os.makedirs(crops_path, exist_ok=True)
                os.makedirs(vis_path, exist_ok=True)
                folder_paths[task][ds].update({
                    "crops": crops_path,
                    "visuals": vis_path
                })

    return folder_paths

def convert_images_to_uint8(image_paths, output_folder=None):
    """
    Convert TIFF images to uint8 and save them to a temporary folder.
    Returns the path to the temp folder and a list of converted filenames.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    base = Path(output_folder or f"./temp_uint8/PEI_{timestamp}")
    base.mkdir(parents=True, exist_ok=True)

    converted_files = []

    for image_path in image_paths:
        try:
            image = imread(image_path).astype("float32")
            min_val, max_val = image.min(), image.max()
            image = (image - min_val) / (max_val - min_val + 1e-6)
            image_uint8 = (image * 255).astype("uint8")

            output_path = base / Path(image_path).name
            imwrite(str(output_path), image_uint8)
            converted_files.append(str(output_path))
        except Exception as e:
            print(f"❌ Error converting {image_path}: {e}")

    return str(base), converted_files