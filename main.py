# ==============================================================================
# File: main.py
# Description: Full inference pipeline: EarGate > Object Detection > Segmentation > EH Ratio.
# Author: @cfusterbarcelo
# Created: 25/03/2025
# ==============================================================================

import os
import torch
import warnings
from datetime import datetime
from utils.pipeline_setup.EarGate import run_eargate_inference
from utils.pipeline_setup.utils import find_model_by_keywords, setup_pipeline_folders

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------------------------
# 🧠 Configuration
# ------------------------------------------------------------------------------

HAS_LABELS = False
# LABELS_CSV = "D:/Data/EHydropsAnalysis/2025-Porcessed/MRC TIFF/MRC_TIFF_Annotations.xlsx"

MODELS_FOLDER = "D:/Models/EHydropsAnalysis/2025/"
RAW_DATA_MRC = "D:/Data/EHydropsAnalysis/2025-Porcessed/MRC-TEST-INFERENCE/cnn_20250326-095831/"
RAW_DATA_PEI = "D:/Data/EHydropsAnalysis/2025-Porcessed/PEI-TEST-INFERENCE/"

RESULTS_FOLDER = "./results/pipeline"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
folder_paths = setup_pipeline_folders(RESULTS_FOLDER, timestamp)

MRC_CLASSIF_DIR = folder_paths["classification"]["mrc"]
PEI_CLASSIF_DIR = folder_paths["classification"]["pei"]


# Other config
BATCH_SIZE = 16
CLASS_THRESHOLD = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Finding models
MRC_CLASSIFICATION_MODEL = find_model_by_keywords(
    root_folder=MODELS_FOLDER,
    required_keywords=["classifier", "MRC"]
)
PEI_CLASSIFICATION_MODEL = find_model_by_keywords(
    root_folder=MODELS_FOLDER,
    required_keywords=["classifier", "PEI"]
)
# MRC_OBJECT_DETECTOR = find_model_by_keywords(MODELS_FOLDER, ["object_detector", "MRC"])
# PEI_SEGMENTATOR = find_model_by_keywords(MODELS_FOLDER, ["segmentator", "PEI"])

# ------------------------------------------------------------------------------
# 👂 EarGate (Slice Classification + Postprocessing)
# ------------------------------------------------------------------------------

print("\n🌀 STEP 1: EarGate – Classifying slices into ear vs. non-ear\n")

results_mrc = run_eargate_inference(
    image_folder=RAW_DATA_MRC,
    model_path=MRC_CLASSIFICATION_MODEL,
    device=DEVICE,
    result_folder=MRC_CLASSIF_DIR["base"],
    label_csv=None,
    dataset_type="MRC",
    class_threshold=CLASS_THRESHOLD,
    batch_size=BATCH_SIZE
)

results_pei = run_eargate_inference(
    image_folder=RAW_DATA_PEI,
    model_path=PEI_CLASSIFICATION_MODEL,
    device=DEVICE,
    result_folder=PEI_CLASSIF_DIR["base"],
    label_csv=None,
    dataset_type="PEI",
    class_threshold=CLASS_THRESHOLD,
    batch_size=BATCH_SIZE
)

# At this point you have two lists:
# - `mrc_cleaned`: [(filename, 0 or 1)] for MRC
# - `pei_cleaned`: [(filename, 0 or 1)] for PEI
# You can now move to object detection using only filenames where label == 1.

print("\n✅ EarGate complete! Ready to proceed to object detection...\n")
