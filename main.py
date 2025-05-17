# ==============================================================================
# File: main.py
# Description: Full inference pipeline: EarGate > Object Detection > Segmentation > EH Ratio.
# Author: @cfusterbarcelo
# Created: 25/03/2025
# ==============================================================================

import torch
import warnings
import time
import pandas as pd
import os

from utils.pipeline_setup.EarGate import run_eargate_inference
from utils.pipeline_setup.utils import find_model_by_keywords, setup_pipeline_folders
from utils.pipeline_setup.AuriBox import run_auribox_inference
from utils.pipeline_setup.EHMasker import run_ehmasker_inference
from utils.pipeline_setup.RatioCalculator import compute_eh_ratios
from utils.pipeline_setup.PostProcess3D import postprocess_all_patients_ears, report_mask_volumes

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
start_time = time.time()

# ------------------------------------------------------------------------------
# 🧠 Configuration
# ------------------------------------------------------------------------------

HAS_LABELS = False
# LABELS_CSV = "D:/Data/EHydropsAnalysis/2025-Porcessed/MRC TIFF/MRC_TIFF_Annotations.xlsx"

MODELS_FOLDER = "D:/Models/EHydropsAnalysis/2025/"
RAW_DATA_MRC = "D:/Data/EHydropsAnalysis/2025-Porcessed/MRC-TEST-INFERENCE/cnn_20250326-095831/"
RAW_DATA_PEI = "D:/Data/EHydropsAnalysis/2025-Porcessed/PEI-TEST-INFERENCE/"

RESULTS_FOLDER = "D:/Results/EHydrops/Pipeline-PostProcessed-v3"
folder_paths = setup_pipeline_folders(RESULTS_FOLDER)

MRC_CLASSIF_DIR = folder_paths["classification"]["mrc"]
PEI_CLASSIF_DIR = folder_paths["classification"]["pei"]
MRC_DETECT_DIR = folder_paths["detection"]["mrc"]["base"]
PEI_DETECT_DIR = folder_paths["detection"]["pei"]["base"]
MRC_SEGMENT_DIR = folder_paths["segmentation"]["mrc"]["base"]
PEI_SEGMENT_DIR = folder_paths["segmentation"]["pei"]["base"]
MRC_POSTPROC_MASKS_DIR = os.path.join(MRC_SEGMENT_DIR, "masks_postprocessed")
PEI_POSTPROC_MASKS_DIR = os.path.join(PEI_SEGMENT_DIR, "masks_postprocessed")


# Other config
BATCH_SIZE = 16
CLASS_THRESHOLD = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE = 0.6

# Finding models
MRC_CLASSIFICATION_MODEL = find_model_by_keywords(
    root_folder=MODELS_FOLDER,
    required_keywords=["classifier", "MRC"]
)
PEI_CLASSIFICATION_MODEL = find_model_by_keywords(
    root_folder=MODELS_FOLDER,
    required_keywords=["classifier", "PEI"]
)
MRC_DETECT_MODEL= find_model_by_keywords(
    root_folder=MODELS_FOLDER, 
    required_keywords=["object_detector", "MRC"]
)
PEI_DETECT_MODEL= find_model_by_keywords(
    root_folder=MODELS_FOLDER, 
    required_keywords=["object_detector", "PEI"]
)
MRC_SEGMENT_MODEL = find_model_by_keywords(
    root_folder=MODELS_FOLDER,
    required_keywords=["segmentator", "MRC"]
)
PEI_SEGMENT_MODEL = find_model_by_keywords(
    root_folder=MODELS_FOLDER,
    required_keywords=["segmentator", "PEI"]
)

# PEI_SEGMENTATOR = find_model_by_keywords(MODELS_FOLDER, ["segmentator", "PEI"])

# ------------------------------------------------------------------------------
# 👂 EarGate (Slice Classification + Postprocessing)
# ------------------------------------------------------------------------------

print("\n🌀 STEP 1: EarGate – Classifying slices into ear vs. non-ear\n")

results_mrc_filtered = run_eargate_inference(
    image_folder=RAW_DATA_MRC,
    model_path=MRC_CLASSIFICATION_MODEL,
    device=DEVICE,
    result_folder=MRC_CLASSIF_DIR["base"],
    label_csv=None,
    dataset_type="MRC",
    class_threshold=CLASS_THRESHOLD,
    batch_size=BATCH_SIZE,
)

results_pei_filtered = run_eargate_inference(
    image_folder=RAW_DATA_PEI,
    model_path=PEI_CLASSIFICATION_MODEL,
    device=DEVICE,
    result_folder=PEI_CLASSIF_DIR["base"],
    label_csv=None,
    dataset_type="PEI",
    class_threshold=CLASS_THRESHOLD,
    batch_size=BATCH_SIZE,
)
print("\n✅ EarGate complete! Ready to proceed to object detection...\n")

# -------------------------------------------------------------------------
# 📦 AuriBox (Object Detection for regions of interest)
# -------------------------------------------------------------------------

detections_mrc = run_auribox_inference(
    image_folder=RAW_DATA_MRC,
    model_path=MRC_DETECT_MODEL,
    device=DEVICE,
    result_folder=MRC_DETECT_DIR,
    selected_images=results_mrc_filtered,
    dataset_type="MRC"
)

detections_pei = run_auribox_inference(
    image_folder=RAW_DATA_PEI,
    model_path=PEI_DETECT_MODEL,
    device=DEVICE,
    result_folder=PEI_DETECT_DIR,
    selected_images=results_pei_filtered,
    dataset_type="PEI"
)

# -------------------------------------------------------------------------
# 🧼 EHMasker (Segmentation of cropped regions)
# -------------------------------------------------------------------------

print("\n🫧 STEP 3: EHMasker – Segmenting cropped regions\n")

mrc_segmentation_masks = run_ehmasker_inference(
    image_folder=RAW_DATA_MRC,
    detections=detections_mrc,
    model_path=MRC_SEGMENT_MODEL,
    device=DEVICE,
    result_folder=MRC_SEGMENT_DIR,
    dataset_type="MRC",
    confidence=CONFIDENCE
)

pei_segmentation_masks = run_ehmasker_inference(
    image_folder=RAW_DATA_PEI,
    detections=detections_pei,
    model_path=PEI_SEGMENT_MODEL,
    device=DEVICE,
    result_folder=PEI_SEGMENT_DIR,
    dataset_type="PEI",
    confidence=CONFIDENCE
)

# TODO: only working for some patients
postprocess_all_patients_ears(mask_folder=os.path.join(MRC_SEGMENT_DIR, "masks"),
                             out_folder=MRC_POSTPROC_MASKS_DIR)
postprocess_all_patients_ears(mask_folder=os.path.join(PEI_SEGMENT_DIR, "masks"),
                             out_folder=PEI_POSTPROC_MASKS_DIR)

# TODO: Add for PEI
report_mask_volumes(
    before_folder=os.path.join(MRC_SEGMENT_DIR, "masks"),
    after_folder=os.path.join(MRC_SEGMENT_DIR, "masks_postprocessed"),
    output_csv=os.path.join(RESULTS_FOLDER, "mrc_mask_postproc_comparison.csv")
)

print("\n✅ EHMasker complete! Segmentation results ready for EH Ratio computation...\n")

# -------------------------------------------------------------------------
# 📊 RatioCalculator (Volume computation and EH ratio)
# -------------------------------------------------------------------------

RATIO_OUTPUT_CSV = os.path.join(RESULTS_FOLDER, "eh_volume_ratios.csv")

print("\n📊 STEP 4: RatioCalculator – Computing EH Ratios from segmented masks\n")

compute_eh_ratios(
    mrc_mask_folder=os.path.join(MRC_SEGMENT_DIR, "masks"),
    pei_mask_folder=os.path.join(PEI_SEGMENT_DIR, "masks"),
    output_csv_path=RATIO_OUTPUT_CSV,
)

elapsed = time.time() - start_time
print(f"\n⏱️ Total pipeline runtime: {elapsed:.2f} seconds")