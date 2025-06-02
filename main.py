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

from pipeline_scripts.EarGate import run_eargate_inference
from pipeline_scripts.utils import find_model_by_keywords, setup_pipeline_folders, save_run_metadata
from pipeline_scripts.AuriBox import run_auribox_inference
from pipeline_scripts.EHMasker import run_ehmasker_inference
from pipeline_scripts.RatioCalculator import compute_eh_ratios
from pipeline_scripts.PostProcessAndEvaluate import postprocess_pred_and_gt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
start_time = time.time()

# ------------------------------------------------------------------------------
# 🧠 Configuration
# ------------------------------------------------------------------------------

HAS_LABELS = True
LABELS_MRC_CSV = "D:/Data/EHydropsAnalysis/2025-Porcessed/NEW_LABELLED_PATIENTS/MRC_TIFF_Annotations.xlsx"
LABELS_PEI_CSV = "D:/Data/EHydropsAnalysis/2025-Porcessed/NEW_LABELLED_PATIENTS/PEI_TIFF_Annotations.xlsx"

MODELS_FOLDER = "D:/Models/EHydropsAnalysis/2025/"
RAW_DATA_MRC = "D:/Data/EHydropsAnalysis/2025-Porcessed/NEW_LABELLED_PATIENTS/images/MRC/"
RAW_DATA_PEI = "D:/Data/EHydropsAnalysis/2025-Porcessed/NEW_LABELLED_PATIENTS/images/PEI/"

HAS_MASKS = True
MASKS_MRC = "D:/Data/EHydropsAnalysis/2025-Porcessed/NEW_LABELLED_PATIENTS/masks/MRC/"
MASKS_PEI = "D:/Data/EHydropsAnalysis/2025-Porcessed/NEW_LABELLED_PATIENTS/masks/PEI/"

RESULTS_FOLDER = "D:/Results/EHydrops/Pipeline-Evaluation"
folder_paths = setup_pipeline_folders(RESULTS_FOLDER)
rel_dir = os.path.join(RESULTS_FOLDER, "REL")
os.makedirs(rel_dir, exist_ok=True)
REL_PATH = os.path.join(rel_dir, "eh_volume_ratios")

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
MRC_CONFIDENCE = 0.7
PEI_CONFIDENCE = 0.55

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
    label_csv=LABELS_MRC_CSV,
    dataset_type="MRC",
    class_threshold=CLASS_THRESHOLD,
    batch_size=BATCH_SIZE,
)

results_pei_filtered = run_eargate_inference(
    image_folder=RAW_DATA_PEI,
    model_path=PEI_CLASSIFICATION_MODEL,
    device=DEVICE,
    result_folder=PEI_CLASSIF_DIR["base"],
    label_csv=LABELS_PEI_CSV,
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
    mrc_confidence=MRC_CONFIDENCE
)

pei_segmentation_masks = run_ehmasker_inference(
    image_folder=RAW_DATA_PEI, 
    detections=detections_pei,
    model_path=PEI_SEGMENT_MODEL,
    device=DEVICE,
    result_folder=PEI_SEGMENT_DIR,
    dataset_type="PEI",
    pei_confidence=PEI_CONFIDENCE
)

print("\n✅ EHMasker complete! Segmentation results ready for EH Ratio computation...\n")

# -------------------------------------------------------------------------
# 🚦 3D Post Process
# -------------------------------------------------------------------------

postprocess_pred_and_gt(
    orig_folder=os.path.join(MRC_SEGMENT_DIR, "tiff"),
    pred_mask_folder=os.path.join(MRC_SEGMENT_DIR, "masks"),
    out_pred_folder=MRC_POSTPROC_MASKS_DIR,
    overlay_pred_folder=os.path.join(MRC_SEGMENT_DIR, "overlays_pp"),
    gt_mask_folder=MASKS_MRC if HAS_MASKS else None,
    out_gt_folder=os.path.join(MRC_SEGMENT_DIR, "masks_gt_postprocessed") if HAS_MASKS else None,
    overlay_gt_folder=os.path.join(MRC_SEGMENT_DIR, "overlays_gt_pp") if HAS_MASKS else None,
)

postprocess_pred_and_gt(
    orig_folder=os.path.join(PEI_SEGMENT_DIR, "tiff"),
    pred_mask_folder=os.path.join(PEI_SEGMENT_DIR, "masks"),
    out_pred_folder=PEI_POSTPROC_MASKS_DIR,
    overlay_pred_folder=os.path.join(PEI_SEGMENT_DIR, "overlays_pp"),
    gt_mask_folder=MASKS_PEI if HAS_MASKS else None,
    out_gt_folder=os.path.join(PEI_SEGMENT_DIR, "masks_gt_postprocessed") if HAS_MASKS else None,
    overlay_gt_folder=os.path.join(PEI_SEGMENT_DIR, "overlays_gt_pp") if HAS_MASKS else None,
)


# -------------------------------------------------------------------------
# 📊 RatioCalculator (Volume computation and EH ratio)
# -------------------------------------------------------------------------

RATIO_OUTPUT_CSV = os.path.join(RESULTS_FOLDER, "eh_volume_ratios.csv")

print("\n📊 STEP 4: RatioCalculator – Computing EH Ratios from segmented masks\n")

compute_eh_ratios(
    mrc_mask_folder=MRC_POSTPROC_MASKS_DIR,
    pei_mask_folder=PEI_POSTPROC_MASKS_DIR,
    output_path=REL_PATH,
    mrc_gt_mask_folder=os.path.join(MRC_SEGMENT_DIR, "masks_gt_postprocessed") if HAS_MASKS else None,
    pei_gt_mask_folder=os.path.join(PEI_SEGMENT_DIR, "masks_gt_postprocessed") if HAS_MASKS else None,
)

# Gather parameters to save
params_dict = {
    "BATCH_SIZE": BATCH_SIZE,
    "CLASS_THRESHOLD": CLASS_THRESHOLD,
    "MRC_CONFIDENCE": MRC_CONFIDENCE,
    "PEI_CONFIDENCE": PEI_CONFIDENCE,
    "DEVICE": DEVICE,
    "MRC_CLASSIFICATION_MODEL": MRC_CLASSIFICATION_MODEL,
    "PEI_CLASSIFICATION_MODEL": PEI_CLASSIFICATION_MODEL,
    "MRC_DETECT_MODEL": MRC_DETECT_MODEL,
    "PEI_DETECT_MODEL": PEI_DETECT_MODEL,
    "MRC_SEGMENT_MODEL": MRC_SEGMENT_MODEL,
    "PEI_SEGMENT_MODEL": PEI_SEGMENT_MODEL,
    "RAW_DATA_MRC": RAW_DATA_MRC,
    "RAW_DATA_PEI": RAW_DATA_PEI,
    "RESULTS_FOLDER": RESULTS_FOLDER,
}
# Save metadata
elapsed = time.time() - start_time
save_run_metadata(
    results_folder=RESULTS_FOLDER,
    params_dict=params_dict,
    elapsed_seconds=elapsed,
    extra_notes=None  # or "Anything else you want to add"
)


print(f"\n⏱️ Total pipeline runtime: {elapsed:.2f} seconds")