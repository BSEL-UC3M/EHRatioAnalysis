# ==============================================================================
# File: main.py
# Description: Full inference pipeline: classification > detection > segmentation
#  > EH Ratio Calculation.
# Author: @cfusterbarcelo
# Created: 25/03/2025
# ==============================================================================

import os
import torch
import numpy as np
from datetime import datetime

from dataloader.dataloader_MRC_classificator import load_inference_dataloader
from models.classificator.five_layer_cnn import FiveLayerCNN
from utils.classification_postprocess import smooth_classification_predictions, plot_comparison, plot_comparison_with_labels, save_comparison_csv

HAS_LABELS = True
LABELS_CSV = "D:/Data/EHydropsAnalysis/2025-Porcessed/MRC TIFF/MRC_TIFF_Annotations.xlsx"
CLASSIFICATION_MODEL = "D:/GitHub/EHRatioAnalysis/results/results_classificator/results_classificator_MRC/cnn_20250402-130217/cnn_best_weights.pt"
MRC_IMAGES_FOLDER = "D:/Data/EHydropsAnalysis/2025-Porcessed/MRC-TEST-INFERENCE/cnn_20250326-095831/"
PEI_IMAGES_FOLDER = "D:/Data/EHydropsAnalysis/2025-Porcessed/PEI TIFF/"
BATCH_SIZE = 16
CLASSIFICATION_OUTPUT = "D:/GitHub/EHRatioAnalysis/results/results_classificator/results_classificator_MRC/cnn_20250402-130217/inference"
THRESHOLD = 0.2 # Confidence threshold to bias towards class 1

os.makedirs(CLASSIFICATION_OUTPUT, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")


# ✅ Update device compatibility for Windows (no MPS support there)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################### CLASSIFICATION ###############################
#----------------------------- Inference -----------------------------
print("🚀 Running inference on classification of slices")
model = FiveLayerCNN(num_classes=2).to(DEVICE)
model.load_state_dict(torch.load(CLASSIFICATION_MODEL, map_location=DEVICE))
model.eval()

# Load test images
inference_loader = load_inference_dataloader(MRC_IMAGES_FOLDER, batch_size=BATCH_SIZE)

# Run inference
results = []
with torch.no_grad():
    for batch in inference_loader:
        images = batch["image"].to(DEVICE)
        filenames = batch["filename"]
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        predictions = (probs[:, 1] > THRESHOLD).long().cpu().numpy()
        
        for fname, pred in zip(filenames, predictions):
            results.append((fname, pred))

#----------------------------- PostProcess -----------------------------
print("🧹 Running post-processing to clean classification results")

# Apply smoothing and continuity enforcement
cleaned_results = smooth_classification_predictions(results)

# Save visual comparison plots per patient
if HAS_LABELS:
    plot_comparison_with_labels(
        before=results,
        after=cleaned_results,
        label_csv=LABELS_CSV,
        save_path=os.path.join(CLASSIFICATION_OUTPUT, "plots_with_labels")
    )
else:
    plot_comparison(results, cleaned_results, save_path=os.path.join(CLASSIFICATION_OUTPUT, "plots"))

# Save comparison CSV
save_comparison_csv(results, cleaned_results, save_path=os.path.join(CLASSIFICATION_OUTPUT, "comparison.csv"))

print(f"✅ Post-processed results saved to: {CLASSIFICATION_OUTPUT}")

