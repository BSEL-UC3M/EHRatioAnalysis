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
from utils.classification_postprocess import smooth_classification_predictions, plot_comparison, save_comparison_csv


CLASSIFICATION_MODEL = "D:/GitHub/EHRatioAnalysis/results/results_classificator/results_classificator_MRC/cnn_20250326-095831/cnn_best_weights.pt"
MRC_IMAGES_FOLDER = "D:/Data/EHydropsAnalysis/2025-Porcessed/MRC-TEST-INFERENCE/cnn_20250326-095831/"
PEI_IMAGES_FOLDER = "D:/Data/EHydropsAnalysis/2025-Porcessed/PEI TIFF/"
BATCH_SIZE = 16
CLASSIFICATION_OUTPUT = "D:/Results/EHydrops/"

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
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        for fname, pred in zip(filenames, predictions):
            results.append((fname, pred))

#----------------------------- PostProcess -----------------------------
print("🧹 Running post-processing to clean classification results")

# Apply smoothing and continuity enforcement
cleaned_results = smooth_classification_predictions(results)


# DEBUGGING: Preview first few before vs after
print("\n🧾 Sample comparison:")
for (fname1, pred1), (_, pred2) in zip(results[:10], cleaned_results[:10]):
    print(f"{fname1}: {pred1} ➡️ {pred2}")

# Save visual comparison plots per patient
plot_comparison(results, cleaned_results, save_path=os.path.join(CLASSIFICATION_OUTPUT, "plots"))

# Save comparison CSV
save_comparison_csv(results, cleaned_results, save_path=os.path.join(CLASSIFICATION_OUTPUT, "comparison.csv"))

print(f"✅ Post-processed results saved to: {CLASSIFICATION_OUTPUT}")

