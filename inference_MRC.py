# ==============================================================================
# File: inference_MRC.py
# Description: Perform inference using a CNN with frozen weights and correct confused labels.
# Author: @claudiacastrillon
# Created: 10/03/2025
# ==============================================================================

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from dataloader.dataloader_MRC_classificator import ClassificationDataLoader
from models.classificator.five_layer_cnn import FiveLayerCNN
from utils.handle_confused_labels import correct_confused_labels
from collections import OrderedDict
import re  # To extract numeric part of patient ID

# ==============================================================================
# ✅ Step 1: Load Model and Weights
def load_model(weights_path, device, num_classes=2):
    """
    Load the trained CNN model with frozen weights for inference.
    """
    model = FiveLayerCNN(num_classes).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print("✅ CNN Model Loaded with Frozen Weights")
    return model

# ==============================================================================
# ✅ Step 2: Perform Inference
def inference(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    predictions_dict = {}

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:  # Case when filenames are missing
                images, labels = batch
                filenames = [f"Unknown_{i}" for i in range(len(labels))]  # Dummy filenames
            else:
                images, labels, filenames = batch  # Expected case

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            for fname, label, pred in zip(filenames, labels.cpu().numpy(), preds.cpu().numpy()):
                patient_id = fname.split("_")[0]  # Extract patient ID
                slice_number = int(fname.split("_")[-1].split(".")[0])  # Extract slice number

                if patient_id not in predictions_dict:
                    predictions_dict[patient_id] = []
                predictions_dict[patient_id].append((slice_number, pred))

    # ✅ Sort predictions within each patient by slice number
    for patient_id in predictions_dict:
        predictions_dict[patient_id].sort(key=lambda x: x[0])  # Sort by slice number

    # ✅ Function to extract numeric part of patient ID for correct ordering
    def extract_patient_number(patient_key):
        match = re.search(r'\d+', patient_key)  # Find the first number in the string
        return int(match.group()) if match else float('inf')  # Default to a high number if no match

    # ✅ Sort dictionary by extracted patient number
    ordered_predictions_dict = OrderedDict(
        sorted(predictions_dict.items(), key=lambda item: extract_patient_number(item[0]))
    )

    return ordered_predictions_dict, y_true, y_pred  # Return sorted predictions



# ==============================================================================
# ✅ Step 3: Compute and Save Confusion Matrices
def plot_confusion_matrix(cm, title, save_path):
    """
    Save confusion matrix as an image.
    """
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==============================================================================
# ✅ Step 4: Main Execution for Inference
if __name__ == "__main__":
    # Define paths and device
    weights_path = "results/results_classificator/results_classificator_MRC/cnn_20250310-131824/cnn_best_weights.pt"
    results_dir = os.path.dirname(weights_path)  # Save results in the same directory as the model
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load dataset
    IMAGES_FOLDER = "/Users/claudiacastrillonalvarez/Desktop/data/MRC_data/MRC_images"
    annotations = ClassificationDataLoader.load_annotations(IMAGES_FOLDER)
    _, _, test_loader = ClassificationDataLoader.train_val_test_split(
        images_folder=IMAGES_FOLDER,
        annotations=annotations,
        splits=(0.7, 0.1, 0.2),
        batch_size=16,
        shuffle=False,
        transform=None
    )

    # Load model
    model = load_model(weights_path, device)

    # Perform inference
    y_true, y_pred, predictions_dict = inference(model, test_loader, device)
    print("✅ Inference Completed")

    # Compute confusion matrix BEFORE post-processing
    conf_matrix_before = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(conf_matrix_before, "Confusion Matrix Before Post-Processing", f"{results_dir}/confusion_matrix_before.png")

    # Apply confused label correction
    corrected_predictions_dict = correct_confused_labels(predictions_dict)

    # Reconstruct y_pred after correction
    y_pred_corrected = [label for patient in corrected_predictions_dict.values() for _, label in patient]

    # Compute confusion matrix AFTER post-processing
    conf_matrix_after = confusion_matrix(y_true, y_pred_corrected)
    plot_confusion_matrix(conf_matrix_after, "Confusion Matrix After Post-Processing", f"{results_dir}/confusion_matrix_after.png")

    print(f"✅ Confusion Matrices Saved in {results_dir}")