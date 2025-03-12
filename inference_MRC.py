# ==============================================================================
# File: inference_MRC.py
# Description: Perform inference using a CNN with frozen weights and correct confused labels.
# Author: @claudiacastrillon
# Created: 10/03/2025
# ==============================================================================

import torch
import platform
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
            probs = torch.softmax(outputs, dim=1)  # Get probabilities
            confidence, preds = torch.max(probs, 1)  # Get highest confidence score

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            for fname, label, pred, conf in zip(filenames, labels.cpu().numpy(), preds.cpu().numpy(), confidence.cpu().numpy()):
                patient_id = fname.split("_")[0]  # Extract patient ID
                slice_number = int(fname.split("_")[-1].split(".")[0])  # Extract slice number

                if patient_id not in predictions_dict:
                    predictions_dict[patient_id] = []
                predictions_dict[patient_id].append((slice_number, pred, conf))  # Store confidence

    return predictions_dict, y_true, y_pred  # ✅ Make sure the order is correct



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
    weights_path = "/Users/claudiacastrillonalvarez/Desktop/weights/cnn_best_weights_MRC.pt"
    from datetime import datetime

    # Define base directory for saving results
    results_root = "./results/results_classificator/results_classificator_MRC/inference_confused_labels"
    os.makedirs(results_root, exist_ok=True)  # Ensure the base directory exists

    # Generate a timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(results_root, f"correction_{timestamp}")  # Timestamped directory
    os.makedirs(results_dir, exist_ok=True)  # Create the directory
    
    # Detect OS
    system_name = platform.system().lower()

    # Select GPU backend based on OS (Windows or macOS)
    if system_name == "darwin":  # macOS
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif system_name in ["windows", "linux"]:  # Windows or Linux
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")  # Fallback to CPU for unknown OS

    print(f"✅ Using device: {device}")

    # Load dataset
    IMAGES_FOLDER = "/Users/claudiacastrillonalvarez/Desktop/data/MRC_data/MRC_images/"
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
    predictions_dict, y_true, y_pred = inference(model, test_loader, device)

    print("✅ Inference Completed")

  

    # # ✅ Now Compute Confusion Matrix AFTER Fixing Length
    conf_matrix_before = confusion_matrix(y_true, y_pred)

    plot_confusion_matrix(conf_matrix_before, "Confusion Matrix Before Post-Processing", f"{results_dir}/confusion_matrix_before.png")

    # Apply confused label correction
    corrected_predictions_dict = correct_confused_labels(predictions_dict)

    # Reconstruct y_pred after correction
    y_pred_corrected = [label for patient in corrected_predictions_dict.values() for _, label in patient]

    # Compute confusion matrix AFTER post-processing
    conf_matrix_after = confusion_matrix(y_true, y_pred_corrected)
    plot_confusion_matrix(conf_matrix_after, "Confusion Matrix After Post-Processing", f"{results_dir}/confusion_matrix_after.png")



from datetime import datetime

# ==============================================================================
# ✅ Step 5: Save Confusion Matrices in a New Folder for Comparison

# Define base directory
results_root = "./results/results_classificator/results_classificator_MRC/inference_confused_labels"
os.makedirs(results_root, exist_ok=True)  # Ensure the base directory exists

# Generate a timestamp for this correction run
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
results_dir = os.path.join(results_root, f"correction_{timestamp}")  # New timestamped directory
os.makedirs(results_dir, exist_ok=True)  # Create the directory

# ✅ Save Confusion Matrix BEFORE Handling Confused Labels
plot_confusion_matrix(conf_matrix_before, 
                      "Confusion Matrix Before Label Correction", 
                      os.path.join(results_dir, "confusion_matrix_before.png"))

# ✅ Save Confusion Matrix AFTER Handling Confused Labels
plot_confusion_matrix(conf_matrix_after, 
                      "Confusion Matrix After Label Correction", 
                      os.path.join(results_dir, "confusion_matrix_after.png"))

# ✅ Save additional comparison data (optional)
with open(os.path.join(results_dir, "results_comparison.txt"), "w") as f:
    f.write("Confusion Matrix Before Correction:\n")
    f.write(str(conf_matrix_before) + "\n\n")
    f.write("Confusion Matrix After Correction:\n")
    f.write(str(conf_matrix_after) + "\n")

print(f"✅ Confusion Matrices saved in {results_dir}")
