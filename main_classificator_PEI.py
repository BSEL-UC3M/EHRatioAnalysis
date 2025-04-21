# ==============================================================================
# File: main_classificator_PEI.py
# Description: Main script for training and evaluating the classification model with PEI data.
# Author: @claudiacastrillon
# Created: 13/02/2025
# ==============================================================================
import torch
import platform
import os
import sys
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix
from dataloader.dataloader_PEI_classificator import ClassificationDataLoader
from models.classificator.five_layer_cnn_PEI import train_model, evaluate_model, FiveLayerCNN
from models.classificator.resnet50 import fine_tune_resnet, train_model, evaluate_model
from utils.preprocessing_all_images import preprocess_all_images


sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# ==============================================================================
# Configuration Parameters
SAVE_RESULTS = input("Save results? (yes/no): ").strip().lower() == "yes"
SAVE_WEIGHTS = input("Save model weights? (yes/no): ").strip().lower() == "yes"
SAVE_PREPROCESSING = False  
LEARNING_RATE = 1e-4  
BATCH_SIZE = 16 
DATA_SPLITS = (0.7, 0.1, 0.2)  
NUM_EPOCHS = 50  

# CAT's paths
# RAW_IMAGES_FOLDER = "D:/Data/EHRatioAnalysis/PEI TIFF"
# ANNOTATIONS_FOLDER = "D:/Data/EHRatioAnalysis"

# CLAUDIA's paths
RAW_IMAGES_FOLDER = "/Users/claudiacastrillonalvarez/Desktop/data/PEI_data/"
ANNOTATIONS_FOLDER = "/Users/claudiacastrillonalvarez/Desktop/data/PEI_data/"

PROCESSED_IMAGES_FOLDER = "/Users/claudiacastrillonalvarez/Desktop/data/PEI_data/"

# Detect OS
system_name = platform.system().lower()

# Select GPU backend based on OS (Windows or macOS)
if system_name == "darwin":  # macOS
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
elif system_name in ["windows", "linux"]:  # Windows or Linux
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")  # Fallback to CPU for unknown OS

print(f" Using device: {device}")

# ==============================================================================
# Step 1: Preprocess Images
if SAVE_PREPROCESSING:
    print("\n Preprocessing PEI images...\n")
    os.makedirs(PROCESSED_IMAGES_FOLDER, exist_ok=True)  # Ensure folder exists
    preprocess_all_images(RAW_IMAGES_FOLDER, PROCESSED_IMAGES_FOLDER)
    print(" Preprocessing complete. Processed images saved in:", PROCESSED_IMAGES_FOLDER)
else:
    print(" Skipping image preprocessing. Using existing processed images.")

# ==============================================================================
# Step 2: Load Dataset


# Load annotations
annotations = ClassificationDataLoader.load_annotations(ANNOTATIONS_FOLDER)

# Step 2.1: obtain all name from annotations
all_patients = list(annotations.keys())

# Step 2.2: use the same patients for validation and test 
val_patients = [
    "PACIENTE 45 PEI TIFF", "PACIENTE 21 PEI TIFF", "PACIENTE 1 PEI TIFF", "PACIENTE 87 PEI TIFF",
    "PACIENTE 58 PEI TIFF", "PACIENTE 85 PEI TIFF", "PACIENTE 54 PEI TIFF", "PACIENTE 90 PEI TIFF",
    "PACIENTE 26 PEI TIFF"
]

test_patients = [
    "PACIENTE 77 PEI TIFF", "PACIENTE 65 PEI TIFF", "PACIENTE 30 PEI TIFF", "PACIENTE 28 PEI TIFF",
    "PACIENTE 81 PEI TIFF", "PACIENTE 88 PEI TIFF", "PACIENTE 5 PEI TIFF", "PACIENTE 55 PEI TIFF",
    "PACIENTE 76 PEI TIFF", "PACIENTE 12 PEI TIFF", "PACIENTE 70 PEI TIFF", "PACIENTE 14 PEI TIFF",
    "PACIENTE 18 PEI TIFF", "PACIENTE 29 PEI TIFF", "PACIENTE 32 PEI TIFF", "PACIENTE 36 PEI TIFF",
    "PACIENTE 4 PEI TIFF", "PACIENTE 15 PEI TIFF", "PACIENTE 82 PEI TIFF"
]

# Step 2.3: the remaining patients are used for training 
train_patients = [p for p in all_patients if p not in val_patients and p not in test_patients]

# Step 2.4: load the dataloaders with full names 
train_loader, val_loader, test_loader = ClassificationDataLoader.train_val_test_split(
    images_folder=PROCESSED_IMAGES_FOLDER,
    annotations=annotations,
    train_patients=train_patients,
    val_patients=val_patients,
    test_patients=test_patients,
    batch_size=BATCH_SIZE,
    transform=None
)



# Step 2.5: Determine the number of classes dynamically
num_classes = len(set(
    annotation 
    for patient_data in annotations.values() 
    for annotation in patient_data['Annotation']
))

# ==============================================================================
# Step 3: User selects the model type
MODEL_TYPE = input("Select model type ('cnn' or 'resnet50'): ").strip().lower()

while MODEL_TYPE not in ["cnn", "resnet50"]:
    MODEL_TYPE = input("Invalid choice. Please select 'cnn' or 'resnet50': ").strip().lower()

print(f"\nTraining {MODEL_TYPE.upper()} model...\n")

# ==============================================================================
# Step 4: Model Definition & Training
if MODEL_TYPE == "cnn":
    model = FiveLayerCNN(num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

elif MODEL_TYPE == "resnet50":
    model, criterion, optimizer, scheduler = fine_tune_resnet(
        num_classes, device, learning_rate=LEARNING_RATE, model_type="resnet50"
    )

# Train the model explicitly
print(f"\n Training {MODEL_TYPE.upper()} model for {NUM_EPOCHS} epochs...\n")
trained_model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=NUM_EPOCHS
)
# Final evaluation over the test set
avg_loss, accuracy, conf_matrix = evaluate_model(
    trained_model, test_loader, device
)

# ==============================================================================
# Step 5: Prepare Result Directory (Ensuring All Outputs Are in the Same Folder)
if SAVE_RESULTS or SAVE_WEIGHTS:
    results_root = "./results/results_classificator/results_classificator_PEI"
    os.makedirs(results_root, exist_ok=True)  # Ensure base directory exists
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(results_root, f"{MODEL_TYPE}_{timestamp}")  # One directory for all results
    os.makedirs(results_dir, exist_ok=True)  # Create timestamped directory

# ==============================================================================
# Step 6: Save Best Weights Based on Validation Loss
if SAVE_WEIGHTS:
    best_epoch = np.argmin(val_losses)  # Find the epoch with the lowest validation loss
    weights_save_path = os.path.join(results_dir, f"{MODEL_TYPE}_best_weights_PEI.pt")  # Ensure it is saved in `results_dir`
    torch.save(trained_model.state_dict(), weights_save_path)
    print(f" Best model weights saved at {weights_save_path} (Epoch {best_epoch + 1})")

# ==============================================================================
# Step 8: Save Results in the Same Directory
if SAVE_RESULTS:
    # Save performance metrics
    with open(os.path.join(results_dir, "results.txt"), "w") as f:
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Number of Epochs: {NUM_EPOCHS}\n")
        f.write(f"Optimizer: {'Adam' if MODEL_TYPE == 'cnn' else 'SGD'}\n")
        f.write(f"Best Epoch: {best_epoch + 1}\n")  # Log the best epoch
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")
        f.write(f"Confusion Matrix:\n{conf_matrix}\n")

    # Save Confusion Matrix Plot
    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Class {i}' for i in range(num_classes)], 
                yticklabels=[f'Class {i}' for i in range(num_classes)])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Save Training & Validation Loss Plot
    epochs_range = range(1, len(train_losses) + 1)  
    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs_range, val_losses, label='Validation Loss', marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "train_val_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Save Training & Validation Accuracy Plot
    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "train_val_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n Results saved in {results_dir}\n")

print("Process completed.")
