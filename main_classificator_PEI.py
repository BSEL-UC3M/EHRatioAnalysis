# ==============================================================================
# File: main_classificator_PEI.py
# Description: Main script for training and evaluating the classification model with PEI data.
# Author: @claudiacastrillon
# Modified: 02/07/2025 by @ChatGPT for dynamic data splits
# ==============================================================================

import torch
import platform
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from dataloader.dataloader_PEI_classificator import ClassificationDataLoader
from models.classificator.five_layer_cnn_PEI import train_model as train_model_cnn, evaluate_model as eval_model_cnn, FiveLayerCNN
from models.classificator.resnet50 import fine_tune_resnet, train_model as train_model_resnet, evaluate_model as eval_model_resnet
from utils.preprocessing_all_images import preprocess_all_images

# ==============================================================================
# Configuration Parameters
# ==============================================================================
SAVE_RESULTS = False
SAVE_WEIGHTS = False
SAVE_PREPROCESSING = False

MODEL_TYPE = "resnet50"  # Choose between 'cnn' and 'resnet50'
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 1

# Dataset paths
RAW_IMAGES_FOLDER = "D:/Data/EHydropsAnalysis/paper-experiments/classification/PEI"
ANNOTATIONS_FOLDER = RAW_IMAGES_FOLDER
PROCESSED_IMAGES_FOLDER = "D:/Data/EHydropsAnalysis/paper-experiments/classification/PEI-PREPROCESSED"
RESULTS_ROOT = "D:/Results/EHydrops/Paper-experiments/classification/PEI"

# ==============================================================================
# Environment Setup
# ==============================================================================
system_name = platform.system().lower()
if system_name == "darwin":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
elif system_name in ["windows", "linux"]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

print(f" Using device: {device}")

# ==============================================================================
# Step 1: Preprocess Images
# ==============================================================================
if SAVE_PREPROCESSING:
    print("\n Preprocessing PEI images...\n")
    os.makedirs(PROCESSED_IMAGES_FOLDER, exist_ok=True)
    preprocess_all_images(RAW_IMAGES_FOLDER, PROCESSED_IMAGES_FOLDER)
    print(" Preprocessing complete. Processed images saved in:", PROCESSED_IMAGES_FOLDER)
else:
    print(" Skipping image preprocessing. Using existing processed images.")

# ==============================================================================
# Step 2: Load Dataset
# ==============================================================================
annotations = ClassificationDataLoader.load_annotations(ANNOTATIONS_FOLDER)
all_patients = list(annotations.keys())

# Split: 70% train, 10% val, 20% test
train_val_patients, test_patients = train_test_split(all_patients, test_size=0.2, random_state=42)
train_patients, val_patients = train_test_split(train_val_patients, test_size=0.125, random_state=42)  # 0.125 * 0.8 = 0.1

train_loader, val_loader, test_loader = ClassificationDataLoader.train_val_test_split(
    images_folder=PROCESSED_IMAGES_FOLDER,
    annotations=annotations,
    train_patients=train_patients,
    val_patients=val_patients,
    test_patients=test_patients,
    batch_size=BATCH_SIZE,
    transform=None
)

num_classes = len(set(
    annotation
    for patient_data in annotations.values()
    for annotation in patient_data['Annotation']
))

# ==============================================================================
# Step 3: Model Setup & Training
# ==============================================================================
print(f"\nTraining {MODEL_TYPE.upper()} model...\n")

if MODEL_TYPE == "cnn":
    model = FiveLayerCNN(num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    train_fn, eval_fn = train_model_cnn, eval_model_cnn

elif MODEL_TYPE == "resnet50":
    model, criterion, optimizer, scheduler = fine_tune_resnet(
        num_classes, device, learning_rate=LEARNING_RATE, model_type="resnet50"
    )
    train_fn, eval_fn = train_model_resnet, eval_model_resnet

print(f"\n Training {MODEL_TYPE.upper()} model for {NUM_EPOCHS} epochs...\n")
trained_model, train_losses, val_losses, train_accuracies, val_accuracies = train_fn(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=NUM_EPOCHS
)

avg_loss, accuracy, conf_matrix = eval_fn(trained_model, test_loader, device)

# ==============================================================================
# Step 4: Save Results
# ==============================================================================
if SAVE_RESULTS or SAVE_WEIGHTS:
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(RESULTS_ROOT, f"{MODEL_TYPE}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

if SAVE_WEIGHTS:
    best_epoch = np.argmin(val_losses)
    weights_save_path = os.path.join(results_dir, f"{MODEL_TYPE}_best_weights_PEI.pt")
    torch.save(trained_model.state_dict(), weights_save_path)
    print(f" Best model weights saved at {weights_save_path} (Epoch {best_epoch + 1})")

if SAVE_RESULTS:
    with open(os.path.join(results_dir, "results.txt"), "w") as f:
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Number of Epochs: {NUM_EPOCHS}\n")
        f.write(f"Optimizer: {'Adam' if MODEL_TYPE == 'cnn' else 'SGD'}\n")
        f.write(f"Best Epoch: {best_epoch + 1}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")
        f.write(f"Confusion Matrix:\n{conf_matrix}\n")

    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Class {i}' for i in range(num_classes)], 
                yticklabels=[f'Class {i}' for i in range(num_classes)])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs_range, val_losses, label='Validation Loss', marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "train_val_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()

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
