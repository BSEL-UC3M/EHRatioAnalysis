# ==============================================================================
# File: main_classificator_PEI.py
# Description: Main script for training and evaluating the classification model with PEI data.
# Author: @claudiacastrillon
# Created: 13/02/2025
# ==============================================================================
import torch
import os
import sys
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix
from dataloader.dataloader_PEI_classificator import ClassificationDataLoader
from trainers.classificator.five_layer_cnn_PEI import train_model, evaluate_model, FiveLayerCNN
from trainers.classificator.resnet50 import fine_tune_resnet, train_model, evaluate_model

# Add utils folder to path to import preprocessing script
sys.path.append("/Users/claudiacastrillonalvarez/Desktop/github/EHRatioAnalysis/utils")
from preprocessing_all_images import preprocess_all_images

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# ==============================================================================
# Configuration Parameters
SAVE_RESULTS = True  # Toggle to save results
SAVE_PREPROCESSING = True  # Toggle to save preprocessed images
LEARNING_RATE = 1e-4  # Learning rate for the optimizer
BATCH_SIZE = 16  # Batch size for training
DATA_SPLITS = (0.7, 0.1, 0.2)  # Train, validation, test splits
RAW_IMAGES_FOLDER = "/Users/claudiacastrillonalvarez/Desktop/github/EHRatioAnalysis/PEI_data/PEI_TIFF/"

PROCESSED_IMAGES_FOLDER = os.path.join(os.path.dirname(RAW_IMAGES_FOLDER), "PEI_processed_data")
NUM_EPOCHS = 20  # Define number of epochs

# Select computing device (use Apple Silicon GPU if available)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ==============================================================================
# ✅ Step 1: Preprocess Images
if SAVE_PREPROCESSING:
    print("\n🔄 Preprocessing PEI images...\n")
    os.makedirs(PROCESSED_IMAGES_FOLDER, exist_ok=True)  # Ensure folder exists
    preprocess_all_images(RAW_IMAGES_FOLDER, PROCESSED_IMAGES_FOLDER)
    print("✅ Preprocessing complete. Processed images saved in:", PROCESSED_IMAGES_FOLDER)
else:
    print("⚠️ Skipping image preprocessing. Using existing processed images.")

# ==============================================================================
# ✅ Step 2: Load Dataset
ANNOTATIONS_FOLDER = "/Users/claudiacastrillonalvarez/Desktop/github/EHRatioAnalysis/PEI_data/"

annotations = ClassificationDataLoader.load_annotations(ANNOTATIONS_FOLDER)

train_loader, val_loader, test_loader = ClassificationDataLoader.train_val_test_split(
    images_folder=PROCESSED_IMAGES_FOLDER,
    annotations=annotations,
    splits=DATA_SPLITS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    transform=None
)

# Determine the number of classes dynamically
num_classes = len(set(
    annotation 
    for patient_data in annotations.values() 
    for annotation in patient_data['Annotation']
))

# ==============================================================================
# ✅ Step 3: User selects the model type
MODEL_TYPE = input("Select model type ('cnn' or 'resnet50'): ").strip().lower()

while MODEL_TYPE not in ["cnn", "resnet50"]:
    MODEL_TYPE = input("Invalid choice. Please select 'cnn' or 'resnet50': ").strip().lower()

print(f"\nTraining {MODEL_TYPE.upper()} model...\n")

# ==============================================================================
# ✅ Step 4: Model Definition & Training
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
print(f"\n🚀 Training {MODEL_TYPE.upper()} model for {NUM_EPOCHS} epochs...\n")
trained_model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=NUM_EPOCHS
)

# ==============================================================================
# ✅ Step 5: Evaluate Model and Compute Confusion Matrix
print(f"\n📊 Evaluating {MODEL_TYPE.upper()} model on the test set...\n")
y_true, y_pred, avg_loss, accuracy = evaluate_model(trained_model, test_loader, device)
print(f"✅ Test Accuracy: {accuracy:.2f}% | Test Loss: {avg_loss:.4f}")

conf_matrix = confusion_matrix(y_true, y_pred)

# ==============================================================================
# ✅ Step 6: Save Results in results/results_classificator/results_classificator_PEI/
if SAVE_RESULTS:
    results_root = "./results"
    results_classificator = os.path.join(results_root, "results_classificator", "results_classificator_PEI")
    os.makedirs(results_classificator, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(results_classificator, f"{MODEL_TYPE}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Save performance metrics
    with open(os.path.join(results_dir, "results.txt"), "w") as f:
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Number of Epochs: {NUM_EPOCHS}\n")
        f.write(f"Optimizer: {'Adam' if MODEL_TYPE == 'cnn' else 'SGD'}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")
        f.write(f"Confusion Matrix:\n{conf_matrix}\n")

    # Generate and save confusion matrix plot
    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Class {i}' for i in range(num_classes)], 
                yticklabels=[f'Class {i}' for i in range(num_classes)])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    epochs_range = range(1, len(train_losses) + 1)  # Match the actual number of epochs

    # Plot training & validation loss
    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs_range, val_losses, label='Validation Loss', marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "train_val_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot training & validation accuracy
    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "train_val_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Results saved in {results_dir}\n")

print("🎉 Process completed.")
