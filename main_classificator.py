# ==============================================================================
# File: main_classificator.py
# Description: Main script for training and evaluating the classification model.
# Author: @cfusterbarcelo
# Created: 09/01/2025
# ==============================================================================
import torch
print(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))


import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix
from dataloader.dataloader_MRC_classificator import ClassificationDataLoader
from models.classificator import SimpleCNN
from trainers.classificator.toy_classificator import train_model, evaluate_model

# Add the current directory to Python's module search path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# ==============================================================================

# Configuration Parameters
SAVE_RESULTS = True  # Toggle to save results
NUM_EPOCHS = 5  # Number of training epochs
LEARNING_RATE = 1e-4  # Learning rate for the optimizer
BATCH_SIZE = 8  # Batch size for training
DATA_SPLITS = (0.34, 0.33, 0.33)  # Train, validation, test splits
IMAGES_FOLDER = "toydataset/classification/"  # Path to the folder containing images

# Full dataset for training (uncomment when needed)
# IMAGES_FOLDER = "D:/Data/VolumetricHydrops/images/MRC"

# ==============================================================================

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Load annotations
annotations = ClassificationDataLoader.load_annotations(IMAGES_FOLDER)

# Create train, val, and test DataLoaders
train_loader, val_loader, test_loader = ClassificationDataLoader.train_val_test_split(
    images_folder=IMAGES_FOLDER,
    annotations=annotations,
    splits=DATA_SPLITS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    transform=None
)

# ==============================================================================

# Print information about the dataset
print("DataLoader Information:")
print(f"Number of training samples: {len(train_loader.dataset)}")
print(f"Number of validation samples: {len(val_loader.dataset)}")
print(f"Number of test samples: {len(test_loader.dataset)}")

# Extract unique class labels from annotations
num_classes = len(set(
    annotation 
    for patient_data in annotations.values() 
    for annotation in patient_data['Annotation']
))
# Initialize the model
model = SimpleCNN(num_classes=num_classes).to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model
print("Starting training...")
trained_model = train_model(model, train_loader, criterion, optimizer, device, NUM_EPOCHS)

# Evaluate the model
print("Evaluating model...")
y_true, y_pred, avg_loss, accuracy = evaluate_model(trained_model, test_loader, device)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
TN, FP, FN, TP = conf_matrix.ravel() if conf_matrix.size == 4 else (0, 0, 0, 0)

# Save results if enabled
if SAVE_RESULTS:
    # Define the root results folder
    results_root = "./results"
    results_classificator = os.path.join(results_root, "results_classificator")

    # Create 'results/results_classificator/' if it doesn't exist
    os.makedirs(results_classificator, exist_ok=True)

    # Generate a subfolder using date-time format YYYYMMDD-HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # Example: 20250207-145528
    results_dir = os.path.join(results_classificator, timestamp)

    # Create the final results directory
    os.makedirs(results_dir, exist_ok=True)

    # Save results in a text file
    with open(os.path.join(results_dir, "results.txt"), "w") as f:
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Number of Epochs: {NUM_EPOCHS}\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Number of Layers: {len(list(model.children()))}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")
        f.write(f"Confusion Matrix:\n{conf_matrix}\n")
        f.write(f"True Positives: {TP}\n")
        f.write(f"False Positives: {FP}\n")
        f.write(f"True Negatives: {TN}\n")
        f.write(f"False Negatives: {FN}\n")

    # Generate and save confusion matrix plot
    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Results saved in {results_dir}")

print("Process completed.")

