# ==============================================================================
# File: main_classificator.py
# Description: Main script for training and evaluating the classification model.
# Author: @cfusterbarcelo
# Created: 09/01/2025
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
from dataloader.dataloader_MRC_classificator import ClassificationDataLoader
from trainers.classificator.toy_classificator import train_model, evaluate_model, fine_tune_resnet, FiveLayerCNN, cross_validate_model

# Add the current directory to Python's module search path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# ==============================================================================

# Configuration Parameters
SAVE_RESULTS = True  # Toggle to save results
# NUM_EPOCHS = 5  # Number of training epochs
LEARNING_RATE = 1e-4  # Learning rate for the optimizer
BATCH_SIZE = 8  # Batch size for training
DATA_SPLITS = (0.7, 0.1, 0.2)  # Train, validation, test splits
IMAGES_FOLDER = "/Users/claudiacastrillonalvarez/Desktop/github/EHRatioAnalysis/MRC_data/MRC_images/" # Path to the folder containing images

# Device configuration: select apple silicon GPU (mps)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# Load dataset annotations 
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

# Extract unique class labels from annotations --> determine num_classes dynamically 
num_classes = len(set(
    annotation 
    for patient_data in annotations.values() 
    for annotation in patient_data['Annotation']
))

# Select model type
MODEL_TYPE = "cnn"  # Change to "cnn" for 5-layer CNN

if MODEL_TYPE == "cnn":
    # ✅ Perform cross-validation (training happens here only)
    best_params, best_model = cross_validate_model(FiveLayerCNN, train_loader, num_classes, device, k_folds=5, num_epochs=5)


    # ✅ No extra training after CV, only testing
    print("Evaluating final model on test set...")
    y_true, y_pred, avg_loss, accuracy = evaluate_model(best_model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.2f}% | Test Loss: {avg_loss:.4f}")

elif MODEL_TYPE == "resnet":
    # ✅ If using ResNet, perform normal training
    model = FiveLayerCNN(num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Default LR, modify if needed
    train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=8, shuffle=True)

    print(f"Training {MODEL_TYPE.upper()} model...")
    trained_model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5
    )okay

    # ✅ Only evaluate after training
    print(f"Evaluating {MODEL_TYPE.upper()} model...")
    y_true, y_pred, avg_loss, accuracy = evaluate_model(trained_model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.2f}% | Test Loss: {avg_loss:.4f}")


train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=BATCH_SIZE, shuffle=True)

# Train the model
print(f"Training {MODEL_TYPE.upper()} model...")
trained_model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, val_loader, criterion, optimizer, device,num_epochs=5)
# Evaluate the model
print(f"Evaluating {MODEL_TYPE.upper()} model...")
y_true, y_pred, avg_loss, accuracy = evaluate_model(trained_model, test_loader, device)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Save results if enabled and plots 
if SAVE_RESULTS:
    results_root = "./results"
    results_classificator = os.path.join(results_root, "results_classificator")
    os.makedirs(results_classificator, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(results_classificator, f"{MODEL_TYPE}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    # save performance metrics 
    with open(os.path.join(results_dir, "results.txt"), "w") as f:
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Number of Epochs: {NUM_EPOCHS}\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Number of Layers: {len(list(model.children()))}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")
        f.write(f"Confusion Matrix:\n{conf_matrix}\n")

    # Generate and save confusion matrix plot
    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[f'Class {i}' for i in range(num_classes)], yticklabels=[f'Class {i}' for i in range(num_classes)])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot training & validation loss
    plt.figure(figsize=(8,6))
    plt.plot(range(1, NUM_EPOCHS+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, NUM_EPOCHS+1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "train_val_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot training & validation accuracy
    plt.figure(figsize=(8,6))
    plt.plot(range(1, NUM_EPOCHS+1), train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(range(1, NUM_EPOCHS+1), val_accuracies, label='Validation Accuracy', marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "train_val_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Results saved in {results_dir}")

print("Process completed.")