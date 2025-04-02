# ==============================================================================
# File: main_classificator.py
# Description: Main script for training and evaluating the classification model.
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
from torchvision import transforms

from dataloader.dataloader_MRC_classificator import ClassificationDataLoader
from models.classificator.five_layer_cnn import FiveLayerCNN
from models.classificator.resnet50 import fine_tune_resnet
from trainers.classificator.trainer import train_model, evaluate_model
from utils.classification_postprocess import threshold_sweep
from utils.custom_plots import plot_threshold_tradeoffs, plot_class1_probability_histogram

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# ==============================================================================
# Configuration Parameters
SAVE_RESULTS = input("Save results? (yes/no): ").strip().lower() == "yes"
SAVE_WEIGHTS = input("Save model weights? (yes/no): ").strip().lower() == "yes"
LEARNING_RATE = 1e-4  # Learning rate for the optimizer
BATCH_SIZE = 16  # Batch size for training
DATA_SPLITS = (0.7, 0.1, 0.2)  # Train, validation, test splits
IMAGES_FOLDER = "D:/Data/EHydropsAnalysis/2025-Porcessed/MRC TIFF" 
NUM_EPOCHS = 50  # Define number of epochs
DA = False
THRESHOLD = 0.2

# Select computing device (use Apple Silicon GPU if available)
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# Load Dataset
annotations = ClassificationDataLoader.load_annotations(IMAGES_FOLDER)

# Determine the number of classes dynamically
num_classes = len(set(
    annotation 
    for patient_data in annotations.values() 
    for annotation in patient_data['Annotation']
))

# User selects the model type
MODEL_TYPE = input("Select model type 'custom' to train a model from scratch or 'pretrained' to use the ResNet50): ").strip().lower()

while MODEL_TYPE not in ["custom", "pretrained"]:
    MODEL_TYPE = input("Invalid choice. Please select 'custom' or 'pretrained': ").strip().lower()

print(f"\nTraining {MODEL_TYPE.upper()} model...\n")

#  Model Definition & Training
if MODEL_TYPE == "custom":
    # Initialize CNN model
    model = FiveLayerCNN(num_classes).to(device)
    # Penalize errors on class 1 more heavily to bias on class 1 classification
    weights = torch.tensor([1.0, 2.0]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    if DA:
        default_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor()
        ])
        val_transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    else: 
        default_transforms = None

elif MODEL_TYPE == "pretrained":
    # Initialize ResNet50 model
    model, criterion, optimizer, scheduler = fine_tune_resnet(
        num_classes, device, learning_rate=LEARNING_RATE, model_type=MODEL_TYPE
    )
    default_transforms = None


train_loader, val_loader, test_loader, train_patients, val_patients, test_patients = ClassificationDataLoader.train_val_test_split(
    images_folder=IMAGES_FOLDER,
    annotations=annotations,
    splits=DATA_SPLITS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    transform=default_transforms
)


# Train the model explicitly
print(f"\n🚀 Training {MODEL_TYPE.upper()} model for {NUM_EPOCHS} epochs...\n")
trained_model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=NUM_EPOCHS, model_type=MODEL_TYPE
)
# ==============================================================================
# ✅ Step 4: Prepare Result Directory
if SAVE_RESULTS or SAVE_WEIGHTS:
    results_root = "./results/results_classificator/results_classificator_MRC"
    os.makedirs(results_root, exist_ok=True)  # Ensure base directory exists
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(results_root, f"cnn_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)  # Create timestamped directory

# ==============================================================================
# ✅ Step 5: Save Best Weights Based on Validation Loss
if SAVE_WEIGHTS:
    best_epoch = np.argmin(val_losses)  # Find the epoch with the lowest validation loss
    weights_save_path = os.path.join(results_dir, "cnn_best_weights.pt")
    torch.save(trained_model.state_dict(), weights_save_path)
    print(f"✅ Best model weights saved at {weights_save_path} (Epoch {best_epoch + 1})")

# ==============================================================================
# ✅ Step 6: Evaluate Model on Test Set
print(f"\n📊 Evaluating {MODEL_TYPE.upper()} model on the test set...\n")
y_true, y_pred, avg_loss, accuracy = evaluate_model(trained_model, test_loader, device, threshold=THRESHOLD)
print(f"✅ {MODEL_TYPE.upper()} Test Accuracy: {accuracy:.2f}% | Test Loss: {avg_loss:.4f}")

# Compute confusion matrix BEFORE post-processing
conf_matrix_before = confusion_matrix(y_true, y_pred)
plot_class1_probability_histogram(trained_model, test_loader, device, threshold = THRESHOLD, results_dir=results_dir)

# ==============================================================================
# ✅ Step 7: Save Results (confusion matrix, train and validation losses/accuracies, .txt file)
if SAVE_RESULTS:
    # Save performance metrics
    with open(os.path.join(results_dir, "results.txt"), "w") as f:
        f.write("\n--- Network Details ---\n")
        f.write(f"Network: {MODEL_TYPE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Number of Epochs: {NUM_EPOCHS}\n")
        f.write(f"Optimizer: Adam\n")
        f.write("\n--- Patient Splits ---\n")
        f.write(f"Train Patients ({len(train_patients)}):\n{', '.join(train_patients)}\n")
        f.write(f"Validation Patients ({len(val_patients)}):\n{', '.join(val_patients)}\n")
        f.write(f"Test Patients ({len(test_patients)}):\n{', '.join(test_patients)}\n")
        f.write("\n--- Results ---\n")
        f.write(f"Best Epoch: {best_epoch + 1}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")
        f.write(f"Confusion Matrix:\n{conf_matrix_before}\n")

    # Save confusion matrix plot
    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix_before, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Class {i}' for i in range(num_classes)], 
                yticklabels=[f'Class {i}' for i in range(num_classes)])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix Before Post-Processing")
    plt.savefig(os.path.join(results_dir, "confusion_matrix_before.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save training & validation loss plot
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
    
    # Save training & validation accuracy plot
    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "train_val_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Evaluating biases towards class 1 PLOTS AND METRICS
    metrics = threshold_sweep(trained_model, test_loader, device, )
    plot_threshold_tradeoffs(metrics, threshold = THRESHOLD, results_dir=results_dir)

    print(f"\n✅ Results saved in {results_dir}\n")

