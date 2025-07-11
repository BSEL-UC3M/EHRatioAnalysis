# ==============================================================================
# File: main_classificator.py
# Description: Modified main script for repeated training with different seeds
# Author: @claudiacastrillon 
# Created: 31/05/2025
# ==============================================================================

import torch
import os
import sys
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from datetime import datetime
from sklearn.metrics import confusion_matrix
from torchvision import transforms

from dataloader.dataloader_MRC_classificator import ClassificationDataLoader
from models.classificator.resnet50 import fine_tune_resnet
#from models.classificator.five_layer_cnn import FiveLayerCNN
from trainers.classificator.trainer import train_model, evaluate_model

# Configuration Parameters
LEARNING_RATE = 0.0005
BATCH_SIZE = 16
DATA_SPLITS = (0.7, 0.1, 0.2)
NUM_EPOCHS = 1
IMAGES_FOLDER = "D:/Data/EHydropsAnalysis/paper-experiments/classification/MRC"
THRESHOLD = 0.2
DROPOUT = 0.5
SEEDS = [42, 123, 456, 789, 1011]

# Set device
system_name = platform.system().lower()
if system_name == "darwin":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
elif system_name in ["windows", "linux"]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(f"\nUsing device: {device}")

# Data transforms (no augmentation for now)
default_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load annotations and determine number of classes
annotations = ClassificationDataLoader.load_annotations(IMAGES_FOLDER)
num_classes = len(set(
    annotation
    for patient_data in annotations.values()
    for annotation in patient_data['Annotation']
))

# Results directory
results_root = "D:/Results/EHydrops/Paper-experiments/classification/MRC"
os.makedirs(results_root, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
results_dir = os.path.join(results_root, f"resnet_mrc_{timestamp}")
os.makedirs(results_dir, exist_ok=True)

# Metrics storage
RUNS = []

# Loop over seeds
for seed in SEEDS:
    print(f"\n\n======================= SEED {seed} =======================")

    # Split dataset
    train_loader, val_loader, test_loader, train_patients, val_patients, test_patients = ClassificationDataLoader.train_val_test_split(
        images_folder=IMAGES_FOLDER,
        annotations=annotations,
        splits=DATA_SPLITS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        transform=default_transforms,
        seed=seed
    )

    # Define model
    # model = FiveLayerCNN(num_classes, dropout_prob=DROPOUT).to(device)
    # weights = torch.tensor([1.0, 2.0]).to(device)
    # criterion = nn.CrossEntropyLoss(weight=weights)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    model, criterion, optimizer, scheduler = fine_tune_resnet(
        num_classes=num_classes,
        device=device,
        learning_rate=LEARNING_RATE,
        model_type='resnet50'
    )
    # Train model
    trained_model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, num_epochs=NUM_EPOCHS, early_stop_patience=5
    )

    # Evaluate
    y_true, y_pred, avg_loss, accuracy = evaluate_model(trained_model, test_loader, device)
    cm = confusion_matrix(y_true, y_pred)
    fn = cm[1][0] if cm.shape == (2, 2) else 0

    # Store metrics
    RUNS.append({
        "seed": seed,
        "accuracy": accuracy,
        "loss": avg_loss,
        "fn": fn,
        "conf_matrix": cm,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies
    })

    # Save confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Seed {seed})")
    plt.savefig(os.path.join(results_dir, f"conf_matrix_seed{seed}.png"), dpi=300)
    plt.close()

    # Save loss curves
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, label='Train')
    plt.plot(epochs_range, val_losses, label='Val')
    plt.title(f"Loss Curve (Seed {seed})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(results_dir, f"loss_curve_seed{seed}.png"), dpi=300)
    plt.close()

    # Save accuracy curves
    plt.plot(epochs_range, train_accuracies, label='Train')
    plt.plot(epochs_range, val_accuracies, label='Val')
    plt.title(f"Accuracy Curve (Seed {seed})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(results_dir, f"accuracy_curve_seed{seed}.png"), dpi=300)
    plt.close()

# Final aggregated metrics
accs = [r["accuracy"] for r in RUNS]
losses = [r["loss"] for r in RUNS]
fns = [r["fn"] for r in RUNS]

summary_path = os.path.join(results_dir, "summary.txt")
with open(summary_path, "w") as f:
    f.write("=== Final Summary ===\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\nBatch Size: {BATCH_SIZE}\nEpochs: {NUM_EPOCHS}\nDropout: {DROPOUT}\n\n")
    f.write(f"Accuracy: {np.mean(accs):.2f} ± {np.std(accs):.2f}\n")
    f.write(f"Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}\n")
    f.write(f"False Negatives: {np.mean(fns):.1f} ± {np.std(fns):.1f}\n")

print("\n✅ Completed all runs. Summary saved to:", summary_path)
