# ==============================================================================
# File: main_classificator.py
# Description: Cleaned and modernized training script for MRC classification.
# Author: @claudiacastrillon
# Modified: 11/07/2025
# ==============================================================================

import os
import torch
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

from dataloader.dataloader_MRC_classificator import ClassificationDataLoader
from trainers.classificator.trainer import train_model, evaluate_model
from models.classificator.resnet50 import fine_tune_resnet
from models.classificator.five_layer_cnn import FiveLayerCNN
from torchvision import transforms

# ==============================================================================
# Configuration
# ==============================================================================
MODEL_TYPE = "resnet50"
SEEDS = [42, 123, 456, 789, 1011]
NUM_EPOCHS = 1
LEARNING_RATE = 5e-4
BATCH_SIZE = 16
DATA_SPLITS = (0.7, 0.1, 0.2)

IMAGES_FOLDER = "D:/Data/EHydropsAnalysis/paper-experiments/classification/MRC"
RESULTS_ROOT = "D:/Results/EHydrops/Paper-experiments/classification/MRC"
WEIGHTS_ROOT = "D:/Models/EHydropsAnalysis/2025/paper-experiments/classification/MRC"

# ==============================================================================
# Setup
# ==============================================================================
system_name = platform.system().lower()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
results_dir = os.path.join(RESULTS_ROOT, f"{MODEL_TYPE}_mrc_{timestamp}")
os.makedirs(results_dir, exist_ok=True)

# Transformations
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

# Storage
all_fold_metrics = []

# ==============================================================================
# Run per seed
# ==============================================================================
for seed in SEEDS:
    print(f"\n🌱 Running seed {seed}")

    train_loader, val_loader, test_loader, *_ = ClassificationDataLoader.train_val_test_split(
        images_folder=IMAGES_FOLDER,
        annotations=annotations,
        splits=DATA_SPLITS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        transform=default_transforms,
        seed=seed
    )

    model, criterion, optimizer, scheduler = fine_tune_resnet(
        num_classes=num_classes,
        device=device,
        learning_rate=LEARNING_RATE,
        model_type="resnet50"
    )

    trained_model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, num_epochs=NUM_EPOCHS, early_stop_patience=5
    )

    preds, targets, test_loss, test_acc = evaluate_model(trained_model, test_loader, device, return_all=True)
    conf_matrix = confusion_matrix(targets, preds)

    # === Metrics ===
    f1 = f1_score(targets, preds, average="binary", zero_division=0)
    precision = precision_score(targets, preds, average="binary", zero_division=0)
    recall = recall_score(targets, preds, average="binary", zero_division=0)
    report = classification_report(targets, preds, digits=4, zero_division=0)

    seed_dir = os.path.join(results_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    # === Save confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix (Seed {seed})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(seed_dir, f"conf_matrix.png"))
    plt.close()

    # === Save training/val curves
    x = range(1, len(train_losses) + 1)

    plt.plot(x, train_losses, label="Train")
    plt.plot(x, val_losses, label="Val")
    plt.title(f"Loss Curve (Seed {seed})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(seed_dir, "loss_curve.png"))
    plt.close()

    plt.plot(x, train_accuracies, label="Train")
    plt.plot(x, val_accuracies, label="Val")
    plt.title(f"Accuracy Curve (Seed {seed})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(seed_dir, "accuracy_curve.png"))
    plt.close()

    # === Save report
    with open(os.path.join(seed_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    with open(os.path.join(seed_dir, "summary.txt"), "w") as f:
        f.write(f"Seed: {seed}\n")
        f.write(f"Accuracy: {test_acc:.4f}\n")
        f.write(f"Loss: {test_loss:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Confusion Matrix:\n{np.array2string(conf_matrix)}\n")

    all_fold_metrics.append({
        "seed": seed,
        "accuracy": test_acc,
        "loss": test_loss,
        "f1": f1,
        "precision": precision,
        "recall": recall
    })

# ==============================================================================
# Save global summary
# ==============================================================================

df = pd.DataFrame(all_fold_metrics)
df.to_csv(os.path.join(results_dir, "all_seeds_summary.csv"), index=False)

with open(os.path.join(results_dir, "final_summary.txt"), "w") as f:
    f.write("==== MRC Classification Summary ====\n")
    f.write(f"Model Type: {MODEL_TYPE}\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Epochs: {NUM_EPOCHS}\n")
    f.write(f"Seeds: {SEEDS}\n\n")

    f.write("==== Aggregated Results ====\n")
    f.write(f"Accuracy: {df['accuracy'].mean():.2f} ± {df['accuracy'].std():.2f}\n")
    f.write(f"Loss: {df['loss'].mean():.4f} ± {df['loss'].std():.4f}\n")
    f.write(f"F1 Score: {df['f1'].mean():.4f}\n")
    f.write(f"Precision: {df['precision'].mean():.4f}\n")
    f.write(f"Recall: {df['recall'].mean():.4f}\n")

print(f"\n✅ Completed all runs. Results saved in: {results_dir}")
