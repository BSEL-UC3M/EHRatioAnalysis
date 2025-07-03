# ==============================================================================
# File: main_classificator_PEI.py
# Description: Main script for k-fold training and evaluation with PEI data.
# Author: @claudiacastrillon
# Modified: 02/07/2025 
# ==============================================================================

import torch
import platform
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

from dataloader.dataloader_PEI_classificator import ClassificationDataLoader
from trainers.classificator.trainer import evaluate_model, train_model
from models.classificator.five_layer_cnn_PEI import FiveLayerCNN
from models.classificator.resnet50 import fine_tune_resnet
from utils.preprocessing_all_images import preprocess_all_images

# ==============================================================================
# Configuration Parameters
# ==============================================================================
SAVE_RESULTS = True
SAVE_WEIGHTS = True
SAVE_PREPROCESSING = False

MODEL_TYPE = "cnn"  # Choose between 'cnn' and 'resnet50'
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 30
NUM_FOLDS = 5
SEED = 42
TEST_SPLIT_RATIO = 0.1  # 10% for final testing

RAW_IMAGES_FOLDER = "D:/Data/EHydropsAnalysis/paper-experiments/classification/PEI"
ANNOTATIONS_FOLDER = RAW_IMAGES_FOLDER
PROCESSED_IMAGES_FOLDER = "D:/Data/EHydropsAnalysis/paper-experiments/classification/PEI-PREPROCESSED"
RESULTS_ROOT = "D:/Results/EHydrops/Paper-experiments/classification/PEI"
WEIGHTS_ROOT = "D:/Models/EHydropsAnalysis/2025/paper-experiments/classification/PEI"

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
# Step 2: Load Dataset & Partition into Train+Val and Test
# ==============================================================================
annotations = ClassificationDataLoader.load_annotations(ANNOTATIONS_FOLDER)

trainval_patients, test_patients = ClassificationDataLoader.split_train_val_test_patients(
    annotations, test_ratio=TEST_SPLIT_RATIO, seed=SEED
)

num_classes = len(set(
    annotation
    for patient_data in annotations.values()
    for annotation in patient_data['Annotation']
))

# Compute class weights from the training+validation patients
all_train_labels = []
for p in trainval_patients:
    if p in annotations:
        annots = annotations[p]["Annotation"].values
        all_train_labels.extend(annots)

label_counts = Counter(all_train_labels)
total_samples = sum(label_counts.values())
class_weights = torch.tensor(
    [total_samples / label_counts[i] for i in range(num_classes)],
    dtype=torch.float32
).to(device)

test_loader = ClassificationDataLoader.get_test_dataloader(
    images_folder=PROCESSED_IMAGES_FOLDER,
    annotations=annotations,
    patient_ids=test_patients,
    batch_size=BATCH_SIZE,
    transform=None
)

folds = ClassificationDataLoader.get_kfold_dataloaders(
    images_folder=PROCESSED_IMAGES_FOLDER,
    annotations=annotations,
    patient_ids=trainval_patients,
    k=NUM_FOLDS,
    batch_size=BATCH_SIZE,
    transform=None,
    seed=SEED
)



fold_accuracies, fold_losses = [], []
model_paths = []

# ==============================================================================
# Step 3: K-Fold Training
# ==============================================================================
for fold_idx, train_loader, val_loader in folds:
    print(f"\n📂 Fold {fold_idx + 1}/{NUM_FOLDS} - Training {MODEL_TYPE.upper()}...\n")

    if MODEL_TYPE == "cnn":
        model = FiveLayerCNN(num_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        train_fn = train_model
        train_args = {"resnet_dropout_schedule": False}

    elif MODEL_TYPE == "resnet50":
        model, criterion, optimizer, scheduler = fine_tune_resnet(
            num_classes, device, learning_rate=LEARNING_RATE, model_type="resnet50"
        )
        train_fn = train_model
        train_args = {"resnet_dropout_schedule": True}

    trained_model, train_losses, val_losses, train_accuracies, val_accuracies = train_fn(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=NUM_EPOCHS,
        **train_args 
    )

    preds, targets, avg_loss, accuracy = evaluate_model(trained_model, val_loader, device, return_all=True)
    conf_matrix = confusion_matrix(targets, preds)
    fold_accuracies.append(accuracy)
    fold_losses.append(avg_loss)

    if SAVE_WEIGHTS:
        os.makedirs(WEIGHTS_ROOT, exist_ok=True)
        model_path = os.path.join(WEIGHTS_ROOT, f"{MODEL_TYPE}_fold{fold_idx + 1}_seed{SEED}.pt").replace("\\", "/")
        torch.save(trained_model.state_dict(), model_path)
        model_paths.append(model_path)

# ==============================================================================
# Step 4: Final Test Evaluation (all models)
# ==============================================================================
import pandas as pd  # Make sure this is imported at the top

test_metrics = []
fold_metrics_summary = []

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
summary_dir = os.path.join(RESULTS_ROOT, f"summary_{MODEL_TYPE}_{timestamp}").replace("\\", "/")
os.makedirs(summary_dir, exist_ok=True)

for fold_idx, model_path in enumerate(model_paths):
    print(f"\n🧪 Evaluating fold {fold_idx + 1} model on test set...")
    if MODEL_TYPE == "cnn":
        model = FiveLayerCNN(num_classes).to(device)
    elif MODEL_TYPE == "resnet50":
        model, _, _, _ = fine_tune_resnet(num_classes, device, learning_rate=LEARNING_RATE, model_type="resnet50")

    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    preds, targets, test_loss, test_acc = evaluate_model(model, test_loader, device, return_all=True)
    test_metrics.append((preds, targets, test_loss, test_acc))

    # Compute classification report
    class_report_str = classification_report(
        targets, preds,
        target_names=[f"Class {i}" for i in range(num_classes)],
        digits=4,
        zero_division=0
    )
    f1 = f1_score(targets, preds, average="binary", zero_division=0)
    precision = precision_score(targets, preds, average="binary", zero_division=0)
    recall = recall_score(targets, preds, average="binary", zero_division=0)

    # Save fold confusion matrix
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Class {i}' for i in range(num_classes)],
                yticklabels=[f'Class {i}' for i in range(num_classes)],
                annot_kws={"size": 14})
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(f"Confusion Matrix - Fold {fold_idx+1}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, f"test_confusion_matrix_fold{fold_idx+1}.png"))
    plt.close()

    # Save classification report
    with open(os.path.join(summary_dir, f"classification_report_fold{fold_idx+1}.txt"), "w") as f:
        f.write(class_report_str)

    # Save fold results
    with open(os.path.join(summary_dir, f"test_results_fold{fold_idx+1}.txt"), "w") as f:
        f.write(f"Fold {fold_idx+1} Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Fold {fold_idx+1} Test Loss: {test_loss:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Confusion Matrix:\n{np.array2string(cm)}\n")

    # Store for final CSV
    fold_metrics_summary.append({
        "fold": fold_idx + 1,
        "accuracy": test_acc,
        "loss": test_loss,
        "f1": f1,
        "precision": precision,
        "recall": recall
    })

# ==============================================================================
# Step 5: Save Final Results and Summary
# ==============================================================================
all_test_acc = [m[3] for m in test_metrics]
all_test_loss = [m[2] for m in test_metrics]
all_test_preds = np.concatenate([m[0] for m in test_metrics])
all_test_targets = np.concatenate([m[1] for m in test_metrics])
test_conf_matrix = confusion_matrix(all_test_targets, all_test_preds)

pred_classes, pred_counts = np.unique(all_test_preds, return_counts=True)
true_classes, true_counts = np.unique(all_test_targets, return_counts=True)

class_report = classification_report(
    all_test_targets, all_test_preds,
    target_names=[f"Class {i}" for i in range(num_classes)],
    digits=4,
    zero_division=0
)
f1 = f1_score(all_test_targets, all_test_preds, average="binary", zero_division=0)
precision = precision_score(all_test_targets, all_test_preds, average="binary", zero_division=0)
recall = recall_score(all_test_targets, all_test_preds, average="binary", zero_division=0)

# Save fold metrics CSV
df_metrics = pd.DataFrame(fold_metrics_summary)
df_metrics.to_csv(os.path.join(summary_dir, "fold_results.csv"), index=False)

# Save overall summary
with open(os.path.join(summary_dir, "final_summary.txt"), "w") as f:
    f.write("==== Final Configuration ====\n")
    f.write(f"Model Type: {MODEL_TYPE}\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Epochs: {NUM_EPOCHS}\n")
    f.write(f"Folds: {NUM_FOLDS}\n")
    f.write(f"Random Seed: {SEED}\n")
    f.write(f"Test Set Size: {len(test_loader.dataset)} images\n\n")
    f.write("Class Weights (used in CrossEntropyLoss):\n")
    f.write(f"{class_weights.cpu().numpy().tolist()}\n\n") 
    
    f.write("==== Validation (K-Fold) Results ====\n")
    f.write(f"Average Accuracy: {np.mean(fold_accuracies):.2f}% ± {np.std(fold_accuracies):.2f}\n")
    f.write(f"Average Loss: {np.mean(fold_losses):.4f} ± {np.std(fold_losses):.4f}\n\n")

    f.write("==== Test Set Results (Averaged over all folds) ====\n")
    f.write(f"Accuracy: {np.mean(all_test_acc):.2f}% ± {np.std(all_test_acc):.2f}\n")
    f.write(f"Loss: {np.mean(all_test_loss):.4f} ± {np.std(all_test_loss):.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n\n")

    f.write("Class Distribution:\n")
    f.write(f"Predicted: {dict(zip(pred_classes, pred_counts))}\n")
    f.write(f"True: {dict(zip(true_classes, true_counts))}\n\n")

    f.write("Confusion Matrix:\n")
    f.write(np.array2string(test_conf_matrix))
    f.write("\n\nClassification Report:\n")
    f.write(class_report)

plt.figure(figsize=(6, 5))
sns.heatmap(
    test_conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=[f'Class {i}' for i in range(num_classes)],
    yticklabels=[f'Class {i}' for i in range(num_classes)],
    annot_kws={"size": 14}
)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Test Set Confusion Matrix (All Models)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(summary_dir, "test_confusion_matrix_all_folds.png"))
plt.close()

print(f"\n📁 Final summary and confusion matrix saved to: {summary_dir}")

# ==============================================================================
# Step 6: Final Console Summary
# ==============================================================================
print("\n📊 Final Summary Across Folds")
print(f"Average Validation Accuracy: {np.mean(fold_accuracies):.2f}% ± {np.std(fold_accuracies):.2f}")
print(f"Average Validation Loss: {np.mean(fold_losses):.4f} ± {np.std(fold_losses):.4f}")
print(f"Test Accuracy (held-out set): {np.mean(all_test_acc):.2f}% ± {np.std(all_test_acc):.2f}")
print(f"Test Loss (held-out set): {np.mean(all_test_loss):.4f} ± {np.std(all_test_loss):.4f}")
print(f"F1 Score: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")


