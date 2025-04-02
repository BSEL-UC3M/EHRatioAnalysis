# ==============================================================================
# File: ultils/classification_postprocess.py
# Description: Groups slices by patient, sort them numerically and apply a smoothing
# filtering rule that enforces blocks of 1s in the middle, suppress isolated 1s or 0s
# and bias toward class 1 if uncertain.
# Author: @cfusterbarcelo
# Created: 25/03/2025
# ==============================================================================

import re
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def extract_patient_and_index(filename):
    """
    Extract patient ID and slice index from filenames like 'MRC_88_75734449.tif'.
    """
    match = re.match(r"([A-Za-z]+_\d+)_(\d+)", filename)
    if match:
        patient_id = match.group(1)  # e.g., 'MRC_88'
        slice_idx = int(match.group(2))  # e.g., 75734449
        return patient_id, slice_idx
    else:
        raise ValueError(f"Unexpected filename format: {filename}")

def smooth_classification_predictions(results, enforce_continuity=True):
    """
    Takes list of (filename, predicted_class) and applies post-processing.
    - Groups by patient.
    - Smooths isolated 1s or 0s.
    - Optionally enforces contiguous region of 1s (ear region).
    Returns a new list of (filename, corrected_prediction)
    """
    grouped = defaultdict(list)
    for fname, pred in results:
        patient_id, slice_idx = extract_patient_and_index(fname)
        grouped[patient_id].append((slice_idx, fname, pred))

    cleaned_results = []

    for patient_id, slices in grouped.items():
        slices.sort()
        predictions = [pred for _, _, pred in slices]
        filenames = [fname for _, fname, _ in slices]

        smoothed = predictions.copy()
        for i in range(1, len(predictions) - 1):
            prev, curr, nxt = predictions[i - 1], predictions[i], predictions[i + 1]
            if curr == 1 and prev == 0 and nxt == 0:
                smoothed[i] = 0
            elif curr == 0 and prev == 1 and nxt == 1:
                smoothed[i] = 1

        if enforce_continuity:
            one_indices = [i for i, val in enumerate(smoothed) if val == 1]
            if one_indices:
                start = min(one_indices)
                end = max(one_indices)
                for i in range(start, end + 1):
                    smoothed[i] = 1

        for fname, new_pred in zip(filenames, smoothed):
            cleaned_results.append((fname, new_pred))

    return cleaned_results

def plot_comparison(before, after, save_path="./comparison_plots"):
    os.makedirs(save_path, exist_ok=True)
    grouped_before = defaultdict(list)
    grouped_after = defaultdict(list)

    for fname, pred in before:
        pid, idx = extract_patient_and_index(fname)
        grouped_before[pid].append((idx, pred))

    for fname, pred in after:
        pid, idx = extract_patient_and_index(fname)
        grouped_after[pid].append((idx, pred))

    for pid in grouped_before:
        b = sorted(grouped_before[pid])
        a = sorted(grouped_after[pid])

        indices = [x[0] for x in b]
        pred_before = [x[1] for x in b]
        pred_after = [x[1] for x in a]

        plt.figure(figsize=(10, 4))
        plt.plot(indices, pred_before, label="Before", linestyle="--", marker="o")
        plt.plot(indices, pred_after, label="After", linestyle="-", marker="x")
        plt.title(f"Classification Smoothing for {pid}")
        plt.xlabel("Slice Index")
        plt.ylabel("Prediction (0=No Ear, 1=Ear)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{pid}_comparison.png"))
        plt.close()

def save_comparison_csv(before, after, save_path="comparison.csv"):
    data = []
    for (fname1, pred1), (_, pred2) in zip(before, after):
        data.append((fname1, pred1, pred2))

    df = pd.DataFrame(data, columns=["filename", "before", "after"])
    df.to_csv(save_path, index=False)
    print(f"✅ CSV comparison saved at: {save_path}")


def threshold_sweep(model, test_loader, device, thresholds=np.arange(0.1, 0.55, 0.05)):
    """
    Evaluates model performance across different thresholds for class 1.

    Returns:
        metrics_list: List of dictionaries with threshold, accuracy, precision, recall, f1
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    metrics_list = []

    with torch.no_grad():
        for threshold in thresholds:
            y_true, y_pred = [], []
            total_correct = 0
            total_loss = 0

            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                preds = (probs[:, 1] > threshold).long()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

                total_correct += (preds == labels).sum().item()

            avg_loss = total_loss / len(test_loader)
            accuracy = 100 * total_correct / len(y_true)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            metrics_list.append({
                "threshold": threshold,
                "accuracy": accuracy,
                "precision": precision * 100,
                "recall": recall * 100,
                "f1": f1* 100
            })

    return metrics_list
