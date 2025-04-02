# ==============================================================================
# File: utils/classification_postprocess.py
# Description: Post-processing, smoothing, plotting and evaluation for classification.
# Author: @cfusterbarcelo
# Updated: 02/04/2025
# ==============================================================================

import os
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

# ------------------------------------------------------------------------------
# 🧩 HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def extract_patient_and_index(filename):
    """Extract patient ID and slice index from filenames like 'MRC_3_59160073.tif'."""
    cleaned = filename.replace("(1)", "")
    match = re.match(r"([A-Za-z]+_\d+)_(\d+)", cleaned)
    if match:
        return match.group(1), int(match.group(2))
    raise ValueError(f"Unexpected filename format: {filename}")

def extract_slice_number(filename):
    """Extract the final numeric slice index from the filename."""
    cleaned = filename.replace("(1)", "")
    try:
        return int(cleaned.split("_")[-1].split(".")[0])
    except Exception:
        raise ValueError(f"Invalid slice number in filename: {filename}")

# ------------------------------------------------------------------------------
# 🧹 POST-PROCESSING
# ------------------------------------------------------------------------------

def smooth_classification_predictions(results, enforce_continuity=True):
    """
    Smooths predictions by:
    - Removing isolated 1s or 0s
    - Optionally enforcing continuous 1s region
    """
    grouped = defaultdict(list)
    for fname, pred in results:
        pid, idx = extract_patient_and_index(fname)
        grouped[pid].append((idx, fname, pred))

    cleaned_results = []

    for pid, slices in grouped.items():
        slices.sort()
        predictions = [p for _, _, p in slices]
        filenames = [f for _, f, _ in slices]

        smoothed = predictions.copy()
        for i in range(1, len(predictions) - 1):
            if smoothed[i] == 1 and smoothed[i-1] == 0 and smoothed[i+1] == 0:
                smoothed[i] = 0
            elif smoothed[i] == 0 and smoothed[i-1] == 1 and smoothed[i+1] == 1:
                smoothed[i] = 1

        if enforce_continuity:
            one_indices = [i for i, val in enumerate(smoothed) if val == 1]
            if one_indices:
                start, end = min(one_indices), max(one_indices)
                for i in range(start, end + 1):
                    smoothed[i] = 1

        cleaned_results += list(zip(filenames, smoothed))

    return cleaned_results

# ------------------------------------------------------------------------------
# 📊 VISUALIZATION AND CSV
# ------------------------------------------------------------------------------

def plot_comparison(before, after, save_path="./comparison_plots"):
    grouped_before = defaultdict(list)
    grouped_after = defaultdict(list)

    # Group predictions by patient ID
    for fname, pred in before:
        pid, _ = extract_patient_and_index(fname)
        grouped_before[pid].append((fname, pred))

    for fname, pred in after:
        pid, _ = extract_patient_and_index(fname)
        grouped_after[pid].append((fname, pred))

    for pid in grouped_before:
        # Sort both lists using the extracted slice number
        b = sorted(grouped_before[pid], key=lambda x: extract_slice_number(x[0]))
        a = sorted(grouped_after[pid], key=lambda x: extract_slice_number(x[0]))

        # Slice numbers as x-axis labels
        x_labels = [str(extract_slice_number(x[0])) for x in b]
        pred_before = [x[1] for x in b]
        pred_after = [x[1] for x in a]

        plt.figure(figsize=(max(12, len(x_labels) * 0.3), 4))
        plt.plot(x_labels, pred_before, label="Before", linestyle="--", marker="o")
        plt.plot(x_labels, pred_after, label="After", linestyle="-", marker="x")

        plt.title(f"Classification Smoothing for {pid}")
        plt.xlabel("Slice Number")
        plt.ylabel("Prediction (0=No Ear, 1=Ear)")
        plt.xticks(rotation=60, ha="right", fontsize=8)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{pid}_comparison.png"))
        plt.close()

def plot_comparison_with_labels(before, after, label_csv, save_path="./comparison_with_labels"):
    """
    Plot predictions before & after smoothing, with ground truth labels overlaid.
    """
    os.makedirs(save_path, exist_ok=True)

    # Read labels from Excel (multi-sheet)
    xls = pd.ExcelFile(label_csv)
    df_list = []
    for sheet in xls.sheet_names:
        df_sheet = pd.read_excel(xls, sheet_name=sheet)
        if "File Name" in df_sheet.columns and "Annotation" in df_sheet.columns:
            df_sheet = df_sheet[["File Name", "Annotation"]]
            df_sheet.columns = ["filename", "label"]
            df_list.append(df_sheet)
    df = pd.concat(df_list, ignore_index=True)

    label_dict = dict(zip(df["filename"], df["label"]))

    # Grouping
    grouped_before = defaultdict(list)
    grouped_after = defaultdict(list)
    grouped_labels = defaultdict(list)

    for fname, pred in before:
        pid, _ = extract_patient_and_index(fname)
        grouped_before[pid].append((fname, pred))

    for fname, pred in after:
        pid, _ = extract_patient_and_index(fname)
        grouped_after[pid].append((fname, pred))

    for fname in df["filename"]:
        pid, _ = extract_patient_and_index(fname)
        label = label_dict.get(fname)
        if label is not None:
            grouped_labels[pid].append((fname, label))

    # ✅ Only valid patients
    patients_before = set(grouped_before.keys())
    patients_after = set(grouped_after.keys())
    patients_labels = set(grouped_labels.keys())
    valid_patients = patients_before & patients_after & patients_labels

    for pid in sorted(valid_patients):
        b = sorted(grouped_before[pid], key=lambda x: extract_slice_number(x[0]))
        a = sorted(grouped_after[pid], key=lambda x: extract_slice_number(x[0]))
        l = sorted(grouped_labels[pid], key=lambda x: extract_slice_number(x[0]))

        # Align by filename
        pred_before_dict = {fname: pred for fname, pred in b}
        pred_after_dict = {fname: pred for fname, pred in a}
        label_dict_local = {fname: label for fname, label in l}

        common_filenames = sorted(set(pred_before_dict) & set(pred_after_dict) & set(label_dict_local),
                                  key=lambda fname: extract_slice_number(fname))

        x_labels = [str(extract_slice_number(fname)) for fname in common_filenames]
        pred_before = [pred_before_dict[fname] for fname in common_filenames]
        pred_after = [pred_after_dict[fname] for fname in common_filenames]
        label_vals = [label_dict_local[fname] for fname in common_filenames]

        # ✅ Plot
        plt.figure(figsize=(max(12, len(x_labels) * 0.3), 4))
        plt.plot(x_labels, pred_before, label="Before", linestyle="--", marker="o")
        plt.plot(x_labels, pred_after, label="After", linestyle="-", marker="x")
        plt.plot(x_labels, label_vals, label="Label", linestyle=":", marker="s")

        plt.title(f"Classification vs Ground Truth for {pid}")
        plt.xlabel("Slice Number")
        plt.ylabel("Prediction")
        plt.xticks(rotation=60, ha="right", fontsize=8)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{pid}_comparison_with_labels.png"))
        plt.close()


def save_comparison_csv(before, after, save_path="comparison.csv"):
    df = pd.DataFrame([
        (fname, pred1, pred2) for (fname, pred1), (_, pred2) in zip(before, after)
    ], columns=["filename", "before", "after"])
    df.to_csv(save_path, index=False)
    print(f"✅ CSV saved to: {save_path}")

# ------------------------------------------------------------------------------
# 📈 THRESHOLD OPTIMIZATION
# ------------------------------------------------------------------------------

def threshold_sweep(model, test_loader, device, thresholds=np.arange(0.1, 0.55, 0.05)):
    """
    Evaluate precision/recall trade-offs at different thresholds on test set.
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    metrics = []

    with torch.no_grad():
        for threshold in thresholds:
            y_true, y_pred = [], []
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

            avg_loss = total_loss / len(test_loader)
            metrics.append({
                "threshold": threshold,
                "accuracy": 100 * np.mean(np.array(y_true) == np.array(y_pred)),
                "precision": precision_score(y_true, y_pred, zero_division=0) * 100,
                "recall": recall_score(y_true, y_pred, zero_division=0) * 100,
                "f1": f1_score(y_true, y_pred, zero_division=0) * 100,
            })

    return metrics
