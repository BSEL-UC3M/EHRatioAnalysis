# ==============================================================================
# File: ultils/classification_postprocess.py
# Description: Groups slices by patient, sort them numerically and apply a smoothing
# filtering rule that enforces blocks of 1s in the middle, suppress isolated 1s or 0s
# and bias toward class 1 if uncertain.
# Author: @cfusterbarcelo
# Created: 25/03/2025
# ==============================================================================

import re
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import defaultdict
import os

def extract_patient_and_index(filename):
    """
    Extract patient ID and slice number from filename.
    Expected format: something like 'patientX_sliceYYY.ext'
    """
    match = re.match(r"(patient\d+).*?(\d+)", filename)
    if match:
        return match.group(1), int(match.group(2))
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