import numpy as np
from scipy.signal import medfilt

def correct_confused_labels(predictions_dict, window_size=3, min_confidence=0.6):
    """
    Corrects misclassified labels in a sorted sequence of slices per patient using median filtering
    and confidence-based correction.

    Args:
        predictions_dict (dict): Dictionary where keys are patient IDs and values are lists of tuples 
                                (slice_number, predicted_label, confidence_score) sorted by slice_number.
        window_size (int): Number of neighboring slices to consider in median filtering.
        min_confidence (float): Minimum confidence score below which corrections will be considered.

    Returns:
        dict: Updated predictions_dict with corrected labels.
    """
    corrected_predictions = {}

    for patient, slices in predictions_dict.items():
        slices.sort()  # Ensure slices are sorted by slice_number
        labels = np.array([label for _, label, _ in slices])  # Extract predicted labels
        confidences = np.array([conf for _, _, conf in slices])  # Extract confidence scores
        
        # ✅ Step 1: Apply Median Filtering for Trend Correction
        filtered_labels = medfilt(labels, kernel_size=window_size)  # Median filter smoothing

        # ✅ Step 2: Confidence-Based Correction
        corrected_labels = labels.copy()
        for i in range(len(labels)):
            if labels[i] != filtered_labels[i] and confidences[i] < min_confidence:
                corrected_labels[i] = filtered_labels[i]  # Correct only low-confidence errors
        
        # ✅ Store corrected predictions
        corrected_predictions[patient] = [(slices[i][0], corrected_labels[i]) for i in range(len(slices))]

    return corrected_predictions

