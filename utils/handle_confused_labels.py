import numpy as np

def correct_confused_labels(predictions_dict, window_size=2):
    """
    Corrects misclassified labels in a sorted sequence of slices per patient.
    If a label is surrounded by a majority of the opposite class, it is flipped.
    
    Args:
        predictions_dict (dict): Dictionary where keys are patient IDs and values are lists of tuples 
                                (slice_number, predicted_label) sorted by slice_number.
        window_size (int): Number of neighboring slices to consider on each side.

    Returns:
        dict: Updated predictions_dict with corrected labels.
    """
    corrected_predictions = {}
    
    for patient, slices in predictions_dict.items():
        slices.sort()  # Ensure slices are sorted by slice_number
        labels = np.array([label for _, label in slices])
        corrected_labels = labels.copy()
        
        for i in range(len(labels)):
            lower_bound = max(0, i - window_size)
            upper_bound = min(len(labels), i + window_size + 1)
            
            surrounding_labels = labels[lower_bound:i].tolist() + labels[i+1:upper_bound].tolist()
            if surrounding_labels:
                majority_label = round(np.mean(surrounding_labels))  # Majority vote
                if majority_label != labels[i]:
                    corrected_labels[i] = majority_label
        
        corrected_predictions[patient] = [(slices[i][0], corrected_labels[i]) for i in range(len(slices))]
    
    return corrected_predictions
