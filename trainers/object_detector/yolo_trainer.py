# ==============================================================================
# File: trainers/object_detector/yolo_trainer.py
# Description: Handles training and evaluation of the YOLO object detector.
# Author: @cfusterbarcelo
# Creation Date: 24/02/2025
# ==============================================================================

import torch
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from models.object_detector.object_detector import YOLOv5


# Define results directory
RESULTS_DIR = "./results/object_detector/MRC_toydataset"

def train_yolo(dataset_yaml, epochs=50, batch_size=8, model_name="yolov5su", save_results=True, verbose=True):
    """
    Train the YOLO model.

    Parameters:
    - dataset_yaml: Path to the dataset YAML file.
    - epochs: Number of training epochs.
    - batch_size: Training batch size.
    - model_name: Name of the YOLO model (e.g., "yolov5s").
    - save_results: If False, no logs or results will be saved.
    - verbose: Controls training verbosity.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = YOLOv5(model_name=model_name, pretrained=True, device=device)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # If save_results is False, we set a temporary run directory that will be discarded
    if save_results:
        project = "./results/object_detector"
        name = f"{model_name}-{timestamp}"
    else:
        project = "/tmp/yolo_no_save"  # Temporary directory for testing
        name = "temp_train"
    
    # Run YOLO training
    model.model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        project=project,
        name=name,
        exist_ok=True,
        verbose=verbose
    )

    print("✅ Training complete!")



def evaluate_yolo(dataset_path, model_path, model_name="yolov5su", verbose=False):
    """
    Evaluates the trained YOLO model on the test set.

    Parameters:
    - dataset_path: (str) Path to dataset with images and annotations.
    - model_path: (str) Path to trained YOLO model file.
    - model_name: (str) YOLO model variant.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"📊 Evaluating {model_name} on {device}...")

    # ✅ Run evaluation using YOLOv5 API
    model = YOLOv5(model_name=model_name, pretrained=False, device=device)
    model.load_model(model_path)
    results = model.model.val(data=dataset_path, batch=8)

    # ✅ Create timestamped results folder
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(RESULTS_DIR, f"{model_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # ✅ Save evaluation results
    with open(os.path.join(results_dir, "results.txt"), "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset Path: {dataset_path}\n")
        f.write(f"Trained Model Path: {model_path}\n")
        f.write(f"Mean Average Precision (mAP@0.5): {results.box.map50:.4f}\n")
        f.write(f"Precision: {results.box.precision:.4f}\n")
        f.write(f"Recall: {results.box.recall:.4f}\n")

    print(f"✅ Evaluation completed! Results saved in {results_dir}")
