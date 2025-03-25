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
RESULTS_DIR = "./results/object_detector/MRC/all_patients"

def train_yolo(
    dataset_yaml,
    epochs=50,
    batch_size=8,
    model_name="yolov5su",
    save_results=True,
    verbose=True,
    img_size=640,
    conf=0.35,
    patience=10,
    # this is done in case data augmentation is desired
    fliplr=0.0,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    augment=False, 
    # added for the custom plots implementation 
    train_loader=None, 
    val_loader=None
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = YOLOv5(model_name=model_name, pretrained=True, device=device)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if save_results:
        project = "./results/object_detector/MRC"
        name = f"{model_name}-{timestamp}"
    else:
        project = "/tmp/yolo_no_save"
        name = "temp_train"

    # Entrenamiento YOLO con parámetros avanzados
    model.model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        conf=conf,
        patience=patience,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
        augment=augment,
        project=project,
        name=name,
        exist_ok=True,
        verbose=verbose
    )

    print("✅ Training complete!")
    ##### ESTO ES PARA LAS CUSTOM PLOTS ####
    ##### ESTO ES PARA LAS CUSTOM PLOTS ####
    from utils.custom_plots import plot_batch_mosaic, format_batch_labels

    if save_results:
        custom_plot_dir = os.path.join(project, name, "custom_plots")
        os.makedirs(custom_plot_dir, exist_ok=True)

        class_names = ["left ear", "right ear"]

        # Mosaicos de entrenamiento
        if train_loader:
            for i, (images, labels, paths) in enumerate(train_loader):
                if i >= 3:
                    break
                paths = list(paths)  # convertimos a lista de strings
                save_path = os.path.join(custom_plot_dir, f"train_batch{i}.jpg")

                batch_targets = format_batch_labels(labels)
                plot_batch_mosaic(images, batch_targets, paths=paths, save_path=save_path, names=class_names)

        # Mosaicos de validación
        if val_loader:
            val_images, val_labels, val_paths = next(iter(val_loader))
            val_paths = list(val_paths)

            batch_val_targets = format_batch_labels(val_labels)

            plot_batch_mosaic(val_images, batch_val_targets, paths=val_paths,
                            save_path=os.path.join(custom_plot_dir, "val_batch0_labels.jpg"),
                            names=class_names)

            # Simular predicciones (de momento igual que labels)
            pred_labels = batch_val_targets.clone()
            plot_batch_mosaic(val_images, pred_labels, paths=val_paths,
                            save_path=os.path.join(custom_plot_dir, "val_batch0_pred.jpg"),
                            names=class_names)






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

