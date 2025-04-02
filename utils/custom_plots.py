import torch
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt

def rescale_image_for_visualization(img):
    """
    Rescala una imagen normalizada o científica (tensor [C, H, W]) para visualización.
    """
    img_np = img.cpu().numpy()
    if img_np.ndim == 3 and img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1, 2, 0))  # (C,H,W) → (H,W,C)

    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    return img_np


def plot_batch_mosaic(images, targets, paths, save_path="mosaic_debug.jpg", names=None):
    """
    Crea un mosaico de imágenes con cajas de anotaciones.
    - images: batch tensor (B, 3, H, W)
    - targets: Nx6 tensor [batch_idx, class_id, x, y, w, h]
    - paths: lista de nombres de archivo
    """
    b, _, h, w = images.shape
    ns = min(b, 16)
    rows = int(np.ceil(ns / 4))
    cols = min(4, ns)

    mosaic_h = rows * h
    mosaic_w = cols * w
    mosaic = np.full((mosaic_h, mosaic_w, 3), 0, dtype=np.uint8)

    for i in range(ns):
        img = rescale_image_for_visualization(images[i])
        r, c = divmod(i, 4)
        top, left = r * h, c * w
        mosaic[top:top + h, left:left + w, :] = img

        # Cajas
        labels = targets[targets[:, 0] == i]
        for label in labels:
            cls, x, y, w_box, h_box = label[1:].cpu().numpy()
            x1 = int((x - w_box / 2) * w) + left
            y1 = int((y - h_box / 2) * h) + top
            x2 = int((x + w_box / 2) * w) + left
            y2 = int((y + h_box / 2) * h) + top
            color = (255, 0, 0) if cls == 0 else (0, 255, 255)
            name = names[int(cls)] if names else str(int(cls))
            cv2.rectangle(mosaic, (x1, y1), (x2, y2), color, 2)
            cv2.putText(mosaic, name, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if paths:
            filename = os.path.basename(paths[i])
            cv2.putText(mosaic, filename, (left + 5, top + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.imwrite(save_path, mosaic)
    print(f"🖼️ Saved custom batch mosaic to {save_path}")

def format_batch_labels(batch_labels):
    """
    Convierte labels tipo [B, N_i, 5] → [N_total, 6] con [image_idx, class, x, y, w, h]
    """
    formatted = []
    for i, boxes in enumerate(batch_labels):
        if boxes.ndim == 1 or boxes.numel() == 0:
            continue  # imagen sin anotaciones
        img_idx = torch.full((boxes.shape[0], 1), i, dtype=boxes.dtype)
        formatted_boxes = torch.cat((img_idx, boxes), dim=1)  # (N, 6)
        formatted.append(formatted_boxes)
    return torch.cat(formatted, dim=0) if formatted else torch.zeros((0, 6))

def plot_threshold_tradeoffs(metrics_list, results_dir=None, filename="threshold_tradeoffs.png"):
    """
    Plots Accuracy, Precision, Recall, F1 vs Threshold and saves to results_dir.
    
    Parameters:
        metrics_list: List of dicts with threshold evaluation metrics
        results_dir: Folder to save the plot into (optional)
        filename: Name of the file to save
    """
    df = pd.DataFrame(metrics_list)

    plt.figure(figsize=(10, 6))
    plt.plot(df["threshold"], df["accuracy"], label="Accuracy", marker='o')
    plt.plot(df["threshold"], df["precision"], label="Precision", marker='o')
    plt.plot(df["threshold"], df["recall"], label="Recall", marker='o')
    plt.plot(df["threshold"], df["f1"], label="F1 Score", marker='o')

    plt.xlabel("Threshold for Class 1")
    plt.ylabel("Metric (%)")
    plt.title("Threshold Sweep - Classification Trade-offs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if results_dir:
        save_path = os.path.join(results_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved threshold trade-off plot to {save_path}")

    plt.close()

def plot_class1_probability_histogram(model, test_loader, device, threshold, results_dir=None, filename="class1_prob_histogram.png"):
    """
    Plots histogram of predicted probabilities for class 1, separated by true label.
    
    Parameters:
    - model: Trained classification model.
    - test_loader: DataLoader for test set.
    - device: 'cuda' or 'cpu'.
    - results_dir: Path to save the plot (if None, just shows it).
    - filename: Name of the output image file (only if results_dir is set).
    """
    model.eval()
    class1_probs = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            class1_probs.extend(probs[:, 1].cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    class1_probs = np.array(class1_probs)
    true_labels = np.array(true_labels)

    # Split by true class
    class0_probs = class1_probs[true_labels == 0]
    class1_probs_true = class1_probs[true_labels == 1]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(class0_probs, bins=40, alpha=0.6, label="True Class 0", color='steelblue')
    plt.hist(class1_probs_true, bins=40, alpha=0.6, label="True Class 1", color='darkorange')
    plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold:.2f}")
    plt.xlabel("Predicted Probability for Class 1")
    plt.ylabel("Number of Samples")
    plt.title("Softmax Probabilities for Class 1")
    plt.legend()
    plt.grid(True)

    if results_dir:
        save_path = os.path.join(results_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved class probability histogram to {save_path}")
        plt.close()