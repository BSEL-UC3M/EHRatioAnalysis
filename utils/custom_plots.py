import torch
import numpy as np
import cv2
import os

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
