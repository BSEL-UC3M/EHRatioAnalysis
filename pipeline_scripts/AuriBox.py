# ======================================================================
# File: pipeline_setup/AuriBox.py
# Description: Object detection step for MRC or PEI using YOLOv5.
# Author: @cfusterbarcelo
# Created: 09/04/2025
# ======================================================================

import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
from pipeline_scripts.utils import convert_images_to_uint8
from pipeline_scripts.plots import plot_confidence_distribution, plot_detection_heatmap


def run_auribox_inference(
    image_folder,
    model_path,
    device,
    result_folder,
    selected_images,  # List of (filename, label)
    dataset_type="MRC"
):
    """
    Run YOLOv5 object detection only on selected ear-containing slices.
    Saves detection visualizations and returns dict of detections.
    """
    print(f"\n📦 Running AuriBox on {dataset_type} images")

    os.makedirs(result_folder, exist_ok=True)
    vis_folder = os.path.join(result_folder, "visualizations")
    os.makedirs(vis_folder, exist_ok=True)

    # Load YOLOv5 model
    model = YOLO(model_path)
    model.to(device)
    model.fuse()

    # Filter filenames with label == 1
    selected_filenames = [fname for fname, label in selected_images if label == 1]
    full_image_paths = [os.path.join(image_folder, fname) for fname in selected_filenames]

    # Convert all to uint8 for YOLO inference
    temp_uint8_dir, converted_paths = convert_images_to_uint8(full_image_paths)

    detections = {}

    for image_path in tqdm(converted_paths, desc=f"{dataset_type} Detection"):
        filename = os.path.basename(image_path)

        try:
            results = model(image_path, verbose=False)
            if results and len(results) > 0:
                result_img = results[0].plot()  # numpy array
                save_path = os.path.join(vis_folder, filename.replace(".tif", "_det.png"))
                Image.fromarray(result_img).save(save_path)

                # Extract one bbox per class (e.g., left/right ear)
                bboxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()

                detection_list = []
                for cls in np.unique(classes):
                    inds = np.where(classes == cls)[0]
                    best_idx = inds[np.argmax(confs[inds])]
                    detection_list.append({
                        "bbox": bboxes[best_idx].tolist(),
                        "conf": float(confs[best_idx]),
                        "class": int(cls)
                    })

                detections[filename] = detection_list
        except Exception as e:
            print(f"⚠️ Skipped {filename} due to error: {e}")
            break

    plot_confidence_distribution(detections, save_path=result_folder)
    plot_detection_heatmap(detections, image_shape=(384, 324), save_path=result_folder)

    print(f"✅ Saved detection results and plots to: {vis_folder}")
    return detections
