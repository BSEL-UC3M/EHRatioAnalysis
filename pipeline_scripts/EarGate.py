# ======================================================================
# File: pipeline_setup/EarGate.py
# Description: Classification + postprocessing step for MRC or PEI.
# Author: @cfusterbarcelo
# Created: 08/04/2025
# ==============================================================================
import torch
import os
from pipeline_scripts.classification_postprocess import (
    smooth_classification_predictions,
    plot_comparison,
    plot_comparison_with_labels,
    save_comparison_csv
)
from models.classificator.five_layer_cnn import FiveLayerCNN as FiveLayerCNN_MRC
from models.classificator.five_layer_cnn_PEI import FiveLayerCNN as FiveLayerCNN_PEI
from dataloader.dataloader_MRC_classificator import load_inference_dataloader as load_mrc
from dataloader.dataloader_PEI_classificator import load_inference_dataloader as load_pei


def run_eargate_inference(
    image_folder,
    model_path,
    device,
    result_folder,
    label_csv=None,
    dataset_type="MRC",
    class_threshold=0.2,
    batch_size=16,
    expand_around_ear_slices: int = 0
):
    """
    Run classification + postprocessing for MRC or PEI.
    Returns: cleaned list of (filename, prediction)
    """
    os.makedirs(result_folder, exist_ok=True)
    print(f"\n📌 Running EarGate on {dataset_type} images")

    # Load model
    # Choose the correct model architecture
    MODEL_ARCHITECTURES = {
        "MRC": FiveLayerCNN_MRC,
        "PEI": FiveLayerCNN_PEI,
    }
    model = MODEL_ARCHITECTURES[dataset_type](num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load dataloader
    dataloader = load_mrc(image_folder, batch_size) if dataset_type == "MRC" else load_pei(image_folder, batch_size)

    # Inference
    raw_preds = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            filenames = batch["filename"]
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = (probs[:, 1] > class_threshold).long().cpu().numpy()
            raw_preds.extend(zip(filenames, preds))

    # Postprocessing
    cleaned = smooth_classification_predictions(raw_preds)
    
    # === EXPAND CONTEXT FOR PEI ===
    if dataset_type == "PEI" and expand_around_ear_slices > 0:
        filenames = [f for f, _ in raw_preds]
        filename_to_index = {f: i for i, f in enumerate(filenames)}
        selected_indices = [filename_to_index[f] for f, p in cleaned if p == 1]

        expanded_indices = set()
        for idx in selected_indices:
            for offset in range(-expand_around_ear_slices, expand_around_ear_slices + 1):
                new_idx = idx + offset
                if 0 <= new_idx < len(filenames):
                    expanded_indices.add(new_idx)

        # Build final expanded cleaned list
        expanded_cleaned = [(filenames[i], 1) for i in sorted(expanded_indices)]
        full_cleaned = []
        for fname in filenames:
            label = 1 if fname in dict(expanded_cleaned) else 0
            full_cleaned.append((fname, label))
        cleaned = full_cleaned

    
    # === PLOTTING ===
    plots_path = os.path.join(result_folder, "plots")

    plots_with_labels_path = os.path.join(result_folder, "plots_with_labels")
    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(plots_with_labels_path, exist_ok=True)

    # Plotting
    if label_csv:
        plot_comparison_with_labels(
            before=raw_preds,
            after=cleaned,
            label_csv=label_csv,
            save_path=plots_with_labels_path
        )
    else:
        plot_comparison(
            before=raw_preds,
            after=cleaned,
            save_path=plots_path
        )

    # Save CSV
    save_comparison_csv(
        before=raw_preds,
        after=cleaned,
        save_path=os.path.join(result_folder, "comparison.csv")
    )

    return cleaned
