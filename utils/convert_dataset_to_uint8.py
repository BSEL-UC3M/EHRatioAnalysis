# ==============================================================================
# File: convert_dataset_to_uint8.py
# Description: Converts all .tif images in YOLO dataset folders (train/val/test)
#              to uint8 .tif format in a new temporary dataset directory.
# Author: @claudiacastrillon
# ==============================================================================

import os
import shutil
import yaml
from pathlib import Path
from tifffile import imread, imwrite
import numpy as np

def convert_tif_to_uint8(image_path):
    """Converts a TIFF image to uint8 format, scaling its intensity."""
    image = imread(image_path).astype('float32')
    min_val = image.min()
    max_val = image.max()
    if max_val - min_val > 0:
        image = (image - min_val) / (max_val - min_val)
    else:
        image *= 0
    image = (image * 255).astype('uint8')
    return image

def convert_yolo_dataset_to_uint8(original_yaml_path, output_base_dir):
    """
    Converts all images in the dataset described by original_yaml_path
    to uint8 .tif format, and copies corresponding .txt files.
    
    Parameters:
        - original_yaml_path (str): path to original dataset.yaml
        - output_base_dir (str): directory to store the converted dataset
    Returns:
        - output_yaml_path (str): path to new dataset.yaml with uint8 images
    """

    with open(original_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Crear estructura de carpetas
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    new_paths = {}

    for split in ['train', 'val', 'test']:
        orig_dir = Path(config[split])
        out_dir = output_base_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)
        new_paths[split] = str(out_dir.resolve())

        for file in orig_dir.glob("*.tif"):
            converted = convert_tif_to_uint8(file)
            imwrite(out_dir / file.name, converted)

            txt_file = file.with_suffix('.txt')
            if txt_file.exists():
                shutil.copy(txt_file, out_dir / txt_file.name)

    # Guardar nuevo dataset.yaml
    output_yaml_path = output_base_dir / "dataset.yaml"
    new_config = {
        "train": new_paths['train'],
        "val": new_paths['val'],
        "test": new_paths['test'],
        "nc": config["nc"],
        "names": config["names"]
    }

    with open(output_yaml_path, 'w') as f:
        yaml.dump(new_config, f)

    print(f"✅ Dataset converted to uint8 and saved in {output_base_dir}")
    return str(output_yaml_path)
