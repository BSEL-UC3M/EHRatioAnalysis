import os
import cv2
from tifffile import imread
from glob import glob
from pathlib import Path
import shutil

def prepare_yolo_proxy_images(original_dataset_path, proxy_dataset_path):
    """
    Convierte todas las imágenes .tif del dataset original a .png 
    en un nuevo directorio proxy, manteniendo la estructura train/val/test.
    """
    for split in ['train', 'val', 'test']:
        original_split = os.path.join(original_dataset_path, split)
        proxy_split = os.path.join(proxy_dataset_path, split)
        os.makedirs(proxy_split, exist_ok=True)

        tif_files = glob(os.path.join(original_split, "*.tif"))

        for tif_path in tif_files:
            filename = Path(tif_path).stem
            png_path = os.path.join(proxy_split, f"{filename}.png")

            try:
                img = imread(tif_path)

                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                img = img.astype('float32')
                img = img / img.max() if img.max() > 0 else img
                img = (img * 255).clip(0, 255).astype("uint8")

                cv2.imwrite(png_path, img)
                shutil.copyfile(tif_path.replace('.tif', '.txt'), png_path.replace('.png', '.txt'))

                print(f"✅ {split}: {filename}.tif → .png")
            except Exception as e:
                print(f"❌ Error converting {tif_path}: {e}")

    # Generar nuevo dataset.yaml
    dataset_yaml_path = os.path.join(proxy_dataset_path, "dataset.yaml")
    with open(dataset_yaml_path, "w") as f:
        f.write(f"train: {os.path.abspath(os.path.join(proxy_dataset_path, 'train'))}\n")
        f.write(f"val: {os.path.abspath(os.path.join(proxy_dataset_path, 'val'))}\n")
        f.write(f"test: {os.path.abspath(os.path.join(proxy_dataset_path, 'test'))}\n")
        f.write("nc: 2\n")
        f.write("names: ['left ear', 'right ear']\n")

    print(f"\n📄 Nuevo dataset.yaml creado: {dataset_yaml_path}")
