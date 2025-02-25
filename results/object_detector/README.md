# Object Detector - YOLO Implementation

## Overview
This folder contains the implementation of an object detector using YOLO. The provided code is designed to train and evaluate a YOLO model on MRC TIFF images, with annotations generated from an ImageJ macro. The objective is to detect specific regions in the images, such as the left and right ear.

## How the Code is Organized
The code is structured as follows:
- **`generate_yolo_annotations.py`**: Converts annotations from the CSV file (generated using the ImageJ macro) into YOLO-compatible format.
- **`main_object_detector.py`**: The main script for training and evaluating the YOLO model.
- **`dataloader_MRC_object_detector.py`**: Defines the dataset class and DataLoader for handling image and annotation loading.
- **`yolo_trainer.py`**: Contains the functions for training the YOLO model.
- **`dataset.yaml`**: Configuration file specifying the paths for train, validation, and test datasets.
- **`utils/`**: Contains the macro to generate the initial annotation CSV from ImageJ.

## Required Steps Before Running the Training
1. **Generate the CSV annotation file**  
   - Use the ImageJ macro (found in `utils/`) to annotate the images.
   - This macro outputs a CSV file containing the bounding box information.

2. **Convert CSV annotations to YOLO format**  
   - Run `generate_yolo_annotations.py` to convert the CSV into YOLO-compatible `.txt` annotation files.

3. **Ensure correct dataset structure**  
   - The dataset should be organized as:
     ```
     toydataset/object_detection/YOLO/
     ├── train/
     │   ├── MRC_1_xxxxxx.tif
     │   ├── MRC_2_xxxxxx.tif
     │   ├── ...
     │   ├── MRC_1_xxxxxx.txt  # Corresponding annotation
     │   ├── MRC_2_xxxxxx.txt
     │   ├── ...
     ├── val/
     ├── test/
     ├── dataset.yaml
     ```

## Training the YOLO Model
To train the YOLO model, simply run:
```bash
python main_object_detector.py
```
The model will train using the specified dataset and parameters.

## Understanding the `dataset.yaml` File
The `dataset.yaml` file contains:
```yaml	
train: ./toydataset/object_detection/YOLO/train
val: ./toydataset/object_detection/YOLO/val
test: ./toydataset/object_detection/YOLO/test
nc: 2
names: ['left ear', 'right ear']
```
* **train**, **val**, and **test**: Paths to the respective dataset splits.
* **nc**: Number of classes (2: left ear, right ear).
* **names*: Names of the classes.

## ✅ Pending Tasks

### 🔹 **Check Bounding Boxes Visualization**
Bounding boxes might not be aligned correctly (e.g., `MRC_5_63210016.tif`). A function should be implemented to display images with bounding boxes and verify correctness.
- [ ] Implement a visualization function for bounding boxes.
- [ ] Verify normalization is correctly applied.

### 🔹 **Test Different YOLO Versions**
Currently, `yolov5s` is being used. Other versions like `yolov5m`, `yolov5l`, or `yolov8` should be tested.
- [ ] Modify `main_object_detector.py` to allow selecting different YOLO versions.
- [ ] Test performance with `yolov5m`, `yolov5l`, and `yolov8`.

### 🔹 **Store Annotations in the Correct Folders**
Annotations should be saved directly in `train/`, `val/`, and `test/` instead of a separate `yolo_annotations/` folder.
- [ ] Update `generate_yolo_annotations.py` to save annotations in the correct folder.
- [ ] Verify that the YOLO model reads the annotations correctly.

### CHange location of `.pt` files
Some `.pt` files are being saved in the root directory. Everything from every run should be saved in the results folder where it belongs.
- [ ] Remove the `yolo11n.pt` and `yolov5su.pt` files from the root directory.
- [ ] Check how are they created and stored and change its location.

