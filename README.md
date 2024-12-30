# Multi-Network Segmentation Pipeline for EH Ratio Analysis

This project implements a full pipeline for image processing and segmentation, integrating three different neural networks to achieve a final segmentation output. The pipeline is designed to analyze both MRC and REAL datasets, process their volumetric images, classify regions of interest, and segment specific areas to compute the EH ratio. The ultimate goal is to provide an end-to-end solution for automated medical image analysis.

---

## **Network Structure**

Below is the architecture of the complete pipeline:

![Network Structure](Network-structure.png)

---

## **Folder Structure**

Below is the folder structure for the GitHub repository, designed to streamline development and testing:
```
root/
├── dataloader/
│   └── dataloader.py   # Classes for loading MRC and REAL images with labels for classification, ROIs, and segmentation.
├── models/
│   ├── classificator.py    # Model architecture for slice classification.
│   ├── object_detector.py  # Model architecture for object detection.
│   └── segmentator.py      # Model architecture for image segmentation.
├── trainers/
│   ├── classificator/
│   │   └── classificator.py    # Training and evaluation functions for the classifier.
│   ├── object_detector/
│   │   └── object_detector.py  # Training and evaluation functions for object detection.
│   └── segmentator/
│       └── segmentator.py      # Training and evaluation functions for segmentation.
├── utils/
│   ├── EHRatioCalculator.py    # Utility to calculate the EH ratio from segmentation results.
│   ├── metrics.py              # Common metrics for evaluation (e.g., Dice score, IoU).
│   └── plots.py                # Visualization tools for predictions and results.
├── losses/
│   └── losses.py    # Custom loss functions for segmentation, object detection, and classification tasks.
├── results/    # Folder to store results, organized by timestamps.
├── toy_dataset/    # Test dataset with MRC images for debugging and prototyping (not included in the repository).
├── main_segmentator.py      # Main script to train and evaluate the segmentation network.
├── main_object_detector.py  # Main script to train and evaluate the object detection network.
├── main_classificator.py    # Main script to train and evaluate the classification network.
└── main.py    # Final pipeline script integrating all three networks with pretrained weights.
```
# Final pipeline script integrating all three networks with pretrained weights
---

## **Pipeline Components**

### **1. Preprocessing**
- Images from MRC and REAL datasets are preprocessed.
- Outputs include:
  - Volumetric images.
  - Slice-level labels for classification.
  - Region of Interest (ROI) labels for object detection.
  - Surface labels for segmentation tasks.

### **2. CNN Classifier**
- Classifies slices into "ear" and "non-ear".
- Designed for robust performance across MRC and REAL datasets.
- Model details are implemented in `models/classificator.py`.

### **3. Object Detector**
- Detects regions of interest in classified "ear" slices.
- Outputs bounding boxes for regions containing relevant structures.
- Model details are implemented in `models/object_detector.py`.

### **4. Image Segmentation**
- Segments the detected regions into fine-grained areas of interest.
- Outputs masks to compute metrics like Dice score, IoU, and the EH ratio.
- Model details are implemented in `models/segmentator.py`.

### **5. Post-Processing**
- Volume-based refinement of segmentation results.
- Computation of the EH ratio using `utils/EHRatioCalculator.py`.

---

TBC

