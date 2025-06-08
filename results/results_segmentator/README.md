# Segmentation Results using UNet

## Overview

This repository contains segmentation results obtained using a UNet model applied to different datasets and preprocessing techniques. The main goal is to compare the performance of various approaches using Mean Dice Score and Mean Intersection over Union (IoU).

## Results

The following table summarizes the results:


| Approach                    |  Mean Dice Score  |      Mean IoU     |
|-----------------------------|-------------------|-------------------|
| MRC with data augmentation  | 0.905 (SD: 0.071) | 0.833 (SD: 0.104) |
| PEI with data augmentation  | 0.742 (SD: 0.224) | 0.563 (SD: 0.267) |
| MRC                         | 0.8560            | 0.7546            |
| PEI                         | 0.6612            | 0.5239            |
| PEI with preprocessing      | 0.7134            | 0.4761            |



- **MRC**: Original images + data augmentation (horizontal flip)
- **PEI**: Original images + data augmentation (horizontal flip)

## Model

- The segmentation model used is **UNet**.
[Model README](MRC/BEST/20250405%20MRC%20NEW%20TRAINING%20MODIEFIED%20LABELS/20250405-170559/README.md)



# Segmentation Results using UNet

## Overview

This repository contains segmentation results obtained using a **UNet** model applied to the **3D-SPACE-MRC** and **3D-REAL-IR** datasets. Multiple training strategies were explored, including variations in normalization, initialization, regularization, data augmentation, and loss functions.

The main performance metrics used are **Dice Score Coefficient (DSC)**, **Intersection over Union (IoU)**, and **Recall**.

---

## Results Summary

The table below summarizes the performance across different experiments with the UNet architecture:

| Experiment | Description                             | DSC (MRC) | IoU (MRC) | Recall (MRC) | DSC (REAL) | IoU (REAL) | Recall (REAL) |
|------------|-----------------------------------------|-----------|-----------|---------------|-------------|-------------|----------------|
| E1         | Baseline UNet                           | 0.85      | 0.76      | 0.86          | 0.68        | 0.46        | 0.58           |
| E2         | + Group and Instance Normalization      | 0.87      | 0.79      | 0.84          | 0.69        | 0.49        | 0.60           |
| E3         | + LeakyReLU                             | 0.87      | 0.79      | 0.86          | 0.70        | 0.52        | 0.67           |
| E4         | + Kaiming Initialization                | 0.87      | 0.79      | 0.83          | 0.69        | 0.50        | 0.61           |
| E5         | + Dropout (No Kaiming Initialization)   | 0.88      | 0.80      | 0.84          | 0.70        | 0.53        | 0.70           |
| E6         | + Dropout (With Kaiming Initialization) | 0.87      | 0.80      | 0.88          | 0.71        | 0.54        | 0.68           |
| E7         | + Data Augmentation                     | 0.89      | 0.81      | 0.89          | 0.73        | 0.55        | 0.69           |
| E8         | Change learning rate from 1e-3 to 1e-4  | **0.91**  | **0.83**  | **0.92**      | 0.75        | 0.55        | 0.68           |
| E9         | Change batch size to 32                 | **0.91**  | 0.83      | 0.89          | 0.73        | 0.55        | 0.74           |
| E10        | Change batch size to 6                  | 0.90      | 0.83      | 0.88          | **0.75**    | **0.56**    | **0.72**       |

---

## Threshold Variation

Different prediction thresholds were tested to evaluate their effect on segmentation performance. A threshold of **0.9** for 3D-SPACE-MRC and **0.8** for 3D-REAL-IR provided the best balance between precision and recall.

| Dataset        | Threshold | Dice     | IoU      | Recall   |
|----------------|-----------|----------|----------|----------|
| 3D-SPACE-MRC   | 0.5       | 0.8997   | 0.8260   | 0.9201   |
|                | 0.7       | 0.9019   | 0.8293   | 0.9344   |
|                | 0.9       | **0.9053** | **0.8339** | **0.9229** |
|                | 0.95      | 0.9051   | 0.8335   | 0.9156   |
|                | 0.98      | 0.9047   | 0.8338   | 0.9048   |
| 3D-REAL-IR     | 0.5       | 0.7194   | 0.5630   | 0.7305   |
|                | 0.6       | 0.7266   | 0.5650   | 0.7370   |
|                | 0.7       | 0.7397   | 0.5671   | 0.7312   |
|                | 0.8       | **0.7454** | **0.5629** | **0.7288** |
|                | 0.9       | 0.7390   | 0.5595   | 0.7065   |

---

## Loss Function Comparison

The impact of different loss functions was also explored. The combination of **Binary Cross Entropy (BCE) + Dice Loss** yielded the most balanced performance, particularly on the 3D-SPACE-MRC dataset.

| Loss Function | Dataset | Avg Loss | Dice | IoU  | Recall |
|---------------|---------|----------|------|------|--------|
| BCE           | MRC     | 0.044    | 0.02 | 0.00 | 0.00   |
| BCE           | REAL    | 0.044    | 0.13 | 0.00 | 0.00   |
| Focal         | MRC     | 0.003    | 0.02 | 0.00 | 0.00   |
| Focal         | REAL    | 0.003    | 0.13 | 0.00 | 0.00   |
| Dice          | MRC     | 0.008    | 0.84 | 0.75 | 0.85   |
| Dice          | REAL    | 0.026    | 0.61 | 0.47 | 0.72   |
| **BCE + Dice**| MRC     | **0.016**| **0.85** | **0.76** | **0.86** |
| **BCE + Dice**| REAL    | **0.048**| **0.68** | **0.52** | **0.71** |
| Tversky       | MRC     | 0.006    | 0.85 | 0.75 | 0.86   |
| Tversky       | REAL    | 0.017    | 0.68 | 0.48 | 0.60   |

---

## Model

- The segmentation model used is **UNet** with variations in normalization (GroupNorm, InstanceNorm), activation functions (ReLU, LeakyReLU), weight initialization (Kaiming), and regularization (Dropout).
- Loss functions evaluated include BCE, Dice, BCE + Dice, Focal, and Tversky.
- [Model README](MRC/BEST/20250405%20MRC%20NEW%20TRAINING%20MODIEFIED%20LABELS/20250405-170559/README.md)

---

