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



