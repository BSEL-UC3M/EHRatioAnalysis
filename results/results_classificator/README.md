# **Classification Results - MRC Image Classifiers**

This folder contains results from two different classification models:

1. **ResNet50-based Classifier**  
   - Initially, I tested **ResNet18** but later switched to **ResNet50** for better accuracy.
   - Implemented **L2 regularization** to prevent overfitting.
   - Used **SGD optimizer** with momentum for improved training stability.
   - Included **Dropout layers** in the fully connected head to improve generalization.

2. **5-Layer CNN Classifier**  
   - Initially tested **2-layer CNN**, but improved to a **5-layer CNN** for better feature extraction.
   - Used **Adam optimizer** instead of SGD (better for CNNs).
   - Added **Dropout (0.3) and Batch Normalization** to improve generalization.

---

## **Implemented Model Selection in `main_classificator.py`**
- The script allows users to **select either `resnet50` or `cnn`** at runtime.
- The classification is performed on **MRC medical images**.

---

## **Results Folder Structure**
Each model's results are saved in a timestamped folder:

📂 `results_classificator/`  
 ├── 📁 `resnet50_20250213-131704/`  
 │   ├── `results.txt` (Training info: epochs, layers, LR, accuracy, confusion matrix)  
 │   ├── `confusion_matrix.png` (Seaborn-generated matrix)  
 │   ├── `train_val_loss.png` (Training/validation loss graph)  
 │   ├── `train_val_accuracy.png` (Training/validation accuracy graph)  
 ├── 📁 `cnn_20250213-122648/`  
 │   ├── (same structure)  
 │
 ├── `README.md` (This file)

---

## **Performance Overview**
| Model     | Optimizer | Regularization | Dropout | BatchNorm | Accuracy |
|-----------|----------|---------------|---------|-----------|---------|
| ResNet50  | SGD      | L2 (1e-4)     | 0.5     | No        | 89.38% |
| 5-Layer CNN | Adam    | L2 (5e-4)     | 0.3     | Yes       | 93.33% |

---

## **Next Steps**
- Experiment with **different data augmentation techniques**.
- Implement **hyperparameter tuning** for CNN architecture.
- Test **ResNet34** as an intermediate solution.

---
