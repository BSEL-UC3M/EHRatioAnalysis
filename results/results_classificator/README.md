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
# **Classification Results - PEI Image Classifiers**

This folder contains results from two different classification models applied to **PEI medical images**:

---

## 🔍 Model Summaries

### 1. **ResNet50-based Classifier**
- Initially tested **ResNet18**, later upgraded to **ResNet50** for better accuracy.
- Implemented **L2 regularization** to prevent overfitting.
- Used **SGD optimizer** with momentum for improved training stability.
- Included **Dropout layers** (0.5) in the fully connected head to improve generalization.

### 2. **5-Layer CNN Classifier**
- Started with a **2-layer CNN**, later expanded to a **5-layer CNN** for deeper feature extraction.
- Used **Adam optimizer** (better suited for CNNs).
- Added **Dropout (0.3)** and **Batch Normalization** to enhance generalization and training performance.

---

## ⚙️ Model Selection via `main_classificator.py`

The script supports runtime selection of model architecture via CLI:

```bash
python main_classificator.py --model resnet50
# or
python main_classificator.py --model cnn
`
## 📊 Performance Overview on PEI Dataset

| Model        | Optimizer | Regularization | Dropout | BatchNorm | Accuracy | Avg. Loss |
|--------------|-----------|----------------|---------|-----------|----------|-----------|
| ResNet50     | SGD       | L2 (1e-4)       | 0.5     | No        | **94.36%** | 0.2546    |
| 5-Layer CNN  | Adam      | L2 (5e-4)       | 0.3     | Yes       | **97.74%** | 0.0649    |

---

## 📂 Results Folder Structure

All experiment outputs are organized by **image type** and **model** inside timestamped folders:

```plaintext
results_classificator_MRC/
├── resnet50_<timestamp>/
│   ├── results.txt
│   ├── confusion_matrix.png
│   ├── train_val_loss.png
│   ├── train_val_accuracy.png
│
├── cnn_<timestamp>/
│   ├── results.txt
│   ├── confusion_matrix.png
│   ├── train_val_loss.png
│   ├── train_val_accuracy.png

results_classificator_PEI/
├── resnet50_<timestamp>/
│   ├── results.txt
│   ├── confusion_matrix.png
│   ├── train_val_loss.png
│   ├── train_val_accuracy.png
│
├── cnn_<timestamp>/
│   ├── results.txt
│   ├── confusion_matrix.png
│   ├── train_val_loss.png
│   ├── train_val_accuracy.png






