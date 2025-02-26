# **PCA Analysis & SMOTE Evaluation for Class Imbalance**

## **Overview**
This experiment investigates the class imbalance problem in our dataset using **PCA visualization** and evaluates the impact of **SMOTE oversampling** on **SVM classification performance** for **PEI images**. The results compare the model's performance **with and without SMOTE** to determine how oversampling affects classification accuracy.

---

## **1️⃣ PCA Visualization of the Dataset**
We reduced the **image feature space** to **2D** using **Principal Component Analysis (PCA)** to observe the class distribution before and after applying **SMOTE**.

### 🔹 **Original Dataset (Before SMOTE)**
<img src="pca_visualization.png" alt="PCA Before SMOTE" width="50%">
- The dataset is **highly imbalanced** (Class 0 dominates).
- The SVM classifier **ignored Class 1 entirely** in this setting.

### 🔹 **SMOTE Oversampled Dataset**
<img src="pca_visualization_smote.png" alt="PCA After SMOTE" width="50%">
- SMOTE **balanced the class distribution**, making Class 1 more visible.
- The SVM classifier was forced to **learn from both classes**.

---

## **2️⃣ Classification Results**
| **Metric**  | **Without SMOTE** | **With SMOTE** | **Change** |
|-------------|------------------|----------------|------------|
| **Accuracy** | **94%** | **65%** | 📉 Drops (expected, as model now recognizes class 1) |
| **Class 0 Recall** | **100%** | **43%** | 📉 Model is misclassifying class 0 more |
| **Class 1 Recall** | **0%** | **86%** | 📈 Massive improvement! Model now detects class 1 |
| **Class 1 Precision** | **0%** | **60%** | 📈 Model now predicts class 1 correctly |

### **Understanding Key Metrics**
- **Macro Avg**: The **unweighted average** of precision, recall, and F1-score across all classes, treating each class equally.
- **Weighted Avg**: The **average weighted by the number of samples in each class**, giving more importance to larger classes.
- **Support**: The **number of actual occurrences of each class** in the dataset (i.e., how many samples belong to each class).

### **🔹 Classification Report WITHOUT SMOTE**
| Metric         | Class 0 | Class 1 | Accuracy | Macro Avg | Weighted Avg |
|---------------|--------|--------|----------|-----------|--------------|
| **Precision**  | 0.94   | 0.00   | -        | 0.47      | 0.88         |
| **Recall**     | 1.00   | 0.00   | 0.94     | 0.50      | 0.94         |
| **F1-score**   | 0.97   | 0.00   | -        | 0.48      | 0.91         |
| **Support**    | 6416   | 416    | 6832     | 6832      | 6832         |

➡️ **Class 1 is completely ignored** by the model because the dataset is **highly imbalanced**.
---
### **🔹 Classification Report WITH SMOTE**
| Metric         | Class 0 | Class 1 | Accuracy | Macro Avg | Weighted Avg |
|---------------|--------|--------|----------|-----------|--------------|
| **Precision**  | 0.75   | 0.60   | -        | 0.68      | 0.68         |
| **Recall**     | 0.43   | 0.86   | 0.65     | 0.65      | 0.65         |
| **F1-score**   | 0.55   | 0.71   | -        | 0.63      | 0.63         |
| **Support**    | 6416   | 6416   | 12832    | 12832     | 12832        |

➡️ **Class 1 recall improved from 0% to 86%!** However, **Class 0 recall dropped significantly**, meaning many Class 0 samples were misclassified as Class 1.

---

## **3️⃣ Conclusions**
- **Without SMOTE**:  
  - The SVM model completely **ignored Class 1**, resulting in misleadingly **high accuracy (94%)**.
  - This is **not a good classifier**, as it fails to detect minority class cases.

- **With SMOTE**:  
  - **Class 1 recall improved dramatically (0% → 86%)**, meaning the model now detects Class 1 well.
  - **Class 0 recall dropped (100% → 43%)**, meaning the model now misclassifies some Class 0 cases.
  - **Overall accuracy decreased (94% → 65%)**, but this is expected since the model is no longer biased.

---

## **4️⃣ Next Steps & Recommendations**
✅ **Try Hybrid Sampling (SMOTE + Undersampling)**  
- **Current issue**: While SMOTE helped Class 1 recall, it caused **Class 0 misclassification**.  
- **Solution**: Use **SMOTE + Undersampling (SMOTEENN)** to balance both classes:
  ```python
  from imblearn.combine import SMOTEENN
  smote_enn = SMOTEENN(random_state=0)
  X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train_pca, y_train)
  ```
- This method **adds synthetic samples to Class 1** while also removing some samples from Class 0 to avoid overfitting to synthetic data.

✅ Evaluate ROC Curves & Precision-Recall Curves
- If false positives (Class 0 misclassified as Class 1) are problematic, we may need cost-sensitive learning or adjust thresholds.

✅ Apply to ResNet50 network
- **Repeat the experiment** using **ResNet50 features** to see if the results are consistent across different feature extraction methods.
