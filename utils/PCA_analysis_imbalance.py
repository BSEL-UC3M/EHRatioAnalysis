# ==============================================================================
# File: PCA_analysis_imbalance.py
# Description: Script to analyze the imbalance in the PEI dataset using PCA,
#              comparing SVM performance with and without SMOTE oversampling.
# Author: @cfusterbarcelo
# Created: 26/02/2025
# ==============================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from torchvision import models
from tqdm import tqdm
import os
import sys
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Add the root directory of the project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataloader.dataloader_PEI_classificator import ClassificationDataLoader

# ==========================================================================
# Configuration: Set Paths
# ==========================================================================
RAW_IMAGES_FOLDER = "D:/Data/EHRatioAnalysis/PEI TIFF"  # Change this to your image folder
ANNOTATIONS_FOLDER = "D:/Data/EHRatioAnalysis"  # Change this to where your annotations are stored
PROCESSED_IMAGES_FOLDER = os.path.join(os.path.dirname(RAW_IMAGES_FOLDER), "PEI_processed_data")
RESULTS_FOLDER = "./results/results_classificator/pca"  # Folder to save results
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ==========================================================================
# Load Dataset and Create Train Loader
# ==========================================================================
train_loader = ClassificationDataLoader.train_val_test_split(
    images_folder=PROCESSED_IMAGES_FOLDER,
    annotations=ClassificationDataLoader.load_annotations(ANNOTATIONS_FOLDER),
    splits=(0.7, 0.15, 0.15),
    batch_size=16,
    shuffle=True
)[0]  # Only keep train_loader

# ==========================================================================
# Load Pretrained ResNet50 for Feature Extraction
# ==========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=True).to(device)
resnet.fc = torch.nn.Identity()  # Remove last FC layer to get feature vectors
resnet.eval()

# ==========================================================================
# Function to Extract Features
# ==========================================================================
def extract_features(data_loader, model, device):
    """Extracts features from images using a pre-trained model."""
    features, labels = [], []
    with torch.no_grad():
        for images, lbls in tqdm(data_loader, desc="Extracting Features"):
            images = images.to(device)
            output = model(images)  # Get CNN embeddings
            features.append(output.cpu().numpy())
            labels.append(lbls.numpy())
    return np.vstack(features), np.hstack(labels)

# ==========================================================================
# Extract Features from Train Loader
# ==========================================================================
X_train_features, y_train = extract_features(train_loader, resnet, device)

# Normalize features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_features)

# Perform PCA (reduce to 2D)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# ==========================================================================
# Function to Plot Dataset
# ==========================================================================
def plot_data(X, y, title, save_path=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5, edgecolors="k", cmap="coolwarm")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(title)
    plt.colorbar(label="Class Label")
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved at {save_path}")
    
    plt.show()

# ==========================================================================
# Apply SMOTE and Train SVM with and without Oversampling
# ==========================================================================
print("\n🔹 Training SVM WITHOUT SMOTE...")
clf_no_smote = SVC(kernel='linear', probability=True)
clf_no_smote.fit(X_train_pca, y_train)
y_pred_no_smote = clf_no_smote.predict(X_train_pca)

# Save PCA Visualization Before SMOTE
save_path = os.path.abspath(os.path.join(RESULTS_FOLDER, "pca_visualization.png"))
plot_data(X_train_pca, y_train, title="Image Dataset - PCA Visualization", save_path=save_path)

# Apply SMOTE
print(f"Before SMOTE: Class distribution: {np.bincount(y_train)}")
smote = SMOTE(random_state=0)
X_train_smote, y_train_smote = smote.fit_resample(X_train_pca, y_train)
print(f"After SMOTE: Class distribution: {np.bincount(y_train_smote)}")

# Save PCA Visualization After SMOTE
smote_save_path = os.path.abspath(os.path.join(RESULTS_FOLDER, "pca_visualization_smote.png"))
plot_data(X_train_smote, y_train_smote, title="PCA Visualization After SMOTE", save_path=smote_save_path)

print("\n🔹 Training SVM WITH SMOTE...")
clf_smote = SVC(kernel='linear', probability=True)
clf_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = clf_smote.predict(X_train_smote)

# ==========================================================================
# Save Classification Reports
# ==========================================================================
report_no_smote = classification_report(y_train, y_pred_no_smote)
report_smote = classification_report(y_train_smote, y_pred_smote)

report_path = os.path.abspath(os.path.join(RESULTS_FOLDER, "classification_report.txt"))
with open(report_path, "w") as f:
    f.write("Classification Report WITHOUT SMOTE:\n")
    f.write(report_no_smote + "\n\n")
    f.write("Classification Report WITH SMOTE:\n")
    f.write(report_smote + "\n")

print(f"Classification reports saved at {report_path}")
