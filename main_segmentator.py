# ==============================================================================
# File: main_segmentator.py
# Description: Main script for training and evaluating the segmentation model.
# Author: @cfusterbarcelo
# Creation Date: 03/09/2024
# ==============================================================================

import os
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
from losses import losses
from dataloader.dataloader_MRC import DataLoaderByPatient
from trainers.segmentator.pretrained_trainers import train_model, evaluate_model
from models.segmentator import Segmentator, UNet, UNet_new, UNetOptimized

# ==============================================================================

# Configuration Parameters
SAVE_RESULTS = False  # Toggle to save results
NUM_EPOCHS = 1  # Number of training epochs
LEARNING_RATE = 1e-4  # Learning rate for the optimizer
BATCH_SIZE = 8  # Batch size for training
DATA_SPLITS = (0.5, 0.25, 0.25)  # Train, validation, test splits

# Dataset Paths
# Toy dataset for testing
#IMAGES_FOLDER = "toydataset/segmentation/MRC/images"
#LABELS_FOLDER = "toydataset/segmentation/MRC/labels"

# Verificar si estamos en Kaggle o en local
if os.path.exists('/kaggle/input'):
    # Si estamos en Kaggle, usar la ruta de Kaggle
    IMAGES_FOLDER = '/kaggle/input/cropped-dataset/CROPPED_DATASET/images/MRC_images'
    LABELS_FOLDER = '/kaggle/input/cropped-dataset/CROPPED_DATASET/labels/MRC_labels'
else:
    # Si estamos en local, usar la ruta local
    #IMAGES_FOLDER = 'toydataset/segmentation/MRC/images/'
    #LABELS_FOLDER = "toydataset/segmentation/MRC/labels"
    IMAGES_FOLDER = "D:\Desktop\CROPPED_DATASET\images\MRC_images"
    LABELS_FOLDER = "D:\Desktop\CROPPED_DATASET\labels\MRC_labels"

# Full dataset for training (uncomment when needed)
# IMAGES_FOLDER = "D:/Data/VolumetricHydrops/images/MRC"
# LABELS_FOLDER = "D:/Data/VolumetricHydrops/labels/MRC"

# ================================================================================

# Initialize the segmentation model
segmentator = UNetOptimized()

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
segmentator = segmentator.to(device)

# Define the loss function (combined BCE and Dice loss)
criterion = losses.BCE_and_Dice_loss(
    bce_kwargs={},  # Default settings for BCELoss
    dice_class=losses.SimpleDiceLoss,  # Simple Dice loss class
    weight_ce=1,  # Weight for BCE loss
    weight_dice=1  # Weight for Dice loss
)

# Define the optimizer (Adam optimizer)
optimizer = optim.Adam(segmentator.parameters(), lr=LEARNING_RATE)

# ==============================================================================

# Create results directory if needed
if SAVE_RESULTS:
    results_folder = "./results/results_segmentator/MRC"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(results_folder, timestamp)
    os.makedirs(results_dir, exist_ok=True)
else:
    results_dir = None

# ================================================================================

# Initialize the data loader with your custom DataLoader class
data_loader = DataLoaderByPatient()
train_loader, val_loader, test_loader= data_loader.train_val_test_split_bypatient(
    images_folder=IMAGES_FOLDER,
    labels_folder=LABELS_FOLDER,
    splits=DATA_SPLITS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    transform=None
)


# ==================

# LET'S VISUALIZE SOME OF OUR INPUT DATA FROM THE DATALOADER
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Obtener un lote del train_loader
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Seleccionar la primera imagen y su correspondiente label
image = images[5]  # Primera imagen
label = labels[5]  # Primer label

# Transponer la imagen de [3, 96, 96] a [96, 96, 3] para visualización
image = image.permute(1, 2, 0)
label = label.permute(1,2,0)

# Normalizar los valores de la imagen para mostrarlos (si es necesario)
image = image.numpy()  # Convertir a numpy
label = label.numpy()

# Crear un plot con dos secciones
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Mostrar la imagen en la primera sección
ax[0].imshow(image, cmap="gray")
ax[0].set_title("Imagen")
ax[0].axis("off")

# Mostrar el label correspondiente en la segunda sección
ax[1].imshow(label, cmap="gray")
ax[1].set_title("Label")
ax[1].axis("off")

# Mostrar el plot
plt.tight_layout()
plt.show()

# LET'S VISUALIZE SOME OF OUR INPUT DATA STRAIGHT FROM THE FOLDER 
import os
from PIL import Image
import matplotlib.pyplot as plt

# Selecciona un archivo de imagen y etiqueta (usa el mismo nombre de base)
image_filename = "PAC5_right_main_right.tif"  # Reemplaza con un nombre de archivo válido
label_filename = "PAC5_right_main_right.tif"  # Reemplaza con el nombre correspondiente

# Construir rutas completas
image_path = os.path.join(IMAGES_FOLDER, image_filename)
label_path = os.path.join(LABELS_FOLDER, label_filename)

# Cargar la imagen y la etiqueta usando PIL
image = Image.open(image_path)
label = Image.open(label_path)

# Mostrar imagen y etiqueta en un subplot de dos secciones
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Mostrar la imagen
ax[0].imshow(image, cmap="gray")
ax[0].set_title("Imagen")
ax[0].axis("off")

# Mostrar la etiqueta
ax[1].imshow(label, cmap="gray")  # Cambia cmap según el formato de la etiqueta
ax[1].set_title("Etiqueta")
ax[1].axis("off")

# Mostrar el plot
plt.tight_layout()
plt.show()

# Convert to numpy array
image = np.array(image)

print("Image min:", image.min())
print("Image max:", image.max())


# ==============================================================================

# Check shape of data: Obtener un batch del train_loader
for images, labels in train_loader:
    print(f"Dimensiones de las imágenes: {images.shape}")
    print(f"Dimensiones de las etiquetas: {labels.shape}")
    break  

# =================================

# Train the model
print("Starting training...")
trained_model = train_model(segmentator, train_loader, criterion, optimizer, device, results_dir, NUM_EPOCHS)

# Evaluate the model
print("Evaluating model...")
avg_loss, mean_dice, mean_iou = evaluate_model(trained_model, test_loader, device, criterion, results_dir)

# Save the trained model if results are being saved
if SAVE_RESULTS:
    model_save_path = os.path.join(results_dir, 'unet_brain_segmentation.pth')
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# ==============================================================================
