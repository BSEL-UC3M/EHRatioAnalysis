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
from models.segmentator.segmentator import Segmentator, UNet, UNet_new, UNetOptimized, UNetOptimizedSE, UNetOptimizedDO


# ==============================================================================

# Configuration Parameters
SAVE_RESULTS = True  # Toggle to save results
SAVE_WEIGHTS = True
NUM_EPOCHS = 40  # Number of training epochs
LEARNING_RATE = 1e-4  # Learning rate for the optimizer
BATCH_SIZE = 16  # Batch size for training
DATA_SPLITS = (0.6, 0.2, 0.2)  # Train, validation, test splits

USE_MRC = False  # Toggle to use the MRC dataset 
USE_PEI = True  # Toggle to use the PEI dataset

# Verificar si estamos en Kaggle o en local
if os.path.exists('/kaggle/input'):
    # Si estamos en Kaggle, usar la ruta de Kaggle
    MRC_IMAGES_FOLDER = '/kaggle/input/cropped-dataset/CROPPED_DATASET/images/MRC_images'
    MRC_LABELS_FOLDER = '/kaggle/input/cropped-dataset/CROPPED_DATASET/labels/MRC_labels'
    PEI_IMAGES_FOLDER = '/kaggle/input/cropped-dataset/CROPPED_DATASET/images/PEI_images_preprocessed'
    PEI_LABELS_FOLDER = '/kaggle/input/cropped-dataset/CROPPED_DATASET/labels/PEI_labels'
else:
    # Si estamos en local, usar la ruta local
    #IMAGES_FOLDER = 'toydataset/segmentation/MRC/images/'
    #LABELS_FOLDER = "toydataset/segmentation/MRC/labels"
    MRC_IMAGES_FOLDER = "D:\Desktop\CROPPED_DATASET\images\MRC_images"
    MRC_LABELS_FOLDER = "D:\Desktop\CROPPED_DATASET\labels\MRC_labels"
    PEI_IMAGES_FOLDER = "D:\Desktop\CROPPED_DATASET\images\PEI_images"
    PEI_LABELS_FOLDER = "D:\Desktop\CROPPED_DATASET\labels\PEI_labels"

if USE_MRC:
    IMAGES_FOLDER = MRC_IMAGES_FOLDER
    LABELS_FOLDER = MRC_LABELS_FOLDER
elif USE_PEI:
    IMAGES_FOLDER = PEI_IMAGES_FOLDER
    LABELS_FOLDER = PEI_LABELS_FOLDER
# ================================================================================

# Initialize the segmentation model
segmentator = Segmentator()

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
segmentator = segmentator.to(device)

LOSS_FUNCTION = "bce_dice"  # Opciones: "bce_dice", "focal"

if LOSS_FUNCTION == "bce_dice":
    criterion = losses.BCE_and_Dice_loss(
        bce_kwargs={},  
        dice_class=losses.SimpleDiceLoss,  
        weight_ce=1,  
        weight_dice=2  
    )
elif LOSS_FUNCTION == "focal":
    criterion = losses.FocalLoss(alpha=0.25, gamma=2, reduction='mean')  
elif LOSS_FUNCTION == "FLProbs": 
    criterion = losses.FocalLossForProbabilities(gamma=3.5, alpha=0.25)
elif LOSS_FUNCTION == "custom_combined":
    criterion = lambda pred, target: (
        0.2 * losses.FocalLossForProbabilities(gamma=3.0, alpha=0.75)(pred, target) +
        0.8 * losses.BCE_and_Dice_loss(bce_kwargs={}, dice_class=losses.SimpleDiceLoss, weight_ce=1, weight_dice=1)(pred, target)
    )

else:
    raise ValueError("Invalid loss function selected. Choose 'bce_dice' or 'focal'.")

# Define the optimizer (Adam optimizer)
optimizer = optim.Adam(segmentator.parameters(), lr=LEARNING_RATE)

# ==============================================================================

# Create results directory if needed
if SAVE_RESULTS:
    results_folder = "./results/results_segmentator/PEI/20250319 PEI no prep Weights BCE_dice"
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

# ============= PLOT IMAHES AND LABELS THAT GO INTO THE MODEL FOR TRAINING
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

# =================== NEW PLOT 

import numpy as np
import matplotlib.pyplot as plt

# Asegurarnos de que 'label' sea 2D
if label.ndim == 3:
    label_red = label[:, :, 0]  # Eliminar dimensión extra si existe

# Crear una imagen RGBA vacía con el mismo tamaño que la imagen
mask_rgba = np.zeros((label_red.shape[0], label_red.shape[1], 4))  # (H, W, 4)

# Asignar color rojo con 50% de transparencia solo a los píxeles blancos de la máscara
mask_rgba[label_red > 0] = [1, 0, 0, 0.4]  # (Rojo, Verde, Azul, Transparencia)

# Crear el plot
fig, ax = plt.subplots(figsize=(6, 6))

# Mostrar la imagen original en escala de grises
ax.imshow(image, cmap="gray")

# Superponer la máscara en rojo semitransparente
ax.imshow(mask_rgba)

# Configurar el título y quitar ejes
ax.set_title("Image with Superposed Label (Red)")
ax.axis("off")

# Mostrar el resultado
plt.show()

 #----------- CONTOUR PLOT

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Asegurar que la máscara es binaria (0 y 255)
label_bin = (label > 0).astype(np.uint8) * 255  

# Encontrar contornos con cv2 (RETR_EXTERNAL para solo el borde externo)
contours, _ = cv2.findContours(label_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Crear el plot
fig, ax = plt.subplots(figsize=(6, 6))

# Mostrar la imagen original
ax.imshow(image, cmap="gray")

# Dibujar los contornos sobre la imagen en rojo, pixel-perfect
for contour in contours:
    contour = contour.squeeze()  # Asegurar que está en el formato correcto
    ax.plot(contour[:, 0], contour[:, 1], color='red', linewidth=1)  # Contorno fino

# Configurar el título y quitar ejes
ax.set_title("Image with contour of the Ground Truth label")
ax.axis("off")

# Mostrar el resultado
plt.show()


# --------------
# # LET'S VISUALIZE SOME OF OUR INPUT DATA STRAIGHT FROM THE FOLDER 
# import os
# from PIL import Image
# import matplotlib.pyplot as plt

# # Selecciona un archivo de imagen y etiqueta (usa el mismo nombre de base)
# image_filename = "PAC5_right_main_right.tif"  # Reemplaza con un nombre de archivo válido
# label_filename = "PAC5_right_main_right.tif"  # Reemplaza con el nombre correspondiente

# # Construir rutas completas
# image_path = os.path.join(IMAGES_FOLDER, image_filename)
# label_path = os.path.join(LABELS_FOLDER, label_filename)

# # Cargar la imagen y la etiqueta usando PIL
# image = Image.open(image_path)
# label = Image.open(label_path)

# # Mostrar imagen y etiqueta en un subplot de dos secciones
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# # Mostrar la imagen
# ax[0].imshow(image, cmap="gray")
# ax[0].set_title("Imagen")
# ax[0].axis("off")

# # Mostrar la etiqueta
# ax[1].imshow(label, cmap="gray")  # Cambia cmap según el formato de la etiqueta
# ax[1].set_title("Etiqueta")
# ax[1].axis("off")

# # Mostrar el plot
# plt.tight_layout()
# plt.show()

# # Convert to numpy array
# image = np.array(image)

# print("Image min:", image.min())
# print("Image max:", image.max())


# ==============================================================================

# Check shape of data: Obtener un batch del train_loader
for images, labels in train_loader:
    print(f"Dimensiones de las imágenes: {images.shape}")
    print(f"Dimensiones de las etiquetas: {labels.shape}")
    break  

# =================================

# Train the model
print("Starting training...")
trained_model = train_model(segmentator, train_loader, criterion, optimizer, device, results_dir, NUM_EPOCHS, val_dataloader=val_loader)

# Evaluate the model
print("Evaluating model...")
avg_loss, mean_dice, mean_iou = evaluate_model(trained_model, test_loader, device, criterion, results_dir)

# Save the trained model if results are being saved
if SAVE_RESULTS:
    model_save_path = os.path.join(results_dir, 'unet_brain_segmentation.pth')
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if SAVE_WEIGHTS:
    #best_epoch = np.argmin(val_losses)  # Find the epoch with the lowest validation loss
    weights_save_path = os.path.join(results_dir, "mrc_segmentator_best_weights.pt")
    torch.save(trained_model.state_dict(), weights_save_path)
    print(f"Model weights saved at {weights_save_path}")

# ==============================================================================
trained_model.visualize_segmentation(test_loader, device=device, results_dir=results_dir, save_results=SAVE_RESULTS)