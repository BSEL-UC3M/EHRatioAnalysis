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
from dataloader.dataloader_MRC import DataLoaderByPatient, DataLoaderByPatientSpecific
from trainers.segmentator.pretrained_trainers import train_model, evaluate_model
from models.segmentator.segmentator import Segmentator, UNet, UNet_new, UNetOptimized, UNetOptimizedSE, UNetOptimizedDO


# ==============================================================================

# Configuration Parameters
SAVE_RESULTS = True  # Toggle to save results
SAVE_WEIGHTS = True
NUM_EPOCHS = 15  # Number of training epochs

LEARNING_RATE = 1e-4  # Learning rate for the optimizer
BATCH_SIZE = 3  # Batch size for training
#DATA_SPLITS = (0.6, 0.2, 0.2)  # Train, validation, test splits

USE_MRC = True  # Toggle to use the MRC dataset 
USE_PEI = False  # Toggle to use the PEI dataset

# Verificar si estamos en Kaggle o en local
if os.path.exists('/kaggle/input'):
    # Si estamos en Kaggle, usar la ruta de Kaggle
    MRC_IMAGES_FOLDER = '/kaggle/input/cropped-dataset/NORMALIZED_CROPPED_DATASET/images/normalized_images_MRC'
    MRC_LABELS_FOLDER = '/kaggle/input/cropped-dataset/NORMALIZED_CROPPED_DATASET/labels/MRC_labels'
    PEI_IMAGES_FOLDER = '/kaggle/input/cropped-dataset/NORMALIZED_CROPPED_DATASET/images/PEI_images_preprocessed'
    PEI_LABELS_FOLDER = '/kaggle/input/cropped-dataset/NORMALIZED_CROPPED_DATASET/labels/PEI_labels'
else:
    # Si estamos en local, usar la ruta local

    # MRC_IMAGES_FOLDER = r"D:\Desktop\NORMALIZED_CROPPED_DATASET\images\MRC_normalized_images"
    # MRC_LABELS_FOLDER = r"D:\Desktop\NORMALIZED_CROPPED_DATASET\labels\MRC_labels"
    # PEI_IMAGES_FOLDER = r"D:\Desktop\NORMALIZED_CROPPED_DATASET\images\PEI_images_Z"
    # PEI_LABELS_FOLDER = r"D:\Desktop\NORMALIZED_CROPPED_DATASET\labels\PEI_labels_Z"
    
    MRC_IMAGES_FOLDER = 'C:\\Users\\TFM1\\Documents\\Data\\EHydropsAnalysis\\NORMALIZED_CROPPED_DATASET\\images\\flipped_images_MRC'
    MRC_LABELS_FOLDER = 'C:\\Users\\TFM1\\Documents\\Data\\EHydropsAnalysis\\NORMALIZED_CROPPED_DATASET\\labels\\flipped_labels_MRC'
    PEI_IMAGES_FOLDER = 'C:\\Users\\TFM1\\Documents\\Data\\EHydropsAnalysis\\NORMALIZED_CROPPED_DATASET\\images\\flipped_clean_images_PEI' #PEI_images_Z #normalized_images_PEI
    PEI_LABELS_FOLDER = 'C:\\Users\\TFM1\\Documents\\Data\\EHydropsAnalysis\\NORMALIZED_CROPPED_DATASET\\labels\\flipped_clean_labels_PEI'

    


if USE_MRC:
    IMAGES_FOLDER = MRC_IMAGES_FOLDER
    LABELS_FOLDER = MRC_LABELS_FOLDER
elif USE_PEI:
    IMAGES_FOLDER = PEI_IMAGES_FOLDER
    LABELS_FOLDER = PEI_LABELS_FOLDER
# ================================================================================

# Initialize the segmentation model
segmentator = UNetOptimizedDO()

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Imprime el dispositivo que se está utilizando
print(f"El dispositivo en uso es: {device}")
print(torch.cuda.is_available()) # Imprime si CUDA está disponible
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
    results_folder = "./results/results_segmentator/MRC/20250401 MRC TRAINING nuevo split"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(results_folder, timestamp)
    os.makedirs(results_dir, exist_ok=True)
else:
    results_dir = None
# ================================================================================
# Sepecific patient Dataloader for MRC dataset

train_patients = [
    "PAC59", "PAC11", "PAC43", "PAC2", "PAC73", "PAC67", "PAC16", "PAC27", "PAC41", "PAC33", "PAC46", "PAC48", 
    "PAC68", "PAC20", "PAC60", "PAC56", "PAC63", "PAC71", "PAC52", "PAC57", "PAC50", "PAC86", "PAC80", "PAC64", 
    "PAC34", "PAC9", "PAC38", "PAC72", "PAC31", "PAC44", "PAC62", "PAC42", "PAC40", "PAC89", "PAC47", "PAC78", 
    "PAC13", "PAC37", "PAC24", "PAC19", "PAC66", "PAC69", "PAC53", "PAC8", "PAC35", "PAC74", "PAC3", "PAC17", 
    "PAC39", "PAC51", "PAC23", "PAC79", "PAC25", "PAC6", "PAC7", "PAC61", "PAC49", "PAC83", "PAC10", "PAC84", 
    "PAC22", "PAC75", "PAC153", "PAC110", "PAC125", "PAC178", "PAC132", "PAC107", "PAC188", "PAC106", "PAC111", 
    "PAC162", "PAC119", "PAC116", "PAC109", "PAC120", "PAC161", "PAC148", "PAC179", "PAC115", "PAC165", "PAC172", 
    "PAC113", "PAC169", "PAC151", "PAC112", "PAC117", "PAC131", "PAC149", "PAC177", "PAC157", "PAC123", "PAC121", 
    "PAC141", "PAC130", "PAC159", "PAC136", "PAC164", "PAC168", "PAC190", "PAC142", "PAC103", "PAC147", "PAC102", 
    "PAC156", "PAC176", "PAC122", "PAC105", "PAC146", "PAC158", "PAC173", "PAC187", "PAC154", "PAC186", "PAC139", 
    "PAC152", "PAC174", "PAC191", "PAC185", "PAC108", "PAC180", "PAC134", "PAC135", "PAC129", "PAC181", "PAC155", 
    "PAC184", "PAC143", "PAC189", "PAC128", "PAC133", "PAC183", "PAC144", "PAC126"
]

val_patients = [
    "PAC45", "PAC21", "PAC1", "PAC87", "PAC58", "PAC85", "PAC54", "PAC90", "PAC26", "PAC114", "PAC118", "PAC163", 
    "PAC150", "PAC182", "PAC145", "PAC137", "PAC138", "PAC104", "PAC171", "PAC127", "PAC140", "PAC167", "PAC166", 
    "PAC124", "PAC175", "PAC160", "PAC170"
]


test_patients = [
    "PAC77", "PAC65", "PAC30", "PAC28", "PAC81", "PAC88", "PAC5", "PAC55", "PAC76", "PAC12", "PAC70", "PAC14", 
    "PAC18", "PAC29", "PAC32", "PAC36", "PAC4", "PAC15", "PAC82"
]


train_loader, val_loader, test_loader = DataLoaderByPatientSpecific.train_val_test_split_bypatient(
    images_folder=IMAGES_FOLDER, 
    labels_folder=LABELS_FOLDER, 
    train_patients=train_patients, 
    val_patients=val_patients, 
    test_patients=test_patients,
    batch_size=BATCH_SIZE, 
    shuffle=True, transform=None
)

# ================================================================================

# # Initialize the data loader with your custom DataLoader class
# data_loader = DataLoaderByPatient()
# train_loader, val_loader, test_loader= data_loader.train_val_test_split_bypatient(
#     images_folder=IMAGES_FOLDER,
#     labels_folder=LABELS_FOLDER,
#     splits=DATA_SPLITS,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     transform=None
# )
print(f"Train loader: {len(train_loader)} batches")
print(f"Validation loader: {len(val_loader)} batches")  
print(f"Test loader: {len(test_loader)} batches")
print(f"Train loader: {len(train_loader.dataset)} images")
print(f"Validation loader: {len(val_loader.dataset)} images")   
print(f"Test loader: {len(test_loader.dataset)} images")

# ==================

# LET'S VISUALIZE SOME OF OUR INPUT DATA FROM THE DATALOADER
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Obtener un lote del train_loader
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Seleccionar la primera imagen y su correspondiente label
image = images[1]  # Primera imagen
label = labels[1]  # Primer label

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

# ===============
# # Evaluate inference MRC
# print("Evaluating model with inference MRC")
# model_path = 'C:\\Users\\TFM1\\Desktop\\mrc_segmentator_best_weights.pt'  # Ajusta el camino a tu archivo .pt
# segmentator.load_state_dict(torch.load(model_path))  # Cargar los pesos en el modelo
# segmentator.eval() 
# trained_model = segmentator  # Asignar el modelo entrenado a trained_model

# ===============
# # Evaluate inference MRC
# print("Evaluating model with inference PEI")
# model_path = 'C:\\Users\\TFM1\\Desktop\\PEI_segmentator_best_weights.pt'  # Ajusta el camino a tu archivo .pt
# segmentator.load_state_dict(torch.load(model_path))  # Cargar los pesos en el modelo
# segmentator.eval() 
# trained_model = segmentator  # Asignar el modelo entrenado a trained_model

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
trained_model.new_visualize_segmentation(test_loader, device=device, results_dir=results_dir, save_results=SAVE_RESULTS)