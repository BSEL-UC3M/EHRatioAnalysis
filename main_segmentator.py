# ==============================================================================
# File: main_segmentator.py
# Description: Main script for training and evaluating the segmentation model.
# Author: @laurarodmmu
# Creation Date: 03/11/2024
# ==============================================================================

import os
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
from losses import losses
from dataloader.dataloader_MRC import DataLoaderByPatientSpecific
from trainers.segmentator.pretrained_trainers import train_model, complete_evaluate_model
from models.segmentator.segmentator import Segmentator, UNetOptimizedDO
from utils.visualize_mask import visualize_sample_with_overlay_and_contour


# ==============================================================================

# Configuration Parameters

USE_MRC = True  
USE_PEI = False  

SAVE_RESULTS = True  
SAVE_WEIGHTS = True
NUM_EPOCHS = 25  

LEARNING_RATE = 1e-4  
BATCH_SIZE = 16

MODE = "train"  # Set the mode: "train", "inference_mrc", or "inference_pei"

LOSS_FUNCTION = "bce_dice" 

# =================================================================================

# Verify wether using Kaggle or local environment

if os.path.exists('/kaggle/input'):
    MRC_IMAGES_FOLDER = '/kaggle/input/cropped-dataset/NORMALIZED_CROPPED_DATASET/images/flipped_images_MRC'
    MRC_LABELS_FOLDER = '/kaggle/input/cropped-dataset/NORMALIZED_CROPPED_DATASET/labels/new_flipped_labels_MRC'
    PEI_IMAGES_FOLDER = '/kaggle/input/cropped-dataset/NORMALIZED_CROPPED_DATASET/images/flipped_images_PEI'
    PEI_LABELS_FOLDER = '/kaggle/input/cropped-dataset/NORMALIZED_CROPPED_DATASET/labels/mod_flipped_labels_PEI'
else:
    MRC_IMAGES_FOLDER = 'C:\\Users\\TFM1\\Documents\\Data\\EHydropsAnalysis\\NORMALIZED_CROPPED_DATASET\\images\\flipped_images_MRC'
    MRC_LABELS_FOLDER = 'C:\\Users\\TFM1\\Documents\\Data\\EHydropsAnalysis\\NORMALIZED_CROPPED_DATASET\\labels\\new_flipped_labels_MRC'
    PEI_IMAGES_FOLDER = 'C:\\Users\\TFM1\\Documents\\Data\\EHydropsAnalysis\\NORMALIZED_CROPPED_DATASET\\images\\flipped_images_PEI' 
    PEI_LABELS_FOLDER = 'C:\\Users\\TFM1\\Documents\\Data\\EHydropsAnalysis\\NORMALIZED_CROPPED_DATASET\\labels\\mod_flipped_labels_PEI'
  
if USE_MRC:
    IMAGES_FOLDER = MRC_IMAGES_FOLDER
    LABELS_FOLDER = MRC_LABELS_FOLDER
elif USE_PEI:
    IMAGES_FOLDER = PEI_IMAGES_FOLDER
    LABELS_FOLDER = PEI_LABELS_FOLDER

# ================================================================================

# Initialize the segmentation model

segmentator = UNetOptimizedDO()
#segmentator = Segmentator()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
segmentator = segmentator.to(device)

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

optimizer = optim.Adam(segmentator.parameters(), lr=LEARNING_RATE)

# ==============================================================================

if SAVE_RESULTS:
    results_folder = "./results/results_segmentator/MRC/20250426 one channel"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(results_folder, timestamp)
    os.makedirs(results_dir, exist_ok=True)
else:
    results_dir = None
# ================================================================================

# Sepecific patient Dataloader for each dataset

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

# ==================================================================================

# Visualize a sample from the training set

visualize_sample_with_overlay_and_contour(train_loader, index=1)

# ==================================================================================

# Train the model or do inference

if MODE == "train":
    print("Starting training...")
    trained_model = train_model(segmentator, train_loader, criterion, optimizer, device, results_dir, NUM_EPOCHS, val_dataloader=val_loader)
elif MODE == "inference": 
    if USE_MRC:
        print("Evaluating model with inference MRC")
        model_path = 'C:\\Users\\TFM1\\Desktop\\mrc_segmentator_best_weights.pt' 
        segmentator.load_state_dict(torch.load(model_path))  
        segmentator.eval() 
        trained_model = segmentator  
    elif USE_PEI:
        # Evaluate inference MRC
        print("Evaluating model with inference PEI")
        model_path = 'C:\\Users\\TFM1\\Desktop\\pei_segmentator_best_weights.pt'  
        segmentator.load_state_dict(torch.load(model_path))  
        segmentator.eval() 
        trained_model = segmentator  


# ==================================================================================   
 
# Evaluate the model

if USE_MRC:
    threshold = 0.98
elif USE_PEI:
    threshold = 0.8
print("Evaluating model...")
avg_loss, mean_dice, mean_iou = complete_evaluate_model(trained_model, test_loader, device, criterion, results_dir, threshold=threshold) 

if SAVE_RESULTS:
    model_save_path = os.path.join(results_dir, 'unet_segmentation.pth')
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if SAVE_WEIGHTS:
    weights_save_path = os.path.join(results_dir, "segmentator_best_weights.pt")
    torch.save(trained_model.state_dict(), weights_save_path)
    print(f"Model weights saved at {weights_save_path}")

# ==================================================================================
