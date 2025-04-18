import os
import torch
import wandb
import numpy as np
from losses import losses
from models.segmentator.segmentator import UNetOptimizedDO
from trainers.segmentator.pretrained_trainers import train_model, complete_evaluate_model
from dataloader.dataloader_MRC import DataLoaderByPatientSpecific
import argparse
import wandb


#wandb.init(project="sweep_tfm", entity="laurarodrigo7")

# Configuración de paths según entorno
if os.path.exists('/kaggle/input'):
    PEI_IMAGES_FOLDER = '/kaggle/input/cropped-dataset/NORMALIZED_CROPPED_DATASET/images/flipped_images_PEI'
    PEI_LABELS_FOLDER = '/kaggle/input/cropped-dataset/NORMALIZED_CROPPED_DATASET/labels/mod_flipped_labels_PEI'
else:
    PEI_IMAGES_FOLDER = 'C:\\Users\\TFM1\\Documents\\Data\\EHydropsAnalysis\\NORMALIZED_CROPPED_DATASET\\images\\flipped_images_PEI' 
    PEI_LABELS_FOLDER = 'C:\\Users\\TFM1\\Documents\\Data\\EHydropsAnalysis\\NORMALIZED_CROPPED_DATASET\\labels\\mod_flipped_labels_PEI'

# Pacientes definidos manualmente
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

# Inicializa una run (esto va justo al principio de tu main)
def sweep_train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        model = UNetOptimizedDO().to(device)

        # Loss function
        if config.loss_function == "bce_dice":
            criterion = losses.BCE_and_Dice_loss(
                bce_kwargs={},
                dice_class=losses.SimpleDiceLoss,
                weight_ce=1,
                weight_dice=2
            )
        elif config.loss_function == "focal":
            criterion = losses.FocalLoss(alpha=0.25, gamma=2)
        elif config.loss_function == "FLProbs":
            criterion = losses.FocalLossForProbabilities(gamma=3.5, alpha=0.25)
        elif config.loss_function == "custom_combined":
            criterion = lambda pred, target: (
                0.2 * losses.FocalLossForProbabilities(gamma=3.0, alpha=0.75)(pred, target) +
                0.8 * losses.BCE_and_Dice_loss(bce_kwargs={}, dice_class=losses.SimpleDiceLoss, weight_ce=1, weight_dice=1)(pred, target)
            )
        else:
            raise ValueError("Invalid loss function")

        # Optimizer
        if config.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        elif config.optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
        elif config.optimizer == "rmsprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
        elif config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        else:
            raise ValueError("Unknown optimizer")

        # Data
        train_loader, val_loader, _ = DataLoaderByPatientSpecific.train_val_test_split_bypatient(
            images_folder=PEI_IMAGES_FOLDER,
            labels_folder=PEI_LABELS_FOLDER,
            train_patients=train_patients,
            val_patients=val_patients,
            test_patients=test_patients,
            batch_size=config.batch_size,
            shuffle=True,
            transform=None
        )

        trained_model = train_model(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            results_dir=None,
            num_epochs=config.num_epochs,
            val_dataloader=val_loader
        )
        # Evalúa y guarda val loss
        model.eval()
        val_loss, _, _ = complete_evaluate_model(trained_model, val_loader, device, criterion, None)
        wandb.log({"val_loss": val_loss})


# Si el script se ejecuta directamente, llamamos a main() para iniciar el entrenamiento.
if __name__ == "__main__":
    # Inicia un sweep con la configuración de wandb
    sweep_train()





