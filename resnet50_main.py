import torch
import os
import sys
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix
from dataloader.dataloader_MRC_classificator import ClassificationDataLoader
from trainers.classificator.resnet50 import train_model, evaluate_model, fine_tune_resnet

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Configuration Parameters
SAVE_RESULTS = True  # Toggle to save results
LEARNING_RATE = 1e-4  # Learning rate for the optimizer
BATCH_SIZE = 16  # Updated batch size
DATA_SPLITS = (0.7, 0.1, 0.2)  # Train, validation, test splits
IMAGES_FOLDER = "/Users/claudiacastrillonalvarez/Desktop/github/EHRatioAnalysis/MRC_data/MRC_images/" 
NUM_EPOCHS = 20  # Define it before using

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

annotations = ClassificationDataLoader.load_annotations(IMAGES_FOLDER)

train_loader, val_loader, test_loader = ClassificationDataLoader.train_val_test_split(
    images_folder=IMAGES_FOLDER,
    annotations=annotations,
    splits=DATA_SPLITS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    transform=None
)

num_classes = len(set(
    annotation 
    for patient_data in annotations.values() 
    for annotation in patient_data['Annotation']
))

MODEL_TYPE = "resnet50"  # Updated model type to ResNet50

print(f"Training {MODEL_TYPE.upper()} model...")
model, criterion, optimizer, scheduler = fine_tune_resnet(num_classes, device, learning_rate=LEARNING_RATE, model_type='resnet50')

trained_model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=NUM_EPOCHS
)

print(f"Evaluating {MODEL_TYPE.upper()} model...")
y_true, y_pred, avg_loss, accuracy = evaluate_model(trained_model, test_loader, device)
print(f"Test Accuracy: {accuracy:.2f}% | Test Loss: {avg_loss:.4f}")

conf_matrix = confusion_matrix(y_true, y_pred)

if SAVE_RESULTS:
    results_root = "./results"
    results_dir = os.path.join(results_root, f"{MODEL_TYPE}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "results.txt"), "w") as f:
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Number of Epochs: {NUM_EPOCHS}\n")
        f.write(f"Optimizer: SGD\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")
        f.write(f"Confusion Matrix:\n{conf_matrix}\n")

    # Generate and save confusion matrix plot
    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[f'Class {i}' for i in range(num_classes)], yticklabels=[f'Class {i}' for i in range(num_classes)])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot training & validation loss
    plt.figure(figsize=(8,6))
    plt.plot(range(1, NUM_EPOCHS+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, NUM_EPOCHS+1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "train_val_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot training & validation accuracy
    plt.figure(figsize=(8,6))
    plt.plot(range(1, NUM_EPOCHS+1), train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(range(1, NUM_EPOCHS+1), val_accuracies, label='Validation Accuracy', marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "train_val_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Results saved in {results_dir}")

print("Process completed.")
