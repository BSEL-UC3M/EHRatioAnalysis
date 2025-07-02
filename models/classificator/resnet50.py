import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

def fine_tune_resnet(num_classes, device, learning_rate=0.0001, model_type='resnet50', weight_decay=1e-4):
    """
    Fine-tune ResNet model (ResNet18 or ResNet50) with dropout and L2 regularization.
    """
    if model_type == 'resnet50':
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    num_ftrs = resnet.fc.in_features
    
    # Modify fully connected layer: add dropout (0.5 initially)
    resnet.fc_layers = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    resnet = resnet.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)  # L2 Regularization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)  # Removed verbose=True

    
    return resnet, criterion, optimizer, scheduler