from sklearn.metrics import jaccard_score

# Define a function to calculate the Dice score
def dice_score(y_pred, y_true, threshold=0.5):
    smooth = 1e-6
    y_pred = (y_pred > threshold).float()
    intersection = (y_pred * y_true).sum()
    dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    return dice.item()

# Define a function to calculate the IoU score
def iou_score(y_pred, y_true, threshold=0.5):
    y_pred = (y_pred > threshold).float()
    y_pred_np = y_pred.cpu().numpy().flatten()
    y_true_np = y_true.cpu().numpy().flatten()
    iou = jaccard_score(y_true_np, y_pred_np)
    return iou
