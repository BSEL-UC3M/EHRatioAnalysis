import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

def visualize_sample_with_overlay_and_contour(
    dataloader, 
    index=0, 
    show_overlay=True, 
    show_contour=True
):
    """
    Visualize a sample image from a DataLoader with its corresponding ground truth mask.
    
    Parameters:
        dataloader (DataLoader): PyTorch DataLoader providing (image, label, name) tuples.
        index (int): Index of the image in the batch to visualize.
        show_overlay (bool): Whether to show a semi-transparent red overlay of the mask.
        show_contour (bool): Whether to plot contours of the mask.
    """
    
    data_iter = iter(dataloader)
    images, labels, names = next(data_iter)

    image = images[index].permute(1, 2, 0).numpy()
    label = labels[index].permute(1, 2, 0).numpy()


    if label.ndim == 3:
        label_red = label[:, :, 0]
    else:
        label_red = label

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(image, cmap="gray")

    if show_overlay:
        mask_rgba = np.zeros((label_red.shape[0], label_red.shape[1], 4))
        mask_rgba[label_red > 0] = [1, 0, 0, 0.4]
        ax.imshow(mask_rgba)

    if show_contour:
        label_bin = (label_red > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(label_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            contour = contour.squeeze()
            if contour.ndim == 2:
                ax.plot(contour[:, 0], contour[:, 1], color='red', linewidth=1)

    ax.set_title(f"Sample: {names[index]}" if names else "Sample Image")
    ax.axis("off")
    plt.show()

    # Debigging information
    #print(f"Image shape: {images[index].shape}")
    #print(f"Label shape: {labels[index].shape}")