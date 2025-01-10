from torch.utils.data import Dataset, DataLoader
import os
import torch
import cv2

class YoloObjectDetectorDataset(Dataset):
    """
    PyTorch Dataset for loading MRC images and YOLO-compatible bounding box annotations.
    """
    def __init__(self, images_folder, labels_folder, transform=None):
        """
        Parameters:
        - images_folder: Path to the folder containing images.
        - labels_folder: Path to the folder containing YOLO annotations (.txt files).
        - transform: Optional transformations to apply to the images.
        """
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.image_files = [f for f in os.listdir(images_folder) if f.endswith('.tif')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        image_path = os.path.join(self.images_folder, file_name)
        label_path = os.path.join(self.labels_folder, file_name.replace('.tif', '.txt'))

        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image file {image_path} not found.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = image.astype('float32') / 255.0

        # Load the annotations
        yolo_annotations = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                yolo_annotations.append([class_id, x_center, y_center, width, height])

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # Change to (C, H, W)
        yolo_annotations = torch.tensor(yolo_annotations, dtype=torch.float32)

        return image, yolo_annotations


# Example Usage
dataset = YoloObjectDetectorDataset(
    images_folder='path/to/images',
    labels_folder='path/to/yolo_annotations',
    transform=None
)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

for images, annotations in dataloader:
    print(images.shape)  # (batch_size, 3, H, W)
    print(annotations)   # List of bounding boxes for each image
