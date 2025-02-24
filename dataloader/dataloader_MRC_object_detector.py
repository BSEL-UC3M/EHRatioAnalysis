import os
import pandas as pd
import cv2

# User-defined parameters
base_dataset_folder = "./toydataset/object_detection/"
# Load all CSV files
csv_files = [f for f in os.listdir(base_dataset_folder) if f.endswith('.csv')]

# Define bounding box size
bbox_size = 56  # Since it will be used as input for U-Net

# Process each CSV file
for csv_file in csv_files:
    patient_folder = os.path.splitext(csv_file)[0]  # Extract patient folder name (remove ".csv")
    csv_path = os.path.join(base_dataset_folder, csv_file)

    # Read the CSV file
    df = pd.read_csv(csv_path, skiprows=1, names=["filename", "x_left", "y_left", "x_right", "y_right"])

    # Process each row (each image annotation)
    for _, row in df.iterrows():
        image_filename = row["filename"]
        image_path = os.path.join(base_dataset_folder, patient_folder, image_filename)
        image_path = image_path.replace("\\", "/")  # Normalize path for compatibility

        # Check if the image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found, skipping...")
            continue

        # Load image to get dimensions
        img = cv2.imread(image_path)
        img_height, img_width, _ = img.shape

        # Normalize bounding box parameters
        x_left_norm = row["x_left"] / img_width
        y_left_norm = row["y_left"] / img_height
        x_right_norm = row["x_right"] / img_width
        y_right_norm = row["y_right"] / img_height
        width_norm = bbox_size / img_width
        height_norm = bbox_size / img_height

        # Create YOLO annotation lines
        yolo_lines = [
            f"0 {x_left_norm:.6f} {y_left_norm:.6f} {width_norm:.6f} {height_norm:.6f}",  # Left ear (class 0)
            f"1 {x_right_norm:.6f} {y_right_norm:.6f} {width_norm:.6f} {height_norm:.6f}"  # Right ear (class 1)
        ]

        # Save annotation file in the same patient folder
        annotations_folder = os.path.join(base_dataset_folder, patient_folder, "yolo_annotations")
        os.makedirs(annotations_folder, exist_ok=True)  # Ensure folder exists

        yolo_filename = os.path.join(annotations_folder, image_filename.replace(".tif", ".txt"))
        with open(yolo_filename, "w") as f:
            f.write("\n".join(yolo_lines))

print("Conversion complete! YOLO annotations are saved.")