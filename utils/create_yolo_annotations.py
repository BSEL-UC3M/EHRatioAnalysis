import os
import pandas as pd
import cv2

#TODO change to get the size of the image for the annotation normalization
def generate_yolo_annotations(images_folder, annotations_file, output_folder, box_size=96):
    """
    Generate YOLO-compatible .txt files for each image based on annotations.
    
    Parameters:
    - images_folder: Path to the folder containing images.
    - annotations_file: Path to the CSV file with annotations.
    - output_folder: Folder to save the YOLO annotation .txt files.
    - box_size: Size of the bounding boxes (default is 96x96).
    """
    annotations = pd.read_csv(annotations_file)
    annotations = annotations.loc[:, ~annotations.columns.str.contains('^Unnamed')]

    os.makedirs(output_folder, exist_ok=True)

    half_box = box_size // 2

    for _, row in annotations.iterrows():
        file_name = row['File Name']
        left_ear_x, left_ear_y = row[' Left Ear X'], row[' Left Ear Y']
        right_ear_x, right_ear_y = row[' Right Ear X'], row[' Right Ear Y']

        # Convert to YOLO format
        yolo_boxes = []
        for class_id, (x, y) in enumerate([(left_ear_x, left_ear_y), (right_ear_x, right_ear_y)]):
            x_min = max(0, x - half_box)
            y_min = max(0, y - half_box)
            x_max = x + half_box
            y_max = y + half_box

            # Normalize coordinates
            x_center = (x_min + x_max) / 2 / box_size
            y_center = (y_min + y_max) / 2 / box_size
            width = (x_max - x_min) / box_size
            height = (y_max - y_min) / box_size

            yolo_boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Write YOLO annotation file
        txt_file = os.path.join(output_folder, file_name.replace('.tif', '.txt'))
        with open(txt_file, 'w') as f:
            f.write('\n'.join(yolo_boxes))

    print(f"YOLO annotations saved in {output_folder}")


images_folder = '../toydataset/object_detection/'
annotations_file = '../toydataset/object_detection/object_detection_annotations.csv'
output_folder = '../toydataset/object_detection/yolo_annotations/'

generate_yolo_annotations(images_folder, annotations_file, output_folder)
