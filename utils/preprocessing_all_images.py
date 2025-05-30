import os
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

def histogram_adjustment(image, lower_threshold_factor=1, upper_threshold_factor=1):
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)

    lower_threshold = mean_intensity - lower_threshold_factor * std_intensity
    upper_threshold = mean_intensity + upper_threshold_factor * std_intensity

    adjusted_image = np.clip(image, lower_threshold, upper_threshold)
    adjusted_image = (adjusted_image - adjusted_image.min()) / (adjusted_image.max() - adjusted_image.min())

    return adjusted_image

def invert_image(image):
    return 1.0 - image

def preprocess_all_images(folder_path, output_folder):
    """
    Process all patient folders in the dataset, saving processed images in `output_folder`.

    Args:
        folder_path (str): Path to the main dataset folder (e.g., PEI_TIFF).
        output_folder (str): Path to save processed images (e.g., PEI_processed_data).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get all patient folders inside the dataset
    patient_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    for patient in patient_folders:
        patient_input_path = os.path.join(folder_path, patient)
        patient_output_path = os.path.join(output_folder, patient)

        # Create patient-specific folder in PEI_processed_data
        if not os.path.exists(patient_output_path):
            os.makedirs(patient_output_path)

        # Get all TIFF images from the patient folder
        tiff_files = [f for f in os.listdir(patient_input_path) if f.endswith('.tif')]
        
        for tiff_file in tiff_files:
            file_path = os.path.join(patient_input_path, tiff_file)
            
            # Open and process the image
            img = Image.open(file_path)
            img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize

            # Apply processing steps
            img_adjusted = 2 * img_array + gaussian_filter(img_array, sigma=6)
            img_adjusted = histogram_adjustment(img_adjusted, lower_threshold_factor=2.4, upper_threshold_factor=2.2)
            img_inverted = invert_image(img_adjusted)

            # Save the processed image in the correct patient folder
            output_path = os.path.join(patient_output_path, tiff_file)
            Image.fromarray((img_inverted * 255).astype(np.uint8)).save(output_path)

            print(f"Saved: {output_path}")  # Debugging output
    
    print(f"\n All processed images saved in {output_folder}")
