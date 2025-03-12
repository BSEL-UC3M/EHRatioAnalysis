# ==============================================================================
# Description: File to calculate the EH (Endolymphatic Hydrops) ratios based on processed MRC and REAL images
# Author: Gloria del Rocío Delicado Correa
# Maintainer: Caterina Fuster-Barceló
# Creation date: 29/08/2023
# ==============================================================================

import os
import cv2
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from utils import normalize, adjust_contrast, crop_images_UNET_MRC, crop_images_UNET_REAL

class EHRatioCalculator:
    def __init__(self, mrc_directory, real_directory, model_MRC, model_REAL_IR, mrc_pixel_size, real_pixel_size):
        self.mrc_directory = mrc_directory
        self.real_directory = real_directory
        self.model_MRC = model_MRC
        self.model_REAL_IR = model_REAL_IR
        self.mrc_pixel_size = mrc_pixel_size
        self.real_pixel_size = real_pixel_size

    def overlay_mask_on_image(self, image, mask, title=None):
        """
        Overlay a binary mask on the original RGB image and display the result.

        Parameters:
            image (numpy.array): The original RGB image.
            mask (numpy.array): The binary mask to overlay on the image (single-channel).
            title (str, optional): Title for the plot.

        Returns:
            None (displays the plot).
        """
        # Convert the grayscale mask to an RGB mask with magenta color
        overlaid_image = image.copy()
        overlaid_image[mask.squeeze() == 1] = [1, 0, 1]  # Red color for predicted mask


        plt.imshow(overlaid_image)
        plt.axis('off')
    

    def transform_REAL(self, image):
        """
        Transform the REAL image by enhancing contrast and inverting intensities.

        Parameters:
            image (numpy.array): The input REAL image.

        Returns:
            transformed_image (numpy.array): The transformed REAL image.
        """
    
        enhanced_image = adjust_contrast(image, alpha=3)

        # Invert the image intensities
        inverted_image = 1.0 - enhanced_image

        return inverted_image

    def preprocess_MRC(self, image):

        """
        Preprocess the MRC image by cropping and normalizing it.

        Parameters:
            image (numpy.array): The input MRC image.

        Returns:
            normalized_mrc_left (numpy.array): Normalized left ear MRC subimage.
            normalized_mrc_right (numpy.array): Normalized right ear MRC subimage.
        """
        image_array = [image]
        mask_array = [np.zeros_like(image)]  # Placeholder for mask, not used in cropping
        cropped_mrc_images, _ = crop_images_UNET_MRC(image_array, mask_array, do_crop=True)
        processed_mrc_images = []
        for cropped_mrc_image in cropped_mrc_images:
            if len(cropped_mrc_image.shape) == 2:  # If the image is grayscale
                cropped_mrc_image = cv2.cvtColor(cropped_mrc_image, cv2.COLOR_GRAY2RGB)
            processed_mrc_images.append(normalize(cropped_mrc_image))
        normalized_mrc_right, normalized_mrc_left = processed_mrc_images
        return normalized_mrc_left, normalized_mrc_right

    def preprocess_REAL(self, image):

        """
        Preprocess the REAL image by transforming, cropping, and normalizing it.

        Parameters:
            image (numpy.array): The input REAL image.

        Returns:
            normalized_real_left (numpy.array): Normalized left ear REAL subimage.
            normalized_real_right (numpy.array): Normalized right ear REAL subimage.
        """
        image_array = [image]
        mask_array = [np.zeros_like(image)]  # Placeholder for mask, not used in cropping
        cropped_real_images, _ = crop_images_UNET_REAL(image_array, mask_array, do_crop=True)
        processed_real_images = []
        for cropped_real_image in cropped_real_images:
            cropped_real_image = self.transform_REAL(cropped_real_image)
            if len(cropped_real_image.shape) == 2:  # If the image is grayscale
                cropped_real_image = cv2.cvtColor(cropped_real_image, cv2.COLOR_GRAY2RGB)
            processed_real_images.append(normalize(cropped_real_image))
        normalized_real_right, normalized_real_left = processed_real_images
        return normalized_real_left, normalized_real_right

    def predict_segmentation(self, model, image):
        """
        Predict a binary mask using the given model for the input image.

        Parameters:
            model: The segmentation model for mask prediction.
            image (numpy.array): The input image for mask prediction.

        Returns:
            mask (numpy.array): The predicted binary mask.
        """
            # Assuming the model expects a batch dimension, expand the dimensions
        expanded_image = np.expand_dims(image, axis=0)
        mask = model.predict(expanded_image)
        # Apply threshold to convert prediction into binary mask
        mask = (mask > 0.5).astype(np.int32)
        return mask[0]  # Remove the batch dimension and return the mask

    def compute_surface_area(self, mask, pixel_size):
        """
        Calculate the surface area from a binary mask.

        Parameters:
            mask (numpy.array): The binary mask.
            pixel_size (float): The size of each pixel in mm^2.

        Returns:
            surface_area (float): The computed surface area in mm^2.
        """
        return np.sum(mask) * pixel_size

    def process_patient_images(self, patient_images, model, preprocess_function, pixel_size):
        """
        Process patient images to obtain surfaces, masks, and images.

        Parameters:
            patient_images (list): List of patient images.
            model: The segmentation model for mask prediction.
            preprocess_function: The image preprocessing function.
            pixel_size (float): The size of each pixel in mm^2.

        Returns:
            left_surfaces (list): List of left ear surfaces.
            right_surfaces (list): List of right ear surfaces.
            left_masks (list): List of left ear masks.
            right_masks (list): List of right ear masks.
            left_images (list): List of normalized left ear images.
            right_images (list): List of normalized right ear images.
        """
        
        
        left_masks = []  # Store the left masks
        right_masks = []  # Store the right masks
        left_surfaces=[]
        right_surfaces=[]
        left_images=[]
        right_images=[]
        
        for image in patient_images:
            left_image, right_image = preprocess_function(image)
            left_mask = self.predict_segmentation(model, left_image)
            right_mask = self.predict_segmentation(model, right_image)
            # Apply threshold to convert prediction into binary mask
            left_mask = (left_mask > 0.8).astype(np.int32)
            # Apply threshold to convert prediction into binary mask
            right_mask = (right_mask > 0.8).astype(np.int32)
            left_surface = self.compute_surface_area(left_mask, pixel_size)
            right_surface = self.compute_surface_area(right_mask, pixel_size)
            left_masks.append(left_mask)  # Store the left mask
            right_masks.append(right_mask)  # Store the right mask  
            left_surfaces.append(left_surface)  # Store the left mask
            right_surfaces.append(right_surface)  # Store the right mask         
            left_images.append(left_image)  # Store the left mask
            right_images.append(right_image)
                
        return left_surfaces, right_surfaces, left_masks, right_masks, left_images, right_images  # Return the patient surfaces, left masks, and right masks



    def load_images(self, directory):
        """
        Load images from the given directory.

        Parameters:
            directory (str): The path to the directory containing images.

        Returns:
            images (dict): A dictionary where the keys are image filenames (without extensions),
                        and the values are the corresponding image arrays.
        """
        images = {}
        for filename in os.listdir(directory):
            if filename.lower().endswith('.tif'):  # Assuming images are in TIFF format
                image_path = os.path.join(directory, filename)
                image = tiff.imread(image_path)
                if image is not None:
                    # Remove file extension and store the image array in the dictionary
                    image_name = os.path.splitext(filename)[0]
                    images[image_name] = image
        return images

    def calculate_eh_ratios(self):
        """
        Calculate the EH (Endolymphatic Hydrops) ratios based on processed MRC and REAL images.

        Returns:
            eh_left_ratio (float): The calculated EH ratio for the left ear.
            eh_right_ratio (float): The calculated EH ratio for the right ear.
        """

        mrc_patients = self.load_images(self.mrc_directory)
        real_patients = self.load_images(self.real_directory)

        mrc_images_list = list(mrc_patients.values())
        real_images_list = list(real_patients.values())

        mrc_left_surfaces, mrc_right_surfaces, mrc_left_masks, mrc_right_masks, mrc_left_images, mrc_right_images = self.process_patient_images(mrc_images_list, self.model_MRC, self.preprocess_MRC, self.mrc_pixel_size)

        real_left_surfaces, real_right_surfaces, real_left_masks, real_right_masks, real_left_images, real_right_images = self.process_patient_images(real_images_list, self.model_REAL_IR, self.preprocess_REAL, self.real_pixel_size)

        mrc_max_left_idx = np.argmax(mrc_left_surfaces)
        mrc_max_right_idx = np.argmax(mrc_right_surfaces)
        real_max_left_idx = np.argmax(real_left_surfaces)
        real_max_right_idx = np.argmax(real_right_surfaces)

        max_mrc_left_surface = max(mrc_left_surfaces)
        max_mrc_right_surface = max(mrc_right_surfaces)
        max_real_left_surface = max(real_left_surfaces)
        max_real_right_surface = max(real_right_surfaces)
        
        eh_left_ratio = max_real_left_surface / max_mrc_left_surface
        eh_right_ratio = max_real_right_surface / max_mrc_right_surface

        print(f"The EH right ratio is: {eh_right_ratio:.2f}, The EH left ratio is: {eh_left_ratio:.2f}")
        
        # Print the results
        print("Right Ear Surfaces (mm^2):")
        print("MRC:", max_mrc_right_surface)
        print("REAL:", max_real_right_surface)

        print("Left Ear Surfaces (mm^2):")
        print("MRC:", max_mrc_left_surface)
        print("REAL:", max_real_left_surface)



        # Plot right ear images for both MRC and REAL
        self.plot_ear_images("Right Ear", eh_right_ratio, mrc_max_right_idx, real_max_right_idx,
                             mrc_right_images, real_right_images,
                             mrc_right_masks, real_right_masks)

        # Plot left ear images for both MRC and REAL
        self.plot_ear_images("Left Ear", eh_left_ratio, mrc_max_left_idx, real_max_left_idx,
                             mrc_left_images, real_left_images,
                             mrc_left_masks, real_left_masks)

        return eh_left_ratio, eh_right_ratio
        
    

    def plot_ear_images(self, ear_name, eh_ratio, mrc_idx, real_idx,
                    mrc_images, real_images, mrc_masks, real_masks):
        """
        Plot original and overlaid images for the specified ear.

        Parameters:
            ear_name (str): The name of the ear (left or right).
            eh_ratio (float): The calculated EH ratio for the ear.
            mrc_idx (int): The index of the MRC image to display.
            real_idx (int): The index of the REAL image to display.
            mrc_images (list): List of MRC images.
            real_images (list): List of REAL images.
            mrc_masks (list): List of MRC masks.
            real_masks (list): List of REAL masks.

        Returns:
            None (displays the plot).
        """    
        
        
        plt.figure(figsize=(12, 8))
        plt.suptitle(f"{ear_name} (EH ratio: {eh_ratio:.2f})")

        plt.subplot(2, 2, 1)
        plt.imshow(mrc_images[mrc_idx], cmap='gray')
        plt.title("Original MRC")
        plt.axis('off')

        plt.subplot(2, 2, 2)
        self.overlay_mask_on_image(mrc_images[mrc_idx], mrc_masks[mrc_idx], title="Overlaid MRC")
        plt.title("Overlaid MRC")

        plt.subplot(2, 2, 3)
        plt.imshow(real_images[real_idx], cmap='gray')
        plt.title("Original REAL")
        plt.axis('off')

        plt.subplot(2, 2, 4)
        self.overlay_mask_on_image(real_images[real_idx], real_masks[real_idx], title="Overlaid REAL")
        plt.title("Overlaid REAL")

        plt.tight_layout()
        plt.show()