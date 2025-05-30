#!/bin/bash

# Define paths
INPUT_FOLDER="/Users/claudiacastrillonalvarez/Desktop/github/EHRatioAnalysis/MRC_data/MRC_images"
OUTPUT_FOLDER="/Users/claudiacastrillonalvarez/Desktop/IMAGES_FIJI/MRC"
JSON_FILE="image_dict.json"

# Ensure the main output folder exists
mkdir -p "$OUTPUT_FOLDER"

# Read JSON file line by line
jq -r 'to_entries[] | "\(.key) \(.value)"' "$JSON_FILE" | while read -r image_name annotation; do
    if [[ "$annotation" -eq 1 ]]; then
        found=0

        # Search for the image in all patient folders
        for patient_folder in "$INPUT_FOLDER"/*/; do
            if [[ -f "$patient_folder$image_name" ]]; then
                # Extract patient folder name
                patient_name=$(basename "$patient_folder")

                # Create patient-specific folder in output (without adding "MRC TIFF" again)
                patient_output_folder="$OUTPUT_FOLDER/$patient_name"
                mkdir -p "$patient_output_folder"

                # Copy the image
                cp "$patient_folder$image_name" "$patient_output_folder/"
                echo " Copied: $patient_folder$image_name → $patient_output_folder/"
                found=1
                break  # Stop searching after the first match
            fi
        done

        # Print warning if the file was not found
        if [[ "$found" -eq 0 ]]; then
            echo "⚠️ Warning: $image_name not found in any patient folder."
        fi
    fi
done

echo " All ear images have been copied and organized!"

