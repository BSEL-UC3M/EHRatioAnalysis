import pandas as pd
import json

# Path to Excel file
excel_path = "/Users/claudiacastrillonalvarez/Desktop/github/EHRatioAnalysis/MRC_data/MRC_images/MRC_TIFF_Annotations.xlsx"

# Load the Excel file
xls = pd.ExcelFile(excel_path)

# Dictionary to store the images and their annotation values
image_dict = {}

for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)
    
    # Rename columns to match the Excel structure
    df.columns = ["File Name", "Annotation"]
    
    # Add valid images to dictionary
    for _, row in df.iterrows():
        image_dict[row["File Name"]] = int(row["Annotation"])

# Save dictionary as JSON
with open("image_dict.json", "w") as f:
    json.dump(image_dict, f, indent=4)

print(" Image dictionary saved as 'image_dict.json'.")
