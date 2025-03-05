import pandas as pd
import json

# Path to PEI Excel file
excel_path = "/Users/claudiacastrillonalvarez/Desktop/github/EHRatioAnalysis/PEI_data/PEI_images/PEI_TIFF_Annotations.xlsx"

# Load the Excel file
xls = pd.ExcelFile(excel_path)

# Dictionary to store the images and their annotation values
image_dict = {}

for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)

    # Ensure correct columns
    if "File Name" in df.columns and "Annotation" in df.columns:
        df = df[["File Name", "Annotation"]]  # Select only these two columns
    else:
        print(f"⚠️ Skipping {sheet_name}: Columns not found")
        continue  # Skip sheet if it doesn't have the expected format

    # Convert data to dictionary
    for _, row in df.iterrows():
        image_dict[row["File Name"]] = int(row["Annotation"])

# Save dictionary as JSON
with open("pei_image_dict.json", "w") as f:
    json.dump(image_dict, f, indent=4)

print("✅ PEI Image dictionary saved as 'pei_image_dict.json'.")
