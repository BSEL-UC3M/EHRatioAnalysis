from utils.EH_ratio import compute_eh_ratios
import os
import csv
import numpy as np
from datetime import datetime
from utils.EH_ratio import procesar_imagenes

RESULTS_FOLDER = "./results/results_segmentator/Ratio"
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
results_dir = os.path.join(RESULTS_FOLDER, timestamp)
os.makedirs(results_dir, exist_ok=True)

print(" RatioCalculator – Computing EH Ratios from segmented masks\n")

RATIO_OUTPUT_CSV = os.path.join(results_dir, "eh_volume_ratios.csv")
MRC_SEGMENT_DIR = "C:/Users/TFM1/Documents/GitHub/EHRatioAnalysis/results/results_segmentator/MRC/20250416/20250416-154715/20250416-154716/binary_masks"
PEI_SEGMENT_DIR = "C:/Users/TFM1/Documents/GitHub/EHRatioAnalysis/results/results_segmentator/PEI/20250411/20250411-113208/20250411-113209/binary_masks"

compute_eh_ratios(
    mrc_mask_folder=os.path.join(MRC_SEGMENT_DIR),
    pei_mask_folder=os.path.join(PEI_SEGMENT_DIR),
    output_csv_path=RATIO_OUTPUT_CSV,
)


mrc_path = "C:/Users/TFM1/Documents/GitHub/EHRatioAnalysis/results/results_segmentator/MRC/20250416/20250416-154715/20250416-154716/binary_masks/PAC5_right_previous_right.tif.png"
pei_path = "C:/Users/TFM1/Documents/GitHub/EHRatioAnalysis/results/results_segmentator/PEI/20250411/20250411-113208/20250411-113209/binary_masks/PAC5_right_previous_right.tif.png"

procesar_imagenes(mrc_path, pei_path)