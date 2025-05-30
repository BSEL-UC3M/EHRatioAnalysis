# eh_ratio_visuals.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# ========================== CONFIG =============================
INPUT_CSV = "../../../Data/Cleaned_EH_Ratio_Data.csv"
OUTPUT_DIR = "../results/RatioAnalysis/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================== 1. LOAD & CLEAN DATA ===================
df = pd.read_csv(INPUT_CSV, sep=",", header=1, encoding="utf-8")
df = df.dropna(axis=1, how='all')
df.columns = df.columns.str.strip()
df = df.applymap(lambda x: str(x).replace('\xa0', '').replace(',', '.').replace('%', '').strip() if isinstance(x, str) else x)
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except:
        pass

df["lado clinica"] = df["lado clinica"].str.strip().str.upper()

# Derived columns
df["Manual_Higher_Ear"] = df.apply(lambda row: "DERECHO" if row["Ratio D Manual"] > row["Ratio I Manual"] else "IZQUIERDO", axis=1)
df["Auto_Higher_Ear"] = df.apply(lambda row: "DERECHO" if row["Ratio D Auto"] > row["Ratio I Auto"] else "IZQUIERDO", axis=1)
df["Manual_Correct"] = df["Manual_Higher_Ear"] == df["lado clinica"]
df["Auto_Correct"] = df["Auto_Higher_Ear"] == df["lado clinica"]
df["Agreement"] = df["Manual_Higher_Ear"] == df["Auto_Higher_Ear"]
df["ELR Manual Impact"] = df["Ratio D Manual"] - df["Ratio I Manual"]
df["ELR Auto Impact"] = df["Ratio D Auto"] - df["Ratio I Auto"]

# ====================== 2. SUMMARY MARKDOWN =====================
summary_md = f"""
# EH Ratio Analysis Summary

- Number of patients: {len(df)}
- Manual Accuracy: {df['Manual_Correct'].mean():.2%}
- Auto Accuracy: {df['Auto_Correct'].mean():.2%}
- Agreement Manual vs Auto: {df['Agreement'].mean():.2%}

## Column Overview:

{df.columns.to_list()}

## PEI vs MRC Volume Ratios (Automatic):

- Mean MRC D Volume: {df['MRC D Auto'].mean():.4f}
- Mean PEI D Volume: {df['PEI D Auto'].mean():.4f}
- Ratio PEI D / MRC D: {(df['PEI D Auto'].mean() / df['MRC D Auto'].mean()):.2f}

- Mean MRC I Volume: {df['MRC I Auto'].mean():.4f}
- Mean PEI I Volume: {df['PEI I Auto'].mean():.4f}
- Ratio PEI I / MRC I: {(df['PEI I Auto'].mean() / df['MRC I Auto'].mean()):.2f}

## ELR Ranges (Automatic):
- Max ELR D: {df['Ratio D Auto'].max():.2f}%
- Max ELR I: {df['Ratio I Auto'].max():.2f}%
- Mean ELR D: {df['Ratio D Auto'].mean():.2f}%
- Mean ELR I: {df['Ratio I Auto'].mean():.2f}%

"""
with open(os.path.join(OUTPUT_DIR, "EH_Ratio_Analysis_Summary.md"), "w") as f:
    f.write(summary_md)


# =======================  PLOT: ELR DISTRIBUTION ========================
elr_long = df.melt(value_vars=["Ratio D Auto", "Ratio I Auto"], 
                   var_name="Ear", value_name="ELR")
elr_long["Ear"] = elr_long["Ear"].map({
    "Ratio D Auto": "Right",
    "Ratio I Auto": "Left"
})

g = sns.displot(
    data=elr_long, x="ELR", hue="Ear", kind="hist", bins=15, kde=True,
    palette={"Right": "#2E86C1", "Left": "#F39C12"},
    height=5, aspect=1.3, edgecolor="black"
)

g.set_axis_labels("ELR (%)", "Count")
g.fig.suptitle("ELR Distribution by Ear (Automatic)", fontsize=14)
g.tight_layout()
g.savefig(os.path.join(OUTPUT_DIR, "elr_distribution_auto.png"), dpi=300)
plt.close()

# =======================  PLOT: VIOLIN PLOTS ========================
elr_long = df.melt(value_vars=["Ratio D Auto", "Ratio I Auto"],
                   var_name="Ear", value_name="ELR")
elr_long["Ear"] = elr_long["Ear"].map({"Ratio D Auto": "Right", "Ratio I Auto": "Left"})

plt.figure(figsize=(6, 5))
sns.violinplot(data=elr_long, x="Ear", y="ELR", palette=["#5DADE2", "#F5B041"])
plt.axhline(100, color='gray', linestyle='--', lw=1)
plt.title("ELR Distribution by Ear (Violin Plot)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "elr_violin_auto.png"), dpi=300)
plt.close()

# ======================= PLOT: SIDE-BY-SIDE IMPACT ========================
impact_df = df[["ELR Manual Impact", "ELR Auto Impact"]].copy()
impact_df["Patient"] = range(1, len(df) + 1)
impact_melted = impact_df.melt(id_vars="Patient", var_name="Method", value_name="Dominance")

plt.figure(figsize=(10, 5))
sns.barplot(data=impact_melted, x="Patient", y="Dominance", hue="Method", palette=["#A3C9A8", "#6D9886"])
plt.axhline(0, color="black", linestyle="--")
plt.title("ELR Dominance Difference (RatioD - RatioI)")
plt.ylabel("Dominance")
plt.xticks([], [])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "elr_impact_comparison.png"), dpi=300)
plt.close()
