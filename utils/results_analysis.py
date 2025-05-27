import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# Output path
output_dir = "../results/RatioAnalysis/"
os.makedirs(output_dir, exist_ok=True)

# Load CSV
path_to_csv = "../../../Data/Cleaned_EH_Ratio_Data.csv"
df = pd.read_csv(path_to_csv, sep=",", header=1, encoding="utf-8")
df = df.dropna(axis=1, how='all')
df.columns = df.columns.str.strip()
df = df.applymap(lambda x: str(x).replace('\xa0', '').replace(',', '.').replace('%', '').strip() if isinstance(x, str) else x)

# Convert to numeric where applicable
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except:
        pass

df["lado clinica"] = df["lado clinica"].str.strip().str.upper()

# Identify affected ear
df["Manual_Higher_Ear"] = df.apply(lambda row: "DERECHO" if row["Ratio D Manual"] > row["Ratio I Manual"] else "IZQUIERDO", axis=1)
df["Auto_Higher_Ear"] = df.apply(lambda row: "DERECHO" if row["Ratio D Auto"] > row["Ratio I Auto"] else "IZQUIERDO", axis=1)

# Evaluate correctness
df["Manual_Correct"] = df["Manual_Higher_Ear"] == df["lado clinica"]
df["Auto_Correct"] = df["Auto_Higher_Ear"] == df["lado clinica"]
df["Agreement"] = df["Manual_Higher_Ear"] == df["Auto_Higher_Ear"]

# Summary stats
means = df.describe().loc["mean"]
stds = df.describe().loc["std"]

# Accuracy
manual_acc = df["Manual_Correct"].mean()
auto_acc = df["Auto_Correct"].mean()
agreement = df["Agreement"].mean()

# Format and print the results cleanly
print("\n=================== EH RATIO ANALYSIS ===================")
print(f"✅ Manual Accuracy (agreement with clinician): {manual_acc:.2%}")
print(f"🤖 Auto Accuracy (agreement with clinician):   {auto_acc:.2%}")
print(f"🤝 Agreement between Manual & Auto:           {agreement:.2%}")
print("==========================================================\n")

# Show means and stds
print("📊 Column-wise Summary Statistics (means and std devs):\n")
summary_df = pd.DataFrame({
    "Mean": means.round(3),
    "Std Dev": stds.round(3)
})
print(summary_df)

# --- Define columns for manual and auto ---
manual_metrics = {
    "MRC D": "MRC D Manual",
    "MRC I": "MRC I Manual",
    "PEI D": "PEI D Manual",
    "PEI I": "PEI I Manual",
    "Ratio D": "Ratio D Manual",
    "Ratio I": "Ratio I Manual"
}

auto_metrics = {
    "MRC D": "MRC D Auto",
    "MRC I": "MRC I Auto",
    "PEI D": "PEI D Auto",
    "PEI I": "PEI I Auto",
    "Ratio D": "Ratio D Auto",
    "Ratio I": "Ratio I Auto"
}

# --- Function to extract mean/std for a dict of columns ---
def extract_stats(metric_dict):
    return pd.DataFrame({
        "Metric": list(metric_dict.keys()),
        "Mean": [df[col].mean() for col in metric_dict.values()],
        "Std": [df[col].std() for col in metric_dict.values()]
    })

# Util function to save bar plot
def save_bar_plot(data, title, ylabel, filename, color="#3498DB"):
    plt.figure(figsize=(6, 4))
    plt.bar(data["Metric"], data["Mean"], yerr=data["Std"], capsize=6, color=color)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_accuracy_by_ear(method_col, title, filename):
    sns.set_theme(style="whitegrid")
    
    acc_stats = df.groupby("lado clinica")[method_col].agg(['mean', 'std']).reset_index()
    acc_stats.columns = ["Ear Affected", "Mean", "Std"]
    acc_stats["Ear Affected"] = acc_stats["Ear Affected"].replace({
        "IZQUIERDO": "Left",
        "DERECHO": "Right"
    })

    # Cap accuracy values for display
    acc_stats["Upper"] = acc_stats["Mean"] + acc_stats["Std"]
    acc_stats["Upper Clipped"] = acc_stats["Upper"].clip(upper=1.0)
    capped_std = acc_stats["Upper Clipped"] - acc_stats["Mean"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        acc_stats["Ear Affected"], acc_stats["Mean"],
        yerr=capped_std, capsize=8,
        color=["#A5D8FF", "#74C0FC"],
        edgecolor="black"
    )
    
    ax.set_title(title, fontsize=14, weight="bold", pad=15)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.2)  # leave headroom
    ax.tick_params(axis='both', labelsize=11)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        label_pos = min(height + 0.05, 1.12)
        ax.text(bar.get_x() + bar.get_width()/2, label_pos, f"{height:.1%}",
                ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

mrc_metrics = {
    "Right MRC Manual": "MRC D Manual",
    "Right MRC Auto": "MRC D Auto",
    "Left MRC Manual": "MRC I Manual",
    "Left MRC Auto": "MRC I Auto"
}
save_bar_plot(extract_stats(mrc_metrics), "MRC Volume by Ear (Manual vs Auto)", "Volume", "mrc_volume.png")

pei_metrics = {
    "Right PEI Manual": "PEI D Manual",
    "Right PEI Auto": "PEI D Auto",
    "Left PEI Manual": "PEI I Manual",
    "Left PEI Auto": "PEI I Auto"
}
save_bar_plot(extract_stats(pei_metrics), "PEI Volume by Ear (Manual vs Auto)", "Volume", "pei_volume.png")

elr_metrics = {
    "Right ELR Manual": "Ratio D Manual",
    "Right ELR Auto": "Ratio D Auto",
    "Left ELR Manual": "Ratio I Manual",
    "Left ELR Auto": "Ratio I Auto"
}
save_bar_plot(extract_stats(elr_metrics), "ELR by Ear (Manual vs Auto)", "ELR (%)", "elr_comparison.png")


plot_accuracy_by_ear("Manual_Correct", "Manual Accuracy by Affected Ear", "manual_accuracy_by_ear.png")
plot_accuracy_by_ear("Auto_Correct", "Automatic Accuracy by Affected Ear", "auto_accuracy_by_ear.png")

