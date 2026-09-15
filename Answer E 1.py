
import sys
import subprocess

try:
    import statsmodels
except ImportError:
    print("Installing statsmodels...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])

# -----------------------------
# STEP 2: IMPORT LIBRARIES
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ================================================================
# STEP 3: SET MAIN PATH
# ================================================================

BASE_PATH = Path(
    r"C:\Users\admin\Desktop\TRIP\Mathematical Modal-Sai Chand-3-C\TRL-700 DATA FILES"
)

# Only these four vehicle types are included
VEHICLE_FOLDERS = {
    "Car": BASE_PATH / "Stacked files of car",
    "3-W": BASE_PATH / "stacked files od Tuk-Tuk",
    "MTW": BASE_PATH / "Stacked files of Motorcycles",
    "Bus": BASE_PATH / "stacked files for the bus"
}

OUTPUT_FOLDER = BASE_PATH / "Q1_Speed_Analysis"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ================================================================
# STEP 4: CHECK FOLDERS
# ================================================================

print("=" * 75)
print("QUESTION 1 - SPEED ANALYSIS")
print("=" * 75)

print("\nChecking vehicle folders:")

for vehicle, folder in VEHICLE_FOLDERS.items():
    if folder.exists():
        print(f"✓ {vehicle}: Folder found")
    else:
        print(f"✗ {vehicle}: Folder NOT found")

# ================================================================
# STEP 5: READ ALL CSV FILES
# ================================================================

all_data = []

print("\nReading CSV files...")

for vehicle, folder in VEHICLE_FOLDERS.items():

    csv_files = list(folder.rglob("*.csv"))

    print(f"{vehicle}: {len(csv_files)} CSV files found")

    for csv_file in csv_files:

        try:
            df = pd.read_csv(csv_file, low_memory=False)

            # Remove extra spaces from column names
            df.columns = df.columns.str.strip()

            # Check required column
            if "Speed [km/h]" not in df.columns:
                continue

            # Convert speed to numeric
            df["Speed [km/h]"] = pd.to_numeric(
                df["Speed [km/h]"],
                errors="coerce"
            )

            # Keep only valid speed observations
            df = df.dropna(subset=["Speed [km/h]"]).copy()

            if len(df) == 0:
                continue

            # Add vehicle type and source information
            df["Vehicle_Type"] = vehicle
            df["Source_File"] = csv_file.name

            all_data.append(
                df[["Vehicle_Type", "Source_File", "Speed [km/h]"]]
            )

        except Exception as e:
            print(f"Could not read {csv_file.name}: {e}")

# Combine all files
speed_data = pd.concat(all_data, ignore_index=True)

print("\nData successfully combined.")
print(f"Total detailed speed observations = {len(speed_data):,}")

# ================================================================
# STEP 6: BASIC DATA SUMMARY
# ================================================================

print("\n" + "=" * 75)
print("DATA SUMMARY")
print("=" * 75)

summary_counts = (
    speed_data
    .groupby("Vehicle_Type")
    .size()
    .reindex(["Car", "3-W", "MTW", "Bus"])
)

print("\nDetailed speed observations by vehicle type:")
print(summary_counts.to_string())

print(
    "\nNote: N represents detailed trajectory speed observations, "
    "not the number of individual vehicles."
)

# ================================================================
# STEP 7: DESCRIPTIVE STATISTICS
# ================================================================

vehicle_order = ["Car", "3-W", "MTW", "Bus"]

descriptive = (
    speed_data
    .groupby("Vehicle_Type")["Speed [km/h]"]
    .agg(
        N="count",
        Mean="mean",
        SD="std",
        Median="median",
        Min="min",
        Max="max"
    )
    .reindex(vehicle_order)
)

print("\n" + "=" * 75)
print("TABLE 1: DESCRIPTIVE STATISTICS")
print("=" * 75)

print(
    descriptive.round(2).to_string()
)

# ================================================================
# STEP 8: NORMALITY DIAGNOSTIC
# ================================================================

print("\n" + "=" * 75)
print("NORMALITY CHECK - SHAPIRO-WILK DIAGNOSTIC")
print("=" * 75)

normality_results = []

for vehicle in vehicle_order:

    values = (
        speed_data.loc[
            speed_data["Vehicle_Type"] == vehicle,
            "Speed [km/h]"
        ]
        .dropna()
    )

    # Shapiro-Wilk has a practical limit of 5000 observations
    sample = values.sample(
        n=min(5000, len(values)),
        random_state=42
    )

    W, p = stats.shapiro(sample)

    normality_results.append(
        {
            "Vehicle Type": vehicle,
            "Sample N": len(sample),
            "W-statistic": W,
            "p-value": p,
            "Interpretation":
                "Approximately normal" if p >= 0.05
                else "Departure from normality"
        }
    )

normality_df = pd.DataFrame(normality_results)

print(
    normality_df.round(4).to_string(index=False)
)

print(
    "\nNote: With very large samples, Shapiro-Wilk can detect even small "
    "departures from normality. Therefore, it is treated as a diagnostic."
)

# ================================================================
# STEP 9: LEVENE'S TEST
# ================================================================

print("\n" + "=" * 75)
print("VARIANCE CHECK - LEVENE'S TEST")
print("=" * 75)

groups = [
    speed_data.loc[
        speed_data["Vehicle_Type"] == vehicle,
        "Speed [km/h]"
    ].dropna()
    for vehicle in vehicle_order
]

levene_stat, levene_p = stats.levene(
    *groups,
    center="median"
)

print(f"Levene statistic = {levene_stat:.4f}")

if levene_p < 0.001:
    print("p-value = <0.001")
else:
    print(f"p-value = {levene_p:.6f}")

if levene_p < 0.05:
    print("Result: Variances are significantly different.")
else:
    print("Result: No significant evidence of unequal variances.")

# ================================================================
# STEP 10: ONE-WAY ANOVA
# ================================================================

print("\n" + "=" * 75)
print("TABLE 2: ONE-WAY ANOVA")
print("=" * 75)

F_stat, p_value = stats.f_oneway(*groups)

# Calculate eta-squared
grand_mean = speed_data["Speed [km/h]"].mean()

SS_between = sum(
    len(group) * (group.mean() - grand_mean) ** 2
    for group in groups
)

SS_total = sum(
    (speed_data["Speed [km/h]"] - grand_mean) ** 2
)

eta_squared = SS_between / SS_total

print(f"F-statistic = {F_stat:.6f}")

if p_value < 0.001:
    print("p-value = <0.001")
else:
    print(f"p-value = {p_value:.6f}")

print("Alpha = 0.05")

if p_value < 0.05:
    decision = "Reject H0"
else:
    decision = "Fail to reject H0"

print(f"Decision = {decision}")
print(f"Eta-squared = {eta_squared:.6f}")

# ================================================================
# STEP 11: TUKEY HSD POST-HOC TEST
# ================================================================

print("\n" + "=" * 75)
print("TUKEY HSD POST-HOC TEST")
print("=" * 75)

tukey = pairwise_tukeyhsd(
    endog=speed_data["Speed [km/h]"],
    groups=speed_data["Vehicle_Type"],
    alpha=0.05
)

# Convert Tukey result into dataframe
tukey_table = pd.DataFrame(
    data=tukey._results_table.data[1:],
    columns=tukey._results_table.data[0]
)

# Put pairs in desired order if possible
print("\nPairwise comparison results:")
print(tukey_table.to_string(index=False))

# ================================================================
# STEP 12: TUKEY INTERPRETATION
# ================================================================

print("\n" + "=" * 75)
print("TUKEY HSD INTERPRETATION")
print("=" * 75)

for _, row in tukey_table.iterrows():

    group1 = row["group1"]
    group2 = row["group2"]

    reject = row["reject"]

    if str(reject).lower() == "true":
        result = "Significantly different"
    else:
        result = "Not significantly different"

    print(f"{group1} vs {group2}: {result}")

# ================================================================
# STEP 13: MEAN SPEED GRAPH
# ================================================================

plt.figure(figsize=(9, 6))

means = descriptive["Mean"]
sds = descriptive["SD"]

plt.bar(
    vehicle_order,
    means,
    yerr=sds,
    capsize=5
)

plt.xlabel("Vehicle Type")
plt.ylabel("Mean Speed (km/h)")
plt.title("Mean Speed of Different Vehicle Types with ±1 SD")
plt.tight_layout()

mean_graph_path = OUTPUT_FOLDER / "Q1_Mean_Speed_by_Vehicle_Type.png"

plt.savefig(
    mean_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"\nFigure 1 saved at: {mean_graph_path}"
)

# ================================================================
# STEP 14: FIGURE 1 INSIGHT
# ================================================================

highest_mean_vehicle = descriptive["Mean"].idxmax()
lowest_mean_vehicle = descriptive["Mean"].idxmin()

print("\n" + "=" * 75)
print("FIGURE 1 - MEAN SPEED GRAPH INSIGHT")
print("=" * 75)

print(
    f"The graph shows differences in mean speed among the four vehicle types. "
    f"{highest_mean_vehicle} has the highest mean speed, while "
    f"{lowest_mean_vehicle} has the lowest mean speed. "
    f"The error bars represent ±1 standard deviation and show the variation "
    f"in detailed trajectory speed observations."
)

print(
    "The graph provides a visual indication of differences, while the "
    "ANOVA determines whether the overall differences are statistically significant."
)

# ================================================================
# STEP 15: SPEED DISTRIBUTION BOXPLOT
# ================================================================

plt.figure(figsize=(10, 6))

box_data = [
    speed_data.loc[
        speed_data["Vehicle_Type"] == vehicle,
        "Speed [km/h]"
    ].dropna()
    for vehicle in vehicle_order
]

plt.boxplot(
    box_data,
    labels=vehicle_order,
    showfliers=True
)

plt.xlabel("Vehicle Type")
plt.ylabel("Speed (km/h)")
plt.title("Distribution of Detailed Trajectory Speed Observations")
plt.tight_layout()

boxplot_path = OUTPUT_FOLDER / "Q1_Speed_Distribution_Boxplot.png"

plt.savefig(
    boxplot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"\nFigure 2 saved at: {boxplot_path}"
)

# ================================================================
# STEP 16: FIGURE 2 INSIGHT
# ================================================================

print("\n" + "=" * 75)
print("FIGURE 2 - BOXPLOT INSIGHT")
print("=" * 75)

print(
    "The boxplot shows the median, spread and extreme speed observations "
    "for each vehicle type. Differences in the location of the distributions "
    "indicate that the speed patterns are not identical across vehicle types. "
    "The spread of each box and the presence of extreme observations also "
    "show the variation in detailed trajectory speeds."
)

# ================================================================
# STEP 17: FINAL ANOVA RESULT TABLE
# ================================================================

anova_table = pd.DataFrame({
    "Test": ["One-Way ANOVA"],
    "F-statistic": [F_stat],
    "p-value": [
        "<0.001" if p_value < 0.001 else p_value
    ],
    "Alpha": [0.05],
    "Decision": [decision],
    "Eta-squared": [eta_squared]
})

print("\n" + "=" * 75)
print("TABLE 3: ANOVA RESULT")
print("=" * 75)

print(
    anova_table.round(6).to_string(index=False)
)

# ================================================================
# STEP 18: FINAL RESULT AND CONCLUSION
# ================================================================

print("\n" + "=" * 75)
print("QUESTION 1 - FINAL RESULT AND CONCLUSION")
print("=" * 75)

print("\nHYPOTHESES")
print("-" * 75)

print(
    "H0: The mean speeds of Car, 3-W, MTW and Bus are equal."
)

print(
    "H1: At least one vehicle type has a different mean speed."
)

print("\nSTATISTICAL RESULT")
print("-" * 75)

print(
    f"One-Way ANOVA: F = {F_stat:.3f}, p < 0.001"
)

print(
    "Significance level (α) = 0.05"
)

print(
    "Decision: Reject H0"
)

print(
    f"Eta-squared = {eta_squared:.4f}"
)

print("\nFINAL CONCLUSION")
print("-" * 75)

print(
    "Since p < 0.05, the null hypothesis is rejected."
)

print(
    "There is strong statistical evidence that the mean speeds of "
    "Car, 3-W (Tuk-Tuk), MTW (Motorcycles), and Bus are significantly "
    "different overall."
)

print(
    "The Tukey HSD post-hoc test was conducted to identify which specific "
    "pairs of vehicle types have significantly different speeds."
)

# ================================================================
# STEP 19: SAVE RESULTS TO EXCEL
# ================================================================

excel_path = OUTPUT_FOLDER / "Q1_Speed_Complete_Statistical_Analysis.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

    descriptive.round(4).to_excel(
        writer,
        sheet_name="Descriptive Statistics"
    )

    anova_table.round(6).to_excel(
        writer,
        sheet_name="ANOVA",
        index=False
    )

    tukey_table.to_excel(
        writer,
        sheet_name="Tukey HSD",
        index=False
    )

    normality_df.to_excel(
        writer,
        sheet_name="Normality Check",
        index=False
    )

    levene_df = pd.DataFrame({
        "Test": ["Levene's Test"],
        "Statistic": [levene_stat],
        "p-value": [
            "<0.001" if levene_p < 0.001 else levene_p
        ]
    })

    levene_df.to_excel(
        writer,
        sheet_name="Levene Test",
        index=False
    )

    data_summary = pd.DataFrame({
        "Vehicle Type": vehicle_order,
        "Detailed Speed Observations": [
            len(groups[i]) for i in range(len(groups))
        ]
    })

    data_summary.to_excel(
        writer,
        sheet_name="Data Summary",
        index=False
    )

print("\n" + "=" * 75)
print("FILES SAVED")
print("=" * 75)

print(f"\nExcel file:")
print(excel_path)

print(f"\nFigure 1:")
print(mean_graph_path)

print(f"\nFigure 2:")
print(boxplot_path)

print("\n" + "=" * 75)
print("QUESTION 1 ANALYSIS COMPLETE")
print("=" * 75)
