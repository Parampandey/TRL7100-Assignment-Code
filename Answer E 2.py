
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ============================================================
# 1. DATA PATH
# ============================================================

BASE_PATH = Path(
    r"C:\Users\admin\Desktop\TRIP\Mathematical Modal-Sai Chand-3-C\TRL-700 DATA FILES"
)

folders = {
    "Car": BASE_PATH / "Stacked files of car",
    "3-W": BASE_PATH / "stacked files od Tuk-Tuk",
    "MTW": BASE_PATH / "Stacked files of Motorcycles",
    "Bus": BASE_PATH / "stacked files for the bus"
}

# ============================================================
# 2. READ ACCELERATION DATA
# ============================================================

data = []

for vehicle, folder in folders.items():

    for file in folder.glob("*.csv"):

        df = pd.read_csv(file, low_memory=False)

        # Remove spaces from column names
        df.columns = df.columns.str.strip()

        if "Tan. Acc. [ms-2]" in df.columns:

            acceleration = pd.to_numeric(
                df["Tan. Acc. [ms-2]"],
                errors="coerce"
            ).dropna()

            for value in acceleration:
                data.append([vehicle, value])

# Create dataframe
acc_data = pd.DataFrame(
    data,
    columns=["Vehicle", "Acceleration"]
)

print("=" * 70)
print("QUESTION 1 - ACCELERATION ANALYSIS")
print("=" * 70)

print(f"\nTotal acceleration observations: {len(acc_data):,}")

# ============================================================
# 3. CREATE VEHICLE GROUPS
# ============================================================

car = acc_data[acc_data["Vehicle"] == "Car"]["Acceleration"]
three_w = acc_data[acc_data["Vehicle"] == "3-W"]["Acceleration"]
mtw = acc_data[acc_data["Vehicle"] == "MTW"]["Acceleration"]
bus = acc_data[acc_data["Vehicle"] == "Bus"]["Acceleration"]

groups = {
    "Car": car,
    "3-W": three_w,
    "MTW": mtw,
    "Bus": bus
}

# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("TABLE 1: DESCRIPTIVE STATISTICS")
print("=" * 70)

descriptive = []

for name, group in groups.items():

    descriptive.append({
        "Vehicle_Type": name,
        "N": len(group),
        "Mean": group.mean(),
        "SD": group.std(),
        "Median": group.median(),
        "Min": group.min(),
        "Max": group.max()
    })

descriptive_df = pd.DataFrame(descriptive)

print(descriptive_df.round(2).to_string(index=False))

# ============================================================
# 5. HYPOTHESES
# ============================================================

print("\n" + "=" * 70)
print("HYPOTHESES")
print("=" * 70)

print("\nH0: The mean accelerations of Car, 3-W, MTW and Bus are equal.")

print(
    "H1: At least one vehicle type has a different mean acceleration."
)

# ============================================================
# 6. ONE-WAY ANOVA
# ============================================================

F, p = f_oneway(
    car,
    three_w,
    mtw,
    bus
)

print("\n" + "=" * 70)
print("ONE-WAY ANOVA RESULT")
print("=" * 70)

print(f"F-statistic = {F:.6f}")

if p < 0.001:
    print("p-value = <0.001")
else:
    print(f"p-value = {p:.6f}")

print("Alpha = 0.05")

if p < 0.05:
    print("Decision = Reject H0")
else:
    print("Decision = Fail to reject H0")

# ============================================================
# 7. TUKEY HSD PAIRWISE TEST
# ============================================================

print("\n" + "=" * 70)
print("TABLE 2: TUKEY HSD PAIRWISE COMPARISON")
print("=" * 70)

tukey = pairwise_tukeyhsd(
    endog=acc_data["Acceleration"],
    groups=acc_data["Vehicle"],
    alpha=0.05
)

print(tukey)

# Convert Tukey result into dataframe
tukey_table = pd.DataFrame(
    tukey._results_table.data[1:],
    columns=tukey._results_table.data[0]
)

# ============================================================
# 8. PAIRWISE INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION FROM PAIRWISE TEST")
print("=" * 70)

for _, row in tukey_table.iterrows():

    group1 = row["group1"]
    group2 = row["group2"]

    if row["reject"] == True:

        print(
            f"{group1} vs {group2}: Significantly different"
        )

    else:

        print(
            f"{group1} vs {group2}: Not significantly different"
        )

# ============================================================
# 9. GRAPH - MEAN ACCELERATION
# ============================================================

plt.figure(figsize=(8, 6))

plt.bar(
    descriptive_df["Vehicle_Type"],
    descriptive_df["Mean"],
    yerr=descriptive_df["SD"],
    capsize=5
)

plt.xlabel("Vehicle Type")
plt.ylabel("Mean Tangential Acceleration (m/s²)")
plt.title("Mean Acceleration by Vehicle Type")

plt.tight_layout()

graph_path = BASE_PATH / "Q1_Mean_Acceleration_by_Vehicle_Type.png"

plt.savefig(
    graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nGraph saved at:")
print(graph_path)

# ============================================================
# 10. GRAPH INSIGHT
# ============================================================

highest = descriptive_df.loc[
    descriptive_df["Mean"].idxmax()
]

lowest = descriptive_df.loc[
    descriptive_df["Mean"].idxmin()
]

print("\n" + "=" * 70)
print("GRAPH INSIGHT")
print("=" * 70)

print(
    f"{highest['Vehicle_Type']} has the highest mean acceleration "
    f"({highest['Mean']:.2f} m/s²), while "
    f"{lowest['Vehicle_Type']} has the lowest mean acceleration "
    f"({lowest['Mean']:.2f} m/s²)."
)

print(
    "The graph shows differences in the average acceleration among "
    "the vehicle types. The error bars represent ±1 standard deviation "
    "and show the variation in acceleration observations."
)

# ============================================================
# 11. FINAL CONCLUSION
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULT AND CONCLUSION")
print("=" * 70)

if p < 0.05:

    print(
        "\nSince p < 0.05, the null hypothesis is rejected."
    )

    print(
        "There is strong statistical evidence that the mean "
        "accelerations of Car, 3-W, MTW and Bus are significantly "
        "different overall."
    )

    print(
        "The Tukey HSD test identifies which specific pairs of "
        "vehicle types have significantly different accelerations."
    )

else:

    print(
        "\nSince p >= 0.05, the null hypothesis is not rejected."
    )

    print(
        "There is not enough statistical evidence to conclude "
        "that the mean accelerations of the vehicle types are "
        "significantly different."
    )

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
