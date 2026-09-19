

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2




BASE_PATH = Path(
    r"C:\Users\admin\Desktop\TRIP\Mathematical Modal-Sai Chand-3-C\TRL-700 DATA FILES"
)


# ================================================================
# 2. VEHICLE FOLDERS
# ================================================================

folders = {
    "Car": BASE_PATH / "Stacked files of car",
    "3-W": BASE_PATH / "stacked files od Tuk-Tuk",
    "MTW": BASE_PATH / "Stacked files of Motorcycles",
    "Bus": BASE_PATH / "stacked files for the bus",
    "Light Truck": BASE_PATH / "Stacked files of Light Truck",
    "Van": BASE_PATH / "Stacked files of Van"
}


# ================================================================
# 3. SAFE VARIANCE
# ================================================================

SAFE_VARIANCE = 100       # (km/h)^2
ALPHA = 0.05


# ================================================================
# 4. OBJECTIVE AND METHODOLOGY
# ================================================================

print("=" * 80)
print("ANSWER E (4)")
print("=" * 80)

print("\nOBJECTIVE AND METHODOLOGY")
print("-" * 80)

print(
    "This analysis tests whether the speed variance of each vehicle "
    "type is within the safe variance limit of 100 (km/h)^2. Detailed "
    "trajectory speed data (Speed [km/h]) are used to calculate the "
    "sample variance for each vehicle type. A one-sample Chi-square "
    "test for variance is conducted separately for each vehicle type "
    "at a 5% significance level (α = 0.05). The observed variance is "
    "compared with the safe variance limit of 100 (km/h)^2. A graph "
    "is also used to compare the observed variance of each vehicle "
    "type with the safe limit."
)


# ================================================================
# 5. HYPOTHESES
# ================================================================

print("\n" + "=" * 80)
print("HYPOTHESES")
print("=" * 80)

print(
    "\nH₀: The speed variance is within the safe limit of "
    "100 (km/h)^2."
)

print(
    "H₁: The speed variance is greater than the safe limit of "
    "100 (km/h)^2."
)


# ================================================================
# 6. READ SPEED DATA
# ================================================================

print("\n" + "=" * 80)
print("READING SPEED DATA")
print("=" * 80)

data = []

for vehicle, folder in folders.items():

    csv_files = list(folder.glob("*.csv"))

    print(f"{vehicle}: {len(csv_files)} CSV files")

    for file in csv_files:

        try:

            df = pd.read_csv(file, low_memory=False)

            # Remove spaces from column names
            df.columns = df.columns.str.strip()

            if "Speed [km/h]" not in df.columns:
                continue

            speed = pd.to_numeric(
                df["Speed [km/h]"],
                errors="coerce"
            ).dropna()

            for value in speed:

                data.append([
                    vehicle,
                    value
                ])

        except Exception as e:

            print(
                f"Error reading {file.name}: {e}"
            )


speed_data = pd.DataFrame(
    data,
    columns=["Vehicle", "Speed"]
)

print(
    f"\nTotal speed observations = {len(speed_data):,}"
)


# ================================================================
# 7. CALCULATE DESCRIPTIVE STATISTICS
# ================================================================

vehicle_order = [
    "Car",
    "3-W",
    "MTW",
    "Bus",
    "Light Truck",
    "Van"
]

descriptive = []

for vehicle in vehicle_order:

    group = speed_data[
        speed_data["Vehicle"] == vehicle
    ]["Speed"].dropna()

    descriptive.append({
        "Vehicle Type": vehicle,
        "N": len(group),
        "Mean Speed": group.mean(),
        "SD": group.std(),
        "Variance": group.var()
    })

descriptive_df = pd.DataFrame(descriptive)


# ================================================================
# 8. TABLE 1 - DESCRIPTIVE STATISTICS
# ================================================================

print("\n" + "=" * 80)
print("TABLE 1: DESCRIPTIVE STATISTICS")
print("=" * 80)

print(
    descriptive_df.round(2).to_string(index=False)
)


# ================================================================
# 9. CHI-SQUARE TEST FOR VARIANCE
# ================================================================
#
# Test statistic:
#
# χ² = (n - 1) × s² / σ₀²
#
# where:
# n  = sample size
# s² = observed sample variance
# σ₀² = safe variance = 100
#
# Since H1 is "variance > 100", this is a RIGHT-TAILED test.
# ================================================================

results = []

for vehicle in vehicle_order:

    group = speed_data[
        speed_data["Vehicle"] == vehicle
    ]["Speed"].dropna()

    n = len(group)

    variance = group.var()

    # Chi-square statistic
    chi_square = (
        (n - 1) * variance / SAFE_VARIANCE
    )

    df = n - 1

    # Right-tail p-value
    p_value = chi2.sf(
        chi_square,
        df
    )

    # Critical value
    critical_value = chi2.ppf(
        1 - ALPHA,
        df
    )

    if p_value < ALPHA:

        decision = "Reject H0"
        result = "Above safe variance"

    else:

        decision = "Fail to reject H0"
        result = "Within safe variance"

    results.append({

        "Vehicle Type": vehicle,
        "N": n,
        "Observed Variance": variance,
        "Safe Variance": SAFE_VARIANCE,
        "Chi-square": chi_square,
        "p-value": p_value,
        "Alpha": ALPHA,
        "Decision": decision,
        "Result": result

    })


results_df = pd.DataFrame(results)


# ================================================================
# 10. TABLE 2 - CHI-SQUARE TEST RESULT
# ================================================================

print("\n" + "=" * 80)
print("TABLE 2: CHI-SQUARE TEST FOR VARIANCE")
print("=" * 80)

display_table = results_df.copy()

display_table["p-value"] = display_table["p-value"].apply(
    lambda x: "<0.001" if x < 0.001 else round(x, 4)
)

print(
    display_table[
        [
            "Vehicle Type",
            "N",
            "Observed Variance",
            "Safe Variance",
            "Chi-square",
            "p-value",
            "Alpha",
            "Decision"
        ]
    ].round(2).to_string(index=False)
)


# ================================================================
# 11. RESULT FOR EACH VEHICLE TYPE
# ================================================================

print("\n" + "=" * 80)
print("TEST RESULT AND INTERPRETATION")
print("=" * 80)

for _, row in results_df.iterrows():

    vehicle = row["Vehicle Type"]
    variance = row["Observed Variance"]
    p = row["p-value"]

    if p < 0.001:
        p_text = "<0.001"
    else:
        p_text = f"{p:.4f}"

    print(
        f"\n{vehicle}:"
    )

    print(
        f"Observed variance = {variance:.2f} (km/h)^2"
    )

    print(
        f"p-value = {p_text}"
    )

    print(
        f"Decision = {row['Decision']}"
    )

    print(
        f"Result = {row['Result']}"
    )


# ================================================================
# 12. COUNT VEHICLE TYPES ABOVE SAFE VARIANCE
# ================================================================

above_safe = results_df[
    results_df["Observed Variance"] > SAFE_VARIANCE
]

within_safe = results_df[
    results_df["Observed Variance"] <= SAFE_VARIANCE
]


# ================================================================
# 13. GRAPH
# ================================================================

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Vehicle Type"],
    results_df["Observed Variance"]
)

plt.axhline(
    SAFE_VARIANCE,
    linestyle="--",
    linewidth=2,
    label="Safe variance = 100 (km/h)²"
)

plt.xlabel("Vehicle Type")
plt.ylabel("Speed Variance (km/h)²")
plt.title("Observed Speed Variance vs Safe Variance Limit")

plt.legend()

plt.tight_layout()

graph_path = BASE_PATH / "Q1_Speed_Variance_Safe_Limit.png"

plt.savefig(
    graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nGraph saved at:")
print(graph_path)


# ================================================================
# 14. GRAPH INSIGHT
# ================================================================

highest = results_df.loc[
    results_df["Observed Variance"].idxmax()
]

lowest = results_df.loc[
    results_df["Observed Variance"].idxmin()
]

print("\n" + "=" * 80)
print("GRAPH INSIGHT")
print("=" * 80)

print(
    f"{highest['Vehicle Type']} has the highest observed speed "
    f"variance of {highest['Observed Variance']:.2f} (km/h)^2."
)

print(
    f"{lowest['Vehicle Type']} has the lowest observed speed "
    f"variance of {lowest['Observed Variance']:.2f} (km/h)^2."
)

print(
    f"The safe variance limit is {SAFE_VARIANCE} (km/h)^2."
)

print(
    f"{len(above_safe)} vehicle type(s) have observed variance "
    "above the safe limit."
)

print(
    f"{len(within_safe)} vehicle type(s) have observed variance "
    "within the safe limit."
)


# ================================================================
# 15. FINAL CONCLUSION
# ================================================================

print("\n" + "=" * 80)
print("FINAL RESULT AND CONCLUSION")
print("=" * 80)

if len(above_safe) == 0:

    print(
        "\nAll vehicle types have observed speed variance "
        "within the safe limit of 100 (km/h)^2."
    )

    print(
        "Therefore, the observed data support the statement "
        "that all vehicle types are within the safe speed variance."
    )

else:

    print(
        f"\n{len(above_safe)} out of {len(vehicle_order)} vehicle "
        "types have observed speed variance above 100 (km/h)^2."
    )

    print(
        "Therefore, complete compliance with the safe variance "
        "limit is not observed for all vehicle types."
    )

    print(
        "Hence, the statement 'All vehicle types are observed to "
        "be within the safe variance of speeds of 100 (km/h)^2' "
        "is not supported by the observed data."
    )

    print("\nVehicle types above the safe variance:")

    for vehicle in above_safe["Vehicle Type"]:
        print(f"- {vehicle}")


# ================================================================
# 16. SAVE RESULTS TO EXCEL
# ================================================================

excel_path = BASE_PATH / "Q1_Speed_Variance_Analysis.xlsx"

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as writer:

    descriptive_df.round(4).to_excel(
        writer,
        sheet_name="Descriptive Statistics",
        index=False
    )

    results_df.round(6).to_excel(
        writer,
        sheet_name="Variance Test",
        index=False
    )


# ================================================================
# 17. COMPLETE
# ================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print("\nExcel file saved at:")
print(excel_path)

print("\nGraph saved at:")
print(graph_path)

print("=" * 80)
