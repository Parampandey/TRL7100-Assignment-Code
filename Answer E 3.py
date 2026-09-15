
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import binomtest


# ================================================================
# 1. MAIN DATA PATH
# ================================================================

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
# 3. SPEED LIMITS
# ================================================================
#
# IMPORTANT:
# Change these values if your teacher/class speed-limit table
# specifies different limits.
#
# Unit = km/h
# ================================================================

speed_limits = {
    "Car": 50,
    "3-W": 40,
    "MTW": 50,
    "Bus": 40,
    "Light Truck": 40,
    "Van": 40
}


# ================================================================
# 4. OBJECTIVE AND METHODOLOGY
# ================================================================

print("=" * 80)
print("ANSWER E (3)")
print("=" * 80)

print("\nOBJECTIVE AND METHODOLOGY")
print("-" * 80)

print(
    "This analysis evaluates whether the observed vehicles comply with "
    "their applicable speed limits. The recorded speed of each vehicle "
    "is compared with the corresponding speed limit for its vehicle type. "
    "A vehicle observation is considered compliant if its speed is less "
    "than or equal to the applicable speed limit and non-compliant if "
    "its speed exceeds the limit. The number and percentage of compliant "
    "and non-compliant observations are calculated for each vehicle type "
    "and for all vehicles combined. A bar chart is used to compare the "
    "compliance percentages. A one-sample proportion test is also used "
    "to assess whether the observed compliance is statistically consistent "
    "with complete compliance."
)


# ================================================================
# 5. HYPOTHESES
# ================================================================

print("\n" + "=" * 80)
print("HYPOTHESES")
print("=" * 80)

print(
    "\nH₀: All observed vehicle speeds are within their applicable "
    "speed limits."
)

print(
    "H₁: Some vehicle observations exceed their applicable speed limits."
)


# ================================================================
# 6. READ SPEED DATA
# ================================================================

print("\n" + "=" * 80)
print("READING DATA")
print("=" * 80)

data = []

for vehicle, folder in folders.items():

    csv_files = list(folder.glob("*.csv"))

    print(f"{vehicle}: {len(csv_files)} CSV files")

    for file in csv_files:

        try:

            df = pd.read_csv(file, low_memory=False)

            # Remove extra spaces from column names
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


# Create dataframe
speed_data = pd.DataFrame(
    data,
    columns=["Vehicle", "Speed"]
)


print(
    f"\nTotal speed observations = {len(speed_data):,}"
)


# ================================================================
# 7. CALCULATE COMPLIANCE
# ================================================================

results = []

vehicle_order = [
    "Car",
    "3-W",
    "MTW",
    "Bus",
    "Light Truck",
    "Van"
]


for vehicle in vehicle_order:

    group = speed_data[
        speed_data["Vehicle"] == vehicle
    ]

    limit = speed_limits[vehicle]

    total = len(group)

    # Speed <= limit = compliant
    compliant = (group["Speed"] <= limit).sum()

    # Speed > limit = non-compliant
    non_compliant = (group["Speed"] > limit).sum()

    compliance_percent = (
        compliant / total * 100
        if total > 0
        else 0
    )

    non_compliance_percent = (
        non_compliant / total * 100
        if total > 0
        else 0
    )

    results.append({
        "Vehicle Type": vehicle,
        "Speed Limit (km/h)": limit,
        "Total N": total,
        "Compliant N": compliant,
        "Non-Compliant N": non_compliant,
        "Compliance %": compliance_percent,
        "Non-Compliance %": non_compliance_percent
    })


compliance_df = pd.DataFrame(results)


# ================================================================
# 8. TABLE 1 - COMPLIANCE RESULTS
# ================================================================

print("\n" + "=" * 80)
print("TABLE 1: SPEED LIMIT COMPLIANCE")
print("=" * 80)

print(
    compliance_df.round(2).to_string(index=False)
)


# ================================================================
# 9. OVERALL COMPLIANCE
# ================================================================

total_all = compliance_df["Total N"].sum()

compliant_all = compliance_df["Compliant N"].sum()

non_compliant_all = compliance_df["Non-Compliant N"].sum()

overall_compliance = (
    compliant_all / total_all * 100
)

overall_non_compliance = (
    non_compliant_all / total_all * 100
)


print("\n" + "=" * 80)
print("OVERALL COMPLIANCE")
print("=" * 80)

print(f"Total observations = {total_all:,}")
print(f"Compliant observations = {compliant_all:,}")
print(f"Non-compliant observations = {non_compliant_all:,}")

print(
    f"Overall compliance = {overall_compliance:.2f}%"
)

print(
    f"Overall non-compliance = {overall_non_compliance:.2f}%"
)


# ================================================================
# 10. STATISTICAL TEST
# ================================================================
#
# We test whether the observed data show complete compliance.
#
# H0: p = 1
# H1: p < 1
#
# However, because a test against p=1 is very strict and any
# non-compliant observation makes complete compliance impossible,
# the practical conclusion is primarily based on the observed
# compliance percentage.
# ================================================================

print("\n" + "=" * 80)
print("STATISTICAL TEST OF COMPLIANCE")
print("=" * 80)

print(
    "\nH0: The compliance proportion is 1 (complete compliance)."
)

print(
    "H1: The compliance proportion is less than 1."
)

# Exact binomial test
test = binomtest(
    compliant_all,
    total_all,
    p=1.0,
    alternative="less"
)

p_test = test.pvalue

print(
    f"\nObserved compliance proportion = "
    f"{compliant_all / total_all:.4f}"
)

if p_test < 0.001:

    print("p-value = <0.001")

else:

    print(
        f"p-value = {p_test:.6f}"
    )

print("Alpha = 0.05")

if p_test < 0.05:

    print("Decision = Reject H0")

else:

    print("Decision = Fail to reject H0")


# ================================================================
# 11. STATISTICAL RESULT
# ================================================================

print("\n" + "=" * 80)
print("STATISTICAL RESULT")
print("=" * 80)

if non_compliant_all > 0:

    print(
        f"{non_compliant_all:,} speed observations exceed "
        "their applicable speed limits."
    )

    print(
        f"Therefore, the observed data do not show complete "
        f"compliance. Overall compliance is {overall_compliance:.2f}%."
    )

else:

    print(
        "No speed observations exceed their applicable speed limits."
    )

    print(
        "The observed data show complete compliance."
    )


# ================================================================
# 12. VEHICLE-WISE INTERPRETATION
# ================================================================

print("\n" + "=" * 80)
print("VEHICLE-WISE RESULT")
print("=" * 80)

for _, row in compliance_df.iterrows():

    vehicle = row["Vehicle Type"]
    limit = row["Speed Limit (km/h)"]
    compliance = row["Compliance %"]
    non_compliance = row["Non-Compliant N"]

    if non_compliance == 0:

        print(
            f"{vehicle}: {compliance:.2f}% compliance "
            f"(no observations above {limit} km/h)"
        )

    else:

        print(
            f"{vehicle}: {compliance:.2f}% compliance "
            f"({int(non_compliance):,} observations exceeded "
            f"{limit} km/h)"
        )


# ================================================================
# 13. FIND HIGHEST AND LOWEST COMPLIANCE
# ================================================================

highest = compliance_df.loc[
    compliance_df["Compliance %"].idxmax()
]

lowest = compliance_df.loc[
    compliance_df["Compliance %"].idxmin()
]


# ================================================================
# 14. GRAPH
# ================================================================

plt.figure(figsize=(10, 6))

plt.bar(
    compliance_df["Vehicle Type"],
    compliance_df["Compliance %"]
)

plt.axhline(
    100,
    linestyle="--",
    linewidth=1.5,
    label="100% compliance"
)

plt.xlabel("Vehicle Type")
plt.ylabel("Compliance (%)")
plt.title("Speed Limit Compliance by Vehicle Type")

plt.ylim(0, 110)

plt.legend()

plt.tight_layout()

graph_path = BASE_PATH / "Q1_Speed_Limit_Compliance.png"

plt.savefig(
    graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nGraph saved at:")
print(graph_path)


# ================================================================
# 15. GRAPH INSIGHT
# ================================================================

print("\n" + "=" * 80)
print("GRAPH INSIGHT")
print("=" * 80)

print(
    f"{highest['Vehicle Type']} has the highest compliance rate "
    f"at {highest['Compliance %']:.2f}%."
)

print(
    f"{lowest['Vehicle Type']} has the lowest compliance rate "
    f"at {lowest['Compliance %']:.2f}%."
)

print(
    f"The overall compliance rate is {overall_compliance:.2f}%, "
    f"while {overall_non_compliance:.2f}% of observations exceed "
    "their applicable speed limits."
)


# ================================================================
# 16. FINAL CONCLUSION
# ================================================================

print("\n" + "=" * 80)
print("FINAL RESULT AND CONCLUSION")
print("=" * 80)

if non_compliant_all == 0:

    print(
        "\nAll observed vehicle speed observations are within "
        "their applicable speed limits."
    )

    print(
        "Therefore, the observed data support the statement that "
        "all vehicle types comply with the speed limit."
    )

else:

    print(
        f"\nOverall compliance = {overall_compliance:.2f}%."
    )

    print(
        f"Overall non-compliance = {overall_non_compliance:.2f}%."
    )

    print(
        f"{non_compliant_all:,} observations exceed their "
        "applicable speed limits."
    )

    print(
        "\nTherefore, complete compliance is not observed for "
        "all vehicle types."
    )

    print(
        "Hence, the statement 'All vehicle types are observed "
        "to comply with the speed limit' is NOT supported by "
        "the observed data."
    )

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
