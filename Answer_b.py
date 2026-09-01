import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
file_path = Path(
    r"C:\Users\admin\Desktop\TRIP\Mathematical Modal-Sai Chand-3-C\TRL-700 DATA FILES\Raw_data_file_1"
)

# If Raw_data_file_1 is a folder, find the CSV file inside
if file_path.is_dir():

    csv_files = list(file_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No CSV file found inside Raw_data_file_1."
        )

    file_path = csv_files[0]

print("Reading file:")
print(file_path)

df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()

# Clean vehicle type values
df["Type"] = (
    df["Type"]
    .astype(str)
    .str.strip()
)

# Convert speed column to numeric
df["Speed [km/h]"] = pd.to_numeric(
    df["Speed [km/h]"],
    errors="coerce"
)

# Remove missing values
df = df.dropna(
    subset=["Type", "Speed [km/h]"]
)

speed_limits = {
    "Car": 60,
    "Motorcycle": 60,
    "Van": 60,
    "Light Truck": 40,
    "Tuk-Tuk": 40,
    "Bus": 40
}

df["Speed Limit [km/h]"] = (
    df["Type"].map(speed_limits)
)

df = df.dropna(
    subset=["Speed Limit [km/h]"]
)

df["Compliant"] = (
    df["Speed [km/h]"]
    <= df["Speed Limit [km/h]"]
)

result = (
    df.groupby("Type")
      .agg(
          Total_Vehicles=("Type", "size"),
          Compliant_Vehicles=("Compliant", "sum"),
          Speed_Limit=("Speed Limit [km/h]", "first")
      )
      .reset_index()
)

result["Non_Compliant_Vehicles"] = (
    result["Total_Vehicles"]
    - result["Compliant_Vehicles"]
)

result["Compliance_Percentage"] = (
    result["Compliant_Vehicles"]
    / result["Total_Vehicles"]
    * 100
)

vehicle_order = [
    "Car",
    "Motorcycle",
    "Light Truck",
    "Tuk-Tuk",
    "Bus",
    "Van"
]

result["Type"] = pd.Categorical(
    result["Type"],
    categories=vehicle_order,
    ordered=True
)

result = result.sort_values("Type")

print("\n")
print("=" * 80)
print("SPEED LIMIT COMPLIANCE RESULTS")
print("=" * 80)

display_table = result[
    [
        "Type",
        "Speed_Limit",
        "Total_Vehicles",
        "Compliant_Vehicles",
        "Non_Compliant_Vehicles",
        "Compliance_Percentage"
    ]
].copy()

display_table["Compliance_Percentage"] = (
    display_table["Compliance_Percentage"].round(2)
)

print(display_table.to_string(index=False))

output_folder = file_path.parent

table_path = (
    output_folder /
    "speed_limit_compliance_results.csv"
)

display_table.to_csv(
    table_path,
    index=False
)

print("\nResult table saved at:")
print(table_path)

vehicle_colors = {
    "Car": "#1f77b4",          # Blue
    "Motorcycle": "#ff7f0e",   # Orange
    "Light Truck": "#2ca02c",  # Green
    "Tuk-Tuk": "#d62728",      # Red
    "Bus": "#9467bd",          # Purple
    "Van": "#8c564b"           # Brown
}

bar_colors = [
    vehicle_colors[vehicle]
    for vehicle in result["Type"].astype(str)
]

fig, ax = plt.subplots(
    figsize=(11, 7)
)

bars = ax.bar(
    result["Type"].astype(str),
    result["Compliance_Percentage"],
    color=bar_colors,
    width=0.65,
    edgecolor="white",
    linewidth=1.5
)

for bar, percentage in zip(
    bars,
    result["Compliance_Percentage"]
):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f"{percentage:.2f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

ax.set_title(
    "Percentage of Vehicles Complying with the Speed Limit by Vehicle Type",
    fontsize=16,
    fontweight="bold",
    pad=18
)
ax.set_xlabel(
    "Vehicle Type",
    fontsize=12,
    fontweight="bold",
    labelpad=10
)

ax.set_ylabel(
    "Compliance with Speed Limit (%)",
    fontsize=12,
    fontweight="bold",
    labelpad=10
)

ax.set_ylim(0, 110)

ax.set_yticks(
    range(0, 101, 10)
)
ax.grid(
    axis="y",
    linestyle="--",
    linewidth=1.0,
    alpha=0.55
)

# Keep grid behind the bars
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_facecolor("#FAFAFA")

fig.patch.set_facecolor("white")

plt.tight_layout()

plot_path = (
    output_folder /
    "speed_limit_compliance.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)
plt.show()

print("\nGraph saved at:")
print(plot_path)
