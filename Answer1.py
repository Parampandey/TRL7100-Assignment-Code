import zipfile
import pandas as pd
import matplotlib.pyplot as plt

zip_path = r"C:\Users\admin\Downloads\TRL7100 Assignment 1 Data.zip"

with zipfile.ZipFile(zip_path, "r") as z:
    csv_files = [
        f for f in z.namelist()
        if f.lower().endswith(".csv")
        and "_file_1" in f.lower()
    ]

    print("CSV file found:")

    for f in csv_files:
        print(f)

    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV found inside _file_1.")

    csv_file = csv_files[0]

    with z.open(csv_file) as file:
        df = pd.read_csv(file)

df.columns = df.columns.str.strip()

print("\nActual column names:")

for i, col in enumerate(df.columns):
    print(i, repr(col))

type_column = None

for col in df.columns:

    cleaned = (
        str(col)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace(".", "")
    )

    if cleaned == "type":
        type_column = col
        break

if type_column is None:

    print("\nCould not automatically find the Type column.")
    print("\nAvailable columns are:")
    print(df.columns.tolist())

    raise KeyError("Vehicle Type column not found.")


print("\nVehicle type column found:")
print(repr(type_column))

vehicle_counts = df[type_column].value_counts(dropna=False)

total_vehicles = vehicle_counts.sum()

vehicle_percentage = (
    vehicle_counts / total_vehicles
) * 100

vehicle_distribution = pd.DataFrame({

    "Vehicle Type": vehicle_counts.index,

    "Number of Vehicles": vehicle_counts.values,

    "Percentage (%)": vehicle_percentage.values

})

vehicle_distribution["Percentage (%)"] = (
    vehicle_distribution["Percentage (%)"].round(2)
)

print("\n==============================================")
print("PERCENTAGE DISTRIBUTION OF VEHICLE TYPES")
print("==============================================")

print(
    vehicle_distribution.to_string(index=False)
)

print("\nTotal number of vehicles:", total_vehicles)

output_csv = r"C:\Users\admin\Downloads\vehicle_type_distribution.csv"

vehicle_distribution.to_csv(
    output_csv,
    index=False
)

print("\nResult table saved at:")
print(output_csv)

plt.figure(figsize=(9, 7))

wedges, texts, autotexts = plt.pie(

    vehicle_counts.values,

    autopct="%1.1f%%",

    startangle=90,

    pctdistance=0.70
)

for i, text in enumerate(autotexts):

    vehicle_type = str(vehicle_counts.index[i]).strip().upper()

    if vehicle_type in ["BUS", "VAN"]:

        # Smaller font only for BUS and VAN
        text.set_fontsize(7)

        text.set_fontweight("bold")

    else:

        # Normal font for all other vehicle types
        text.set_fontsize(10)

        text.set_fontweight("bold")

plt.legend(

    wedges,

    vehicle_counts.index,

    title="Vehicle Type",

    loc="center left",

    bbox_to_anchor=(1, 0.5)
)

plt.title(

    "Percentage Distribution of Vehicle Types",

    fontsize=15
)

plt.axis("equal")

plt.tight_layout()

plot_path = r"C:\Users\admin\Downloads\vehicle_type_distribution.png"

plt.savefig(

    plot_path,

    dpi=300,

    bbox_inches="tight"
)

print("\nPlot saved at:")
print(plot_path)

plt.show()
