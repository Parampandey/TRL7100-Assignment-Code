
from pathlib import Path
import sys
import subprocess
import warnings
import re
import html
import math

# -------------------- Dependency check ---------------------------
# pandas/openpyxl/matplotlib/scipy are normally installed.
# If openpyxl is missing, try installing it automatically so that
# the Excel workbook can also be produced.
REQUIRED = {
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "openpyxl": "openpyxl",
}

def ensure_packages():
    missing = []
    for import_name, package_name in REQUIRED.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        print("\nMissing Python package(s):", ", ".join(missing))
        print("Attempting automatic installation...\n")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", *missing]
            )
        except Exception as e:
            print("\nAutomatic installation failed.")
            print("Run this once in your terminal:")
            print(f"  {sys.executable} -m pip install {' '.join(missing)}")
            raise e

ensure_packages()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

# ================================================================
# 1. MAIN DATA PATH
# ================================================================

DATA_ROOT = Path(
    r"C:\Users\admin\Desktop\TRIP\Mathematical Modal-Sai Chand-3-C"
    r"\TRL-700 DATA FILES"
)

# ================================================================
# 2. EXACT SIX VEHICLE FOLDERS
# ================================================================
# Do NOT replace this with automatic "all folders containing CSVs".
# This is the correction that prevents Raw_data_file_1 from entering
# the Question (c) analysis.

VEHICLE_FOLDERS = {
    "Bus": "stacked files for the bus",
    "Tuk-Tuk": "stacked files od Tuk-Tuk",
    "Car": "Stacked files of car",
    "Light Truck": "Stacked files of Light Truck",
    "Motorcycles": "stacked files of Motorcycles",
    "Van": "Stacked files of Van",
}

VEHICLE_ORDER = [
    "Bus",
    "Tuk-Tuk",
    "Car",
    "Light Truck",
    "Motorcycles",
    "Van",
]

# Explicitly excluded folders/files.
EXCLUDED_NAMES = {
    "raw_data_file_1",
    "trl7100_question_c_final",
}

# ================================================================
# 3. OUTPUT FOLDERS
# ================================================================

OUT = DATA_ROOT / "TRL7100_Question_C_FINAL"

TABLES = OUT / "01_Detailed_Tables"
FIGURES = OUT / "02_High_Quality_Figures"
PROCESSED = OUT / "03_Clean_and_Processed_Data"
REPORT = OUT / "04_Final_Report"
TEXT = OUT / "05_Assignment_Ready_Text"

for p in [OUT, TABLES, FIGURES, PROCESSED, REPORT, TEXT]:
    p.mkdir(parents=True, exist_ok=True)

# ================================================================
# 4. COLUMN NAMES
# ================================================================

TYPE_COL = "Type"
TRACK_COL = "Track ID"
SPEED_COL = "Avg. Speed [km/h]"
ACC_COL = "Tan. Acc. [ms-2]"

# ================================================================
# 5. PLOT SETTINGS
# ================================================================

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 400,
    "font.size": 10,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

# ================================================================
# 6. HELPER FUNCTIONS
# ================================================================

def find_col(df, target):
    """Find a column by exact name, then case-insensitive name."""
    if target in df.columns:
        return target

    target_clean = str(target).strip().lower()

    for col in df.columns:
        if str(col).strip().lower() == target_clean:
            return col

    return None


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def safe_filename(value):
    return re.sub(r'[<>:"/\\|?*]+', "_", str(value)).strip()[:120]


def read_csv_file(path):
    """Read CSV with two common encodings."""
    try:
        return pd.read_csv(
            path,
            encoding="utf-8-sig",
            low_memory=False
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            path,
            encoding="latin1",
            low_memory=False
        )


def round_numeric(df, decimals=4):
    result = df.copy()
    numeric_cols = result.select_dtypes(include=np.number).columns
    result[numeric_cols] = result[numeric_cols].round(decimals)
    return result


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(
        FIGURES / filename,
        dpi=400,
        bbox_inches="tight"
    )
    plt.close(fig)


def descriptive_statistics(series):
    """
    Calculate detailed descriptive statistics.
    Sample SD and sample variance use ddof=1.
    """
    x = pd.Series(series).dropna().astype(float)
    n = len(x)

    columns = [
        "N",
        "Mean",
        "Median",
        "Std. Deviation",
        "Variance",
        "Minimum",
        "Q1 (25%)",
        "Q3 (75%)",
        "Maximum",
        "Range",
        "IQR",
        "CV (%)",
        "Skewness",
        "95% CI Lower",
        "95% CI Upper",
    ]

    if n == 0:
        return {column: np.nan for column in columns}

    mean = x.mean()
    median = x.median()

    if n >= 2:
        sd = x.std(ddof=1)
        variance = x.var(ddof=1)
        standard_error = sd / np.sqrt(n)
        t_critical = stats.t.ppf(0.975, n - 1)
        ci_lower = mean - t_critical * standard_error
        ci_upper = mean + t_critical * standard_error
    else:
        sd = np.nan
        variance = np.nan
        ci_lower = np.nan
        ci_upper = np.nan

    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1

    if n >= 3 and x.nunique() > 1:
        skewness = stats.skew(x, bias=False)
    else:
        skewness = np.nan

    if n >= 2 and mean != 0:
        cv = 100 * sd / abs(mean)
    else:
        cv = np.nan

    return {
        "N": n,
        "Mean": mean,
        "Median": median,
        "Std. Deviation": sd,
        "Variance": variance,
        "Minimum": x.min(),
        "Q1 (25%)": q1,
        "Q3 (75%)": q3,
        "Maximum": x.max(),
        "Range": x.max() - x.min(),
        "IQR": iqr,
        "CV (%)": cv,
        "Skewness": skewness,
        "95% CI Lower": ci_lower,
        "95% CI Upper": ci_upper,
    }


def outlier_statistics(series):
    """1.5 x IQR outlier rule."""
    x = pd.Series(series).dropna().astype(float)
    n = len(x)

    if n == 0:
        return {
            "N": 0,
            "Q1": np.nan,
            "Q3": np.nan,
            "IQR": np.nan,
            "Lower Fence": np.nan,
            "Upper Fence": np.nan,
            "Outlier Count": 0,
            "Outlier (%)": np.nan,
        }

    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    outlier_mask = (x < lower_fence) | (x > upper_fence)
    outlier_count = int(outlier_mask.sum())

    return {
        "N": n,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower Fence": lower_fence,
        "Upper Fence": upper_fence,
        "Outlier Count": outlier_count,
        "Outlier (%)": 100 * outlier_count / n,
    }


def actual_outliers(df, value_column, vehicle_types):
    """Return actual observations outside the 1.5 x IQR fences."""
    records = []

    for vehicle_type in vehicle_types:
        subset = df[
            df["Vehicle Type"] == vehicle_type
        ].copy()

        if subset.empty:
            continue

        x = subset[value_column].dropna()

        if x.empty:
            continue

        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1

        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        mask = (
            (subset[value_column] < lower_fence)
            | (subset[value_column] > upper_fence)
        )

        for _, row in subset.loc[mask].iterrows():
            direction = (
                "Low"
                if row[value_column] < lower_fence
                else "High"
            )

            records.append({
                "Vehicle Type": vehicle_type,
                "Track ID": row["Track ID"],
                "Source Folder": row["Source Folder"],
                "Source File": row["Source File"],
                "Observed Value": row[value_column],
                "Lower Fence": lower_fence,
                "Upper Fence": upper_fence,
                "Direction": direction,
            })

    return pd.DataFrame(records)


def make_stats_dataframe(data_by_vehicle):
    records = []

    for vehicle_type in VEHICLE_ORDER:
        if vehicle_type in data_by_vehicle:
            row = {
                "Vehicle Type": vehicle_type,
                **descriptive_statistics(
                    data_by_vehicle[vehicle_type]
                ),
            }
        else:
            row = {
                "Vehicle Type": vehicle_type,
                **descriptive_statistics([]),
            }

        records.append(row)

    return pd.DataFrame(records)


def make_outlier_dataframe(data_by_vehicle):
    records = []

    for vehicle_type in VEHICLE_ORDER:
        if vehicle_type in data_by_vehicle:
            row = {
                "Vehicle Type": vehicle_type,
                **outlier_statistics(
                    data_by_vehicle[vehicle_type]
                ),
            }
        else:
            row = {
                "Vehicle Type": vehicle_type,
                **outlier_statistics([]),
            }

        records.append(row)

    return pd.DataFrame(records)


def write_text_file(path, text):
    path.write_text(text, encoding="utf-8")


# ================================================================
# 7. CHECK ROOT AND EXACT FOLDERS
# ================================================================

if not DATA_ROOT.exists():
    raise FileNotFoundError(
        "\nDATA_ROOT was not found.\n\n"
        f"Expected folder:\n{DATA_ROOT}\n\n"
        "Please check DATA_ROOT at the top of this script."
    )

print("\n" + "=" * 100)
print("TRL7100 - ASSIGNMENT 1 - QUESTION (c)")
print("DETAILED DESCRIPTIVE STATISTICS OF SPEED AND ACCELERATION")
print("=" * 100)

print("\nMain data folder:")
print(DATA_ROOT)

print("\nONLY these six vehicle folders will be analysed:")
print("-" * 70)

vehicle_folder_info = []
all_csv_files = []

for vehicle_type in VEHICLE_ORDER:
    folder_name = VEHICLE_FOLDERS[vehicle_type]
    folder_path = DATA_ROOT / folder_name

    if not folder_path.exists():
        raise FileNotFoundError(
            f"\nRequired vehicle folder was not found:\n"
            f"Vehicle type: {vehicle_type}\n"
            f"Expected folder: {folder_path}\n\n"
            "Check the folder name in VEHICLE_FOLDERS."
        )

    csv_files = sorted(
        [
            p for p in folder_path.rglob("*.csv")
            if p.is_file()
            and not p.name.startswith("~$")
            and OUT not in p.parents
        ]
    )

    vehicle_folder_info.append(
        {
            "Vehicle Type": vehicle_type,
            "Folder": folder_name,
            "Folder Path": str(folder_path),
            "CSV Files Found": len(csv_files),
        }
    )

    all_csv_files.extend(
        [(vehicle_type, folder_path, p) for p in csv_files]
    )

    print(
        f"{vehicle_type:<15} : {len(csv_files):>5} CSV files"
    )

print("-" * 70)
print(
    "IMPORTANT: Raw_data_file_1 is NOT included in the above list."
)

expected_total_files = sum(
    x["CSV Files Found"] for x in vehicle_folder_info
)

print(
    f"\nTotal CSV files to process: {expected_total_files:,}"
)

# ================================================================
# 8. READ ALL SIX VEHICLE FOLDERS
# ================================================================

speed_parts = []
acceleration_parts = []
file_log = []

raw_rows_total = 0
successful_files = 0
failed_files = 0

# Data-quality counters
missing_type = 0
missing_track = 0
missing_speed = 0
missing_acceleration = 0
type_mismatch_files = 0

print("\n" + "=" * 100)
print("READING CSV FILES")
print("=" * 100)
print(
    "Progress is displayed every 100 files to keep the console readable."
)

processed_count = 0

for vehicle_type, folder_path, csv_path in all_csv_files:

    processed_count += 1

    try:
        df = read_csv_file(csv_path)

        raw_rows = len(df)
        raw_rows_total += raw_rows

        type_col = find_col(df, TYPE_COL)
        track_col = find_col(df, TRACK_COL)
        speed_col = find_col(df, SPEED_COL)
        acceleration_col = find_col(df, ACC_COL)

        # --------------------------------------------------------
        # Column availability
        # --------------------------------------------------------

        if type_col is None:
            missing_type += 1

        if track_col is None:
            missing_track += 1

        if speed_col is None:
            missing_speed += 1

        if acceleration_col is None:
            missing_acceleration += 1

        # --------------------------------------------------------
        # SPEED
        # --------------------------------------------------------

        speed_added = 0
        acceleration_added = 0
        type_values = []

        if (
            type_col is not None
            and track_col is not None
            and speed_col is not None
        ):

            z = df[
                [type_col, track_col, speed_col]
            ].copy()

            z.columns = [
                "Original Type",
                "Track ID",
                "Average Speed",
            ]

            z["Original Type"] = (
                z["Original Type"]
                .astype(str)
                .str.strip()
            )

            # The exact source folder determines the final vehicle
            # category. This safely handles Motorcycle/Motorcycles
            # spelling differences in the Type column.
            z["Vehicle Type"] = vehicle_type

            # Track ID is kept as a string to avoid accidental
            # changes such as 001 -> 1.
            z["Track ID"] = (
                z["Track ID"]
                .astype(str)
                .str.strip()
            )

            z["Average Speed"] = numeric(
                z["Average Speed"]
            )

            z = z.dropna(
                subset=[
                    "Vehicle Type",
                    "Track ID",
                    "Average Speed",
                ]
            )

            # Ignore blank/string Track IDs such as "nan".
            z = z[
                ~z["Track ID"].isin(
                    ["", "nan", "None", "NaN"]
                )
            ]

            # Check the original Type column for data quality only.
            # Motorcycle/Motorcycles and common Tuk-Tuk/Light Truck
            # spelling variants are treated as equivalent.
            type_values = sorted(
                set(z["Original Type"].unique())
            )

            aliases = {
                "motorcycle": "motorcycles",
                "motorcycles": "motorcycles",
                "tuk tuk": "tuk-tuk",
                "tuk-tuk": "tuk-tuk",
                "tuktuk": "tuk-tuk",
                "light truck": "light truck",
                "lighttruck": "light truck",
            }

            expected_norm = aliases.get(
                vehicle_type.strip().lower(),
                vehicle_type.strip().lower()
            )

            def type_matches_folder(value):
                value_norm = aliases.get(
                    str(value).strip().lower(),
                    str(value).strip().lower()
                )
                return value_norm == expected_norm

            file_type_matches = (
                len(type_values) == 0
                or all(type_matches_folder(x) for x in type_values)
            )

            if not file_type_matches:
                type_mismatch_files += 1

            # Avg. Speed is vehicle-level.
            # Retain ONE speed value per Track ID INSIDE THIS FILE.
            #
            # Critical:
            # We do NOT globally deduplicate Track ID because
            # Track IDs can restart in different source CSV files.
            z = z.drop_duplicates(
                subset=["Track ID"],
                keep="first"
            )

            z["Source Folder"] = folder_path.name
            z["Source File"] = csv_path.name

            z["Vehicle Key"] = (
                z["Source Folder"].astype(str)
                + " | "
                + z["Source File"].astype(str)
                + " | Track "
                + z["Track ID"].astype(str)
            )

            speed_parts.append(z)
            speed_added = len(z)

        # --------------------------------------------------------
        # TANGENTIAL ACCELERATION
        # --------------------------------------------------------

        if (
            type_col is not None
            and track_col is not None
            and acceleration_col is not None
        ):

            z = df[
                [type_col, track_col, acceleration_col]
            ].copy()

            z.columns = [
                "Original Type",
                "Track ID",
                "Tangential Acceleration",
            ]

            z["Original Type"] = (
                z["Original Type"]
                .astype(str)
                .str.strip()
            )

            # The exact source folder determines the final category.
            z["Vehicle Type"] = vehicle_type

            z["Track ID"] = (
                z["Track ID"]
                .astype(str)
                .str.strip()
            )

            z["Tangential Acceleration"] = numeric(
                z["Tangential Acceleration"]
            )

            z = z.dropna(
                subset=[
                    "Vehicle Type",
                    "Track ID",
                    "Tangential Acceleration",
                ]
            )

            z = z[
                ~z["Track ID"].isin(
                    ["", "nan", "None", "NaN"]
                )
            ]

            # IMPORTANT:
            # No deduplication is performed for acceleration.
            # Every valid trajectory observation is retained.
            z["Source Folder"] = folder_path.name
            z["Source File"] = csv_path.name

            acceleration_parts.append(z)
            acceleration_added = len(z)

        successful_files += 1

        file_log.append({
            "Expected Vehicle Type": vehicle_type,
            "Folder": folder_path.name,
            "File": csv_path.name,
            "Raw Rows": raw_rows,
            "Speed Observations Added": speed_added,
            "Acceleration Observations Added": acceleration_added,
            "Type Column": type_col is not None,
            "Track ID Column": track_col is not None,
            "Speed Column": speed_col is not None,
            "Acceleration Column": acceleration_col is not None,
            "Type Values Found": ", ".join(type_values),
            "Type Matches Expected": (
                len(type_values) == 0
                or all(
                    str(x).strip().lower()
                    == vehicle_type.strip().lower()
                    for x in type_values
                )
            ),
            "Status": "OK",
        })

    except Exception as exc:

        failed_files += 1

        file_log.append({
            "Expected Vehicle Type": vehicle_type,
            "Folder": folder_path.name,
            "File": csv_path.name,
            "Raw Rows": np.nan,
            "Speed Observations Added": 0,
            "Acceleration Observations Added": 0,
            "Type Column": False,
            "Track ID Column": False,
            "Speed Column": False,
            "Acceleration Column": False,
            "Original Type Values Found": "",
            "Type Matches Expected": False,
            "Folder Used For Final Vehicle Type": vehicle_type,
            "Status": f"ERROR: {exc}",
        })

        print(
            f"\nWARNING - Could not read: "
            f"{vehicle_type} / {csv_path.name}\n"
            f"Reason: {exc}"
        )

    if (
        processed_count % 100 == 0
        or processed_count == expected_total_files
    ):
        print(
            f"Processed {processed_count:,} / "
            f"{expected_total_files:,} files"
        )

# ================================================================
# 9. COMBINE PROCESSED DATA
# ================================================================

if not speed_parts:
    raise ValueError(
        "No valid speed observations were found."
    )

if not acceleration_parts:
    raise ValueError(
        "No valid tangential acceleration observations were found."
    )

speed = pd.concat(
    speed_parts,
    ignore_index=True
)

acceleration = pd.concat(
    acceleration_parts,
    ignore_index=True
)

# Safety check.
# Vehicle Key is already unique by source folder + source file + Track ID.
speed = (
    speed
    .drop_duplicates(
        subset=["Vehicle Key"],
        keep="first"
    )
    .reset_index(drop=True)
)

# ================================================================
# 10. FINAL VEHICLE-TYPE SAFETY CHECK
# ================================================================
# Vehicle Type was assigned from the exact six source folders.
# Therefore Raw_data_file_1 cannot enter the analysis, and a spelling
# difference such as Motorcycle/Motorcycles cannot remove data.

unexpected_speed_types = sorted(
    set(speed["Vehicle Type"].dropna().unique()) - set(VEHICLE_ORDER)
)

unexpected_acceleration_types = sorted(
    set(acceleration["Vehicle Type"].dropna().unique())
    - set(VEHICLE_ORDER)
)

if unexpected_speed_types:
    raise ValueError(
        "Unexpected vehicle type(s) in final speed data: "
        + ", ".join(map(str, unexpected_speed_types))
    )

if unexpected_acceleration_types:
    raise ValueError(
        "Unexpected vehicle type(s) in final acceleration data: "
        + ", ".join(map(str, unexpected_acceleration_types))
    )

unexpected_speed_type_rows = 0
unexpected_acceleration_type_rows = 0

# Hard validation: all six categories MUST contain both speed and
# acceleration observations. This catches any future data issue
# before an incorrect assignment result is produced.
missing_final_vehicle_types = []

for vehicle_type in VEHICLE_ORDER:
    speed_n = int(
        (speed["Vehicle Type"] == vehicle_type).sum()
    )
    acceleration_n = int(
        (acceleration["Vehicle Type"] == vehicle_type).sum()
    )

    if speed_n == 0 or acceleration_n == 0:
        missing_final_vehicle_types.append(
            f"{vehicle_type} (speed={speed_n}, "
            f"acceleration={acceleration_n})"
        )

if missing_final_vehicle_types:
    raise ValueError(
        "FINAL VALIDATION FAILED. Every one of the six vehicle "
        "categories must have observations.\n"
        + "\n".join(missing_final_vehicle_types)
    )

# ================================================================
# 11. VEHICLE-TYPE DATA DICTIONARIES
# ================================================================

speed_by_vehicle = {
    vehicle_type: speed.loc[
        speed["Vehicle Type"] == vehicle_type,
        "Average Speed"
    ].dropna()
    for vehicle_type in VEHICLE_ORDER
}

acceleration_by_vehicle = {
    vehicle_type: acceleration.loc[
        acceleration["Vehicle Type"] == vehicle_type,
        "Tangential Acceleration"
    ].dropna()
    for vehicle_type in VEHICLE_ORDER
}

# ================================================================
# 12. PRINT DATA SUMMARY
# ================================================================

print("\n" + "=" * 100)
print("DATA SUMMARY")
print("=" * 100)

print(
    f"CSV files expected from six vehicle folders : "
    f"{expected_total_files:,}"
)

print(
    f"CSV files successfully read                  : "
    f"{successful_files:,}"
)

print(
    f"CSV files with errors                        : "
    f"{failed_files:,}"
)

print(
    f"Raw trajectory rows read                     : "
    f"{raw_rows_total:,}"
)

print(
    f"Vehicle-level speed observations             : "
    f"{len(speed):,}"
)

print(
    f"Tangential acceleration observations         : "
    f"{len(acceleration):,}"
)

print(
    f"Unexpected speed-type rows excluded          : "
    f"{unexpected_speed_type_rows:,}"
)

print(
    f"Unexpected acceleration-type rows excluded   : "
    f"{unexpected_acceleration_type_rows:,}"
)

print(
    f"Files with Type mismatch                     : "
    f"{type_mismatch_files:,}"
)

print("\nObservations by vehicle type:")
print("-" * 70)

for vehicle_type in VEHICLE_ORDER:
    print(
        f"{vehicle_type:<15} "
        f"Speed = {len(speed_by_vehicle[vehicle_type]):>10,}   "
        f"Acceleration = {len(acceleration_by_vehicle[vehicle_type]):>12,}"
    )

# ================================================================
# 13. DESCRIPTIVE STATISTICS
# ================================================================

speed_stats = make_stats_dataframe(
    speed_by_vehicle
)

acceleration_stats = make_stats_dataframe(
    acceleration_by_vehicle
)

speed_outliers = make_outlier_dataframe(
    speed_by_vehicle
)

acceleration_outliers = make_outlier_dataframe(
    acceleration_by_vehicle
)

# ================================================================
# 14. ACTUAL OUTLIER OBSERVATIONS
# ================================================================

speed_actual_outliers = actual_outliers(
    speed,
    "Average Speed",
    VEHICLE_ORDER
)

acceleration_actual_outliers = actual_outliers(
    acceleration,
    "Tangential Acceleration",
    VEHICLE_ORDER
)

# ================================================================
# 15. DATA-QUALITY AUDIT
# ================================================================

file_log_df = pd.DataFrame(file_log)

folder_summary = pd.DataFrame(vehicle_folder_info)

quality_summary = pd.DataFrame([
    {
        "Check": "Expected vehicle folders",
        "Value": len(VEHICLE_ORDER),
    },
    {
        "Check": "Expected vehicle types",
        "Value": ", ".join(VEHICLE_ORDER),
    },
    {
        "Check": "Raw_data_file_1 included?",
        "Value": "NO",
    },
    {
        "Check": "CSV files found in six vehicle folders",
        "Value": expected_total_files,
    },
    {
        "Check": "CSV files successfully read",
        "Value": successful_files,
    },
    {
        "Check": "CSV files with errors",
        "Value": failed_files,
    },
    {
        "Check": "Files missing Type column",
        "Value": missing_type,
    },
    {
        "Check": "Files missing Track ID column",
        "Value": missing_track,
    },
    {
        "Check": "Files missing Avg. Speed column",
        "Value": missing_speed,
    },
    {
        "Check": "Files missing Tan. Acc. column",
        "Value": missing_acceleration,
    },
    {
        "Check": "Files with Type mismatch",
        "Value": type_mismatch_files,
    },
    {
        "Check": "Unexpected speed rows excluded",
        "Value": unexpected_speed_type_rows,
    },
    {
        "Check": "Unexpected acceleration rows excluded",
        "Value": unexpected_acceleration_type_rows,
    },
    {
        "Check": "Raw trajectory rows read",
        "Value": raw_rows_total,
    },
    {
        "Check": "Final speed observations",
        "Value": len(speed),
    },
    {
        "Check": "Final acceleration observations",
        "Value": len(acceleration),
    },
])

# ================================================================
# 16. SAVE TABLES
# ================================================================

speed_stats_rounded = round_numeric(
    speed_stats,
    4
)

acceleration_stats_rounded = round_numeric(
    acceleration_stats,
    4
)

speed_outliers_rounded = round_numeric(
    speed_outliers,
    4
)

acceleration_outliers_rounded = round_numeric(
    acceleration_outliers,
    4
)

speed_actual_rounded = (
    round_numeric(speed_actual_outliers, 4)
    if not speed_actual_outliers.empty
    else speed_actual_outliers
)

acceleration_actual_rounded = (
    round_numeric(
        acceleration_actual_outliers,
        4
    )
    if not acceleration_actual_outliers.empty
    else acceleration_actual_outliers
)

speed_stats_rounded.to_csv(
    TABLES / "01_SPEED_Detailed_Descriptive_Statistics.csv",
    index=False
)

acceleration_stats_rounded.to_csv(
    TABLES / "02_ACCELERATION_Detailed_Descriptive_Statistics.csv",
    index=False
)

speed_outliers_rounded.to_csv(
    TABLES / "03_SPEED_Outlier_Analysis.csv",
    index=False
)

acceleration_outliers_rounded.to_csv(
    TABLES / "04_ACCELERATION_Outlier_Analysis.csv",
    index=False
)

speed_actual_rounded.to_csv(
    TABLES / "05_SPEED_Actual_Outlier_Observations.csv",
    index=False
)

acceleration_actual_rounded.to_csv(
    TABLES / "06_ACCELERATION_Actual_Outlier_Observations.csv",
    index=False
)

# ================================================================
# 17. MASTER COMPARISON TABLE
# ================================================================

master = speed_stats.merge(
    acceleration_stats,
    on="Vehicle Type",
    how="outer",
    suffixes=(" - Speed", " - Acceleration")
)

master = master.set_index(
    "Vehicle Type"
).reindex(
    VEHICLE_ORDER
).reset_index()

round_numeric(master, 4).to_csv(
    TABLES / "07_MASTER_Vehicle_Type_Comparison.csv",
    index=False
)

# ================================================================
# 18. SAVE PROCESSED DATA
# ================================================================

speed.to_csv(
    PROCESSED / "01_Processed_Speed_Vehicle_Level.csv",
    index=False
)

acceleration.to_csv(
    PROCESSED / "02_Processed_Acceleration_Trajectory_Level.csv",
    index=False
)

file_log_df.to_csv(
    PROCESSED / "03_File_Processing_Log.csv",
    index=False
)

folder_summary.to_csv(
    PROCESSED / "04_Vehicle_Folder_Summary.csv",
    index=False
)

quality_summary.to_csv(
    PROCESSED / "05_Data_Quality_Audit.csv",
    index=False
)

# ================================================================
# 19. PLOTS - SPEED
# ================================================================

# 19.1 Comparative histogram
fig, ax = plt.subplots(figsize=(11, 7))

for vehicle_type in VEHICLE_ORDER:
    x = speed_by_vehicle[vehicle_type]

    if len(x) > 0:
        ax.hist(
            x,
            bins="auto",
            density=True,
            alpha=0.35,
            label=vehicle_type
        )

ax.set_title(
    "Distribution of Average Speed by Vehicle Type"
)

ax.set_xlabel(
    "Average Speed (km/h)"
)

ax.set_ylabel(
    "Density"
)

ax.legend()
ax.grid(alpha=0.2)

save_figure(
    fig,
    "01_SPEED_Comparative_Histogram.png"
)

# 19.2 Boxplot
fig, ax = plt.subplots(figsize=(11, 7))

ax.boxplot(
    [
        speed_by_vehicle[v].values
        for v in VEHICLE_ORDER
    ],
    labels=VEHICLE_ORDER,
    showfliers=True
)

ax.set_title(
    "Average Speed Distribution by Vehicle Type"
)

ax.set_xlabel("Vehicle Type")
ax.set_ylabel("Average Speed (km/h)")

ax.grid(axis="y", alpha=0.2)

plt.setp(
    ax.get_xticklabels(),
    rotation=30,
    ha="right"
)

save_figure(
    fig,
    "02_SPEED_Boxplot.png"
)

# 19.3 Density plot
fig, ax = plt.subplots(figsize=(11, 7))

for vehicle_type in VEHICLE_ORDER:
    x = speed_by_vehicle[vehicle_type]

    if len(x) > 1 and x.nunique() > 1:
        try:
            kde = stats.gaussian_kde(x)

            grid = np.linspace(
                x.min(),
                x.max(),
                500
            )

            ax.plot(
                grid,
                kde(grid),
                linewidth=2,
                label=vehicle_type
            )

        except Exception:
            pass

ax.set_title(
    "Kernel Density Estimate of Average Speed"
)

ax.set_xlabel(
    "Average Speed (km/h)"
)

ax.set_ylabel("Density")

ax.legend()
ax.grid(alpha=0.2)

save_figure(
    fig,
    "03_SPEED_Density_Plot.png"
)

# 19.4 Mean with 95% CI
fig, ax = plt.subplots(figsize=(11, 7))

means = speed_stats["Mean"].to_numpy(dtype=float)
ci_lower = speed_stats["95% CI Lower"].to_numpy(dtype=float)
ci_upper = speed_stats["95% CI Upper"].to_numpy(dtype=float)

valid = (
    np.isfinite(means)
    & np.isfinite(ci_lower)
    & np.isfinite(ci_upper)
)

positions = np.arange(len(VEHICLE_ORDER))

if valid.any():
    ax.errorbar(
        positions[valid],
        means[valid],
        yerr=[
            means[valid] - ci_lower[valid],
            ci_upper[valid] - means[valid],
        ],
        fmt="o",
        capsize=5
    )

ax.set_xticks(positions)
ax.set_xticklabels(
    VEHICLE_ORDER,
    rotation=30,
    ha="right"
)

ax.set_title(
    "Mean Average Speed with 95% Confidence Intervals"
)

ax.set_xlabel("Vehicle Type")
ax.set_ylabel("Mean Average Speed (km/h)")

ax.grid(axis="y", alpha=0.2)

save_figure(
    fig,
    "04_SPEED_Mean_95CI.png"
)

# 19.5 Mean vs median
fig, ax = plt.subplots(figsize=(11, 7))

positions = np.arange(len(VEHICLE_ORDER))

ax.plot(
    positions,
    speed_stats["Mean"],
    marker="o",
    label="Mean"
)

ax.plot(
    positions,
    speed_stats["Median"],
    marker="s",
    label="Median"
)

ax.set_xticks(positions)
ax.set_xticklabels(
    VEHICLE_ORDER,
    rotation=30,
    ha="right"
)

ax.set_title(
    "Mean and Median Average Speed"
)

ax.set_xlabel("Vehicle Type")
ax.set_ylabel("Speed (km/h)")

ax.legend()
ax.grid(alpha=0.2)

save_figure(
    fig,
    "05_SPEED_Mean_vs_Median.png"
)

# 19.6 Standard deviation
fig, ax = plt.subplots(figsize=(11, 7))

ax.bar(
    VEHICLE_ORDER,
    speed_stats["Std. Deviation"]
)

ax.set_title(
    "Standard Deviation of Average Speed"
)

ax.set_xlabel("Vehicle Type")
ax.set_ylabel("SD (km/h)")

ax.grid(axis="y", alpha=0.2)

plt.setp(
    ax.get_xticklabels(),
    rotation=30,
    ha="right"
)

save_figure(
    fig,
    "06_SPEED_Standard_Deviation.png"
)

# ================================================================
# 20. PLOTS - TANGENTIAL ACCELERATION
# ================================================================

# 20.1 Comparative histogram
fig, ax = plt.subplots(figsize=(11, 7))

for vehicle_type in VEHICLE_ORDER:
    x = acceleration_by_vehicle[vehicle_type]

    if len(x) > 0:
        ax.hist(
            x,
            bins="auto",
            density=True,
            alpha=0.35,
            label=vehicle_type
        )

ax.set_title(
    "Distribution of Tangential Acceleration by Vehicle Type"
)

ax.set_xlabel(
    "Tangential Acceleration (m/s²)"
)

ax.set_ylabel("Density")

ax.legend()
ax.grid(alpha=0.2)

save_figure(
    fig,
    "07_ACCELERATION_Comparative_Histogram.png"
)

# 20.2 Boxplot
fig, ax = plt.subplots(figsize=(11, 7))

ax.boxplot(
    [
        acceleration_by_vehicle[v].values
        for v in VEHICLE_ORDER
    ],
    labels=VEHICLE_ORDER,
    showfliers=True
)

ax.set_title(
    "Tangential Acceleration Distribution by Vehicle Type"
)

ax.set_xlabel("Vehicle Type")
ax.set_ylabel(
    "Tangential Acceleration (m/s²)"
)

ax.grid(axis="y", alpha=0.2)

plt.setp(
    ax.get_xticklabels(),
    rotation=30,
    ha="right"
)

save_figure(
    fig,
    "08_ACCELERATION_Boxplot.png"
)

# 20.3 Density plot
fig, ax = plt.subplots(figsize=(11, 7))

for vehicle_type in VEHICLE_ORDER:
    x = acceleration_by_vehicle[vehicle_type]

    if len(x) > 1 and x.nunique() > 1:
        try:
            kde = stats.gaussian_kde(x)

            grid = np.linspace(
                x.min(),
                x.max(),
                500
            )

            ax.plot(
                grid,
                kde(grid),
                linewidth=2,
                label=vehicle_type
            )

        except Exception:
            pass

ax.set_title(
    "Kernel Density Estimate of Tangential Acceleration"
)

ax.set_xlabel(
    "Tangential Acceleration (m/s²)"
)

ax.set_ylabel("Density")

ax.legend()
ax.grid(alpha=0.2)

save_figure(
    fig,
    "09_ACCELERATION_Density_Plot.png"
)

# 20.4 Mean with 95% CI
fig, ax = plt.subplots(figsize=(11, 7))

means = acceleration_stats["Mean"].to_numpy(dtype=float)
ci_lower = acceleration_stats[
    "95% CI Lower"
].to_numpy(dtype=float)

ci_upper = acceleration_stats[
    "95% CI Upper"
].to_numpy(dtype=float)

valid = (
    np.isfinite(means)
    & np.isfinite(ci_lower)
    & np.isfinite(ci_upper)
)

positions = np.arange(len(VEHICLE_ORDER))

if valid.any():
    ax.errorbar(
        positions[valid],
        means[valid],
        yerr=[
            means[valid] - ci_lower[valid],
            ci_upper[valid] - means[valid],
        ],
        fmt="o",
        capsize=5
    )

ax.set_xticks(positions)
ax.set_xticklabels(
    VEHICLE_ORDER,
    rotation=30,
    ha="right"
)

ax.set_title(
    "Mean Tangential Acceleration with 95% Confidence Intervals"
)

ax.set_xlabel("Vehicle Type")

ax.set_ylabel(
    "Mean Tangential Acceleration (m/s²)"
)

ax.grid(axis="y", alpha=0.2)

save_figure(
    fig,
    "10_ACCELERATION_Mean_95CI.png"
)

# 20.5 Mean vs median
fig, ax = plt.subplots(figsize=(11, 7))

positions = np.arange(len(VEHICLE_ORDER))

ax.plot(
    positions,
    acceleration_stats["Mean"],
    marker="o",
    label="Mean"
)

ax.plot(
    positions,
    acceleration_stats["Median"],
    marker="s",
    label="Median"
)

ax.set_xticks(positions)
ax.set_xticklabels(
    VEHICLE_ORDER,
    rotation=30,
    ha="right"
)

ax.set_title(
    "Mean and Median Tangential Acceleration"
)

ax.set_xlabel("Vehicle Type")

ax.set_ylabel(
    "Tangential Acceleration (m/s²)"
)

ax.legend()
ax.grid(alpha=0.2)

save_figure(
    fig,
    "11_ACCELERATION_Mean_vs_Median.png"
)

# 20.6 Standard deviation
fig, ax = plt.subplots(figsize=(11, 7))

ax.bar(
    VEHICLE_ORDER,
    acceleration_stats["Std. Deviation"]
)

ax.set_title(
    "Standard Deviation of Tangential Acceleration"
)

ax.set_xlabel("Vehicle Type")

ax.set_ylabel("SD (m/s²)")

ax.grid(axis="y", alpha=0.2)

plt.setp(
    ax.get_xticklabels(),
    rotation=30,
    ha="right"
)

save_figure(
    fig,
    "12_ACCELERATION_Standard_Deviation.png"
)

# ================================================================
# 21. INDIVIDUAL HISTOGRAMS
# ================================================================

for vehicle_type in VEHICLE_ORDER:

    # Speed
    x = speed_by_vehicle[vehicle_type]

    if len(x) > 0:

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        ax.hist(
            x,
            bins="auto",
            edgecolor="black"
        )

        ax.axvline(
            x.mean(),
            linestyle="--",
            linewidth=2,
            label=f"Mean = {x.mean():.2f}"
        )

        ax.axvline(
            x.median(),
            linestyle=":",
            linewidth=2,
            label=f"Median = {x.median():.2f}"
        )

        ax.set_title(
            f"Average Speed Distribution - {vehicle_type}"
        )

        ax.set_xlabel(
            "Average Speed (km/h)"
        )

        ax.set_ylabel("Frequency")

        ax.legend()
        ax.grid(alpha=0.2)

        save_figure(
            fig,
            f"13_SPEED_Histogram_{safe_filename(vehicle_type)}.png"
        )

    # Acceleration
    x = acceleration_by_vehicle[vehicle_type]

    if len(x) > 0:

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        ax.hist(
            x,
            bins="auto",
            edgecolor="black"
        )

        ax.axvline(
            x.mean(),
            linestyle="--",
            linewidth=2,
            label=f"Mean = {x.mean():.3f}"
        )

        ax.axvline(
            x.median(),
            linestyle=":",
            linewidth=2,
            label=f"Median = {x.median():.3f}"
        )

        ax.set_title(
            f"Tangential Acceleration Distribution - "
            f"{vehicle_type}"
        )

        ax.set_xlabel(
            "Tangential Acceleration (m/s²)"
        )

        ax.set_ylabel("Frequency")

        ax.legend()
        ax.grid(alpha=0.2)

        save_figure(
            fig,
            f"14_ACCELERATION_Histogram_"
            f"{safe_filename(vehicle_type)}.png"
        )

# ================================================================
# 22. DATA-DRIVEN INTERPRETATION
# ================================================================

def generate_interpretation(
    statistics_df,
    outlier_df,
    variable_name,
    unit
):
    """
    Generate interpretation only from calculated results.
    No numerical result is invented.
    """

    valid = statistics_df.dropna(
        subset=["Mean"]
    ).copy()

    if valid.empty:
        return [
            f"No valid {variable_name.lower()} statistics "
            "were available."
        ]

    sentences = []

    highest_mean = valid.loc[
        valid["Mean"].idxmax()
    ]

    lowest_mean = valid.loc[
        valid["Mean"].idxmin()
    ]

    highest_sd = valid.loc[
        valid["Std. Deviation"].idxmax()
    ]

    lowest_sd = valid.loc[
        valid["Std. Deviation"].idxmin()
    ]

    highest_iqr = valid.loc[
        valid["IQR"].idxmax()
    ]

    lowest_iqr = valid.loc[
        valid["IQR"].idxmin()
    ]

    skew_valid = valid.dropna(
        subset=["Skewness"]
    )

    if not skew_valid.empty:

        strongest_skew = skew_valid.loc[
            skew_valid["Skewness"].abs().idxmax()
        ]

        skew_value = strongest_skew["Skewness"]

        if skew_value > 0.1:
            skew_direction = "positive"
        elif skew_value < -0.1:
            skew_direction = "negative"
        else:
            skew_direction = "approximately symmetric"

        skew_sentence = (
            f"The strongest distributional asymmetry was observed "
            f"for {strongest_skew['Vehicle Type']} "
            f"(skewness = {skew_value:.3f}), indicating "
            f"{skew_direction} skewness."
        )

    else:
        skew_sentence = (
            "Skewness could not be reliably evaluated for the "
            "available sample sizes."
        )

    max_outlier_row = outlier_df.loc[
        outlier_df["Outlier Count"].idxmax()
    ]

    sentences.append(
        f"The highest mean {variable_name.lower()} was observed "
        f"for {highest_mean['Vehicle Type']} "
        f"({highest_mean['Mean']:.3f} {unit}), while the lowest "
        f"mean was observed for {lowest_mean['Vehicle Type']} "
        f"({lowest_mean['Mean']:.3f} {unit})."
    )

    sentences.append(
        f"The greatest standard deviation was observed for "
        f"{highest_sd['Vehicle Type']} "
        f"(SD = {highest_sd['Std. Deviation']:.3f} {unit}), "
        f"indicating the greatest absolute dispersion. The "
        f"smallest standard deviation occurred for "
        f"{lowest_sd['Vehicle Type']} "
        f"(SD = {lowest_sd['Std. Deviation']:.3f} {unit})."
    )

    sentences.append(
        f"The largest IQR was observed for "
        f"{highest_iqr['Vehicle Type']} "
        f"({highest_iqr['IQR']:.3f} {unit}), whereas the smallest "
        f"IQR was observed for "
        f"{lowest_iqr['Vehicle Type']} "
        f"({lowest_iqr['IQR']:.3f} {unit})."
    )

    sentences.append(skew_sentence)

    sentences.append(
        f"Using the 1.5 × IQR criterion, "
        f"{max_outlier_row['Vehicle Type']} had the largest number "
        f"of flagged outliers "
        f"({int(max_outlier_row['Outlier Count'])}), representing "
        f"{max_outlier_row['Outlier (%)']:.2f}% of its valid "
        f"observations."
    )

    sentences.append(
        "Outliers were retained rather than automatically deleted. "
        "In traffic trajectory data, extreme speed or acceleration "
        "values may represent genuine vehicle behaviour or "
        "manoeuvring; deletion should therefore require an "
        "independent data-quality justification."
    )

    return sentences


speed_interpretation = generate_interpretation(
    speed_stats,
    speed_outliers,
    "speed",
    "km/h"
)

acceleration_interpretation = generate_interpretation(
    acceleration_stats,
    acceleration_outliers,
    "tangential acceleration",
    "m/s²"
)

# ================================================================
# 23. ASSIGNMENT-READY TEXT
# ================================================================

assignment_text = []

assignment_text.append(
    "QUESTION (c): DETAILED DESCRIPTIVE STATISTICS OF "
    "SPEED AND ACCELERATION"
)

assignment_text.append("")
assignment_text.append("1. METHODOLOGY")
assignment_text.append("")

assignment_text.append(
    """The speed and tangential acceleration variables were analysed
separately for Bus, Tuk-Tuk, Car, Light Truck, Motorcycles and Van.
The Raw_data_file_1 folder was explicitly excluded from the Question
(c) analysis. The source vehicle folder was used as the authoritative
vehicle category, while the Type column was retained for data-quality
checking. This prevents loss of Motorcycle observations when the
source Type field uses the singular spelling "Motorcycle" rather than
"Motorcycles"."""
)

assignment_text.append(
    "Avg. Speed [km/h] was treated as a vehicle-level variable. "
    "Therefore, one average-speed observation was retained per "
    "Track ID within each source CSV. The source folder and source "
    "file were included in the vehicle key so that Track IDs that "
    "restart in different files were not incorrectly treated as "
    "duplicate vehicles."
)

assignment_text.append(
    "Tangential acceleration was treated as a trajectory-level "
    "variable. Every valid Tan. Acc. [ms-2] observation was "
    "retained rather than deduplicated."
)

assignment_text.append(
    "For each vehicle type, N, mean, median, sample standard "
    "deviation, sample variance, minimum, first quartile (Q1), "
    "third quartile (Q3), maximum, range, interquartile range "
    "(IQR), coefficient of variation, skewness and the 95% "
    "Student-t confidence interval for the mean were calculated."
)

assignment_text.append(
    "Potential outliers were identified using the 1.5 × IQR rule. "
    "The identified outliers were flagged but not automatically "
    "removed."
)

assignment_text.append("")
assignment_text.append("2. SPEED RESULTS")
assignment_text.append("")
assignment_text.append(
    "Table 1. Detailed descriptive statistics of average speed"
)
assignment_text.append("")
assignment_text.append(
    speed_stats_rounded.to_string(index=False)
)

assignment_text.append("")
assignment_text.append("Interpretation:")

for sentence in speed_interpretation:
    assignment_text.append("• " + sentence)

assignment_text.append("")
assignment_text.append(
    "Recommended main figures: "
    "01_SPEED_Comparative_Histogram.png, "
    "02_SPEED_Boxplot.png, "
    "03_SPEED_Density_Plot.png and "
    "04_SPEED_Mean_95CI.png."
)

assignment_text.append("")
assignment_text.append("3. SPEED OUTLIERS")
assignment_text.append("")
assignment_text.append(
    "Table 2. Speed outlier analysis"
)
assignment_text.append("")
assignment_text.append(
    speed_outliers_rounded.to_string(index=False)
)

assignment_text.append("")
assignment_text.append(
    "4. TANGENTIAL ACCELERATION RESULTS"
)
assignment_text.append("")
assignment_text.append(
    "Table 3. Detailed descriptive statistics of "
    "tangential acceleration"
)
assignment_text.append("")
assignment_text.append(
    acceleration_stats_rounded.to_string(index=False)
)

assignment_text.append("")
assignment_text.append("Interpretation:")

for sentence in acceleration_interpretation:
    assignment_text.append("• " + sentence)

assignment_text.append("")
assignment_text.append(
    "Recommended main figures: "
    "07_ACCELERATION_Comparative_Histogram.png, "
    "08_ACCELERATION_Boxplot.png, "
    "09_ACCELERATION_Density_Plot.png and "
    "10_ACCELERATION_Mean_95CI.png."
)

assignment_text.append("")
assignment_text.append(
    "5. TANGENTIAL ACCELERATION OUTLIERS"
)
assignment_text.append("")
assignment_text.append(
    "Table 4. Tangential acceleration outlier analysis"
)
assignment_text.append("")
assignment_text.append(
    acceleration_outliers_rounded.to_string(index=False)
)

assignment_text.append("")
assignment_text.append(
    "6. PROMINENT TRENDS AND OUTLIERS"
)
assignment_text.append("")

assignment_text.append(
    "The prominent trends should be interpreted using the combined "
    "evidence from central tendency, variability, distribution "
    "shape, confidence intervals and outlier analysis. Differences "
    "between the mean and median can indicate distributional "
    "asymmetry. Standard deviation and IQR describe variability, "
    "while histograms and density plots show the distributional "
    "form. Box plots provide a direct visual comparison of spread "
    "and potential extreme observations."
)

assignment_text.append(
    "The 1.5 × IQR outliers were retained because extreme "
    "observations can represent genuine traffic behaviour, "
    "including strong acceleration/deceleration or unusually high "
    "or low vehicle speeds. They should only be removed if the "
    "original trajectory data demonstrate a measurement or "
    "recording error."
)

assignment_text.append("")
assignment_text.append("7. DATA-PROCESSING SUMMARY")
assignment_text.append("")

assignment_text.append(
    f"Six vehicle categories were analysed: "
    f"{', '.join(VEHICLE_ORDER)}."
)

assignment_text.append(
    f"Total CSV files processed successfully: "
    f"{successful_files:,} out of {expected_total_files:,}."
)

assignment_text.append(
    f"Total raw trajectory rows read: "
    f"{raw_rows_total:,}."
)

assignment_text.append(
    f"Final vehicle-level speed observations: "
    f"{len(speed):,}."
)

assignment_text.append(
    f"Final tangential acceleration observations: "
    f"{len(acceleration):,}."
)

assignment_text.append(
    "Raw_data_file_1 was explicitly excluded."
)

assignment_text.append("")
assignment_text.append(
    "END OF QUESTION (c) RESULTS"
)

assignment_text_path = (
    TEXT / "Q3_Assignment_Ready_Text.txt"
)

write_text_file(
    assignment_text_path,
    "\n".join(assignment_text)
)

# Also create a more clearly named copy.
write_text_file(
    TEXT / "Question_c_Assignment_Ready_Text.txt",
    "\n".join(assignment_text)
)

# ================================================================
# 24. HTML REPORT
# ================================================================

def html_table(df):
    return df.to_html(
        index=False,
        border=0,
        classes="tbl"
    )


html_parts = [
    "<!doctype html>",
    "<html>",
    "<head>",
    '<meta charset="utf-8">',
    "<title>TRL7100 Question (c)</title>",
    "<style>",
    "body{font-family:Arial,sans-serif;margin:35px;"
    "line-height:1.5;color:#222}",
    ".tbl{border-collapse:collapse;width:100%;"
    "font-size:12px;margin:15px 0 30px}",
    ".tbl th,.tbl td{border:1px solid #999;"
    "padding:6px;text-align:center}",
    ".tbl th{font-weight:bold}",
    "img{max-width:100%;height:auto}",
    ".figure{margin:25px 0}",
    ".warning{font-weight:bold}",
    "</style>",
    "</head>",
    "<body>",
    "<h1>TRL7100 - Assignment 1 - Question (c)</h1>",
    "<p><b>Detailed descriptive statistics of speed and "
    "acceleration for each vehicle type</b></p>",
    f"<p><b>Vehicle categories:</b> "
    f"{', '.join(VEHICLE_ORDER)}</p>",
    "<p class='warning'>Raw_data_file_1 was explicitly excluded "
    "from this analysis.</p>",
    f"<p>CSV files successfully read: {successful_files:,} | "
    f"Raw rows: {raw_rows_total:,} | "
    f"Speed observations: {len(speed):,} | "
    f"Acceleration observations: {len(acceleration):,}</p>",
    "<h2>Speed descriptive statistics</h2>",
    html_table(speed_stats_rounded),
    "<h3>Interpretation</h3>",
    "<ul>",
]

for sentence in speed_interpretation:
    html_parts.append(
        f"<li>{html.escape(sentence)}</li>"
    )

html_parts.extend([
    "</ul>",
    "<h2>Speed outlier analysis</h2>",
    html_table(speed_outliers_rounded),
    "<h2>Tangential acceleration descriptive statistics</h2>",
    html_table(acceleration_stats_rounded),
    "<h3>Interpretation</h3>",
    "<ul>",
])

for sentence in acceleration_interpretation:
    html_parts.append(
        f"<li>{html.escape(sentence)}</li>"
    )

html_parts.extend([
    "</ul>",
    "<h2>Tangential acceleration outlier analysis</h2>",
    html_table(acceleration_outliers_rounded),
    "<h2>Data-quality audit</h2>",
    html_table(quality_summary),
    "<h2>Main figures</h2>",
])

main_figures = [
    "01_SPEED_Comparative_Histogram.png",
    "02_SPEED_Boxplot.png",
    "03_SPEED_Density_Plot.png",
    "04_SPEED_Mean_95CI.png",
    "07_ACCELERATION_Comparative_Histogram.png",
    "08_ACCELERATION_Boxplot.png",
    "09_ACCELERATION_Density_Plot.png",
    "10_ACCELERATION_Mean_95CI.png",
]

for figure_name in main_figures:
    html_parts.append(
        f"<div class='figure'>"
        f"<h3>{html.escape(figure_name)}</h3>"
        f"<img src='../02_High_Quality_Figures/"
        f"{html.escape(figure_name)}'>"
        f"</div>"
    )

html_parts.extend([
    "<h2>Outlier treatment</h2>",
    "<p>Potential outliers were identified using the "
    "1.5 × IQR criterion and flagged, not automatically "
    "deleted. Their validity should be assessed against the "
    "original trajectory data if further data cleaning is "
    "required.</p>",
    "</body>",
    "</html>",
])

html_report_path = (
    REPORT / "TRL7100_Question_C_COMPLETE_REPORT.html"
)

write_text_file(
    html_report_path,
    "\n".join(html_parts)
)

# ================================================================
# 25. EXCEL WORKBOOK
# ================================================================

excel_path = (
    OUT / "TRL7100_Question_C_Complete_Results.xlsx"
)

try:

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl"
    ) as writer:

        speed_stats_rounded.to_excel(
            writer,
            sheet_name="Speed Statistics",
            index=False
        )

        acceleration_stats_rounded.to_excel(
            writer,
            sheet_name="Acceleration Statistics",
            index=False
        )

        speed_outliers_rounded.to_excel(
            writer,
            sheet_name="Speed Outliers",
            index=False
        )

        acceleration_outliers_rounded.to_excel(
            writer,
            sheet_name="Acceleration Outliers",
            index=False
        )

        round_numeric(
            master,
            4
        ).to_excel(
            writer,
            sheet_name="Master Comparison",
            index=False
        )

        if not speed_actual_rounded.empty:
            speed_actual_rounded.to_excel(
                writer,
                sheet_name="Speed Actual Outliers",
                index=False
            )

        if not acceleration_actual_rounded.empty:
            acceleration_actual_rounded.to_excel(
                writer,
                sheet_name="Acceleration Actual Outliers",
                index=False
            )

        file_log_df.to_excel(
            writer,
            sheet_name="File Processing Log",
            index=False
        )

        folder_summary.to_excel(
            writer,
            sheet_name="Vehicle Folder Summary",
            index=False
        )

        quality_summary.to_excel(
            writer,
            sheet_name="Data Quality Audit",
            index=False
        )

        # Basic readable column widths.
        for worksheet in writer.book.worksheets:

            for column_cells in worksheet.columns:

                maximum_length = 0

                for cell in column_cells:
                    value = cell.value

                    if value is not None:
                        maximum_length = max(
                            maximum_length,
                            len(str(value))
                        )

                width = min(
                    max(maximum_length + 2, 10),
                    45
                )

                worksheet.column_dimensions[
                    column_cells[0].column_letter
                ].width = width

    excel_message = "Excel workbook created successfully."

except Exception as exc:

    excel_message = (
        "Excel workbook could not be created: "
        f"{exc}"
    )

# ================================================================
# 26. RUN SUMMARY FILE
# ================================================================

run_summary = [
    "TRL7100 QUESTION (c) - RUN SUMMARY",
    "=" * 70,
    "",
    f"Data root: {DATA_ROOT}",
    "",
    "Vehicle folders analysed:",
]

for vehicle_type in VEHICLE_ORDER:
    info = next(
        item
        for item in vehicle_folder_info
        if item["Vehicle Type"] == vehicle_type
    )

    run_summary.append(
        f"  {vehicle_type}: "
        f"{info['CSV Files Found']} CSV files"
    )

run_summary.extend([
    "",
    "EXCLUSION CHECK:",
    "  Raw_data_file_1: EXCLUDED; only six exact vehicle folders analysed",
    "",
    f"Expected CSV files: {expected_total_files:,}",
    f"Successfully read: {successful_files:,}",
    f"Files with errors: {failed_files:,}",
    f"Raw rows read: {raw_rows_total:,}",
    f"Final speed observations: {len(speed):,}",
    f"Final acceleration observations: {len(acceleration):,}",
    f"Type mismatch files: {type_mismatch_files:,}",
    "",
    "Excel status:",
    f"  {excel_message}",
    "",
    "Main output folder:",
    str(OUT),
])

write_text_file(
    OUT / "RUN_SUMMARY.txt",
    "\n".join(run_summary)
)

# ================================================================
# 27. FINAL CONSOLE OUTPUT
# ================================================================

print("\n" + "=" * 100)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 100)

print("\nFINAL VEHICLE CATEGORIES:")
for vehicle_type in VEHICLE_ORDER:
    print(f"  ✓ {vehicle_type}")

print("\nMOTORCYCLE VALIDATION:")
print(
    f"  ✓ Motorcycle-folder speed observations: "
    f"{len(speed_by_vehicle['Motorcycles']):,}"
)
print(
    f"  ✓ Motorcycle-folder acceleration observations: "
    f"{len(acceleration_by_vehicle['Motorcycles']):,}"
)

print("\nEXCLUSION:")
print("  ✓ Raw_data_file_1 was NOT analysed.")

print("\nFINAL OBSERVATION COUNTS:")
for vehicle_type in VEHICLE_ORDER:
    print(
        f"  {vehicle_type:<15} "
        f"Speed: {len(speed_by_vehicle[vehicle_type]):>10,} | "
        f"Acceleration: "
        f"{len(acceleration_by_vehicle[vehicle_type]):>12,}"
    )

print("\nOUTPUT FOLDER:")
print(OUT)

print("\nIMPORTANT FILES:")
print(
    "  1. Assignment-ready text:"
)
print(
    f"     {assignment_text_path}"
)

print(
    "  2. HTML report:"
)
print(
    f"     {html_report_path}"
)

print(
    "  3. Excel workbook:"
)
print(
    f"     {excel_path}"
)

print(
    "  4. Speed statistics:"
)
print(
    f"     {TABLES / '01_SPEED_Detailed_Descriptive_Statistics.csv'}"
)

print(
    "  5. Acceleration statistics:"
)
print(
    f"     {TABLES / '02_ACCELERATION_Detailed_Descriptive_Statistics.csv'}"
)

print(
    "  6. Data-quality audit:"
)
print(
    f"     {PROCESSED / '05_Data_Quality_Audit.csv'}"
)

print("\nEXCEL STATUS:")
print(excel_message)

print("\nMAIN FIGURES FOR THE ASSIGNMENT:")
print("  SPEED")
print("    - 01_SPEED_Comparative_Histogram.png")
print("    - 02_SPEED_Boxplot.png")
print("    - 03_SPEED_Density_Plot.png")
print("    - 04_SPEED_Mean_95CI.png")

print("  TANGENTIAL ACCELERATION")
print("    - 07_ACCELERATION_Comparative_Histogram.png")
print("    - 08_ACCELERATION_Boxplot.png")
print("    - 09_ACCELERATION_Density_Plot.png")
print("    - 10_ACCELERATION_Mean_95CI.png")

print("\n" + "=" * 100)
print("DONE - Question (c) analysis is complete.")
print("=" * 100)
