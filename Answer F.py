

import os
import glob
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


# ================================================================
# 1. ROOT FOLDER
# ================================================================

ROOT_FOLDER = r"C:\Users\admin\Desktop\TRIP\Mathematical Modal-Sai Chand-3-C\TRL-700 DATA FILES"


# ================================================================
# 2. VEHICLE FOLDERS
# ================================================================

VEHICLE_FOLDERS = {
    "Bus": "stacked files for the bus",
    "3-W (Tuk-Tuk)": "stacked files od Tuk-Tuk",
    "Car": "Stacked files of car",
    "Light Truck": "Stacked files of Light Truck",
    "MTW (Motorcycles)": "stacked files of Motorcycles",
    "Van": "Stacked files of Van"
}


# ================================================================
# 3. SIGNIFICANCE LEVEL
# ================================================================

ALPHA = 0.05


# ================================================================
# 4. IMPORTANT ROAD DIRECTION
# ================================================================
#
# If HIGHER Y = closer to MEDIAN:
#
#     "higher_y"
#
# If LOWER Y = closer to MEDIAN:
#
#     "lower_y"
#
# Change this according to your actual road layout.
# ================================================================

MEDIAN_SIDE = "higher_y"


# ================================================================
# 5. NORMALIZE COLUMN NAME
# ================================================================

def normalize_column_name(col):

    text = str(col).strip().lower()

    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    # Handles:
    # Trajectory.1
    # y [m].1
    # Speed [km/h].1

    text = re.sub(
        r"\.\d+$",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ================================================================
# 6. IDENTIFY TRAJECTORY
# ================================================================

def is_x_column(col):

    name = normalize_column_name(col)

    return (
        name == "trajectory"
        or
        name.startswith("trajectory")
    )


# ================================================================
# 7. IDENTIFY Y POSITION
# ================================================================

def is_y_column(col):

    name = normalize_column_name(col)

    return (
        name == "y [m]"
        or
        name == "y"
        or
        name.startswith("y [m]")
    )


# ================================================================
# 8. IDENTIFY SPEED
# ================================================================

def is_speed_column(col):

    name = normalize_column_name(col)

    return (
        name == "speed [km/h]"
        or
        name.startswith("speed [km/h]")
        or
        name == "speed"
    )


# ================================================================
# 9. READ FILE
# ================================================================

def read_file(file_path):

    extension = os.path.splitext(
        file_path
    )[1].lower()

    try:

        if extension in [".xlsx", ".xls"]:

            return pd.read_excel(
                file_path
            )

        else:

            try:

                df = pd.read_csv(
                    file_path,
                    low_memory=False
                )

                # If incorrectly detected as one column
                if len(df.columns) == 1:

                    df = pd.read_csv(
                        file_path,
                        sep=None,
                        engine="python",
                        low_memory=False
                    )

                return df

            except Exception:

                return pd.read_csv(
                    file_path,
                    sep=None,
                    engine="python",
                    encoding="latin1",
                    low_memory=False
                )

    except Exception:

        return None


# ================================================================
# 10. START
# ================================================================

print("=" * 80)
print("TRL7100 - SPEED AND LATERAL POSITION ANALYSIS")
print("SPEARMAN CORRELATION ONLY")
print("=" * 80)

print()

print("Research question:")

print(
    "Do slow-moving MTWs and 3-Ws tend to travel closer "
    "to the shoulder, while faster vehicles tend to travel "
    "closer to the median?"
)

print()

print(
    "Median side setting:",
    MEDIAN_SIDE
)


# ================================================================
# 11. CHECK FOLDERS
# ================================================================

print()
print("=" * 80)
print("CHECKING VEHICLE FOLDERS")
print("=" * 80)


for vehicle, folder in VEHICLE_FOLDERS.items():

    path = os.path.join(
        ROOT_FOLDER,
        folder
    )

    if os.path.exists(path):

        print(
            "OK   :",
            vehicle
        )

    else:

        print(
            "ERROR:",
            vehicle
        )


# ================================================================
# 12. EXTRACT DATA
# ================================================================

all_data = []

total_files = 0

files_read = 0

files_with_data = 0

total_blocks = 0


print()
print("=" * 80)
print("READING FILES")
print("=" * 80)


for vehicle, folder in VEHICLE_FOLDERS.items():

    folder_path = os.path.join(
        ROOT_FOLDER,
        folder
    )

    files = []

    files.extend(
        glob.glob(
            os.path.join(
                folder_path,
                "**",
                "*.csv"
            ),
            recursive=True
        )
    )

    files.extend(
        glob.glob(
            os.path.join(
                folder_path,
                "**",
                "*.xlsx"
            ),
            recursive=True
        )
    )

    files.extend(
        glob.glob(
            os.path.join(
                folder_path,
                "**",
                "*.xls"
            ),
            recursive=True
        )
    )


    print()
    print(
        vehicle,
        ":",
        len(files),
        "files"
    )


    total_files += len(files)


    # ------------------------------------------------------------
    # FILE LOOP
    # ------------------------------------------------------------

    for file_path in files:

        df = read_file(
            file_path
        )

        if df is None:

            continue


        files_read += 1


        # --------------------------------------------------------
        # Find columns
        # --------------------------------------------------------

        x_columns = [
            col
            for col in df.columns
            if is_x_column(col)
        ]


        y_columns = [
            col
            for col in df.columns
            if is_y_column(col)
        ]


        speed_columns = [
            col
            for col in df.columns
            if is_speed_column(col)
        ]


        # --------------------------------------------------------
        # Number of complete X-Y-Speed blocks
        # --------------------------------------------------------

        blocks = min(
            len(x_columns),
            len(y_columns),
            len(speed_columns)
        )


        if blocks == 0:

            continue


        files_with_data += 1

        total_blocks += blocks


        # --------------------------------------------------------
        # Track ID
        # --------------------------------------------------------

        track_column = None

        for col in df.columns:

            if normalize_column_name(
                col
            ) == "track id":

                track_column = col

                break


        # --------------------------------------------------------
        # Extract every block
        # --------------------------------------------------------

        for block in range(
            blocks
        ):

            temp = pd.DataFrame()


            temp["X"] = df[
                x_columns[block]
            ]


            temp["Y"] = df[
                y_columns[block]
            ]


            temp["Speed"] = df[
                speed_columns[block]
            ]


            temp["Vehicle_Type"] = vehicle


            temp["Source_File"] = os.path.basename(
                file_path
            )


            if track_column is not None:

                temp["Track_ID"] = df[
                    track_column
                ]

            else:

                temp["Track_ID"] = np.arange(
                    len(df)
                )


            temp["Block"] = block + 1


            all_data.append(
                temp
            )


# ================================================================
# 13. READING SUMMARY
# ================================================================

print()
print("=" * 80)
print("READING SUMMARY")
print("=" * 80)

print(
    "Total files:",
    total_files
)

print(
    "Files successfully read:",
    files_read
)

print(
    "Files containing usable X-Y-Speed data:",
    files_with_data
)

print(
    "Total trajectory blocks:",
    total_blocks
)


# ================================================================
# 14. CHECK WHETHER DATA WERE EXTRACTED
# ================================================================

if len(all_data) == 0:

    print()
    print("=" * 80)
    print("ERROR - NO DATA EXTRACTED")
    print("=" * 80)

    print(
        "No complete Trajectory + y [m] + Speed [km/h] "
        "blocks were found."
    )

else:

    # ============================================================
    # 15. COMBINE ALL DATA
    # ============================================================

    data = pd.concat(
        all_data,
        ignore_index=True
    )


    # ============================================================
    # 16. CONVERT TO NUMERIC
    # ============================================================

    data["X"] = pd.to_numeric(
        data["X"],
        errors="coerce"
    )

    data["Y"] = pd.to_numeric(
        data["Y"],
        errors="coerce"
    )

    data["Speed"] = pd.to_numeric(
        data["Speed"],
        errors="coerce"
    )


    # ============================================================
    # 17. REMOVE INVALID DATA
    # ============================================================

    data = data.dropna(
        subset=[
            "X",
            "Y",
            "Speed"
        ]
    )


    data = data[
        np.isfinite(data["X"])
        &
        np.isfinite(data["Y"])
        &
        np.isfinite(data["Speed"])
    ].copy()


    print()
    print("=" * 80)
    print("VALID DATA")
    print("=" * 80)

    print(
        "Valid trajectory observations:",
        len(data)
    )


    # ============================================================
    # 18. CREATE VEHICLE ID
    # ============================================================

    data["Vehicle_ID"] = (
        data["Source_File"].astype(str)
        + "_"
        + data["Track_ID"].astype(str)
    )


    # ============================================================
    # 19. VEHICLE-LEVEL MEANS
    # ============================================================
    #
    # One row = one vehicle.
    #
    # This avoids vehicles with more trajectory observations
    # receiving greater weight.
    # ============================================================

    vehicle_level = (
        data
        .groupby(
            [
                "Vehicle_ID",
                "Vehicle_Type"
            ]
        )
        .agg(
            Mean_Speed=(
                "Speed",
                "mean"
            ),

            Mean_Y=(
                "Y",
                "mean"
            ),

            Observations=(
                "Speed",
                "count"
            )
        )
        .reset_index()
    )


    # ============================================================
    # 20. CREATE MEDIAN-DIRECTION VARIABLE
    # ============================================================
    #
    # Higher value always means closer to median.
    # ============================================================

    if MEDIAN_SIDE == "higher_y":

        vehicle_level["Median_Position"] = (
            vehicle_level["Mean_Y"]
        )

    elif MEDIAN_SIDE == "lower_y":

        vehicle_level["Median_Position"] = (
            -vehicle_level["Mean_Y"]
        )

    else:

        raise ValueError(
            "MEDIAN_SIDE must be 'higher_y' or 'lower_y'"
        )


    # ============================================================
    # 21. TABLE 1 - VEHICLE TYPE SUMMARY
    # ============================================================

    vehicle_summary = (
        vehicle_level
        .groupby("Vehicle_Type")
        .agg(
            Number_of_Vehicles=(
                "Vehicle_ID",
                "count"
            ),

            Mean_Speed=(
                "Mean_Speed",
                "mean"
            ),

            SD_Speed=(
                "Mean_Speed",
                "std"
            ),

            Mean_Y_Position=(
                "Mean_Y",
                "mean"
            ),

            SD_Y_Position=(
                "Mean_Y",
                "std"
            )
        )
        .reindex(
            VEHICLE_FOLDERS.keys()
        )
    )


    print()
    print("=" * 80)
    print("TABLE 1 - VEHICLE TYPE SUMMARY")
    print("=" * 80)

    print(
        vehicle_summary.round(
            3
        ).to_string()
    )


    # ============================================================
    # 22. SPEARMAN CORRELATION
    # ============================================================

    rho, p_value = spearmanr(
        vehicle_level["Mean_Speed"],
        vehicle_level["Median_Position"]
    )


    print()
    print("=" * 80)
    print("SPEARMAN RANK CORRELATION TEST")
    print("=" * 80)

    print()

    print("H0:")
    print(
        "There is no significant relationship between "
        "vehicle speed and lateral position."
    )

    print()

    print("H1:")
    print(
        "There is a significant relationship between "
        "vehicle speed and lateral position."
    )

    print()

    print(
        "Spearman correlation coefficient (rho) =",
        round(
            rho,
            4
        )
    )


    if p_value < 0.001:

        print(
            "p-value = < 0.001"
        )

    else:

        print(
            "p-value =",
            round(
                p_value,
                6
            )
        )


    print(
        "Significance level (alpha) =",
        ALPHA
    )


    # ============================================================
    # 23. HYPOTHESIS DECISION
    # ============================================================

    print()
    print("=" * 80)
    print("HYPOTHESIS DECISION")
    print("=" * 80)


    if p_value < ALPHA:

        print(
            "REJECT H0"
        )

        print(
            "There is a statistically significant relationship "
            "between speed and lateral position."
        )

    else:

        print(
            "FAIL TO REJECT H0"
        )

        print(
            "There is no statistically significant relationship "
            "between speed and lateral position."
        )


    # ============================================================
    # 24. DIRECTION
    # ============================================================

    print()
    print("=" * 80)
    print("DIRECTION OF RELATIONSHIP")
    print("=" * 80)


    if rho > 0:

        print(
            "Positive relationship."
        )

        print(
            "As speed increases, vehicles tend to move "
            "toward the median side."
        )


    elif rho < 0:

        print(
            "Negative relationship."
        )

        print(
            "As speed increases, vehicles tend to move "
            "toward the shoulder side."
        )


    else:

        print(
            "No clear relationship."
        )


    # ============================================================
    # 25. GRAPH 1 - X-Y TRAJECTORY MAP
    # ============================================================

    plt.figure(
        figsize=(13, 8)
    )


    for vehicle in VEHICLE_FOLDERS.keys():

        subset = data[
            data["Vehicle_Type"] == vehicle
        ]


        if len(subset) == 0:
            continue


        plt.scatter(
            subset["X"],
            subset["Y"],
            s=2,
            alpha=0.25,
            label=vehicle
        )


    plt.xlabel(
        "Trajectory X-coordinate (m)"
    )

    plt.ylabel(
        "Y-position (m)"
    )

    plt.title(
        "X-Y Vehicle Trajectory Map"
    )

    plt.legend(
        markerscale=4
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "Spearman_01_XY_Trajectory_Map.png"
        ),
        dpi=300
    )


    plt.show()


    print()
    print("=" * 80)
    print("GRAPH 1 INSIGHT")
    print("=" * 80)

    print(
        "The X-Y trajectory map shows the spatial paths "
        "followed by different vehicle types."
    )

    print(
        "The Y-axis represents the lateral position across "
        "the road."
    )

    print(
        "The graph helps visually identify whether vehicle "
        "types occupy different lateral positions."
    )


    # ============================================================
    # 26. GRAPH 2 - SPEED SPATIAL MAP
    # ============================================================

    plt.figure(
        figsize=(13, 8)
    )


    speed_plot = plt.scatter(
        data["X"],
        data["Y"],
        c=data["Speed"],
        s=4,
        alpha=0.45
    )


    plt.xlabel(
        "Trajectory X-coordinate (m)"
    )

    plt.ylabel(
        "Y-position (m)"
    )

    plt.title(
        "Vehicle Trajectory Map Coloured by Speed"
    )


    colorbar = plt.colorbar(
        speed_plot
    )

    colorbar.set_label(
        "Speed (km/h)"
    )


    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "Spearman_02_Speed_Spatial_Map.png"
        ),
        dpi=300
    )


    plt.show()


    print()
    print("=" * 80)
    print("GRAPH 2 INSIGHT")
    print("=" * 80)

    print(
        "The speed spatial map shows where slow and fast "
        "vehicle observations occur."
    )

    print(
        "It provides visual evidence about whether higher "
        "speeds are concentrated toward the median side."
    )


    # ============================================================
    # 27. GRAPH 3 - MAIN SPEARMAN GRAPH
    # ============================================================

    plt.figure(
        figsize=(11, 7)
    )


    for vehicle in VEHICLE_FOLDERS.keys():

        subset = vehicle_level[
            vehicle_level["Vehicle_Type"] == vehicle
        ]


        if len(subset) == 0:
            continue


        plt.scatter(
            subset["Mean_Speed"],
            subset["Median_Position"],
            s=30,
            alpha=0.5,
            label=vehicle
        )


    # ------------------------------------------------------------
    # Trend line
    # ------------------------------------------------------------

    x_values = vehicle_level[
        "Mean_Speed"
    ].values


    y_values = vehicle_level[
        "Median_Position"
    ].values


    if len(x_values) >= 2:

        slope, intercept = np.polyfit(
            x_values,
            y_values,
            1
        )


        x_line = np.linspace(
            x_values.min(),
            x_values.max(),
            100
        )


        y_line = (
            intercept
            +
            slope * x_line
        )


        plt.plot(
            x_line,
            y_line,
            linewidth=2,
            label="Overall trend"
        )


    plt.xlabel(
        "Mean Speed (km/h)"
    )

    plt.ylabel(
        "Position toward Median"
    )

    plt.title(
        "Spearman Analysis: Speed vs Lateral Position"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "Spearman_03_Speed_vs_Lateral_Position.png"
        ),
        dpi=300
    )


    plt.show()


    # ============================================================
    # 28. GRAPH 3 INSIGHT
    # ============================================================

    print()
    print("=" * 80)
    print("GRAPH 3 INSIGHT - MAIN GRAPH")
    print("=" * 80)

    print(
        "This is the main graph for the research question."
    )

    print(
        "Each point represents one vehicle using its mean "
        "speed and mean lateral position."
    )


    if rho > 0:

        print(
            "The overall trend is upward."
        )

        print(
            "Higher-speed vehicles tend to occupy positions "
            "closer to the median."
        )


    elif rho < 0:

        print(
            "The overall trend is downward."
        )

        print(
            "Higher-speed vehicles tend to occupy positions "
            "closer to the shoulder."
        )


    else:

        print(
            "No clear trend is visible."
        )


    print()

    print(
        "Spearman rho =",
        round(
            rho,
            4
        )
    )


    # ============================================================
    # 29. AUTOMATIC FINAL CONCLUSION
    # ============================================================

    print()
    print("=" * 80)
    print("FINAL RESULT AND CONCLUSION")
    print("=" * 80)


    if (
        p_value < ALPHA
        and
        rho > 0
    ):

        print(
            "RESULT: THE ASSUMPTION IS SUPPORTED."
        )

        print()

        print(
            "The Spearman test shows a statistically significant "
            "positive relationship between vehicle speed and "
            "lateral position."
        )

        print(
            "Therefore, higher-speed vehicles tend to travel "
            "closer to the median side."
        )

        print(
            "This supports the assumption that slower MTWs and "
            "3-Ws tend to travel closer to the shoulder."
        )


    elif (
        p_value < ALPHA
        and
        rho < 0
    ):

        print(
            "RESULT: THE ASSUMPTION IS NOT SUPPORTED."
        )

        print()

        print(
            "A statistically significant relationship exists, "
            "but the direction is opposite to the proposed assumption."
        )

        print(
            "Higher-speed vehicles tend to be closer to the "
            "shoulder side."
        )


    else:

        print(
            "RESULT: THE ASSUMPTION IS NOT STATISTICALLY CONFIRMED."
        )

        print()

        print(
            "The Spearman test does not provide sufficient "
            "statistical evidence of a relationship between "
            "vehicle speed and lateral position."
        )

        print(
            "Therefore, the available data do not statistically "
            "confirm that slow-moving MTWs and 3-Ws travel closer "
            "to the shoulder while faster vehicles travel closer "
            "to the median."
        )


    # ============================================================
    # 30. SAVE RESULTS
    # ============================================================

    vehicle_summary.round(
        4
    ).to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "Spearman_Vehicle_Type_Statistics.csv"
        )
    )


    vehicle_level.round(
        4
    ).to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "Spearman_Vehicle_Level_Data.csv"
        ),
        index=False
    )


    data.round(
        4
    ).to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "Spearman_XY_Speed_Data.csv"
        ),
        index=False
    )


    # ============================================================
    # 31. SAVE SPEARMAN RESULT
    # ============================================================

    result_table = pd.DataFrame(
        {
            "Test": [
                "Spearman Rank Correlation"
            ],

            "Spearman_Rho": [
                rho
            ],

            "p_value": [
                p_value
            ],

            "Alpha": [
                ALPHA
            ],

            "Decision": [
                "Reject H0"
                if p_value < ALPHA
                else
                "Fail to reject H0"
            ]
        }
    )


    result_table.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "Spearman_Test_Result.csv"
        ),
        index=False
    )


    # ============================================================
    # 32. SAVE EXCEL
    # ============================================================

    excel_file = os.path.join(
        OUTPUT_FOLDER,
        "Spearman_Speed_Lateral_Position_Analysis.xlsx"
    )


    with pd.ExcelWriter(
        excel_file,
        engine="openpyxl"
    ) as writer:

        vehicle_summary.round(
            4
        ).to_excel(
            writer,
            sheet_name="Vehicle Summary"
        )


        vehicle_level.round(
            4
        ).to_excel(
            writer,
            sheet_name="Vehicle Level",
            index=False
        )


        result_table.round(
            6
        ).to_excel(
            writer,
            sheet_name="Spearman Test",
            index=False
        )


    # ============================================================
    # 33. FINAL OUTPUT
    # ============================================================

    print()
    print("=" * 80)
    print("OUTPUT FILES CREATED")
    print("=" * 80)

    print(
        "Spearman_01_XY_Trajectory_Map.png"
    )

    print(
        "Spearman_02_Speed_Spatial_Map.png"
    )

    print(
        "Spearman_03_Speed_vs_Lateral_Position.png"
    )

    print(
        "Spearman_Vehicle_Type_Statistics.csv"
    )

    print(
        "Spearman_Vehicle_Level_Data.csv"
    )

    print(
        "Spearman_XY_Speed_Data.csv"
    )

    print(
        "Spearman_Test_Result.csv"
    )

    print(
        "Spearman_Speed_Lateral_Position_Analysis.xlsx"
    )


    print()
    print("=" * 80)
    print("ANALYSIS COMPLETED")
    print("=" * 80)
