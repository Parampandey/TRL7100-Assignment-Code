# ================================================================
# CORRECT ROAD-POSITION ACCELERATION / DECELERATION GRAPH
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ------------------------------------------------
# 1. USE ACTUAL TRAJECTORY POSITION
# ------------------------------------------------

data["Trajectory"] = pd.to_numeric(
    data["Trajectory"],
    errors="coerce"
)

data["Tan_Acceleration"] = pd.to_numeric(
    data["Tan_Acceleration"],
    errors="coerce"
)

data = data.dropna(
    subset=["Trajectory", "Tan_Acceleration"]
)


# ------------------------------------------------
# 2. ACCELERATION / DECELERATION CLASSIFICATION
# ------------------------------------------------

ACC_THRESHOLD = 0.10
DEC_THRESHOLD = -0.10

data["Motion_State"] = np.where(
    data["Tan_Acceleration"] > ACC_THRESHOLD,
    "Acceleration",
    np.where(
        data["Tan_Acceleration"] < DEC_THRESHOLD,
        "Deceleration",
        "Approximately Constant"
    )
)


# ------------------------------------------------
# 3. CREATE ACTUAL ROAD POSITION SEGMENTS
# ------------------------------------------------
#
# IMPORTANT:
# Do NOT subtract the minimum separately for every file.
# We use the actual Trajectory coordinate.
#
# Change this if you want 25 m or 100 m sections.
# ------------------------------------------------

SEGMENT_LENGTH = 10   # 10 metre road sections

min_position = np.floor(
    data["Trajectory"].min() / SEGMENT_LENGTH
) * SEGMENT_LENGTH

data["Road_Segment_Start"] = (
    np.floor(
        (data["Trajectory"] - min_position)
        / SEGMENT_LENGTH
    ) * SEGMENT_LENGTH
    + min_position
)

data["Road_Segment_End"] = (
    data["Road_Segment_Start"]
    + SEGMENT_LENGTH
)

data["Road_Segment"] = (
    data["Road_Segment_Start"].round(1).astype(str)
    + "–"
    + data["Road_Segment_End"].round(1).astype(str)
    + " m"
)


# ------------------------------------------------
# 4. CALCULATE STATISTICS FOR EACH ROAD PORTION
# ------------------------------------------------

road_stats = (
    data.groupby(
        [
            "Vehicle_Type",
            "Road_Segment_Start",
            "Road_Segment_End"
        ]
    )
    .agg(
        Observations=("Tan_Acceleration", "size"),

        Mean_Acceleration=(
            "Tan_Acceleration",
            "mean"
        ),

        Median_Acceleration=(
            "Tan_Acceleration",
            "median"
        ),

        Maximum_Acceleration=(
            "Tan_Acceleration",
            "max"
        ),

        Minimum_Acceleration=(
            "Tan_Acceleration",
            "min"
        ),

        Acceleration_Count=(
            "Motion_State",
            lambda x: (x == "Acceleration").sum()
        ),

        Deceleration_Count=(
            "Motion_State",
            lambda x: (x == "Deceleration").sum()
        )
    )
    .reset_index()
)


# ------------------------------------------------
# 5. CALCULATE PERCENTAGES
# ------------------------------------------------

road_stats["Acceleration_Percent"] = (
    road_stats["Acceleration_Count"]
    / road_stats["Observations"]
    * 100
)

road_stats["Deceleration_Percent"] = (
    road_stats["Deceleration_Count"]
    / road_stats["Observations"]
    * 100
)


# ------------------------------------------------
# 6. ROAD POSITION MIDPOINT
# ------------------------------------------------

road_stats["Road_Position"] = (
    road_stats["Road_Segment_Start"]
    + SEGMENT_LENGTH / 2
)


# ------------------------------------------------
# 7. SORT DATA
# ------------------------------------------------

road_stats = road_stats.sort_values(
    [
        "Vehicle_Type",
        "Road_Position"
    ]
)


# ================================================================
# GRAPH 1
# MAIN GRAPH: ACTUAL ROAD POSITION VS MEAN ACCELERATION
# ================================================================

plt.figure(figsize=(16, 9))

vehicle_types = [
    "Car",
    "3-W (Tuk-Tuk)",
    "MTW (Motorcycles)",
    "Bus",
    "Light Truck"
]

for vehicle in vehicle_types:

    temp = road_stats[
        road_stats["Vehicle_Type"] == vehicle
    ].copy()

    if len(temp) == 0:
        continue

    plt.plot(
        temp["Road_Position"],
        temp["Mean_Acceleration"],
        marker="o",
        linewidth=2,
        markersize=4,
        label=vehicle
    )


# Zero line = boundary between acceleration and deceleration
plt.axhline(
    0,
    linestyle="--",
    linewidth=2,
    label="Zero acceleration"
)

plt.axhline(
    ACC_THRESHOLD,
    linestyle=":",
    linewidth=1
)

plt.axhline(
    DEC_THRESHOLD,
    linestyle=":",
    linewidth=1
)

plt.xlabel(
    "Actual road position / Trajectory (m)",
    fontsize=13
)

plt.ylabel(
    "Mean tangential acceleration (m/s²)",
    fontsize=13
)

plt.title(
    "Acceleration and Deceleration at Different Road Portions",
    fontsize=16,
    fontweight="bold"
)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


main_graph = os.path.join(
    OUTPUT_FOLDER,
    "FINAL_Actual_Road_Position_Acceleration_Deceleration.png"
)

plt.savefig(
    main_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ================================================================
# GRAPH 2
# ACCELERATION PERCENTAGE BY ACTUAL ROAD POSITION
# ================================================================

plt.figure(figsize=(16, 9))

for vehicle in vehicle_types:

    temp = road_stats[
        road_stats["Vehicle_Type"] == vehicle
    ].copy()

    if len(temp) == 0:
        continue

    plt.plot(
        temp["Road_Position"],
        temp["Acceleration_Percent"],
        marker="o",
        linewidth=2,
        markersize=4,
        label=vehicle
    )

plt.xlabel(
    "Actual road position / Trajectory (m)",
    fontsize=13
)

plt.ylabel(
    "Accelerating observations (%)",
    fontsize=13
)

plt.title(
    "Acceleration Activity at Different Road Portions",
    fontsize=16,
    fontweight="bold"
)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

acc_graph = os.path.join(
    OUTPUT_FOLDER,
    "FINAL_Acceleration_Percentage_Road_Position.png"
)

plt.savefig(
    acc_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ================================================================
# GRAPH 3
# DECELERATION PERCENTAGE BY ACTUAL ROAD POSITION
# ================================================================

plt.figure(figsize=(16, 9))

for vehicle in vehicle_types:

    temp = road_stats[
        road_stats["Vehicle_Type"] == vehicle
    ].copy()

    if len(temp) == 0:
        continue

    plt.plot(
        temp["Road_Position"],
        temp["Deceleration_Percent"],
        marker="o",
        linewidth=2,
        markersize=4,
        label=vehicle
    )

plt.xlabel(
    "Actual road position / Trajectory (m)",
    fontsize=13
)

plt.ylabel(
    "Decelerating observations (%)",
    fontsize=13
)

plt.title(
    "Deceleration Activity at Different Road Portions",
    fontsize=16,
    fontweight="bold"
)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

dec_graph = os.path.join(
    OUTPUT_FOLDER,
    "FINAL_Deceleration_Percentage_Road_Position.png"
)

plt.savefig(
    dec_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ================================================================
# 8. IDENTIFY MAIN ACCELERATION / DECELERATION PORTIONS
# ================================================================

print("\n")
print("=" * 100)
print("FINAL ROAD PORTION RESULTS")
print("=" * 100)

final_results = []

for vehicle in vehicle_types:

    temp = road_stats[
        road_stats["Vehicle_Type"] == vehicle
    ].copy()

    if len(temp) == 0:
        continue

    # Highest acceleration percentage
    max_acc = temp.loc[
        temp["Acceleration_Percent"].idxmax()
    ]

    # Highest deceleration percentage
    max_dec = temp.loc[
        temp["Deceleration_Percent"].idxmax()
    ]

    # Highest positive mean acceleration
    strongest_acc = temp.loc[
        temp["Mean_Acceleration"].idxmax()
    ]

    # Most negative mean acceleration
    strongest_dec = temp.loc[
        temp["Mean_Acceleration"].idxmin()
    ]

    result = {
        "Vehicle_Type": vehicle,

        "Main_Acceleration_Portion":
            f"{max_acc['Road_Segment_Start']:.1f}–"
            f"{max_acc['Road_Segment_End']:.1f} m",

        "Acceleration_Percent":
            max_acc["Acceleration_Percent"],

        "Mean_Acceleration_There":
            max_acc["Mean_Acceleration"],

        "Strongest_Acceleration_Portion":
            f"{strongest_acc['Road_Segment_Start']:.1f}–"
            f"{strongest_acc['Road_Segment_End']:.1f} m",

        "Highest_Mean_Acceleration":
            strongest_acc["Mean_Acceleration"],

        "Main_Deceleration_Portion":
            f"{max_dec['Road_Segment_Start']:.1f}–"
            f"{max_dec['Road_Segment_End']:.1f} m",

        "Deceleration_Percent":
            max_dec["Deceleration_Percent"],

        "Mean_Acceleration_There_Dec":
            max_dec["Mean_Acceleration"],

        "Strongest_Deceleration_Portion":
            f"{strongest_dec['Road_Segment_Start']:.1f}–"
            f"{strongest_dec['Road_Segment_End']:.1f} m",

        "Lowest_Mean_Acceleration":
            strongest_dec["Mean_Acceleration"]
    }

    final_results.append(result)


final_results_df = pd.DataFrame(
    final_results
)

final_results_df = final_results_df.round(3)


# ================================================================
# 9. PRINT FINAL TABLE
# ================================================================

print(
    final_results_df.to_string(index=False)
)


# ================================================================
# 10. SAVE RESULTS
# ================================================================

results_file = os.path.join(
    OUTPUT_FOLDER,
    "FINAL_Road_Portions_Acceleration_Deceleration.csv"
)

final_results_df.to_csv(
    results_file,
    index=False
)


road_stats_file = os.path.join(
    OUTPUT_FOLDER,
    "FINAL_Road_Position_Segment_Statistics.csv"
)

road_stats.to_csv(
    road_stats_file,
    index=False
)


print("\n")
print("=" * 100)
print("FILES SAVED")
print("=" * 100)

print("\nMain road-position graph:")
print(main_graph)

print("\nAcceleration percentage graph:")
print(acc_graph)

print("\nDeceleration percentage graph:")
print(dec_graph)

print("\nFinal results:")
print(results_file)

print("\nComplete road-segment statistics:")
print(road_stats_file)

print("\nAnalysis completed successfully.")
