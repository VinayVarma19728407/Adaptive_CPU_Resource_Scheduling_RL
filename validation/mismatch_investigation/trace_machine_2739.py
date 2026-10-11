"""
Trace Machine 2739 Workload and Resource Utilization.

This script performs a deep-dive trace audit on a specific representative node (machine_2739) to verify temporal load
profile accuracy. It filters machine usage and instance assignments for machine 2739 and analyzes resource load dynamics.
"""

import os
import pandas as pd

PROJECT_PATH = (
    "C:/Users/gvkva/Desktop/My ongoing projects/"
    "Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL"
)

RAW_MACHINE_META = os.path.join(
    PROJECT_PATH,
    "dataset",
    "raw",
    "machine_meta.csv"
)

CLEAN_MACHINE_META = os.path.join(
    PROJECT_PATH,
    "dataset",
    "cleaned",
    "machine_meta.csv"
)

PREPROCESSED_MACHINE_META = os.path.join(
    PROJECT_PATH,
    "dataset",
    "preprocessed",
    "machine_meta.csv"
)

COLUMNS = [
    "machine_id",
    "time_stamp",
    "failure_domain_1",
    "failure_domain_2",
    "cpu_num",
    "mem_size",
    "status"
]

TARGET = "2739"

print()
print("MACHINE 2739 SOURCE TRACE")



# RAW


print()
print("RAW MACHINE_META")

raw = pd.read_csv(
    RAW_MACHINE_META,
    header=None,
    names=COLUMNS,
    dtype="string",
    low_memory=False
)

raw["machine_id"] = (
    raw["machine_id"]
    .astype("string")
    .str.strip()
    .str.replace(r"^m_", "", regex=True)
)

raw_match = raw.loc[
    raw["machine_id"].eq(TARGET)
].copy()

print("Rows for machine 2739:", len(raw_match))

if len(raw_match) > 0:
    print()
    print(raw_match.to_string(index=False))

    print()
    print("STATUS COUNTS")
    print(raw_match["status"].value_counts(dropna=False).to_string())

    print()
    print("DUPLICATE ROWS")
    print(
        "Duplicate rows:",
        int(raw_match.duplicated().sum())
    )

else:
    print("Machine 2739 does NOT exist in raw machine_meta.")


# CLEANING EFFECT


print()
print("CHECKING CLEANING CONDITIONS")

if len(raw_match) > 0:

    importing = raw_match["status"].eq("IMPORT_INSTALLING")

    print(
        "Rows with status IMPORT_INSTALLING:",
        int(importing.sum())
    )

    print(
        "Rows that would remain after status filtering:",
        int((~importing).sum())
    )

    print(
        "Duplicate rows within machine 2739:",
        int(raw_match.duplicated().sum())
    )


# CLEANED


print()
print("CLEANED MACHINE_META")

clean = pd.read_csv(
    CLEAN_MACHINE_META,
    dtype="string",
    low_memory=False
)

clean["machine_id"] = (
    clean["machine_id"]
    .astype("string")
    .str.strip()
)

clean_match = clean.loc[
    clean["machine_id"].eq(TARGET)
].copy()

print("Rows for machine 2739:", len(clean_match))

if len(clean_match) > 0:
    print()
    print(clean_match.to_string(index=False))


# PREPROCESSED


print()
print("PREPROCESSED MACHINE_META")

preprocessed = pd.read_csv(
    PREPROCESSED_MACHINE_META,
    dtype="string",
    low_memory=False
)

preprocessed["machine_id"] = (
    preprocessed["machine_id"]
    .astype("string")
    .str.strip()
)

preprocessed_match = preprocessed.loc[
    preprocessed["machine_id"].eq(TARGET)
].copy()

print(
    "Rows for machine 2739:",
    len(preprocessed_match)
)

if len(preprocessed_match) > 0:
    print()
    print(preprocessed_match.to_string(index=False))


# FINAL CLASSIFICATION


print()
print("CLASSIFICATION")


if len(raw_match) == 0:

    print(
        "RESULT: Machine 2739 is absent from the RAW machine_meta dataset."
    )

elif len(clean_match) == 0:

    if (raw_match["status"] == "IMPORT_INSTALLING").all():
        print(
            "RESULT: Machine 2739 existed in RAW, "
            "but all its records were removed because "
            "their status was IMPORT_INSTALLING."
        )

    elif raw_match.duplicated().all():
        print(
            "RESULT: Machine 2739 existed in RAW, "
            "but its records were removed as duplicates."
        )

    else:
        print(
            "RESULT: Machine 2739 existed in RAW but is absent "
            "from CLEANED metadata. Additional investigation required."
        )

elif len(preprocessed_match) == 0:

    print(
        "RESULT: Machine 2739 exists in CLEANED metadata "
        "but is absent from PREPROCESSED metadata."
    )

else:

    print(
        "RESULT: Machine 2739 exists through the full "
        "raw → cleaned → preprocessed pipeline."
    )


print()
print("TRACE COMPLETED")
