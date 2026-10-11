"""
Exploratory Analysis of Machine Usage Time-Series.

This exploratory script inspects time-series resource utilization metrics in machine_usage. It analyzes CPU and memory
load dynamics over time across physical nodes in the cluster trace.
"""

import pandas as pd
import numpy as np
import os

PROJECT_PATH = ("C:/Users/gvkva/Desktop/My ongoing projects/""Machine Learning Project/""Adaptive-Multi-Core-CPU-Scheduler-RL")
MACHINE_USAGE_PATH = os.path.join(PROJECT_PATH,"dataset","raw","machine_usage.csv")
UNDERSTANDING_PATH = os.path.join(PROJECT_PATH,"dataset","understanding_of_dataset")
MACHINE_USAGE_UNDERSTANDING_PATH = os.path.join(UNDERSTANDING_PATH,"machine_usage")
os.makedirs(MACHINE_USAGE_UNDERSTANDING_PATH,exist_ok=True)

CHUNK_SIZE = 100000

print("DATASET PATH")
print(MACHINE_USAGE_PATH)

print("\nFILE EXISTENCE CHECK")
if not os.path.exists(MACHINE_USAGE_PATH):
    print("Dataset file does not exist.")
    raise FileNotFoundError(MACHINE_USAGE_PATH)
print("Dataset file exists.")

file_size_bytes = os.path.getsize(MACHINE_USAGE_PATH)
file_size_gb = file_size_bytes / (1024 ** 3)
print("Dataset size in GB:",round(file_size_gb,2))

columns = ["machine_id","time_stamp","cpu_util_percent","mem_util_percent","mem_gps","mkpi","net_in","net_out","disk_io_percent"]
number_of_columns = len(columns)

numeric_columns = ["time_stamp","cpu_util_percent","mem_util_percent","mem_gps","mkpi","net_in","net_out","disk_io_percent"]
id_columns = ["machine_id"]
timestamp_column = "time_stamp"
resource_columns = ["cpu_util_percent","mem_util_percent","mem_gps","mkpi","net_in","net_out","disk_io_percent"]

print("\nFULL-FILE ANALYSIS")

number_of_rows = 0
full_missing_values = pd.Series(0,dtype="int64",index=columns)

numeric_min_values = pd.Series(np.inf,index=numeric_columns,dtype="float64")
numeric_max_values = pd.Series(-np.inf,index=numeric_columns,dtype="float64")
numeric_sum_values = pd.Series(0.0,index=numeric_columns,dtype="float64")
numeric_sum_squares = pd.Series(0.0,index=numeric_columns,dtype="float64")
numeric_valid_counts = pd.Series(0,dtype="int64",index=numeric_columns)
numeric_invalid_counts = pd.Series(0,dtype="int64",index=numeric_columns)
numeric_negative_counts = pd.Series(0,dtype="int64",index=numeric_columns)
numeric_above_100_counts = pd.Series(0,dtype="int64",index=numeric_columns)

unique_machine_ids = set()

timestamp_min = None
timestamp_max = None
timestamp_valid_count = 0
timestamp_invalid_count = 0
timestamp_negative_count = 0

progress_rows = 0
chunk_number = 0
first_chunk_dtypes = None

try:
    for chunk in pd.read_csv(MACHINE_USAGE_PATH,names=columns,header=None,chunksize=CHUNK_SIZE,low_memory=False):
        chunk_number += 1
        chunk_rows = len(chunk)
        number_of_rows += chunk_rows
        progress_rows += chunk_rows

        if first_chunk_dtypes is None:
            first_chunk_dtypes = chunk.dtypes.astype(str)

        chunk_missing_values = chunk.isnull().sum()
        full_missing_values = full_missing_values.add(chunk_missing_values,fill_value=0)

        for column in numeric_columns:
            values = pd.to_numeric(chunk[column],errors="coerce")
            original_missing = chunk[column].isna()
            valid_values = values.dropna()
            invalid_count = values.isna().sum() - original_missing.sum()
            numeric_invalid_counts[column] += int(invalid_count)

            if len(valid_values) > 0:
                numeric_min_values[column] = min(numeric_min_values[column],valid_values.min())
                numeric_max_values[column] = max(numeric_max_values[column],valid_values.max())
                numeric_sum_values[column] += valid_values.sum()
                numeric_sum_squares[column] += np.square(valid_values).sum()
                numeric_valid_counts[column] += len(valid_values)
                numeric_negative_counts[column] += int((valid_values < 0).sum())
                numeric_above_100_counts[column] += int((valid_values > 100).sum())

        values = chunk["machine_id"].dropna()
        if len(values) > 0:
            unique_machine_ids.update(values.tolist())

        values = pd.to_numeric(chunk[timestamp_column],errors="coerce")
        original_missing = chunk[timestamp_column].isna()
        valid_values = values.dropna()
        invalid_count = values.isna().sum() - original_missing.sum()
        timestamp_invalid_count += int(invalid_count)

        if len(valid_values) > 0:
            current_min = valid_values.min()
            current_max = valid_values.max()

            if timestamp_min is None or current_min < timestamp_min:
                timestamp_min = current_min

            if timestamp_max is None or current_max > timestamp_max:
                timestamp_max = current_max

            timestamp_valid_count += len(valid_values)
            timestamp_negative_count += int((valid_values < 0).sum())

        if chunk_number % 10 == 0:
            print("Processed chunks:",chunk_number,"| Rows processed:",progress_rows)

except Exception as e:
    print("\nError during full-file analysis:")
    print(type(e).__name__,":",e)
    raise

print("\nFULL-FILE ANALYSIS COMPLETED")
print("Total chunks processed:",chunk_number)
print("Total rows:",number_of_rows)

print("\n1. COMPLETE NUMBER OF ROWS")
print("Total rows:",number_of_rows)

print("\n2. COMPLETE NUMBER OF COLUMNS")
print("Total columns:",number_of_columns)

print("\n3. COMPLETE COLUMN NAMES")
for index,column in enumerate(columns,1):
    print(index,".",column)

print("\n4. COMPLETE DATA TYPES")
print(first_chunk_dtypes.to_string())
first_chunk_dtypes.to_csv(os.path.join(MACHINE_USAGE_UNDERSTANDING_PATH,"machine_usage_dtypes.csv"),header=["dtype"])

print("\n5. COMPLETE MISSING VALUES")
full_missing_values = full_missing_values.astype("int64")
print(full_missing_values.to_string())
full_missing_values.to_csv(os.path.join(MACHINE_USAGE_UNDERSTANDING_PATH,"machine_usage_full_missing_values.csv"),header=["missing_count"])

print("\n6. COMPLETE UNIQUE VALUES FOR ID COLUMNS")
complete_id_unique_counts = {}
for column in id_columns:
    complete_id_unique_counts[column] = len(unique_machine_ids)
    print(column,":",complete_id_unique_counts[column])

pd.Series(complete_id_unique_counts).to_csv(os.path.join(MACHINE_USAGE_UNDERSTANDING_PATH,"machine_usage_id_unique_counts.csv"),header=["unique_count"])

print("\n7. DUPLICATE RECORDS")
print("Duplicate analysis is not performed on the complete file because it needs a lot of memory.")

print("\n8. COMPLETE MINIMUM AND MAXIMUM VALUES")
complete_min_max = pd.DataFrame({"min":numeric_min_values,"max":numeric_max_values})
print(complete_min_max.to_string())
complete_min_max.to_csv(os.path.join(MACHINE_USAGE_UNDERSTANDING_PATH,"machine_usage_full_min_max.csv"))

print("\n9. COMPLETE BASIC STATISTICS")
complete_statistics = pd.DataFrame(index=numeric_columns)
complete_statistics["count"] = numeric_valid_counts
complete_statistics["mean"] = numeric_sum_values / numeric_valid_counts.replace(0,np.nan)
variance_values = (numeric_sum_squares - (numeric_sum_values ** 2) / numeric_valid_counts.replace(0,np.nan)) / (numeric_valid_counts.replace(0,np.nan) - 1)
variance_values = variance_values.clip(lower=0)
complete_statistics["std"] = np.sqrt(variance_values)
complete_statistics["min"] = numeric_min_values
complete_statistics["max"] = numeric_max_values
print(complete_statistics.to_string())
complete_statistics.to_csv(os.path.join(MACHINE_USAGE_UNDERSTANDING_PATH,"machine_usage_full_statistics.csv"))

print("\n10. COMPLETE ID RELATIONSHIPS")
for column in id_columns:
    print(column,":",complete_id_unique_counts[column],"unique values across the complete file.")

print("\n11. COMPLETE TIMESTAMP RANGE")
complete_timestamp_statistics = pd.DataFrame({
    "min":[timestamp_min],
    "max":[timestamp_max],
    "valid_count":[timestamp_valid_count],
    "invalid_count":[timestamp_invalid_count],
    "negative_count":[timestamp_negative_count]
},index=[timestamp_column])
print(complete_timestamp_statistics.to_string())
complete_timestamp_statistics.to_csv(os.path.join(MACHINE_USAGE_UNDERSTANDING_PATH,"machine_usage_full_timestamp_statistics.csv"))

print("\n12. COMPLETE RESOURCE-RELATED FIELD ANALYSIS")
for column in resource_columns:
    print(column,": minimum =",numeric_min_values[column],"| maximum =",numeric_max_values[column],"| negative values =",numeric_negative_counts[column],"| values above 100 =",numeric_above_100_counts[column])


print("\n13. IMPOSSIBLE OR INVALID VALUES")
invalid_summary = pd.DataFrame(index=numeric_columns)
invalid_summary["invalid_numeric_values"] = numeric_invalid_counts
invalid_summary["negative_values"] = numeric_negative_counts
invalid_summary["values_above_100"] = numeric_above_100_counts
print("\nInvalid numeric values:")
print(invalid_summary.to_string())
invalid_summary.to_csv(os.path.join(MACHINE_USAGE_UNDERSTANDING_PATH,"machine_usage_invalid_numeric_values.csv"))

timestamp_invalid_summary = pd.DataFrame({
    "invalid_timestamp_values":[timestamp_invalid_count],
    "negative_timestamp_values":[timestamp_negative_count]
},index=[timestamp_column])
print("\nInvalid timestamp values:")
print(timestamp_invalid_summary.to_string())
timestamp_invalid_summary.to_csv(os.path.join(MACHINE_USAGE_UNDERSTANDING_PATH,"machine_usage_invalid_timestamps.csv"))

disk_io_invalid = pd.DataFrame({
    "negative_values":[numeric_negative_counts["disk_io_percent"]],
    "values_above_100":[numeric_above_100_counts["disk_io_percent"]]
},index=["disk_io_percent"])
print("\nDisk I/O invalid values:")
print(disk_io_invalid.to_string())
disk_io_invalid.to_csv(os.path.join(MACHINE_USAGE_UNDERSTANDING_PATH,"machine_usage_disk_io_invalid_values.csv"))

print("\nANALYSIS OUTPUT LOCATION")
print(MACHINE_USAGE_UNDERSTANDING_PATH)

print("\nDATASET UNDERSTANDING STAGE COMPLETED")
