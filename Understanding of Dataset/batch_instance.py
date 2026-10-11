"""
Exploratory Analysis of Batch Instance Data.

This exploratory script reads and inspects raw batch_instance trace files. It displays schema information, sample rows,
basic statistical distributions, and missing value counts for task instance records.
"""

import os
import tarfile
import pandas as pd
import numpy as np

PROJECT_PATH = ("C:/Users/gvkva/Desktop/My ongoing projects/""Machine Learning Project/""Adaptive-Multi-Core-CPU-Scheduler-RL")
BATCH_INSTANCE_TAR_PATH = os.path.join("C:/Users/gvkva/Desktop/My ongoing projects/","Machine Learning Project","batch_instance.tar.gz")
UNDERSTANDING_PATH = os.path.join(PROJECT_PATH,"Understanding of Dataset")

os.makedirs(UNDERSTANDING_PATH,exist_ok=True)

CHUNK_SIZE = 250000

columns = ["instance_name","task_name","job_name","task_type","status","start_time","end_time","machine_id","seq_no","total_seq_no","cpu_avg","cpu_max","mem_avg","mem_max"]

dtype_map = {
    "instance_name":"string",
    "task_name":"string",
    "job_name":"string",
    "task_type":"string",
    "status":"string",
    "start_time":"float64",
    "end_time":"float64",
    "machine_id":"string",
    "seq_no":"float32",
    "total_seq_no":"float32",
    "cpu_avg":"float32",
    "cpu_max":"float32",
    "mem_avg":"float32",
    "mem_max":"float32"
}

numeric_columns = ["start_time","end_time","seq_no","total_seq_no","cpu_avg","cpu_max","mem_avg","mem_max"]
resource_columns = ["cpu_avg","cpu_max","mem_avg","mem_max"]

if not os.path.exists(BATCH_INSTANCE_TAR_PATH):
    raise FileNotFoundError(BATCH_INSTANCE_TAR_PATH)

print("BATCH INSTANCE DATASET ANALYSIS")
print("Archive:",BATCH_INSTANCE_TAR_PATH)
print("Opening archive...")

rows = 0
chunks = 0
physical_rows_read = 0
header_rows_detected = 0

missing_counts = pd.Series(0,index=columns,dtype="int64")
unique_counts = {column:set() for column in ["instance_name","task_name","job_name","task_type","status","machine_id"]}
status_counts = {}
task_type_counts = {}

minimums = {column:np.inf for column in numeric_columns}
maximums = {column:-np.inf for column in numeric_columns}
sums = {column:0.0 for column in numeric_columns}
sum_squares = {column:0.0 for column in numeric_columns}
negative_counts = {column:0 for column in numeric_columns}

invalid_resource_counts = {column:0 for column in resource_columns}

negative_duration = 0
zero_duration = 0
positive_duration = 0
invalid_timestamp_rows = 0

with tarfile.open(BATCH_INSTANCE_TAR_PATH,"r:gz") as tar:
    csv_members = [member for member in tar.getmembers() if member.name.endswith(".csv")]

    if not csv_members:
        raise FileNotFoundError("batch_instance.csv not found inside archive.")

    csv_member = csv_members[0]

    print("CSV member:",csv_member.name)
    print("Starting chunk processing...")

    with tar.extractfile(csv_member) as csv_file:
        reader = pd.read_csv(
            csv_file,
            header=None,
            names=columns,
            chunksize=CHUNK_SIZE,
            dtype=dtype_map,
            low_memory=False,
            on_bad_lines="error"
        )

        for chunk in reader:
            physical_rows_read += len(chunk)

            header_mask = (
                chunk["instance_name"].eq("instance_name") &
                chunk["task_name"].eq("task_name") &
                chunk["job_name"].eq("job_name") &
                chunk["task_type"].eq("task_type") &
                chunk["status"].eq("status") &
                chunk["machine_id"].eq("machine_id")
            )

            if header_mask.any():
                header_rows_detected += int(header_mask.sum())
                chunk = chunk.loc[~header_mask].copy()

            if len(chunk) == 0:
                continue

            chunks += 1
            rows += len(chunk)

            missing_counts += chunk.isna().sum().astype("int64")

            for column in unique_counts:
                unique_counts[column].update(chunk[column].dropna().unique())

            status_values = chunk["status"].value_counts(dropna=True)
            for status,value in status_values.items():
                status_counts[status] = status_counts.get(status,0) + int(value)

            task_type_values = chunk["task_type"].value_counts(dropna=True)
            for task_type,value in task_type_values.items():
                task_type_counts[task_type] = task_type_counts.get(task_type,0) + int(value)

            for column in numeric_columns:
                values = chunk[column].dropna()

                if len(values) > 0:
                    minimum = float(values.min())
                    maximum = float(values.max())

                    if minimum < minimums[column]:
                        minimums[column] = minimum

                    if maximum > maximums[column]:
                        maximums[column] = maximum

                    sums[column] += float(values.sum())
                    sum_squares[column] += float(np.square(values).sum())
                    negative_counts[column] += int((values < 0).sum())

            for column in resource_columns:
                invalid_resource_counts[column] += int(((chunk[column] < 0) | (chunk[column] > 100)).sum())

            valid_timestamps = chunk["start_time"].notna() & chunk["end_time"].notna()
            invalid_timestamp_rows += int((~valid_timestamps).sum())

            duration = chunk["end_time"] - chunk["start_time"]
            negative_duration += int((duration < 0).sum())
            zero_duration += int((duration == 0).sum())
            positive_duration += int((duration > 0).sum())

            if chunks % 10 == 0:
                print("Processed chunks:",chunks,"| Rows processed:",rows,"| Physical rows read:",physical_rows_read)

print()
print("COMPLETE NUMBER OF ROWS")
print("Data rows:",rows)
print("Physical rows read:",physical_rows_read)
print("Header rows detected:",header_rows_detected)

print()
print("COMPLETE NUMBER OF COLUMNS")
print(len(columns))

print()
print("COMPLETE COLUMN NAMES")
for column in columns:
    print(column)

print()
print("COMPLETE DATA TYPES")
for column in columns:
    print(column,":",dtype_map[column])

print()
print("COMPLETE MISSING VALUES")
for column in columns:
    print(column,":",int(missing_counts[column]))

print()
print("UNIQUE VALUES")
for column in unique_counts:
    print(column,":",len(unique_counts[column]))

print()
print("STATUS VALUES")
for key in sorted(status_counts):
    print(key,":",status_counts[key])

print()
print("TASK TYPE VALUES")
for key in sorted(task_type_counts,key=str):
    print(key,":",task_type_counts[key])

print()
print("COMPLETE MINIMUM AND MAXIMUM VALUES")
for column in numeric_columns:
    print(column,":",minimums[column],maximums[column])

print()
print("COMPLETE BASIC STATISTICS")
for column in numeric_columns:
    count = rows - int(missing_counts[column])

    if count > 0:
        mean = sums[column] / count
        variance = (sum_squares[column] / count) - (mean * mean)
        std = np.sqrt(max(variance,0))
    else:
        mean = np.nan
        std = np.nan

    print(column,": count =",count,"mean =",mean,"std =",std,"min =",minimums[column],"max =",maximums[column])

print()
print("COMPLETE TIMESTAMP RANGE")
print("start_time:",minimums["start_time"],"to",maximums["start_time"])
print("end_time:",minimums["end_time"],"to",maximums["end_time"])

print()
print("TIMESTAMP DURATION CHECKS")
print("Negative duration:",negative_duration)
print("Zero duration:",zero_duration)
print("Positive duration:",positive_duration)

print()
print("COMPLETE RESOURCE-RELATED FIELD ANALYSIS")
for column in resource_columns:
    print(column,": min =",minimums[column],"max =",maximums[column],"missing =",int(missing_counts[column]))

print()
print("RESOURCE VALUE VALIDATION")
for column in resource_columns:
    print(column,"outside [0,100]:",invalid_resource_counts[column])

print()
print("DATASET SUITABILITY FOR SCHEDULING")
print("Unique machines:",len(unique_counts["machine_id"]))
print("Unique tasks:",len(unique_counts["task_name"]))
print("Unique instances:",len(unique_counts["instance_name"]))
print("Unique jobs:",len(unique_counts["job_name"]))

print()
print("OUTLIER DETECTION")
for column in numeric_columns:
    print(column,": minimum =",minimums[column],"maximum =",maximums[column])

print()
print("IMPOSSIBLE OR INVALID VALUES")
for column in numeric_columns:
    print(column,": negative values =",negative_counts[column])

print()
print("INVALID TIMESTAMP VALUES")
print("Missing or invalid timestamp rows:",invalid_timestamp_rows)

print()
print("ANALYSIS OUTPUT LOCATION")
print(UNDERSTANDING_PATH)

column_summary = pd.DataFrame({
    "column":columns,
    "dtype":[dtype_map[column] for column in columns],
    "missing_values":[int(missing_counts[column]) for column in columns],
    "unique_values":[len(unique_counts[column]) if column in unique_counts else np.nan for column in columns]
})
column_summary.to_csv(os.path.join(UNDERSTANDING_PATH,"batch_instance_column_summary.csv"),index=False)

numeric_summary = pd.DataFrame({
    "column":numeric_columns,
    "minimum":[minimums[column] for column in numeric_columns],
    "maximum":[maximums[column] for column in numeric_columns],
    "mean":[sums[column] / (rows-int(missing_counts[column])) if rows-int(missing_counts[column]) else np.nan for column in numeric_columns]
})
numeric_summary.to_csv(os.path.join(UNDERSTANDING_PATH,"batch_instance_numeric_summary.csv"),index=False)

status_output = pd.DataFrame({"status":list(status_counts.keys()),"count":list(status_counts.values())})
status_output.to_csv(os.path.join(UNDERSTANDING_PATH,"batch_instance_status_values.csv"),index=False)

task_type_output = pd.DataFrame({"task_type":list(task_type_counts.keys()),"count":list(task_type_counts.values())})
task_type_output.to_csv(os.path.join(UNDERSTANDING_PATH,"batch_instance_task_type_values.csv"),index=False)

print()
print("Analysis completed successfully.")
