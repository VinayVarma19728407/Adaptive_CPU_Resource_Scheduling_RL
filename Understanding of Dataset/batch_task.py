"""
Exploratory Analysis of Batch Task Metadata.

This exploratory script reads and inspects batch_task metadata files. It analyzes task counts, job distributions,
planned resource request ranges, and task dependency structures in the workload dataset.
"""

import pandas as pd
import numpy as np
import os

PROJECT_PATH = ("C:/Users/gvkva/Desktop/My ongoing projects/""Machine Learning Project/""Adaptive-Multi-Core-CPU-Scheduler-RL")
BATCH_TASK_PATH = os.path.join(PROJECT_PATH,"dataset","raw","batch_task.csv")
UNDERSTANDING_PATH = os.path.join(PROJECT_PATH,"dataset","understanding_of_datasets")
BATCH_TASK_UNDERSTANDING_PATH = os.path.join(UNDERSTANDING_PATH,"batch_task")
os.makedirs(BATCH_TASK_UNDERSTANDING_PATH,exist_ok=True)

CHUNK_SIZE = 100000
COLUMNS = ["task_name","instance_num","job_name","task_type","status","start_time","end_time","plan_cpu","plan_mem"]
ID_COLUMNS = ["task_name","job_name"]
CATEGORICAL_COLUMNS = ["task_type","status"]
NUMERIC_COLUMNS = ["instance_num","start_time","end_time","plan_cpu","plan_mem"]
RESOURCE_COLUMNS = ["plan_cpu","plan_mem"]
TIMESTAMP_COLUMNS = ["start_time","end_time"]

print("BATCH TASK DATASET FULL ANALYSIS")
print("\nFile:",BATCH_TASK_PATH)

if not os.path.exists(BATCH_TASK_PATH):
    raise FileNotFoundError(BATCH_TASK_PATH)

print("File size:",round(os.path.getsize(BATCH_TASK_PATH)/(1024**3),2),"GB")
print("\nThe complete CSV will be processed in chunks.")

total_rows = 0
chunk_number = 0
missing_counts = pd.Series(0,dtype="int64",index=COLUMNS)
numeric_min = pd.Series(np.inf,index=NUMERIC_COLUMNS,dtype="float64")
numeric_max = pd.Series(-np.inf,index=NUMERIC_COLUMNS,dtype="float64")
numeric_sum = pd.Series(0.0,index=NUMERIC_COLUMNS,dtype="float64")
numeric_sum_squares = pd.Series(0.0,index=NUMERIC_COLUMNS,dtype="float64")
numeric_count = pd.Series(0,dtype="int64",index=NUMERIC_COLUMNS)
numeric_invalid = pd.Series(0,dtype="int64",index=NUMERIC_COLUMNS)
numeric_negative = pd.Series(0,dtype="int64",index=NUMERIC_COLUMNS)

category_counts = {column:{} for column in CATEGORICAL_COLUMNS}

timestamp_min = {column:None for column in TIMESTAMP_COLUMNS}
timestamp_max = {column:None for column in TIMESTAMP_COLUMNS}
timestamp_valid = {column:0 for column in TIMESTAMP_COLUMNS}
timestamp_invalid = {column:0 for column in TIMESTAMP_COLUMNS}
end_before_start = 0
zero_duration = 0
positive_duration = 0

task_name_unique = set()
job_name_unique = set()

for chunk in pd.read_csv(BATCH_TASK_PATH,header=None,names=COLUMNS,chunksize=CHUNK_SIZE,low_memory=False):
    chunk_number += 1
    total_rows += len(chunk)
    missing_counts = missing_counts.add(chunk.isna().sum(),fill_value=0)

    for column in CATEGORICAL_COLUMNS:
        values = chunk[column].dropna().value_counts()
        for value,count in values.items():
            category_counts[column][value] = category_counts[column].get(value,0) + int(count)

    task_name_unique.update(chunk["task_name"].dropna().unique())
    job_name_unique.update(chunk["job_name"].dropna().unique())

    numeric = chunk[NUMERIC_COLUMNS].apply(pd.to_numeric,errors="coerce")

    for column in NUMERIC_COLUMNS:
        values = numeric[column]
        valid = values.dropna()
        numeric_invalid[column] += int(values.isna().sum()-chunk[column].isna().sum())
        if len(valid) > 0:
            numeric_min[column] = min(numeric_min[column],valid.min())
            numeric_max[column] = max(numeric_max[column],valid.max())
            numeric_sum[column] += valid.sum()
            numeric_sum_squares[column] += np.square(valid).sum()
            numeric_count[column] += len(valid)
            numeric_negative[column] += int((valid < 0).sum())

    for column in TIMESTAMP_COLUMNS:
        values = numeric[column]
        valid = values.dropna()
        timestamp_invalid[column] += int(values.isna().sum()-chunk[column].isna().sum())
        if len(valid) > 0:
            current_min = valid.min()
            current_max = valid.max()
            if timestamp_min[column] is None or current_min < timestamp_min[column]:
                timestamp_min[column] = current_min
            if timestamp_max[column] is None or current_max > timestamp_max[column]:
                timestamp_max[column] = current_max
            timestamp_valid[column] += len(valid)

    valid_times = numeric["start_time"].notna() & numeric["end_time"].notna()
    if valid_times.any():
        duration = numeric.loc[valid_times,"end_time"]-numeric.loc[valid_times,"start_time"]
        end_before_start += int((duration < 0).sum())
        zero_duration += int((duration == 0).sum())
        positive_duration += int((duration > 0).sum())

    if chunk_number % 10 == 0:
        print("Processed chunks:",chunk_number,"| Rows processed:",total_rows)

print("\nFULL-FILE ANALYSIS COMPLETED")
print("Total chunks:",chunk_number)
print("Total rows:",total_rows)

print("\n1. NUMBER OF ROWS")
print(total_rows)

print("\n2. NUMBER OF COLUMNS")
print(len(COLUMNS))

print("\n3. COLUMN NAMES")
for index,column in enumerate(COLUMNS,1):
    print(index,".",column)

print("\n4. DATA TYPES")
dtype_report = pd.DataFrame({"column":COLUMNS,"dtype":chunk[COLUMNS].dtypes.astype(str).values})
print(dtype_report.to_string(index=True))
dtype_report.to_csv(os.path.join(BATCH_TASK_UNDERSTANDING_PATH,"batch_task_full_dtypes.csv"),index=True)

print("\n5. MISSING VALUES")
missing_report = pd.DataFrame({"column":COLUMNS,"missing_count":missing_counts.values,"missing_percentage":(missing_counts.values/total_rows*100).round(2)})
print(missing_report.to_string(index=False))
missing_report.to_csv(os.path.join(BATCH_TASK_UNDERSTANDING_PATH,"batch_task_full_missing_values.csv"),index=False)

print("\n6. UNIQUE VALUES")
print("task_name:",len(task_name_unique))
print("job_name:",len(job_name_unique))
for column in CATEGORICAL_COLUMNS:
    print(column,":",len(category_counts[column]))
    print("Available values:")
    for value,count in sorted(category_counts[column].items(),key=lambda x:x[1],reverse=True):
        print(value,":",count)

unique_report = pd.DataFrame({
    "column":ID_COLUMNS+CATEGORICAL_COLUMNS,
    "unique_count":[len(task_name_unique),len(job_name_unique)]+[len(category_counts[column]) for column in CATEGORICAL_COLUMNS]
})
unique_report.to_csv(os.path.join(BATCH_TASK_UNDERSTANDING_PATH,"batch_task_full_unique_values.csv"),index=False)

for column in CATEGORICAL_COLUMNS:
    category_report = pd.DataFrame({
        "value":list(category_counts[column].keys()),
        "count":list(category_counts[column].values())
    }).sort_values("count",ascending=False)
    category_report.to_csv(os.path.join(BATCH_TASK_UNDERSTANDING_PATH,f"batch_task_{column}_counts.csv"),index=False)

print("\n7. DUPLICATE RECORDS")
print("Exact duplicate counting is not performed on the complete file because retaining fingerprints for every record would require excessive memory.")

print("\n8. MINIMUM AND MAXIMUM VALUES")
min_max_report = pd.DataFrame({
    "column":NUMERIC_COLUMNS,
    "minimum":[numeric_min[column] for column in NUMERIC_COLUMNS],
    "maximum":[numeric_max[column] for column in NUMERIC_COLUMNS]
})
print(min_max_report.to_string(index=False))
min_max_report.to_csv(os.path.join(BATCH_TASK_UNDERSTANDING_PATH,"batch_task_full_min_max.csv"),index=False)

print("\n9. BASIC STATISTICS")
statistics = []
for column in NUMERIC_COLUMNS:
    count = numeric_count[column]
    mean = numeric_sum[column]/count if count > 0 else np.nan
    variance = ((numeric_sum_squares[column]-(numeric_sum[column]**2)/count)/(count-1)) if count > 1 else np.nan
    if not np.isnan(variance):
        variance = max(variance,0)
    statistics.append({
        "column":column,
        "count":count,
        "missing":total_rows-count,
        "mean":mean,
        "std":np.sqrt(variance) if not np.isnan(variance) else np.nan,
        "min":numeric_min[column],
        "max":numeric_max[column]
    })

statistics_report = pd.DataFrame(statistics)
print(statistics_report.to_string(index=False))
statistics_report.to_csv(os.path.join(BATCH_TASK_UNDERSTANDING_PATH,"batch_task_full_statistics.csv"),index=False)

print("\n10. RELATIONSHIPS BETWEEN IDS")
print("task_name unique values:",len(task_name_unique))
print("job_name unique values:",len(job_name_unique))
print("The exact task-to-job mapping is not stored because it can become very large for the complete dataset.")

print("\n11. TIMESTAMP FORMAT AND RANGE")
timestamp_report = pd.DataFrame({
    "column":TIMESTAMP_COLUMNS,
    "minimum":[timestamp_min[column] for column in TIMESTAMP_COLUMNS],
    "maximum":[timestamp_max[column] for column in TIMESTAMP_COLUMNS],
    "valid_count":[timestamp_valid[column] for column in TIMESTAMP_COLUMNS],
    "invalid_count":[timestamp_invalid[column] for column in TIMESTAMP_COLUMNS]
})
print(timestamp_report.to_string(index=False))
timestamp_report.to_csv(os.path.join(BATCH_TASK_UNDERSTANDING_PATH,"batch_task_full_timestamp_range.csv"),index=False)

print("\n12. RESOURCE-RELATED FIELDS")
resource_report = statistics_report[statistics_report["column"].isin(RESOURCE_COLUMNS)].copy()
print(resource_report.to_string(index=False))
resource_report.to_csv(os.path.join(BATCH_TASK_UNDERSTANDING_PATH,"batch_task_full_resource_analysis.csv"),index=False)

print("\n13. OUTLIER DETECTION")
print("Complete-file min/max/statistics are available above.")
print("Exact full-file IQR detection is not performed because storing all values is not memory efficient.")
print("Outliers will NOT be deleted or modified at the dataset-understanding stage.")

print("\n14. IMPOSSIBLE OR INVALID VALUES")
invalid_report = pd.DataFrame({
    "column":NUMERIC_COLUMNS,
    "invalid_numeric_values":[numeric_invalid[column] for column in NUMERIC_COLUMNS],
    "negative_values":[numeric_negative[column] for column in NUMERIC_COLUMNS]
})
print(invalid_report.to_string(index=False))
invalid_report.to_csv(os.path.join(BATCH_TASK_UNDERSTANDING_PATH,"batch_task_full_invalid_values.csv"),index=False)

duration_report = pd.DataFrame({
    "condition":["end_time < start_time","end_time == start_time","end_time > start_time"],
    "count":[end_before_start,zero_duration,positive_duration]
})
print("\nTimestamp consistency:")
print(duration_report.to_string(index=False))
duration_report.to_csv(os.path.join(BATCH_TASK_UNDERSTANDING_PATH,"batch_task_duration_consistency.csv"),index=False)

print("\nANALYSIS OUTPUT LOCATION")
print(BATCH_TASK_UNDERSTANDING_PATH)

print("\nDATASET UNDERSTANDING STAGE COMPLETED")
