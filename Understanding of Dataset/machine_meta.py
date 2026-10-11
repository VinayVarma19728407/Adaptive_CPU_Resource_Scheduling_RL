"""
Exploratory Analysis of Machine Metadata.

This exploratory script inspects physical cluster node specifications in machine_meta. It analyzes cluster size,
core count distributions, memory capacities, and hardware heterogeneity across cluster nodes.
"""

import pandas as pd
import numpy as np
import os

PROJECT_PATH = ("C:/Users/gvkva/Desktop/My ongoing projects/""Machine Learning Project/""Adaptive-Multi-Core-CPU-Scheduler-RL")
MACHINE_META_PATH = os.path.join(PROJECT_PATH,"dataset","raw","machine_meta.csv")
UNDERSTANDING_PATH = os.path.join(PROJECT_PATH,"dataset","understanding_of_dataset")
MACHINE_META_UNDERSTANDING_PATH = os.path.join(UNDERSTANDING_PATH,"machine_meta")
os.makedirs(MACHINE_META_UNDERSTANDING_PATH,exist_ok=True)

COLUMNS = ["machine_id","time_stamp","failure_domain_1","failure_domain_2","cpu_num","mem_size","status"]
NUMERIC_COLUMNS = ["time_stamp","cpu_num","mem_size"]
RESOURCE_COLUMNS = ["cpu_num","mem_size"]
CATEGORICAL_COLUMNS = ["status","failure_domain_1","failure_domain_2"]

print("DATASET PATH")
print(MACHINE_META_PATH)

if not os.path.exists(MACHINE_META_PATH):
    raise FileNotFoundError(MACHINE_META_PATH)

print("\nDATASET INFORMATION")
file_size_bytes = os.path.getsize(MACHINE_META_PATH)
print("File size:",round(file_size_bytes/(1024**2),2),"MB")

machine_meta = pd.read_csv(MACHINE_META_PATH,header=None,names=COLUMNS,low_memory=False)

print("\n1. NUMBER OF ROWS")
print("Total rows:",len(machine_meta))

print("\n2. NUMBER OF COLUMNS")
print("Total columns:",len(machine_meta.columns))

print("\n3. COLUMN NAMES")
for index,column in enumerate(COLUMNS,1):
    print(index,".",column)

print("\n4. DATA TYPES")
dtype_report = pd.DataFrame({"column":COLUMNS,"dtype":machine_meta.dtypes.values})
print(dtype_report.to_string(index=False))
dtype_report.to_csv(os.path.join(MACHINE_META_UNDERSTANDING_PATH,"machine_meta_dtypes.csv"),index=False)

print("\n5. MISSING VALUES")
missing_counts = machine_meta.isnull().sum()
missing_report = pd.DataFrame({"column":COLUMNS,"missing_count":missing_counts.values,"missing_percentage":(missing_counts.values/len(machine_meta)*100).round(2)})
print(missing_report.to_string(index=False))
missing_report.to_csv(os.path.join(MACHINE_META_UNDERSTANDING_PATH,"machine_meta_missing_values.csv"),index=False)

print("\n6. UNIQUE VALUES")
unique_counts = machine_meta.nunique(dropna=True)
unique_report = pd.DataFrame({"column":COLUMNS,"unique_count":unique_counts.values})
print(unique_report.to_string(index=False))
unique_report.to_csv(os.path.join(MACHINE_META_UNDERSTANDING_PATH,"machine_meta_unique_values.csv"),index=False)

for column in CATEGORICAL_COLUMNS:
    category_report = machine_meta[column].value_counts(dropna=False).rename_axis("value").reset_index(name="count")
    print("\n",column)
    print(category_report.to_string(index=False))
    category_report.to_csv(os.path.join(MACHINE_META_UNDERSTANDING_PATH,f"machine_meta_{column}_counts.csv"),index=False)

print("\n7. DUPLICATE RECORDS")
duplicate_count = machine_meta.duplicated().sum()
print("Duplicate records:",duplicate_count)

print("\n8. MINIMUM AND MAXIMUM VALUES")
numeric_data = machine_meta[NUMERIC_COLUMNS].apply(pd.to_numeric,errors="coerce")
min_max_report = pd.DataFrame({"min":numeric_data.min(),"max":numeric_data.max()})
print(min_max_report.to_string())
min_max_report.to_csv(os.path.join(MACHINE_META_UNDERSTANDING_PATH,"machine_meta_min_max.csv"))

print("\n9. BASIC STATISTICS")
statistics_report = numeric_data.describe().T
print(statistics_report.to_string())
statistics_report.to_csv(os.path.join(MACHINE_META_UNDERSTANDING_PATH,"machine_meta_statistics.csv"))

print("\n10. ID RELATIONSHIPS")
for column in ["machine_id","failure_domain_1","failure_domain_2"]:
    print(column,":",machine_meta[column].nunique(dropna=True),"unique values")

id_relationship_report = pd.DataFrame({
    "field":["machine_id","failure_domain_1","failure_domain_2"],
    "unique_count":[machine_meta[column].nunique(dropna=True) for column in ["machine_id","failure_domain_1","failure_domain_2"]]
})
id_relationship_report.to_csv(os.path.join(MACHINE_META_UNDERSTANDING_PATH,"machine_meta_id_relationships.csv"),index=False)

print("\n11. TIMESTAMP RANGE")
timestamp_values = pd.to_numeric(machine_meta["time_stamp"],errors="coerce")
timestamp_report = pd.DataFrame({
    "min":[timestamp_values.min()],
    "max":[timestamp_values.max()],
    "valid_count":[timestamp_values.notna().sum()],
    "invalid_count":[timestamp_values.isna().sum()],
    "negative_count":[int((timestamp_values.dropna()<0).sum())]
})
print(timestamp_report.to_string(index=False))
timestamp_report.to_csv(os.path.join(MACHINE_META_UNDERSTANDING_PATH,"machine_meta_timestamp_analysis.csv"),index=False)

print("\n12. RESOURCE-RELATED FIELDS")
resource_report = pd.DataFrame({
    "field":RESOURCE_COLUMNS,
    "min":[numeric_data[column].min() for column in RESOURCE_COLUMNS],
    "max":[numeric_data[column].max() for column in RESOURCE_COLUMNS],
    "negative_count":[int((numeric_data[column].dropna()<0).sum()) for column in RESOURCE_COLUMNS]
})
print(resource_report.to_string(index=False))
resource_report.to_csv(os.path.join(MACHINE_META_UNDERSTANDING_PATH,"machine_meta_resource_analysis.csv"),index=False)

print("\n13. DATASET SUITABILITY FOR SCHEDULING")
print("The dataset contains machine identity, timestamp, failure-domain information, CPU capacity, memory capacity and machine status.")
print("These fields support capacity-aware and failure-domain-aware scheduling.")

print("\n14. OUTLIER DETECTION")
outlier_report = pd.DataFrame(index=NUMERIC_COLUMNS)
outlier_report["Q1"] = numeric_data.quantile(0.25)
outlier_report["Q3"] = numeric_data.quantile(0.75)
outlier_report["IQR"] = outlier_report["Q3"]-outlier_report["Q1"]
outlier_report["lower_bound"] = outlier_report["Q1"]-1.5*outlier_report["IQR"]
outlier_report["upper_bound"] = outlier_report["Q3"]+1.5*outlier_report["IQR"]
outlier_report["outlier_count"] = [(numeric_data[column]<outlier_report.loc[column,"lower_bound"]).sum()+(numeric_data[column]>outlier_report.loc[column,"upper_bound"]).sum() for column in NUMERIC_COLUMNS]
print(outlier_report.to_string())
outlier_report.to_csv(os.path.join(MACHINE_META_UNDERSTANDING_PATH,"machine_meta_outliers.csv"))

print("\n15. IMPOSSIBLE OR INVALID VALUES")
invalid_report = pd.DataFrame({
    "field":NUMERIC_COLUMNS,
    "invalid_count":[int(machine_meta[column].notna().sum()-numeric_data[column].notna().sum()) for column in NUMERIC_COLUMNS],
    "negative_count":[int((numeric_data[column].dropna()<0).sum()) for column in NUMERIC_COLUMNS]
})
print(invalid_report.to_string(index=False))
invalid_report.to_csv(os.path.join(MACHINE_META_UNDERSTANDING_PATH,"machine_meta_invalid_values.csv"),index=False)

print("\nANALYSIS OUTPUT LOCATION")
print(MACHINE_META_UNDERSTANDING_PATH)

print("\nDATASET UNDERSTANDING STAGE COMPLETED")
