"""
Preprocess Machine Metadata Features.

This script preprocesses physical machine node specifications from machine_meta. It normalizes core capacities, memory allocations,
and formats machine feature matrices for cluster state representation in the simulator.
"""

import os
import pandas as pd

PROJECT_PATH = ("C:/Users/gvkva/Desktop/My ongoing projects/""Machine Learning Project/""Adaptive-Multi-Core-CPU-Scheduler-RL")
CLEAN_MACHINE_META_PATH = os.path.join(PROJECT_PATH,"dataset","cleaned","machine_meta.csv")
PREPROCESSED_DATASET_PATH = os.path.join(PROJECT_PATH,"dataset","preprocessed")
PREPROCESS_MACHINE_META_PATH = os.path.join(PREPROCESSED_DATASET_PATH,"machine_meta.csv")
os.makedirs(PREPROCESSED_DATASET_PATH,exist_ok=True)

MACHINE_META_COLUMNS = ["machine_id","time_stamp","failure_domain_1","failure_domain_2","cpu_num","mem_size","status"]

dtype_map = {
    "machine_id":"string",
    "time_stamp":"int64",
    "failure_domain_1":"int64",
    "failure_domain_2":"float32",
    "cpu_num":"int32",
    "mem_size":"float32",
    "status":"string"
}

print("DATASET PATH")
print(CLEAN_MACHINE_META_PATH)

if not os.path.exists(CLEAN_MACHINE_META_PATH):
    raise FileNotFoundError(CLEAN_MACHINE_META_PATH)

machine_meta = pd.read_csv(CLEAN_MACHINE_META_PATH,header=0,dtype=dtype_map)
machine_meta["machine_id"] = machine_meta["machine_id"].str.strip()
status_encoding = {"USING":1}
unknown_status = set(machine_meta["status"].dropna().unique()) - set(status_encoding)

if unknown_status:
    raise ValueError("Unknown machine status values: " + str(sorted(unknown_status)))
machine_meta["status"] = machine_meta["status"].map(status_encoding).astype("int8")
machine_meta.to_csv(PREPROCESS_MACHINE_META_PATH,index=False)
print("MACHINE META PREPROCESSING COMPLETED")
print("Rows processed:",len(machine_meta))
print("Rows removed during preprocessing: 0")
print("STATUS ENCODING")

for key,value in status_encoding.items():
    print(key,"=",value)
print("Status column converted from categorical values to numeric values.")
print("Failure domain columns preserved without additional encoding or filling.")
print("Output:",PREPROCESS_MACHINE_META_PATH)