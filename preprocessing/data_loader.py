#The job is to load the 103GB file by making  it into  smaller pieces
"""
=========================================================
File: data_loader.py
=========================================================
Purpose:
--------
This module reads the Alibaba Cluster Trace dataset
(batch_instance.csv) in chunks.
Since the dataset is over 100 GB, we never load the
entire file into memory.
Responsibilities:
-----------------
1. Read CSV in chunks
2. Assign proper column names
3. Return one chunk at a time
Author:
--------
Adaptive Multi-Core CPU Scheduler using RL
"""

import pandas as pd

# =========================================================
# Column Names for batch_instance.csv
# =========================================================

COLUMN_NAMES = [
    "Process_ID",          # Instance ID
    "Task_ID",             # Task ID
    "Job_ID",              # Job ID
    "Task_Type",           # Task Type
    "Status",              # Execution Status
    "Start_Time",          # Arrival Time
    "End_Time",            # Completion Time
    "PC_ID",               # Machine ID
    "Sequence_No",         # Sequence Number
    "Total_Sequence",      # Total Number of Sequences
    "CPU_Avg",             # Average CPU Usage
    "CPU_Max",             # Maximum CPU Usage
    "Memory_Avg",          # Average Memory Usage
    "Memory_Max"           # Maximum Memory Usage
]


# =========================================================
# Data Loader Function
# =========================================================

def load_batch_instance(
        file_path,
        chunk_size=100000
):
    """
    Reads batch_instance.csv in chunks.
    Parameters
    ----------
    file_path : str
        Path to CSV file
    chunk_size : int
        Number of rows to read at a time
    Returns
    -------
    Generator
        Returns one DataFrame chunk at a time
    """
    reader = pd.read_csv(
        file_path,
        header=None,
        names=COLUMN_NAMES,
        chunksize=chunk_size,
        low_memory=False
    )
    for chunk in reader:
        yield chunk


# =========================================================
# Test the Loader
# =========================================================

if __name__ == "__main__":
    FILE_PATH = "C:/Users/gvkva/Desktop/My ongoing projects/Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL/data/raw/batch_instance.csv"
    print("=" * 60)
    print("Reading Dataset...")
    print("=" * 60)

    for i, chunk in enumerate(load_batch_instance(FILE_PATH)):
        print(f"\nChunk : {i + 1}")
        print(f"Rows  : {len(chunk)}")
        print("\nColumns:")
        print(chunk.columns.tolist())
        print("\nFirst Five Rows:")
        print(chunk.head())
        print("\nData Types:")
        print(chunk.dtypes)
        # Stop after first chunk
        break
    print("\nDataset Loader Working Successfully!")