"""
Custom Exploratory Analysis of Batch Instances.

This custom exploratory script inspects specific subsets of batch_instance trace data. It performs custom filtering,
column inspections, and preliminary feature evaluation on task execution traces.
"""

import os

file_path = os.path.join(
    r"C:\Users\gvkva\Desktop\My ongoing projects\Machine Learning Project\Adaptive-Multi-Core-CPU-Scheduler-RL","dataset","raw","batch_instance.csv")

# Column names in exact order
columns = [
    "instance_name", "task_name", "job_name", "task_type", "status",
    "start_time", "end_time", "machine_id", "seq_no", "total_seq_no",
    "cpu_avg", "cpu_max", "mem_avg", "mem_max"
]

# Print the header once at the top
print("HEADER:", ", ".join(columns))

# Open and read only the first 5 lines
with open(file_path, "r", encoding="utf-8") as f:
    for i in range(5):
        line = f.readline()
        if not line:
            break
        
        # Split line by comma and pair each value with its column name
        values = line.strip().split(",")
        row_dict = dict(zip(columns, values))
        
        print(f"Row {i + 1}: {row_dict}")