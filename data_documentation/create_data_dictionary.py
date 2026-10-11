"""
Create Data Dictionary and Metadata Schema.

This script generates a comprehensive data dictionary document and JSON schema mapping for all dataset tables.
It inspects column types, value ranges, constraints, and descriptions across batch_task, batch_instance, machine_meta,
and machine_usage tables to maintain schema documentation.
"""

import os
import pandas as pd


OUTPUT_DIR = "dataset"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "data_dictionary.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


rows = []


def add(dataset, column, data_type, role, description,
        preprocessing, missing_allowed, downstream_use, notes=""):
    rows.append({
        "dataset": dataset,
        "column_name": column,
        "data_type": data_type,
        "role": role,
        "description": description,
        "preprocessing_status": preprocessing,
        "missing_allowed": missing_allowed,
        "downstream_use": downstream_use,
        "notes": notes
    })


add(
    "batch_task",
    "task_name",
    "object",
    "identifier",
    "Task identifier used to relate batch_task with batch_instance.",
    "Original value preserved. No task-name filtering applied.",
    "Task identity and cross-dataset relationship validation.",
    "No",
    "416,208 unique task names in the final preprocessed batch_task dataset."
)

add(
    "batch_task",
    "instance_num",
    "numeric",
    "identifier",
    "Instance number associated with the task record.",
    "Missing values filled with 1.",
    "Task instance representation.",
    "No"
)

add(
    "batch_task",
    "job_name",
    "object",
    "identifier",
    "Job identifier associated with the task record.",
    "Original value preserved.",
    "Task and job analysis.",
    "No"
)

add(
    "batch_task",
    "task_type",
    "object",
    "categorical",
    "Task type associated with the task record.",
    "Original value preserved and encoded.",
    "Task characterization and RL workload analysis.",
    "No"
)

add(
    "batch_task",
    "status",
    "object",
    "categorical",
    "Original task status.",
    "Original status retained.",
    "Task state information.",
    "No"
)

add(
    "batch_task",
    "start_time",
    "numeric",
    "temporal",
    "Task start timestamp.",
    "Rows with end_time earlier than start_time removed.",
    "Temporal analysis and task duration.",
    "No"
)

add(
    "batch_task",
    "end_time",
    "numeric",
    "temporal",
    "Task end timestamp.",
    "Rows with end_time earlier than start_time removed.",
    "Temporal analysis and task duration.",
    "No"
)

add(
    "batch_task",
    "plan_cpu",
    "numeric",
    "resource",
    "Planned CPU resource value.",
    "Missing values filled using the global mode.",
    "Task resource requirement.",
    "No"
)

add(
    "batch_task",
    "plan_mem",
    "numeric",
    "resource",
    "Planned memory resource value.",
    "Missing values filled using the global mode.",
    "Task resource requirement.",
    "No"
)

add(
    "batch_task",
    "task_name_id",
    "numeric",
    "encoded identifier",
    "Numeric encoding of task_name.",
    "Generated during preprocessing.",
    "Model input preparation and categorical encoding.",
    "No"
)

add(
    "batch_task",
    "job_name_id",
    "numeric",
    "encoded identifier",
    "Numeric encoding of job_name.",
    "Generated during preprocessing.",
    "Model input preparation and categorical encoding.",
    "No"
)

add(
    "batch_task",
    "task_type_id",
    "numeric",
    "encoded identifier",
    "Numeric encoding of task_type.",
    "Generated during preprocessing.",
    "Model input preparation and categorical encoding.",
    "No"
)

add(
    "batch_task",
    "status_id",
    "numeric",
    "encoded categorical",
    "Numeric encoding of task status.",
    "Generated during preprocessing.",
    "Model input preparation and categorical encoding.",
    "No",
    "Waiting=0, Running=1, Terminated=2, Failed=3."
)

add(
    "batch_task",
    "duration",
    "numeric",
    "derived temporal feature",
    "Task duration calculated from end_time minus start_time.",
    "Calculated after invalid negative-duration rows were removed.",
    "Task runtime and workload analysis.",
    "No",
    "Zero duration is retained."
)


add(
    "machine_meta",
    "machine_id",
    "numeric",
    "identifier",
    "Machine identifier.",
    "m_ prefix removed and numeric representation retained.",
    "Relationship with batch_instance and machine_usage.",
    "No",
    "4,033 unique machine IDs in the final machine metadata."
)

add(
    "machine_meta",
    "time_stamp",
    "numeric",
    "temporal",
    "Timestamp associated with the machine metadata record.",
    "Converted to numeric representation.",
    "Machine metadata temporal analysis.",
    "No"
)

add(
    "machine_meta",
    "failure_domain_1",
    "numeric",
    "machine metadata",
    "First failure-domain attribute.",
    "Retained from source.",
    "Machine characterization.",
    "No"
)

add(
    "machine_meta",
    "failure_domain_2",
    "numeric",
    "machine metadata",
    "Second failure-domain attribute.",
    "Missing values filled using same-machine neighboring records where the established rule allowed it.",
    "Machine characterization and failure-domain information.",
    "No",
    "34 missing values were filled during preprocessing."
)

add(
    "machine_meta",
    "cpu_num",
    "numeric",
    "machine resource",
    "CPU capacity associated with the machine.",
    "Retained from source.",
    "Machine resource characterization.",
    "No"
)

add(
    "machine_meta",
    "mem_size",
    "numeric",
    "machine resource",
    "Memory capacity associated with the machine.",
    "Retained from source.",
    "Machine resource characterization.",
    "No"
)

add(
    "machine_meta",
    "status",
    "numeric",
    "machine status",
    "Machine status value.",
    "IMPORT_INSTALLING records removed.",
    "Machine availability/state characterization.",
    "No"
)


add(
    "machine_usage",
    "machine_id",
    "numeric",
    "identifier",
    "Machine identifier associated with the utilization record.",
    "m_ prefix removed and numeric representation retained.",
    "Relationship with machine_meta and batch_instance.",
    "No"
)

add(
    "machine_usage",
    "time_stamp",
    "numeric",
    "temporal",
    "Timestamp associated with the machine usage record.",
    "Retained from source.",
    "Temporal machine usage analysis.",
    "No"
)

add(
    "machine_usage",
    "cpu_util_percent",
    "numeric",
    "resource utilization",
    "CPU utilization measurement.",
    "Rows with missing CPU utilization removed.",
    "Machine load and resource availability analysis.",
    "No"
)

add(
    "machine_usage",
    "mem_util_percent",
    "numeric",
    "resource utilization",
    "Memory utilization measurement.",
    "Retained from source.",
    "Machine load and resource availability analysis.",
    "No"
)

add(
    "machine_usage",
    "net_in",
    "numeric",
    "resource utilization",
    "Network input measurement.",
    "Retained from source.",
    "Machine usage characterization.",
    "No"
)

add(
    "machine_usage",
    "net_out",
    "numeric",
    "resource utilization",
    "Network output measurement.",
    "Retained from source.",
    "Machine usage characterization.",
    "No"
)

add(
    "machine_usage",
    "disk_io_percent",
    "numeric",
    "resource utilization",
    "Disk I/O utilization measurement.",
    "Invalid values below 0 or above 100 removed.",
    "Machine usage characterization.",
    "No"
)


batch_instance_columns = [
    (
        "instance_name",
        "object",
        "identifier",
        "Instance identifier associated with the batch instance.",
        "Retained from source.",
        "No",
        "Task instance representation.",
        ""
    ),
    (
        "task_name",
        "object",
        "identifier",
        "Task identifier.",
        "Retained from source.",
        "No",
        "Task relationship and task representation.",
        ""
    ),
    (
        "job_name",
        "object",
        "identifier",
        "Job identifier.",
        "Retained from source.",
        "No",
        "Job characterization.",
        ""
    ),
    (
        "task_type",
        "numeric",
        "categorical",
        "Task type value.",
        "Retained from source.",
        "No",
        "Task characterization.",
        ""
    ),
    (
        "status",
        "numeric",
        "categorical",
        "Batch instance status.",
        "Mapped from the established source status categories.",
        "No",
        "Task state representation.",
        "Failed=0, Interrupted=1, Ready=2, Running=3, Terminated=4, Terminating=5."
    ),
    (
        "start_time",
        "numeric",
        "temporal",
        "Batch instance start timestamp.",
        "Invalid time ordering removed.",
        "No",
        "Temporal scheduling information.",
        ""
    ),
    (
        "end_time",
        "numeric",
        "temporal",
        "Batch instance end timestamp.",
        "Invalid time ordering removed.",
        "No",
        "Temporal scheduling information.",
        ""
    ),
    (
        "machine_id",
        "numeric",
        "identifier",
        "Machine assigned to the batch instance.",
        "Numeric machine identifier retained.",
        "No",
        "Machine relationship and scheduling.",
        ""
    ),
    (
        "seq_no",
        "numeric",
        "sequence",
        "Sequence number of the task instance.",
        "Invalid sequence relationships removed.",
        "No",
        "Task sequence representation.",
        ""
    ),
    (
        "total_seq_no",
        "numeric",
        "sequence",
        "Total sequence count associated with the task instance.",
        "Invalid sequence relationships removed.",
        "No",
        "Task sequence representation.",
        ""
    ),
    (
        "cpu_avg",
        "numeric",
        "resource usage",
        "Average CPU resource value associated with the batch instance.",
        "Missing values remain NaN. Extreme values retained. No clipping or normalization.",
        "Yes",
        "RL state construction and resource-aware scheduling.",
        ""
    ),
    (
        "cpu_max",
        "numeric",
        "resource usage",
        "Maximum CPU resource value associated with the batch instance.",
        "Missing values remain NaN. Extreme values retained. No clipping or normalization.",
        "Yes",
        "RL state construction and resource-aware scheduling.",
        ""
    ),
    (
        "mem_avg",
        "numeric",
        "resource usage",
        "Average memory resource value associated with the batch instance.",
        "Missing values remain NaN. Extreme values retained. No clipping or normalization.",
        "Yes",
        "RL state construction and resource-aware scheduling.",
        ""
    ),
    (
        "mem_max",
        "numeric",
        "resource usage",
        "Maximum memory resource value associated with the batch instance.",
        "Missing values remain NaN. Extreme values retained. No clipping or normalization.",
        "Yes",
        "RL state construction and resource-aware scheduling.",
        ""
    ),
    (
        "duration",
        "numeric",
        "derived temporal feature",
        "Batch instance duration calculated from end_time minus start_time.",
        "Calculated after invalid negative-duration rows were removed. Zero duration retained.",
        "No",
        "Runtime estimation and scheduling.",
        ""
    )
]


for column, data_type, role, description, preprocessing, missing_allowed, downstream_use, notes in batch_instance_columns:
    add(
        "batch_instance",
        column,
        data_type,
        role,
        description,
        preprocessing,
        missing_allowed,
        downstream_use,
        notes
    )


df = pd.DataFrame(rows)

column_order = [
    "dataset",
    "column_name",
    "data_type",
    "role",
    "description",
    "preprocessing_status",
    "missing_allowed",
    "downstream_use",
    "notes"
]

df = df[column_order]

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("Data dictionary created")
print("Output file:", OUTPUT_FILE)
print("Total documented columns:", len(df))
print("Datasets:", ", ".join(df["dataset"].unique()))