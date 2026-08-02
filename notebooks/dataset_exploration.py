import pandas as pd

columns = [
    "instance_name",
    "task_name",
    "job_name",
    "task_type",
    "status",
    "start_time",
    "end_time",
    "machine_id",
    "seq_no",
    "total_seq_no",
    "cpu_avg",
    "cpu_max",
    "mem_avg",
    "mem_max"
]

df = pd.read_csv(
    "C:/Users/gvkva/Desktop/My ongoing projects/Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL/data/raw/batch_instance.csv",
    header=None,
    names=columns,
    nrows=1000
)

print(df.head())
print(df.dtypes)

print(df.describe())
print(df["status"].value_counts())
print(df.isnull().sum())