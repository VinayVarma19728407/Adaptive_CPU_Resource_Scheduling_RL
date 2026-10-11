import os
import pandas as pd

PROJECT_PATH = r"C:\Users\gvkva\Desktop\My ongoing projects\Machine Learning Project\Adaptive-Multi-Core-CPU-Scheduler-RL"

FILES = [
    os.path.join(PROJECT_PATH, "dataset", "preprocessed", "batch_task.csv"),
    os.path.join(PROJECT_PATH, "dataset", "preprocessed", "batch_instance_training.csv"),
    os.path.join(PROJECT_PATH, "dataset", "preprocessed", "batch_instance_validation.csv"),
    os.path.join(PROJECT_PATH, "dataset", "preprocessed", "batch_instance_testing.csv"),
    os.path.join(PROJECT_PATH, "dataset", "preprocessed", "machine_meta.csv"),
    os.path.join(PROJECT_PATH, "dataset", "cleaned", "machine_usage.csv"),
]

print("STAGE 1: DATA CONTRACT VERIFICATION")
print("READ-ONLY CHECK - NO DATASETS WILL BE MODIFIED")

for file_path in FILES:
    print("FILE:", os.path.basename(file_path))

    if not os.path.isfile(file_path):
        print("ERROR: FILE NOT FOUND")
        print("EXPECTED PATH:", file_path)
        continue

    print("SIZE (GB):", round(os.path.getsize(file_path) / (1024 ** 3), 4))

    try:
        sample = pd.read_csv(file_path, nrows=1000, low_memory=False)

        print("COLUMNS:", len(sample.columns))
        print("SCHEMA:", list(sample.columns))
        print("\nSAMPLE DTYPES:")
        print(sample.dtypes.to_string())
        print("\nSAMPLE MISSING VALUES:")
        print(sample.isna().sum().to_string())
        print("\nFIRST 3 ROWS:")
        print(sample.head(3).to_string(index=False))

        del sample

    except Exception as error:
        print("ERROR WHILE READING:", repr(error))

print("STAGE 1 INITIAL FILE CHECK FINISHED")
print("NO FILES WERE WRITTEN OR MODIFIED")
