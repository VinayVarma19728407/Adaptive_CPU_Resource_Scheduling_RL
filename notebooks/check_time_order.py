import pandas as pd

FILE_PATH = "C:/Users/gvkva/Desktop/My ongoing projects/Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL/data/raw/batch_instance.csv"

COLUMNS = [
    "Process_ID",
    "Task_ID",
    "Job_ID",
    "Task_Type",
    "Status",
    "Start_Time",
    "End_Time",
    "PC_ID",
    "Sequence_No",
    "Total_Sequence",
    "CPU_Avg",
    "CPU_Max",
    "Memory_Avg",
    "Memory_Max"
]

CHUNK_SIZE = 100000

previous_max_time = None
total_rows = 0
out_of_order_rows = 0
first_out_of_order = None

reader = pd.read_csv(
    FILE_PATH,
    header=None,
    names=COLUMNS,
    chunksize=CHUNK_SIZE,
    low_memory=False
)

for chunk_number, chunk in enumerate(reader, start=1):

    current_min_time = chunk["Start_Time"].min()
    current_max_time = chunk["Start_Time"].max()

    if previous_max_time is not None:

        if current_min_time < previous_max_time:
            out_of_order_rows += 1

            if first_out_of_order is None:
                first_out_of_order = {
                    "previous_max_time": previous_max_time,
                    "current_min_time": current_min_time,
                    "chunk": chunk_number
                }

    previous_max_time = current_max_time
    total_rows += len(chunk)

    print(
        f"Chunk {chunk_number}: "
        f"rows={len(chunk)}, "
        f"min_start={current_min_time}, "
        f"max_start={current_max_time}"
    )

print("\n" + "=" * 60)
print("TIME ORDER CHECK")
print("=" * 60)

print(f"Total rows checked: {total_rows}")
print(f"Out-of-order chunk boundaries: {out_of_order_rows}")

if first_out_of_order:
    print("\nFirst out-of-order boundary:")
    print(first_out_of_order)
else:
    print("\nDataset appears chronologically ordered by Start_Time.")