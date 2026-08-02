from preprocessing.data_loader import load_batch_instance
from preprocessing.clean_data import clean_chunk
from preprocessing.feature_engineering import engineer_features
FILE_PATH = "C:/Users/gvkva/Desktop/My ongoing projects/Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL/data/raw/batch_instance.csv"
loader = load_batch_instance(FILE_PATH)
chunk = next(loader)
print("=" * 50)
print("Before Feature Engineering")
print(chunk.columns.tolist())
chunk = clean_chunk(chunk)
chunk = engineer_features(chunk)
print("\n" + "=" * 50)
print("After Feature Engineering")
print(chunk.columns.tolist())
print("\nFirst 5 Rows")
print(
    chunk[
        [
            "Arrival_Time",
            "Completion_Time",
            "Burst_Time",
            "CPU_Requirement",
            "Memory_Requirement",
            "Required_Cores"
        ]
    ].head()
)