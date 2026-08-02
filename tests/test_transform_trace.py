from preprocessing.data_loader import load_batch_instance
from preprocessing.clean_data import clean_chunk
from preprocessing.feature_engineering import engineer_features
from preprocessing.transform_trace import transform_trace
FILE_PATH = "C:/Users/gvkva/Desktop/My ongoing projects/Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL/data/raw/batch_instance.csv"
loader = load_batch_instance(FILE_PATH)
chunk = next(loader)
chunk = clean_chunk(chunk)
chunk = engineer_features(chunk)
chunk = transform_trace(chunk)
print("=" * 60)
print(chunk.columns)
print("=" * 60)
print(chunk.head())
print("=" * 60)
print(chunk.shape)