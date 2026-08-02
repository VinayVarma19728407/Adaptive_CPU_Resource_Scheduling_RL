from preprocessing.data_loader import load_batch_instance
from preprocessing.clean_data import clean_chunk
FILE_PATH = "C:/Users/gvkva/Desktop/My ongoing projects/Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL/data/raw/batch_instance.csv"
loader = load_batch_instance(FILE_PATH)
chunk = next(loader)
print("Before Cleaning")
print(chunk.shape)
chunk = clean_chunk(chunk)
print("\nAfter Cleaning")
print(chunk.shape)
print("\nFirst Five Rows")
print(chunk.head())