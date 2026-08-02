#This is used to write output files
"""
=========================================================
File : save_processed.py
=========================================================
Purpose
-------
Reads the Alibaba batch_instance.csv file in chunks,
cleans the data, performs feature engineering,
transforms it into the scheduler format,
and saves the processed dataset.
Author
------
Adaptive Multi-Core CPU Scheduler using RL
"""
import os
import pandas as pd
from preprocessing.data_loader import load_batch_instance
from preprocessing.clean_data import clean_chunk
from preprocessing.feature_engineering import engineer_features
from preprocessing.transform_trace import transform_trace
# ==========================================================
# File Paths
# ==========================================================
INPUT_FILE = "C:/Users/gvkva/Desktop/My ongoing projects/Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL/data/raw/batch_instance.csv"
OUTPUT_FILE = "C:/Users/gvkva/Desktop/My ongoing projects/Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL/data/processedscheduler_dataset.csv"
# ==========================================================
# Maximum number of chunks to process
# IMPORTANT
# During development DO NOT process the
# entire 103GB dataset.
# ==========================================================
MAX_CHUNKS = 10
# ==========================================================
# Delete old processed dataset
# ==========================================================
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)
# ==========================================================
# Start Processing
# ==========================================================
print("=" * 70)
print("Starting Dataset Preprocessing...")
print("=" * 70)
for chunk_number, chunk in enumerate(load_batch_instance(INPUT_FILE), start=1):
    print(f"\nProcessing Chunk {chunk_number}")
    # ------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------
    chunk = clean_chunk(chunk)
    # ------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------
    chunk = engineer_features(chunk)
    # ------------------------------------------------------
    # Transform
    # ------------------------------------------------------
    chunk = transform_trace(chunk)
    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------
    if chunk_number == 1:
        chunk.to_csv(
            OUTPUT_FILE,
            mode="w",
            index=False
        )
    else:
        chunk.to_csv(
            OUTPUT_FILE,
            mode="a",
            index=False,
            header=False
        )
    print(f"Saved Chunk {chunk_number}")
    # ------------------------------------------------------
    # Stop after MAX_CHUNKS
    # ------------------------------------------------------
    if chunk_number >= MAX_CHUNKS:
        break
print("\n" + "=" * 70)
print("Dataset Successfully Processed")
print("=" * 70)
print(f"\nSaved File : {OUTPUT_FILE}")