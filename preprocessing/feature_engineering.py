#This helps to select the features which might effect our output and the rest of them are ommited
"""
=========================================================
File : feature_engineering.py
=========================================================
Purpose
-------
Create scheduling features from the cleaned Alibaba trace.
Input
-----
Clean DataFrame
Output
------
DataFrame with engineered scheduling features.
"""
import pandas as pd
def engineer_features(df):
    """
    Create scheduling-related features.
    """
    # --------------------------------------------------
    # Arrival Time
    # --------------------------------------------------
    df["Arrival_Time"] = df["Start_Time"]
    # --------------------------------------------------
    # Completion Time
    # --------------------------------------------------
    df["Completion_Time"] = df["End_Time"]
    # --------------------------------------------------
    # Burst Time
    # --------------------------------------------------
    df["Burst_Time"] = (
        df["Completion_Time"] -
        df["Arrival_Time"]
    )
    # --------------------------------------------------
    # CPU Requirement
    # --------------------------------------------------
    df["CPU_Requirement"] = df["CPU_Avg"]
    # --------------------------------------------------
    # Memory Requirement
    # --------------------------------------------------
    df["Memory_Requirement"] = df["Memory_Avg"]
    # --------------------------------------------------
    # Required CPU Cores
    # --------------------------------------------------
    def estimate_required_cores(cpu_usage):
        if cpu_usage <= 25:
            return 1
        elif cpu_usage <= 50:
            return 2
        elif cpu_usage <= 75:
            return 4
        else:
            return 8
    df["Required_Cores"] = (
        df["CPU_Requirement"]
        .apply(estimate_required_cores)
    )
    return df