"""
=========================================================
File : transform_trace.py
=========================================================
Purpose
-------
Transform the cleaned Alibaba trace into the final
CPU scheduling dataset.
"""
import pandas as pd
def transform_trace(df):
    """
    Keep only the columns required by the scheduler.
    """
    final_columns = [
        # ----------------------------
        # Process Information
        # ----------------------------
        "Process_ID",
        "Task_ID",
        "Job_ID",
        # ----------------------------
        # Machine
        # ----------------------------
        "PC_ID",
        # ----------------------------
        # Scheduling
        # ----------------------------
        "Arrival_Time",
        "Completion_Time",
        "Burst_Time",
        # ----------------------------
        # Resource Requirements
        # ----------------------------
        "CPU_Requirement",
        "Memory_Requirement",
        "Required_Cores",
    ]
    df = df[final_columns]
    return df