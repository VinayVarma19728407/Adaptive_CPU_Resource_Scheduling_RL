#Cleaning the 103GB file data this helps in reducing the file size by a lot
"""
=======================================================
File : clean_data.py
=========================================================
Purpose
-------
Cleans one chunk of Alibaba Cluster Trace.
Input
-----
Raw DataFrame
Output
------
Clean DataFrame
"""
import pandas as pd
def clean_chunk(df):
    """
    Clean one chunk of data.
    """
    # ---------------------------------------
    # Remove duplicate rows
    # ---------------------------------------
    df = df.drop_duplicates()
    # ---------------------------------------
    # Keep only completed processes
    # ---------------------------------------
    df = df[df["Status"] == "Terminated"]
    # ---------------------------------------
    # Remove rows with missing Machine ID
    # ---------------------------------------
    df = df.dropna(subset=["PC_ID"])
    # ---------------------------------------
    # Remove rows where
    # End Time <= Start Time
    # ---------------------------------------
    df = df[df["End_Time"] > df["Start_Time"]]
    # ---------------------------------------
    # Remove invalid CPU values
    # ---------------------------------------
    df = df[df["CPU_Avg"] >= 0]
    df = df[df["CPU_Max"] >= 0]
    # ---------------------------------------
    # Remove invalid Memory values
    # ---------------------------------------
    df = df[df["Memory_Avg"] >= 0]
    df = df[df["Memory_Max"] >= 0]
    return df