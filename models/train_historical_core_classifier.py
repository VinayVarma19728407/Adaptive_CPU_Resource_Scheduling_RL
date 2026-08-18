"""
Historical Core Allocation Classifier

Predict:
    1, 2, 4, or 8 CPU cores

using leakage-free historical workload features.

Evaluation:
    Chronological 80/20 split
"""

import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# FILE PATH
# =========================================================

DATA_FILE = (
    "C:/Users/gvkva/Desktop/My ongoing projects/"
    "Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL/"
    "data/processed/ml_cpu_historical.csv"
)


# =========================================================
# LOAD
# =========================================================

print("=" * 60)
print("LOADING HISTORICAL DATASET")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print(f"Rows: {len(df)}")


# =========================================================
# CREATE TARGET
# =========================================================

def cpu_to_cores(cpu):

    if cpu <= 25:
        return 1
    elif cpu <= 50:
        return 2
    elif cpu <= 75:
        return 4
    else:
        return 8


df["Core_Target"] = (
    df["CPU_Requirement"]
    .apply(cpu_to_cores)
)


# =========================================================
# SORT BY TIME
# =========================================================

df = df.sort_values(
    "Arrival_Time",
    kind="mergesort"
).reset_index(drop=True)


# =========================================================
# FEATURES
# =========================================================

features = [
    "Arrival_Time",
    "Memory_Requirement",

    "Task_Previous_Count",
    "Task_Historical_Mean",
    "Task_Historical_Std",
    "Task_Historical_Min",
    "Task_Historical_Max",

    "PC_Previous_Count",
    "PC_Historical_Mean",
    "PC_Historical_Std",
    "PC_Historical_Min",
    "PC_Historical_Max",
]

target = "Core_Target"

X = df[features].copy()
y = df[target].copy()


# =========================================================
# TEMPORAL SPLIT
# =========================================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_train = y.iloc[:split_index].copy()
y_test = y.iloc[split_index:].copy()


print("\n" + "=" * 60)
print("TEMPORAL DATA SPLIT")
print("=" * 60)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# =========================================================
# TRAIN
# =========================================================

print("\n" + "=" * 60)
print("TRAINING HISTORICAL CORE CLASSIFIER")
print("=" * 60)

model = CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    loss_function="MultiClass",
    random_seed=42,
    verbose=100
)

model.fit(
    X_train,
    y_train
)


# =========================================================
# PREDICT
# =========================================================

print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

y_pred = model.predict(X_test).flatten()


# =========================================================
# RESULTS
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("HISTORICAL CORE CLASSIFIER RESULTS")
print("=" * 60)

print(f"Accuracy          : {accuracy:.4f}")
print(f"Balanced Accuracy : {balanced_accuracy:.4f}")


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        labels=[1, 2, 4, 8],
        zero_division=0
    )
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=[1, 2, 4, 8]
)

matrix_df = pd.DataFrame(
    matrix,
    index=["Actual_1", "Actual_2", "Actual_4", "Actual_8"],
    columns=["Pred_1", "Pred_2", "Pred_4", "Pred_8"]
)

print(matrix_df)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

importance = model.get_feature_importance()

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)

print(
    feature_importance.to_string(index=False)
)