"""
Core Allocation Classification Model

Target:
--------
Required CPU cores: 1, 2, 4, 8

Evaluation:
-----------
Chronological 80/20 split.

Important:
----------
Core_Target is derived from CPU_Requirement and is therefore
the TARGET, not an input feature.
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
    "data/processed/ml_cpu_dataset.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 60)
print("LOADING ML DATASET")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print(f"Rows: {len(df)}")


# =========================================================
# CREATE CORE TARGET
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
    "Task_ID",
    "Job_ID",
    "PC_ID",
    "Arrival_Time",
    "Memory_Requirement"
]

target = "Core_Target"

X = df[features].copy()
y = df[target].copy()


# =========================================================
# CATEGORICAL FEATURES
# =========================================================

categorical_features = [
    "Task_ID",
    "Job_ID",
    "PC_ID"
]

for column in categorical_features:
    X[column] = X[column].astype(str)


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

print(
    f"\nTraining time range: "
    f"{X_train['Arrival_Time'].min()} -> "
    f"{X_train['Arrival_Time'].max()}"
)

print(
    f"Testing time range: "
    f"{X_test['Arrival_Time'].min()} -> "
    f"{X_test['Arrival_Time'].max()}"
)


# =========================================================
# TRAIN MODEL
# =========================================================

print("\n" + "=" * 60)
print("TRAINING CATBOOST CLASSIFIER")
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
    y_train,
    cat_features=categorical_features
)


# =========================================================
# PREDICTION
# =========================================================

print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

y_pred = model.predict(X_test)

y_pred = y_pred.flatten()


# =========================================================
# ACCURACY
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
print("CORE CLASSIFICATION RESULTS")
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