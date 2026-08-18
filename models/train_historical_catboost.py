"""
Historical Feature CatBoost Model

Purpose:
--------
Predict CPU requirement using historical CPU behavior.

Evaluation:
-----------
Chronological 80/20 split.

Important:
----------
No Job_ID, Task_ID, or PC_ID are directly supplied to the model.
Their historical CPU behavior is represented through
historical statistics.
"""

import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# FILE PATH
# =========================================================

DATA_FILE = (
    "C:/Users/gvkva/Desktop/My ongoing projects/"
    "Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL/"
    "data/processed/ml_cpu_historical.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 60)
print("LOADING HISTORICAL FEATURE DATASET")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print(f"Rows: {len(df)}")


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

target = "CPU_Requirement"

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
print("TRAINING HISTORICAL CATBOOST")
print("=" * 60)

model = CatBoostRegressor(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    loss_function="RMSE",
    random_seed=42,
    verbose=100
)

model.fit(
    X_train,
    y_train
)


# =========================================================
# PREDICTIONS
# =========================================================

print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

y_pred = model.predict(X_test)


# =========================================================
# EVALUATION
# =========================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("HISTORICAL CATBOOST RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


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