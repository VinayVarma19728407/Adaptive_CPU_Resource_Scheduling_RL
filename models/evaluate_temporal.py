"""
Temporal Evaluation of CatBoost CPU Prediction Model

Purpose:
--------
Evaluate CPU requirement prediction using a time-based split.

Earlier processes -> Training
Later processes  -> Testing

This is more realistic for the CPU scheduler than
a random train/test split.
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
# SORT BY ARRIVAL TIME
# =========================================================

print("\n" + "=" * 60)
print("SORTING BY ARRIVAL TIME")
print("=" * 60)

df = df.sort_values("Arrival_Time").reset_index(drop=True)

print(
    f"Time range: "
    f"{df['Arrival_Time'].min()} -> {df['Arrival_Time'].max()}"
)


# =========================================================
# FEATURES AND TARGET
# =========================================================

features = [
    "Task_ID",
    "Job_ID",
    "PC_ID",
    "Arrival_Time",
    "Memory_Requirement"
]

target = "CPU_Requirement"

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
    f"Testing time range : "
    f"{X_test['Arrival_Time'].min()} -> "
    f"{X_test['Arrival_Time'].max()}"
)


# =========================================================
# CHECK ID OVERLAP
# =========================================================

print("\n" + "=" * 60)
print("TRAIN / TEST ID OVERLAP")
print("=" * 60)

for column in ["Task_ID", "Job_ID", "PC_ID"]:

    train_values = set(X_train[column].unique())
    test_values = set(X_test[column].unique())

    overlap = train_values.intersection(test_values)

    print(
        f"{column}: "
        f"{len(overlap)} overlapping values"
    )


# =========================================================
# TRAIN MODEL
# =========================================================

print("\n" + "=" * 60)
print("TRAINING TEMPORAL CATBOOST MODEL")
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
    y_train,
    cat_features=categorical_features
)


# =========================================================
# PREDICTION
# =========================================================

print("\n" + "=" * 60)
print("GENERATING TEMPORAL PREDICTIONS")
print("=" * 60)

y_pred = model.predict(X_test)


# =========================================================
# EVALUATION
# =========================================================

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)


print("\n" + "=" * 60)
print("TEMPORAL CATBOOST RESULTS")
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

print(feature_importance.to_string(index=False))