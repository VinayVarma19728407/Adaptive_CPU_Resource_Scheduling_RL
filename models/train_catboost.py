"""
CatBoost Model
--------------
Predict CPU_Requirement using categorical and numerical features.
"""

import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
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
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 60)
print("DATA SPLIT")
print("=" * 60)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# =========================================================
# TRAIN CATBOOST
# =========================================================

print("\n" + "=" * 60)
print("TRAINING CATBOOST")
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
print("GENERATING PREDICTIONS")
print("=" * 60)

y_pred = model.predict(X_test)


# =========================================================
# EVALUATION
# =========================================================

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)


print("\n" + "=" * 60)
print("CATBOOST MODEL RESULTS")
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