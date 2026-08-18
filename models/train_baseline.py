"""
Baseline ML Model
-----------------
Predict CPU_Requirement using Memory_Requirement.

This is our first simple baseline.
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

X = df[["Memory_Requirement"]]

y = df["CPU_Requirement"]


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
# TRAIN MODEL
# =========================================================

print("\n" + "=" * 60)
print("TRAINING LINEAR REGRESSION")
print("=" * 60)

model = LinearRegression()

model.fit(X_train, y_train)


# =========================================================
# PREDICTION
# =========================================================

y_pred = model.predict(X_test)


# =========================================================
# EVALUATION
# =========================================================

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)


print("\n" + "=" * 60)
print("BASELINE MODEL RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# =========================================================
# MODEL COEFFICIENT
# =========================================================

print("\n" + "=" * 60)
print("MODEL INFORMATION")
print("=" * 60)

print(f"Coefficient : {model.coef_[0]:.6f}")
print(f"Intercept   : {model.intercept_:.6f}")