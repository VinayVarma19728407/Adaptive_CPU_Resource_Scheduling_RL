import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# FILE
# =========================================================

DATA_FILE = (
    "C:/Users/gvkva/Desktop/My ongoing projects/"
    "Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL/"
    "data/processed/ml_cpu_dataset.csv"
)


# =========================================================
# LOAD
# =========================================================

print("=" * 60)
print("LOADING ML DATASET")
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
# FEATURES - NO IDS
# =========================================================

features = [
    "Arrival_Time",
    "Memory_Requirement"
]

target = "CPU_Requirement"

X = df[features]
y = df[target]


print("\n" + "=" * 60)
print("FEATURES USED")
print("=" * 60)

for feature in features:
    print(f"- {feature}")


print("\nIDs deliberately excluded:")
print("- Task_ID")
print("- Job_ID")
print("- PC_ID")


# =========================================================
# TEMPORAL SPLIT
# =========================================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\n" + "=" * 60)
print("TEMPORAL DATA SPLIT")
print("=" * 60)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

print(
    f"Training time range: "
    f"{X_train['Arrival_Time'].min()} -> "
    f"{X_train['Arrival_Time'].max()}"
)

print(
    f"Testing time range: "
    f"{X_test['Arrival_Time'].min()} -> "
    f"{X_test['Arrival_Time'].max()}"
)


# =========================================================
# TRAIN
# =========================================================

print("\n" + "=" * 60)
print("TRAINING CATBOOST WITHOUT IDS")
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
# PREDICT
# =========================================================

print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

y_pred = model.predict(X_test)


# =========================================================
# EVALUATE
# =========================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("CATBOOST WITHOUT IDS - RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# =========================================================
# COMPARISON
# =========================================================

print("\n" + "=" * 60)
print("COMPARISON WITH PREVIOUS TEMPORAL MODEL")
print("=" * 60)

print(
    f"{'Metric':<10}"
    f"{'With IDs':>15}"
    f"{'Without IDs':>18}"
)

print("-" * 43)

print(
    f"{'MAE':<10}"
    f"{25.1969:>15.4f}"
    f"{mae:>18.4f}"
)

print(
    f"{'RMSE':<10}"
    f"{32.0241:>15.4f}"
    f"{rmse:>18.4f}"
)

print(
    f"{'R²':<10}"
    f"{0.1007:>15.4f}"
    f"{r2:>18.4f}"
)