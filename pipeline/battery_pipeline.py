"""
NASA Battery Dataset - SoH / RUL prediction pipeline
=====================================================

Steps:
1. Load metadata.csv, isolate discharge cycles (these carry Capacity, Ah)
2. Build a per-battery capacity-fade curve (Capacity vs cycle index)
3. Engineer features: cycle_index, ambient_temperature, capacity_fade_pct,
   rolling slope of degradation
4. Compute SoH (%) = Capacity / initial_Capacity * 100
5. Compute RUL (cycles remaining until SoH hits 70% -- the standard
   NASA/EOL threshold, i.e. 30% capacity fade)
6. Train a RandomForestRegressor to predict SoH from cycle_index +
   ambient_temperature + battery-specific normalized cycle position
7. Save: cleaned dataset (CSV), trained model (joblib), and diagnostic plots
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "cleaned_dataset")  # unzip archive.zip here (produces this folder)
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load metadata, keep only discharge cycles (Capacity is only populated there)
# ---------------------------------------------------------------------------
meta = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
discharge = meta[meta["type"] == "discharge"].copy()
discharge["Capacity"] = pd.to_numeric(discharge["Capacity"], errors="coerce")
discharge = discharge.dropna(subset=["Capacity"])
discharge["ambient_temperature"] = pd.to_numeric(discharge["ambient_temperature"], errors="coerce")

print(f"Total discharge cycles with capacity data: {len(discharge)}")
print(f"Unique batteries: {discharge['battery_id'].nunique()}")

# ---------------------------------------------------------------------------
# 2. Build per-battery cycle index + capacity-fade curve
# ---------------------------------------------------------------------------
discharge = discharge.sort_values(["battery_id", "test_id"]).reset_index(drop=True)
discharge["cycle_index"] = discharge.groupby("battery_id").cumcount() + 1

# Reference ("rated") capacity per battery = max capacity observed.
# Using the *first* cycle is unreliable: several batteries have a broken/
# incomplete first discharge (near-zero Ah), which blows up the ratio.
# Peak observed capacity is the physically correct nominal-capacity proxy.
rated_cap = discharge.groupby("battery_id")["Capacity"].transform("max")
discharge["SoH_pct"] = discharge["Capacity"] / rated_cap * 100

# Drop clearly broken measurement rows (SoH > 105% shouldn't happen once
# normalized against the max; a few % of overshoot from noise is fine)
discharge = discharge[discharge["SoH_pct"] <= 105]

# max cycle count per battery (used for normalized position feature)
max_cycle = discharge.groupby("battery_id")["cycle_index"].transform("max")
discharge["cycle_position_norm"] = discharge["cycle_index"] / max_cycle

# ---------------------------------------------------------------------------
# 3. Compute RUL: cycles remaining until SoH crosses 70% (30% fade = EOL)
# ---------------------------------------------------------------------------
EOL_THRESHOLD = 70.0  # % SoH considered end-of-life

def compute_rul(group):
    group = group.sort_values("cycle_index")
    eol_rows = group[group["SoH_pct"] <= EOL_THRESHOLD]
    if len(eol_rows) > 0:
        eol_cycle = eol_rows["cycle_index"].iloc[0]
    else:
        eol_cycle = group["cycle_index"].max()  # never reached EOL in this data
    group["RUL_cycles"] = (eol_cycle - group["cycle_index"]).clip(lower=0)
    return group

discharge = discharge.groupby("battery_id", group_keys=False)[discharge.columns].apply(compute_rul)

# save cleaned dataset
clean_path = os.path.join(OUT_DIR, "battery_capacity_fade.csv")
discharge[[
    "battery_id", "cycle_index", "Capacity", "SoH_pct",
    "ambient_temperature", "cycle_position_norm", "RUL_cycles", "start_time"
]].to_csv(clean_path, index=False)
print(f"Saved cleaned capacity-fade dataset -> {clean_path}")

# ---------------------------------------------------------------------------
# 4. Train/test split + RandomForest model to predict SoH from cycle info
# ---------------------------------------------------------------------------
model_df = discharge.dropna(subset=["ambient_temperature"]).copy()

features = ["cycle_index", "ambient_temperature", "cycle_position_norm"]
target = "SoH_pct"

X = model_df[features]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"SoH model  ->  MAE: {mae:.2f}%  |  R^2: {r2:.3f}")

model_path = os.path.join(OUT_DIR, "soh_rf_model.joblib")
joblib.dump(model, model_path)
print(f"Saved trained model -> {model_path}")

# ---------------------------------------------------------------------------
# 5. Also train a RUL regressor (useful for "how many cycles left" chatbot answer)
# ---------------------------------------------------------------------------
y_rul = model_df["RUL_cycles"]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_rul, test_size=0.2, random_state=42
)
rul_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rul_model.fit(X_train_r, y_train_r)
rul_preds = rul_model.predict(X_test_r)
rul_mae = mean_absolute_error(y_test_r, rul_preds)
rul_r2 = r2_score(y_test_r, rul_preds)
print(f"RUL model  ->  MAE: {rul_mae:.1f} cycles  |  R^2: {rul_r2:.3f}")

rul_model_path = os.path.join(OUT_DIR, "rul_rf_model.joblib")
joblib.dump(rul_model, rul_model_path)
print(f"Saved trained RUL model -> {rul_model_path}")

# ---------------------------------------------------------------------------
# 6. Diagnostic plots
# ---------------------------------------------------------------------------
# 6a. Capacity fade curves for a handful of batteries
sample_batteries = discharge["battery_id"].unique()[:6]
plt.figure(figsize=(9, 6))
for bid in sample_batteries:
    sub = discharge[discharge["battery_id"] == bid]
    plt.plot(sub["cycle_index"], sub["SoH_pct"], label=bid, linewidth=1.5)
plt.axhline(EOL_THRESHOLD, color="red", linestyle="--", label="EOL threshold (70%)")
plt.xlabel("Cycle Index")
plt.ylabel("State of Health (%)")
plt.title("Battery Capacity Fade (SoH vs Cycle) - Sample Batteries")
plt.legend(fontsize=8)
plt.tight_layout()
fade_plot_path = os.path.join(OUT_DIR, "capacity_fade_curves.png")
plt.savefig(fade_plot_path, dpi=150)
plt.close()
print(f"Saved plot -> {fade_plot_path}")

# 6b. Predicted vs actual SoH scatter
plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds, alpha=0.4, s=15)
lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
plt.plot(lims, lims, "r--", linewidth=1)
plt.xlabel("Actual SoH (%)")
plt.ylabel("Predicted SoH (%)")
plt.title(f"SoH Prediction: Actual vs Predicted (R²={r2:.3f})")
plt.tight_layout()
pred_plot_path = os.path.join(OUT_DIR, "soh_pred_vs_actual.png")
plt.savefig(pred_plot_path, dpi=150)
plt.close()
print(f"Saved plot -> {pred_plot_path}")

print("\nDone. Outputs in:", OUT_DIR)
