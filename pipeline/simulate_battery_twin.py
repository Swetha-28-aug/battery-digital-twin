"""
Battery Digital Twin - Simulation Layer
========================================
This is the "brain" behind the dashboard. Two jobs:

1. ACCURACY: train a RandomForest on each dataset (NASA, CALCE) using
   5-fold cross-validation, so every logged cycle gets an *out-of-fold*
   predicted SoH -- i.e. a genuinely honest prediction, not the model
   grading its own training data. This is what backs the "model accuracy"
   numbers shown in the dashboard.

2. SIMULATION: for each individual battery, fit a capacity-fade curve
   (exponential decay, the standard Li-ion aging model) to its own
   logged history with scipy.optimize.curve_fit, then extrapolate that
   curve forward to simulate future cycles until end-of-life. This
   replaces naive client-side linear extrapolation with an actual
   physically-motivated simulation.

Output: dashboard_data.json, consumed directly by the HTML dashboard.
No computation happens in the browser -- it only renders what Python
already computed and validated here.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.optimize import curve_fit
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(SCRIPT_DIR, "outputs", "dashboard_data.json")
EOL_THRESHOLD = 70.0

# ---------------------------------------------------------------------------
# Load cleaned datasets (produced by battery_pipeline.py / calce_pipeline.py)
# Both write into a sibling "outputs/" folder next to each script -- if you
# ran battery_pipeline.py and calce_pipeline.py from a different folder than
# this script, update these two paths to point there instead.
# ---------------------------------------------------------------------------
nasa = pd.read_csv(os.path.join(SCRIPT_DIR, "outputs", "battery_capacity_fade.csv"))
calce = pd.read_csv(os.path.join(SCRIPT_DIR, "outputs", "calce_capacity_fade.csv"))

nasa = nasa.dropna(subset=["ambient_temperature"]).copy()
calce = calce.dropna(subset=["avg_discharge_current_A"]).copy()

# cycle_position_norm needed for CALCE (NASA's CSV doesn't carry it, recompute for both)
for df in (nasa, calce):
    max_cycle = df.groupby("battery_id")["cycle_index"].transform("max")
    df["cycle_position_norm"] = df["cycle_index"] / max_cycle


# ---------------------------------------------------------------------------
# 1. Cross-validated accuracy per dataset
# ---------------------------------------------------------------------------
def cross_validated_predictions(df, feature_cols, target_col, categorical_col=None):
    X = df[feature_cols].copy()
    if categorical_col:
        dummies = pd.get_dummies(df[categorical_col], prefix=categorical_col)
        X = pd.concat([X, dummies], axis=1)
    X = X.values
    y = df[target_col].values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    preds = cross_val_predict(model, X, y, cv=kf)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    return preds, mae, r2


nasa_features = ["cycle_index", "ambient_temperature"]
calce_features = ["cycle_index", "avg_discharge_current_A"]

nasa_soh_pred, nasa_soh_mae, nasa_soh_r2 = cross_validated_predictions(nasa, nasa_features, "SoH_pct", categorical_col="battery_id")
nasa_rul_pred, nasa_rul_mae, nasa_rul_r2 = cross_validated_predictions(nasa, nasa_features, "RUL_cycles", categorical_col="battery_id")
nasa["SoH_pred"] = nasa_soh_pred
nasa["RUL_pred"] = nasa_rul_pred

calce_soh_pred, calce_soh_mae, calce_soh_r2 = cross_validated_predictions(calce, calce_features, "SoH_pct", categorical_col="battery_id")
calce["SoH_pred"] = calce_soh_pred

print(f"NASA  SoH: MAE={nasa_soh_mae:.2f}%  R2={nasa_soh_r2:.3f}   |   RUL: MAE={nasa_rul_mae:.1f}cyc  R2={nasa_rul_r2:.3f}")
print(f"CALCE SoH: MAE={calce_soh_mae:.2f}%  R2={calce_soh_r2:.3f}")

accuracy = {
    "nasa": {"soh_mae": round(nasa_soh_mae, 2), "soh_r2": round(nasa_soh_r2, 3),
              "rul_mae": round(nasa_rul_mae, 1), "rul_r2": round(nasa_rul_r2, 3)},
    "calce": {"soh_mae": round(calce_soh_mae, 2), "soh_r2": round(calce_soh_r2, 3)},
}


# ---------------------------------------------------------------------------
# 2. Per-battery capacity-fade simulation (exponential decay curve fit)
# ---------------------------------------------------------------------------
def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c


def simulate_future(cycles, soh, extend_to=40.0, max_extra_cycles=1500):
    """Fit an exponential decay curve to this battery's own SoH history and
    extrapolate forward until it crosses `extend_to` percent SoH."""
    cycles = np.array(cycles, dtype=float)
    soh = np.array(soh, dtype=float)

    try:
        p0 = [soh[0] - soh[-1], 1.0 / max(cycles[-1], 1), soh[-1]]
        popt, _ = curve_fit(exp_decay, cycles, soh, p0=p0, maxfev=8000)
        fit_fn = lambda x: exp_decay(x, *popt)
    except Exception:
        # fallback: simple linear fit on the last 40% of points
        tail = cycles[int(len(cycles) * 0.6):]
        tail_y = soh[int(len(cycles) * 0.6):]
        slope, intercept = np.polyfit(tail, tail_y, 1)
        fit_fn = lambda x: slope * x + intercept

    last_cycle = cycles[-1]
    step = max(1, round(last_cycle / len(cycles)))
    proj_cycles, proj_soh = [], []
    c = last_cycle
    guard = 0
    while guard < max_extra_cycles:
        c += step
        val = float(fit_fn(c))
        if val <= extend_to or val > 105:
            break
        proj_cycles.append(int(c))
        proj_soh.append(round(max(0.0, val), 2))
        guard += 1

    return proj_cycles, proj_soh


def build_battery_series(df, id_col="battery_id", extra_col=None, extra_key=None):
    out = {}
    for bid, grp in df.groupby(id_col):
        grp = grp.sort_values("cycle_index")
        cycles = grp["cycle_index"].tolist()
        soh = grp["SoH_pct"].round(2).tolist()
        soh_pred = grp["SoH_pred"].round(2).tolist()
        proj_cycles, proj_soh = simulate_future(cycles, soh)

        entry = {
            "cycles": cycles,
            "soh": soh,
            "soh_pred": soh_pred,
            "proj_cycles": proj_cycles,
            "proj_soh": proj_soh,
        }
        if extra_col:
            entry[extra_key] = [round(x, 2) if pd.notna(x) else None for x in grp[extra_col].tolist()]
        if "RUL_pred" in grp.columns:
            entry["rul_pred_latest"] = round(float(grp["RUL_pred"].iloc[-1]), 1)
        out[bid] = entry
    return out


dashboard = {
    "nasa": build_battery_series(nasa, extra_col="ambient_temperature", extra_key="temp"),
    "calce": build_battery_series(calce, extra_col="avg_discharge_current_A", extra_key="current"),
    "accuracy": accuracy,
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(dashboard, f)

print(f"\nSaved -> {OUT_PATH}  ({os.path.getsize(OUT_PATH)/1024:.1f} KB)")
