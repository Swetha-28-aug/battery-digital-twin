# CellTwin — Battery Health Digital Twin

A digital twin for EV battery health: real degradation data, cross-validated
machine learning models, and a live simulation dashboard for state-of-health
(SoH) and remaining-useful-life (RUL) prediction.

Built for the e-mobility hackathon (Sept 2026).

## What it does

- Tracks a battery cell's health over its logged charge/discharge cycles
- Predicts current SoH and RUL using a cross-validated Random Forest model
- Simulates future degradation beyond the logged data using a physics-based
  exponential decay curve fit
- Flags alarms when a cell crosses unsafe or end-of-life thresholds
- Answers "what-if" usage questions (fast charging, deep discharge, heat)
  grounded in real dataset patterns

## Datasets

| Dataset | Source | Cells | What it's used for |
|---|---|---|---|
| NASA Li-ion Battery Aging | [NASA Ames PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) | 34 cells | Primary SoH/RUL training data, varied conditions |
| CALCE CS2 | [CALCE, University of Maryland](https://calce.umd.edu/battery-data) | 4 cells (CS2_35–38) | Second, independent validation source |

Raw data isn't checked into this repo (see `.gitignore`) — download separately
and follow the folder layout in `pipeline/`.

## Pipeline

Run in this order:

```bash
pip install pandas numpy scikit-learn scipy matplotlib joblib openpyxl fastapi uvicorn

python pipeline/battery_pipeline.py     # cleans NASA data -> data/battery_capacity_fade.csv
python pipeline/calce_pipeline.py       # cleans CALCE data -> data/calce_capacity_fade.csv
python pipeline/simulate_battery_twin.py  # trains models, runs simulation -> dashboard_data.json
```

Then open `site/celltwin.html` directly in a browser — no server required.

To run the live prediction API:
```bash
cd pipeline
uvicorn battery_api:app --reload --port 8000
# docs at http://127.0.0.1:8000/docs
```

## Model accuracy (5-fold cross-validated)

| Dataset | SoH MAE | SoH R² | RUL MAE | RUL R² |
|---|---|---|---|---|
| NASA (34 cells) | 3.48% | 0.871 | 0.9 cycles | 0.996 |
| CALCE (4 cells) | 0.89% | 0.974 | — | — |

Predictions are out-of-fold — the model never sees a cycle during its own
training fold before predicting it.

## Bugs caught along the way

Worth knowing if you're extending this:

- **CALCE capacity column is cumulative, not per-cycle.** The raw
  `Discharge_Capacity(Ah)` column resets only once per session file, not per
  cycle. Taking `.max()` per cycle gives a running total; the fix is
  `max - min` within each cycle's rows.
- **Data leakage in the first model version.** An early feature
  (`cycle_position_norm`) was calculated from each battery's own known final
  cycle count — information a real deployed twin wouldn't have in advance.
  Removing it dropped NASA's R² from an inflated 0.64 to an honest 0.21.
  Accuracy was then improved properly by adding battery identity as a
  feature instead (legitimate — it's known, not future information),
  bringing R² to 0.871.
- **Truncated session files.** Some CALCE session files end mid-discharge,
  producing artificially low capacity readings for that cycle. Filtered out
  before training.

## Tech stack

- Python: pandas, scikit-learn (RandomForestRegressor), scipy (curve fitting)
- FastAPI backend for live predictions
- Single-file HTML/CSS/JS dashboard (Chart.js) — no build step, no server
  dependency for the demo itself
