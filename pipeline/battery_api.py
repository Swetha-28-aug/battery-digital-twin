"""
Battery Digital Twin - Backend API
===================================
Single-file FastAPI server exposing:
  GET  /health/predict          -> current SoH % + RUL cycles for given inputs
  GET  /health/curve/{battery_id} -> full historical capacity-fade curve
  POST /whatif                  -> "what if" scenario answer (stub until
                                     CALCE discharge-rate data is loaded)

Run with:  uvicorn battery_api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# Load trained models + cleaned dataset once at startup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

soh_model = joblib.load(os.path.join(BASE_DIR, "soh_rf_model.joblib"))
rul_model = joblib.load(os.path.join(BASE_DIR, "rul_rf_model.joblib"))
capacity_df = pd.read_csv(os.path.join(BASE_DIR, "battery_capacity_fade.csv"))

app = FastAPI(title="Battery Digital Twin API")

# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    cycle_index: int
    ambient_temperature: float
    battery_id: str | None = None  # optional, used to look up cycle_position_norm


class PredictResponse(BaseModel):
    soh_pct: float
    rul_cycles: float
    status: str  # "healthy" | "degrading" | "replace_soon" | "end_of_life"


class WhatIfRequest(BaseModel):
    battery_id: str
    scenario: str  # e.g. "fast_charge_daily", "deep_discharge", "high_temp"


class WhatIfResponse(BaseModel):
    scenario: str
    summary: str
    estimated_extra_fade_pct: float | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def classify_status(soh_pct: float) -> str:
    if soh_pct >= 90:
        return "healthy"
    elif soh_pct >= 80:
        return "degrading"
    elif soh_pct >= 70:
        return "replace_soon"
    return "end_of_life"


def get_cycle_position_norm(battery_id: str | None, cycle_index: int) -> float:
    """Look up how far through its life this cycle is, for a known battery.
    Falls back to a rough estimate (against median max-cycle) for unknown IDs."""
    if battery_id and battery_id in capacity_df["battery_id"].values:
        max_cycle = capacity_df.loc[
            capacity_df["battery_id"] == battery_id, "cycle_index"
        ].max()
    else:
        max_cycle = capacity_df.groupby("battery_id")["cycle_index"].max().median()
    return min(cycle_index / max_cycle, 1.0) if max_cycle else 0.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Battery Digital Twin API running"}


@app.post("/health/predict", response_model=PredictResponse)
def predict_health(req: PredictRequest):
    cycle_position_norm = get_cycle_position_norm(req.battery_id, req.cycle_index)

    features = pd.DataFrame([{
        "cycle_index": req.cycle_index,
        "ambient_temperature": req.ambient_temperature,
        "cycle_position_norm": cycle_position_norm,
    }])

    soh_pred = float(soh_model.predict(features)[0])
    rul_pred = float(rul_model.predict(features)[0])
    soh_pred = max(0.0, min(100.0, soh_pred))
    rul_pred = max(0.0, rul_pred)

    return PredictResponse(
        soh_pct=round(soh_pred, 1),
        rul_cycles=round(rul_pred, 1),
        status=classify_status(soh_pred),
    )


@app.get("/health/curve/{battery_id}")
def get_curve(battery_id: str):
    sub = capacity_df[capacity_df["battery_id"] == battery_id]
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"No data for battery_id={battery_id}")
    return sub[["cycle_index", "SoH_pct", "RUL_cycles"]].to_dict(orient="records")


@app.get("/batteries")
def list_batteries():
    return sorted(capacity_df["battery_id"].unique().tolist())


@app.post("/whatif", response_model=WhatIfResponse)
def whatif(req: WhatIfRequest):
    """
    Stub logic for now -- once CALCE (CS2_35/36/37/38) discharge-rate data
    is loaded, replace this with an actual model trained on rate-vs-fade.
    Currently gives directionally-correct, literature-informed estimates
    so the chatbot has something real to say today.
    """
    scenario_effects = {
        "fast_charge_daily": (
            "Fast charging daily accelerates capacity fade, mainly from "
            "extra heat and lithium-plating risk at high charge rates.",
            8.0,
        ),
        "deep_discharge": (
            "Regularly discharging below the recommended cutoff voltage "
            "increases fade rate and raises internal resistance faster.",
            6.0,
        ),
        "high_temp": (
            "Operating at high ambient temperature is one of the strongest "
            "accelerators of capacity fade in Li-ion cells.",
            10.0,
        ),
    }

    if req.scenario not in scenario_effects:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario. Options: {list(scenario_effects.keys())}",
        )

    summary, extra_fade = scenario_effects[req.scenario]
    return WhatIfResponse(
        scenario=req.scenario,
        summary=f"[Preliminary estimate, pending CALCE data] {summary}",
        estimated_extra_fade_pct=extra_fade,
    )
