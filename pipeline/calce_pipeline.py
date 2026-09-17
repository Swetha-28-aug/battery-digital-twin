"""
CALCE CS2 Battery Dataset - Extraction Pipeline
================================================
Parses raw Arbin-cycler Excel exports (one file per test session, many
sessions per battery) for cells CS2_35, CS2_36, CS2_37, CS2_38.

For each session file:
  - finds the data sheet (named 'Channel_1-XXX')
  - groups rows by Cycle_Index (local, resets each file)
  - takes end-of-cycle Discharge_Capacity(Ah) as that cycle's capacity
  - takes mean |Current(A)| during discharging rows as the discharge rate
    used in that session (this is what varies across CALCE sessions and
    is the basis for the "what-if" logic)

Sessions are then sorted chronologically per battery and cycle indices
are made continuous (global_cycle_index) so we get one clean fade curve
per battery, same shape as the NASA pipeline output.
"""

import pandas as pd
import numpy as np
import glob
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # unzip CS2_35.zip, CS2_36.zip, etc. here (each produces a CS2_XX folder)
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

BATTERIES = {
    "CS2_35": "Channel_1-008",
    "CS2_36": "Channel_1-009",
    "CS2_37": "Channel_1-010",
    "CS2_38": "Channel_1-011",
}

USECOLS = ["Date_Time", "Cycle_Index", "Current(A)", "Discharge_Capacity(Ah)"]


def extract_session(filepath: str, sheet_name: str) -> pd.DataFrame:
    """Read one Excel session file, return per-cycle summary rows."""
    df = pd.read_excel(filepath, sheet_name=sheet_name, usecols=USECOLS)
    if df.empty:
        return pd.DataFrame()

    session_start = df["Date_Time"].min()

    rows = []
    for cyc, grp in df.groupby("Cycle_Index"):
        # Discharge_Capacity(Ah) is a CUMULATIVE counter across the whole
        # session file, not per-cycle -- this cycle's actual discharge
        # capacity is the increase (max - min) within this cycle's rows.
        discharge_cap = grp["Discharge_Capacity(Ah)"].max() - grp["Discharge_Capacity(Ah)"].min()
        discharge_rows = grp[grp["Current(A)"] < -0.01]
        avg_discharge_current = (
            discharge_rows["Current(A)"].abs().mean() if len(discharge_rows) else np.nan
        )
        rows.append({
            "local_cycle_index": cyc,
            "discharge_capacity_Ah": discharge_cap,
            "avg_discharge_current_A": avg_discharge_current,
        })

    out = pd.DataFrame(rows)
    out["session_start"] = session_start
    out["session_file"] = os.path.basename(filepath)
    return out


def process_battery(battery_id: str, sheet_name: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(BASE_DIR, battery_id, "*.xlsx"))
    print(f"\n{battery_id}: {len(files)} session files")

    sessions = []
    for i, f in enumerate(files):
        t0 = time.time()
        try:
            s = extract_session(f, sheet_name)
            if not s.empty:
                sessions.append(s)
            print(f"  [{i+1}/{len(files)}] {os.path.basename(f)} -> "
                  f"{len(s)} cycles ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  [{i+1}/{len(files)}] {os.path.basename(f)} -> FAILED: {e}")

    if not sessions:
        return pd.DataFrame()

    combined = pd.concat(sessions, ignore_index=True)

    # Order sessions deterministically: by start time, then filename as tiebreak
    # (two sessions in this data share an identical logged start timestamp,
    # which breaks a naive sort -- filename as secondary key fixes that)
    session_order = (
        combined.groupby("session_file")
        .agg(session_start=("session_start", "first"),
             local_max=("local_cycle_index", "max"))
        .sort_values(["session_start", "session_file"])
    )
    session_order["offset"] = session_order["local_max"].cumsum().shift(fill_value=0)

    combined = combined.merge(
        session_order[["offset"]], left_on="session_file", right_index=True
    )
    combined["cycle_index"] = combined["offset"] + combined["local_cycle_index"]
    combined = combined.sort_values("cycle_index").reset_index(drop=True)
    combined = combined.drop(columns=["offset"])
    combined["battery_id"] = battery_id

    # Drop truncated/incomplete cycles: a session file sometimes ends
    # mid-discharge, producing an artificially low capacity reading for
    # that final cycle. These show up as sharp drop-to-zero spikes that
    # aren't real degradation -- filter anything under half the nominal
    # ~1.1 Ah capacity.
    combined = combined[combined["discharge_capacity_Ah"] >= 0.5]

    # SoH normalized against peak observed capacity (same approach as NASA pipeline)
    rated_cap = combined["discharge_capacity_Ah"].max()
    combined["SoH_pct"] = combined["discharge_capacity_Ah"] / rated_cap * 100
    combined = combined[combined["SoH_pct"] <= 105]

    return combined


# ---------------------------------------------------------------------------
# Run extraction for all four batteries
# ---------------------------------------------------------------------------
all_batteries = []
for bid, sheet in BATTERIES.items():
    result = process_battery(bid, sheet)
    if not result.empty:
        all_batteries.append(result)

calce_df = pd.concat(all_batteries, ignore_index=True)
calce_df = calce_df[[
    "battery_id", "cycle_index", "local_cycle_index", "session_file",
    "session_start", "discharge_capacity_Ah", "SoH_pct", "avg_discharge_current_A"
]]

out_path = os.path.join(OUT_DIR, "calce_capacity_fade.csv")
calce_df.to_csv(out_path, index=False)
print(f"\nSaved combined CALCE dataset -> {out_path}")
print(f"Total cycles across 4 batteries: {len(calce_df)}")
print(calce_df.groupby("battery_id")["cycle_index"].max())
