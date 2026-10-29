"""
src/geo_utils.py
================
Geospatial data loader, coordinate validator, and update utility.
Handles reading delivery stops and vehicle fleet from CSVs, managing missing GPS coordinates,
deriving explicitly labeled simulated coordinates, and persisting user updates.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Bengaluru Central Depot reference point
BENGALURU_DEPOT_LAT = 12.9716
BENGALURU_DEPOT_LNG = 77.5946

# Scaling factors for Bengaluru latitude (~12.97 deg N)
# 1 degree of latitude ~ 110.8 km
# 1 degree of longitude ~ 110.8 * cos(12.9716 deg) ~ 107.97 km
KM_PER_LAT_DEG = 110.8
KM_PER_LNG_DEG = 107.97

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
EVIDENCE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evidence")

STOPS_CSV_PATH = os.path.join(DATA_DIR, "stops.csv")
VEHICLES_CSV_PATH = os.path.join(DATA_DIR, "vehicles.csv")
SOLUTION_JSON_PATH = os.path.join(EVIDENCE_DIR, "03_qpiai_opt_dispatch_solution.json")

def load_vehicles(csv_path: str = VEHICLES_CSV_PATH) -> pd.DataFrame:
    """Loads vehicle fleet details from vehicles.csv."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Vehicles CSV not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    return df

def load_stops(csv_path: str = STOPS_CSV_PATH) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads stops from CSV.
    Inspects whether true 'latitude' and 'longitude' columns exist and are populated.
    If missing or null, derives simulated coordinates from (x_coord, y_coord) km offsets,
    and returns a summary report of missing/simulated records.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Stops CSV not found at: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    has_lat = "latitude" in df.columns
    has_lng = "longitude" in df.columns
    
    records_needing_coords = []
    
    if not (has_lat and has_lng):
        df["latitude"] = np.nan
        df["longitude"] = np.nan
    
    # Check each row
    is_simulated_list = []
    coord_status_list = []
    
    for idx, row in df.iterrows():
        lat_val = row["latitude"]
        lng_val = row["longitude"]
        
        # If coordinates are missing or NaN
        if pd.isna(lat_val) or pd.isna(lng_val) or (lat_val == 0.0 and lng_val == 0.0 and row["stop_id"] != 0):
            records_needing_coords.append({
                "stop_id": int(row["stop_id"]),
                "name": str(row["name"]),
                "x_coord_km": float(row.get("x_coord", 0.0)),
                "y_coord_km": float(row.get("y_coord", 0.0)),
                "demand_kg": int(row.get("demand_kg", 0)),
                "sla_penalty_inr": float(row.get("sla_penalty_inr", 0.0))
            })
            
            # Derive explicitly labeled simulated coordinates
            x_km = float(row.get("x_coord", 0.0))
            y_km = float(row.get("y_coord", 0.0))
            
            sim_lat = round(BENGALURU_DEPOT_LAT + (y_km / KM_PER_LAT_DEG), 6)
            sim_lng = round(BENGALURU_DEPOT_LNG + (x_km / KM_PER_LNG_DEG), 6)
            
            df.at[idx, "latitude"] = sim_lat
            df.at[idx, "longitude"] = sim_lng
            is_simulated_list.append(True)
            coord_status_list.append("Simulated Data (from km offset)")
        else:
            is_simulated_list.append(False)
            coord_status_list.append("Real GPS Coordinates")

    df["is_simulated"] = is_simulated_list
    df["coord_status"] = coord_status_list
    
    status_summary = {
        "total_stops": len(df),
        "real_coordinates_count": len(df) - len(records_needing_coords),
        "simulated_coordinates_count": len(records_needing_coords),
        "records_needing_coords": records_needing_coords,
        "is_all_simulated": len(records_needing_coords) == len(df),
        "reference_depot": {"lat": BENGALURU_DEPOT_LAT, "lng": BENGALURU_DEPOT_LNG, "city": "Bengaluru, Karnataka"}
    }
    
    return df, status_summary

def update_stop_coordinates(
    stop_id: int,
    lat: float,
    lng: float,
    csv_path: str = STOPS_CSV_PATH
) -> bool:
    """
    Updates or adds real latitude and longitude for a specific stop_id in stops.csv.
    Saves directly to disk.
    """
    if not os.path.exists(csv_path):
        return False
        
    df = pd.read_csv(csv_path)
    if "latitude" not in df.columns:
        df["latitude"] = np.nan
    if "longitude" not in df.columns:
        df["longitude"] = np.nan
        
    matched = df[df["stop_id"] == stop_id]
    if matched.empty:
        return False
        
    idx = matched.index[0]
    df.at[idx, "latitude"] = round(float(lat), 6)
    df.at[idx, "longitude"] = round(float(lng), 6)
    
    df.to_csv(csv_path, index=False)
    return True

def load_dispatch_solution(solution_path: str = SOLUTION_JSON_PATH) -> Optional[Dict[str, Any]]:
    """Loads pre-computed or saved QpiAI Opt dispatch solution."""
    if os.path.exists(solution_path):
        try:
            with open(solution_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load solution JSON: {e}")
    return None

def get_stop_risk_badge(breach_prob: float) -> Tuple[str, str]:
    """Returns human-readable risk category and badge color."""
    if breach_prob >= 0.65:
        return "High Risk", "#EF4444"       # Red
    elif breach_prob >= 0.45:
        return "Moderate Risk", "#F59E0B"   # Amber
    else:
        return "Low Risk", "#10B981"        # Emerald Green
