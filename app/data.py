"""
app/data.py
===========
Logistics definitions for QML-01: Delay-Risk-Aware Fleet Dispatch.
Single source of truth: loads directly from data/*.csv.
- 12 commercial stops + Depot 0 across Bengaluru, Karnataka.
- 4 heterogeneous vehicles (V1_EcoVan, V2_CityTruck, V3_HeavyHauler, V4_MaxFreight).
- 156 directed legs with physical distances, travel times, and risk metrics.
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

STOPS_CSV = DATA_DIR / "stops.csv"
VEHICLES_CSV = DATA_DIR / "vehicles.csv"
EDGES_RISK_CSV = DATA_DIR / "edges_with_qcloud_risk.csv"
EDGES_UNCERTAINTY_CSV = DATA_DIR / "edges_uncertainty.csv"

# Central reference for Bengaluru Depot
BENGALURU_DEPOT_LAT = 12.9716
BENGALURU_DEPOT_LNG = 77.5946
KM_PER_LAT = 110.8
KM_PER_LNG = 107.97

def _load_stops() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    depot = None
    stops = []
    if not STOPS_CSV.exists():
        raise FileNotFoundError(f"Missing required stops file: {STOPS_CSV}")

    with open(STOPS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row["stop_id"])
            x_km = float(row.get("x_coord", 0.0))
            y_km = float(row.get("y_coord", 0.0))
            lat = float(row["latitude"]) if "latitude" in row and row["latitude"] else round(BENGALURU_DEPOT_LAT + y_km / KM_PER_LAT, 6)
            lng = float(row["longitude"]) if "longitude" in row and row["longitude"] else round(BENGALURU_DEPOT_LNG + x_km / KM_PER_LNG, 6)
            tw_s = float(row["tw_start_min"])
            tw_e = float(row["tw_end_min"])
            
            stop_dict = {
                "id": sid,
                "name": row["name"],
                "area": row["name"].replace("_", " "),
                "x_coord": x_km,
                "y_coord": y_km,
                "lat": lat,
                "lng": lng,
                "demand_kg": int(row["demand_kg"]),
                "window_start_min": tw_s,
                "window_end_min": tw_e,
                "service_time_min": float(row["service_time_min"]),
                "sla_penalty_rs": float(row["sla_penalty_inr"]),
                "is_depot": (sid == 0)
            }
            if sid == 0:
                depot = stop_dict
            else:
                stops.append(stop_dict)

    if depot is None:
        depot = {
            "id": 0, "name": "Depot_Central", "area": "Central Bengaluru Hub",
            "x_coord": 0.0, "y_coord": 0.0, "lat": BENGALURU_DEPOT_LAT, "lng": BENGALURU_DEPOT_LNG,
            "demand_kg": 0, "window_start_min": 0, "window_end_min": 480,
            "service_time_min": 0, "sla_penalty_rs": 0, "is_depot": True
        }
    return depot, stops

def _load_vehicles() -> Dict[str, Dict[str, Any]]:
    vehicles = {}
    if not VEHICLES_CSV.exists():
        raise FileNotFoundError(f"Missing required vehicles file: {VEHICLES_CSV}")

    colors = {
        "V1_EcoVan": "#2563eb",       # Blue
        "V2_CityTruck": "#059669",    # Emerald
        "V3_HeavyHauler": "#d97706",  # Amber
        "V4_MaxFreight": "#7c3aed"    # Purple
    }

    with open(VEHICLES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["vehicle_id"]
            vehicles[vid] = {
                "id": vid,
                "name": vid.replace("_", " "),
                "capacity_kg": int(row["capacity_kg"]),
                "fuel_cost_per_km": float(row["fuel_cost_per_km"]),
                "cost_per_km": float(row["fuel_cost_per_km"]),
                "driver_shift_cost": float(row["driver_shift_cost"]),
                "labor_cost_per_hour": round(float(row["driver_shift_cost"]) / 8.0, 2),
                "avg_speed_kmh": float(row["avg_speed_kmh"]),
                "speed_kmh": float(row["avg_speed_kmh"]),
                "max_shift_minutes": float(row["max_shift_minutes"]),
                "color": colors.get(vid, "#64748b")
            }
    return vehicles

def _load_edges() -> Dict[Tuple[int, int], Dict[str, Any]]:
    edges_file = EDGES_RISK_CSV if EDGES_RISK_CSV.exists() else EDGES_UNCERTAINTY_CSV
    edges = {}
    if not edges_file.exists():
        return edges

    with open(edges_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u, v = int(row["from_stop"]), int(row["to_stop"])
            edges[(u, v)] = {
                "from_stop": u,
                "to_stop": v,
                "distance_km": float(row["distance_km"]),
                "base_travel_time_min": float(row["base_travel_time_min"]),
                "congestion_index": float(row["congestion_index"]),
                "loading_dock_variance": float(row["loading_dock_variance"]),
                "weather_risk": float(row["weather_risk"]),
                "calibrated_p_ij": float(row.get("calibrated_p_ij", row.get("historical_sla_breach", 0.3))),
                "expected_exposure_inr": float(row.get("expected_financial_exposure_inr", 0.0)),
                "sla_penalty": float(row.get("target_sla_penalty", 0.0))
            }
    return edges

# Module-level singletons
DEPOT, STOPS = _load_stops()
VEHICLES = _load_vehicles()
EDGES = _load_edges()

def get_distance(node_a: int, node_b: int) -> float:
    """Returns road/network distance between two stop IDs in km."""
    if node_a == node_b:
        return 0.0
    if (node_a, node_b) in EDGES:
        return EDGES[(node_a, node_b)]["distance_km"]
    if (node_b, node_a) in EDGES:
        return EDGES[(node_b, node_a)]["distance_km"]
    # Fallback to Euclidean on stops coordinates
    all_nodes = {DEPOT["id"]: DEPOT, **{s["id"]: s for s in STOPS}}
    if node_a in all_nodes and node_b in all_nodes:
        na, nb = all_nodes[node_a], all_nodes[node_b]
        dx = na["x_coord"] - nb["x_coord"]
        dy = na["y_coord"] - nb["y_coord"]
        return round((dx*dx + dy*dy)**0.5, 2)
    return 10.0

def reload_dataset():
    """Reloads dataset singletons from disk if files were regenerated."""
    global DEPOT, STOPS, VEHICLES, EDGES
    DEPOT, STOPS = _load_stops()
    VEHICLES = _load_vehicles()
    EDGES = _load_edges()
