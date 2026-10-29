"""
tests/test_feasibility.py
=========================
Independent Behavioral Unit Tests for Physical & Contractual Route Feasibility:
- Recomputes vehicle capacity limits directly from raw data/stops.csv and data/vehicles.csv.
- Recomputes timeline and SLA delivery window constraints directly from raw distances and speeds.
- Recomputes driver shift limit directly from raw return travel times.
- Zero reliance on internal candidate properties: verifies actual calculation behavior.
"""

import sys
import csv
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.candidate_generator import generate_candidate_route_pool, evaluate_route_candidate
from app.data import DEPOT, STOPS, VEHICLES, EDGES


def load_raw_csv_fixtures():
    """Independently parses data/*.csv files without using app data structures."""
    stops_file = BASE_DIR / "data" / "stops.csv"
    vehicles_file = BASE_DIR / "data" / "vehicles.csv"
    edges_file = BASE_DIR / "data" / "edges_with_qcloud_risk.csv"
    if not edges_file.exists():
        edges_file = BASE_DIR / "data" / "edges_uncertainty.csv"

    raw_stops = {}
    with open(stops_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sid = int(r["stop_id"])
            raw_stops[sid] = {
                "demand_kg": float(r["demand_kg"]),
                "tw_start_min": float(r["tw_start_min"]),
                "tw_end_min": float(r["tw_end_min"]),
                "service_time_min": float(r["service_time_min"]),
                "name": r["name"]
            }

    raw_vehicles = {}
    with open(vehicles_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            vid = r["vehicle_id"]
            raw_vehicles[vid] = {
                "capacity_kg": float(r["capacity_kg"]),
                "avg_speed_kmh": float(r["avg_speed_kmh"]),
                "max_shift_minutes": float(r["max_shift_minutes"])
            }

    raw_edges = {}
    with open(edges_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            u, v = int(r["from_stop"]), int(r["to_stop"])
            raw_edges[(u, v)] = float(r["distance_km"])

    return raw_stops, raw_vehicles, raw_edges


def test_candidate_route_feasibility_recomputed_from_csvs():
    """
    For every generated route candidate in the pool, independently recompute:
    1. Total demand <= Vehicle capacity.
    2. Leg-by-leg arrival times <= Delivery SLA window deadline.
    3. Final return arrival time at central depot <= Driver max shift limit.
    """
    raw_stops, raw_vehicles, raw_edges = load_raw_csv_fixtures()
    stops_dict = {DEPOT["id"]: DEPOT, **{s["id"]: s for s in STOPS}}

    candidates, _ = generate_candidate_route_pool(
        stops_dict, EDGES, VEHICLES, risk_weight=1.0, max_stops_per_route=4
    )
    assert len(candidates) > 0, "Candidate route pool must not be empty"

    for cand in candidates:
        vid = cand["vehicle_id"]
        v_info = raw_vehicles[vid]
        stop_seq = cand["stops"]
        assert len(stop_seq) >= 1, "Candidate must visit at least one customer stop"

        # 1. Independently recompute demand from stops.csv
        recomputed_demand = sum(raw_stops[s]["demand_kg"] for s in stop_seq)
        assert recomputed_demand <= v_info["capacity_kg"] + 1e-6, (
            f"Candidate for {vid} exceeds capacity: {recomputed_demand} > {v_info['capacity_kg']}"
        )

        # 2. Independently simulate timeline from raw distance and speed
        t = 0.0
        curr = 0  # Depot
        speed = v_info["avg_speed_kmh"]

        for sid in stop_seq:
            leg_key = (curr, sid)
            assert leg_key in raw_edges, f"Missing edge {leg_key} in raw edges CSV"
            dist = raw_edges[leg_key]
            transit_min = (dist / speed) * 60.0
            arrival_time = t + transit_min

            tw_start = raw_stops[sid]["tw_start_min"]
            tw_end = raw_stops[sid]["tw_end_min"]

            # Must arrive before or at SLA deadline
            assert arrival_time <= tw_end + 1e-4, (
                f"Candidate arrived late at stop {sid}: {arrival_time:.2f} > {tw_end}"
            )

            service_start = max(arrival_time, tw_start)
            service_dur = raw_stops[sid]["service_time_min"]
            departure_time = service_start + service_dur

            t = departure_time
            curr = sid

        # 3. Independently check return trip to depot
        ret_key = (curr, 0)
        assert ret_key in raw_edges, f"Missing return edge {ret_key} in raw edges CSV"
        ret_dist = raw_edges[ret_key]
        ret_arrival = t + (ret_dist / speed) * 60.0

        assert ret_arrival <= v_info["max_shift_minutes"] + 1e-4, (
            f"Candidate route exceeds driver shift limit: {ret_arrival:.2f} > {v_info['max_shift_minutes']}"
        )


def test_invalid_candidate_rejected_by_capacity():
    """Verify that an overloaded stop combination is rejected by the candidate evaluator."""
    raw_stops, raw_vehicles, _ = load_raw_csv_fixtures()
    stops_dict = {DEPOT["id"]: DEPOT, **{s["id"]: s for s in STOPS}}

    # Find a stop combination exceeding EcoVan capacity (1200 kg)
    # Stop 3 (750 kg) + Stop 4 (640 kg) = 1390 kg > 1200 kg
    overloaded_seq = [3, 4]
    tot_dem = raw_stops[3]["demand_kg"] + raw_stops[4]["demand_kg"]
    assert tot_dem > raw_vehicles["V1_EcoVan"]["capacity_kg"]

    res = evaluate_route_candidate("V1_EcoVan", overloaded_seq, stops_dict, EDGES, VEHICLES)
    assert res is None, "Overloaded sequence must be rejected by evaluate_route_candidate"


def test_invalid_candidate_rejected_by_time_window():
    """Verify that a sequence arriving past SLA deadline is rejected."""
    stops_dict = {DEPOT["id"]: DEPOT, **{s["id"]: s for s in STOPS}}

    # Stop 1 (deadline 180 min) visited after late stops [8, 7, 5, 1]
    # Visiting 8 -> 7 -> 5 first causes severe delay before reaching stop 1
    late_seq = [8, 7, 5, 1]
    res = evaluate_route_candidate("V1_EcoVan", late_seq, stops_dict, EDGES, VEHICLES)
    assert res is None, "Sequence violating customer SLA deadline must be rejected"
