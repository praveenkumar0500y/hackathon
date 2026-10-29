"""
tests/test_qubo_energy.py
=========================
Unit tests for explicit Set-Partitioning QUBO matrix formulation:
1. Scale-derived penalties: P_cov = 2 * C_max + 10,000, P_veh = 2 * P_cov.
2. Exact energy equivalence on known feasible route selections.
3. Violation of coverage constraint raises QUBO energy by at least P_cov.
4. Exhaustive brute-force verification on a 5-stop / 2-vehicle instance.
"""

import pytest
import sys
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.data import DEPOT, STOPS, VEHICLES, EDGES
from src.candidate_generator import generate_candidate_route_pool
from src.step4_opt_dispatch import (
    calculate_scale_derived_penalties,
    build_explicit_qubo_matrix,
    evaluate_qubo_energy
)

STOPS_DICT = {DEPOT["id"]: DEPOT, **{s["id"]: s for s in STOPS}}


def test_scale_derived_penalties():
    """Verify that scale-derived penalties are mathematically sound and dominate costs."""
    candidates, _ = generate_candidate_route_pool(
        STOPS_DICT, EDGES, VEHICLES, risk_weight=1.0, max_stops_per_route=4
    )
    C_max, P_cov, P_veh = calculate_scale_derived_penalties(candidates)

    assert C_max > 0, "C_max must be positive"
    assert P_cov >= 2.0 * C_max + 10000.0, "P_cov must satisfy derivation formula"
    assert P_veh >= 2.0 * P_cov, "P_veh must strictly dominate coverage penalty"
    assert P_cov > C_max, "Stop coverage penalty must strictly dominate maximum route cost"
    assert P_veh > C_max, "Vehicle conflict penalty must exceed maximum route cost"


def test_qubo_energy_matches_true_cost_on_feasible_solution():
    """Verify that x^T * Q * x + const_offset exactly equals the true cost for feasible selections."""
    cand1 = {
        "id": 0,
        "total_route_cost_inr": 2500.0,
        "stops": [1, 2],
        "vehicle_id": "Truck-1",
        "timeline": []
    }
    cand2 = {
        "id": 1,
        "total_route_cost_inr": 3200.0,
        "stops": [3, 4],
        "vehicle_id": "Truck-2",
        "timeline": []
    }
    cand3 = {
        "id": 2,
        "total_route_cost_inr": 4100.0,
        "stops": [5],
        "vehicle_id": "Truck-3",
        "timeline": []
    }
    candidates = [cand1, cand2, cand3]
    couplings = [(0, 1, 15.50)]  # coupling between cand1 and cand2
    num_stops = 5

    C_max, P_cov, P_veh = calculate_scale_derived_penalties(candidates)
    Q, const_offset = build_explicit_qubo_matrix(candidates, couplings, P_cov, P_veh, num_stops=num_stops)

    # Feasible selection: all 3 candidates active => covers stops {1, 2, 3, 4, 5} exactly once, 1 per vehicle
    x_feas = np.array([1, 1, 1], dtype=np.float64)
    qubo_energy = evaluate_qubo_energy(Q, const_offset, x_feas)

    # True cost = cand1 + cand2 + cand3 + coupling(0, 1)
    true_cost = cand1["total_route_cost_inr"] + cand2["total_route_cost_inr"] + cand3["total_route_cost_inr"] + 15.50

    assert abs(qubo_energy - true_cost) < 1e-6, f"QUBO energy {qubo_energy} != True cost {true_cost}"


def test_coverage_violation_penalty():
    """Verify that violating coverage constraint raises QUBO energy by at least P_cov."""
    cand1 = {"id": 0, "total_route_cost_inr": 2000.0, "stops": [1, 2], "vehicle_id": "V1"}
    cand2 = {"id": 1, "total_route_cost_inr": 3000.0, "stops": [3, 4], "vehicle_id": "V2"}
    candidates = [cand1, cand2]
    couplings = []
    num_stops = 4

    C_max, P_cov, P_veh = calculate_scale_derived_penalties(candidates)
    Q, const_offset = build_explicit_qubo_matrix(candidates, couplings, P_cov, P_veh, num_stops=num_stops)

    # Feasible selection: [1, 1]
    x_feas = np.array([1, 1], dtype=np.float64)
    energy_feas = evaluate_qubo_energy(Q, const_offset, x_feas)
    assert abs(energy_feas - 5000.0) < 1e-6

    # Infeasible: omit cand2 -> stops 3 and 4 are uncovered (2 missing stops)
    x_infeas = np.array([1, 0], dtype=np.float64)
    energy_infeas = evaluate_qubo_energy(Q, const_offset, x_infeas)

    # Missing 2 stops adds 2 * P_cov penalty, minus saving cand2's cost
    expected_diff = (2 * P_cov) - cand2["total_route_cost_inr"]
    actual_diff = energy_infeas - energy_feas

    assert abs(actual_diff - expected_diff) < 1e-6
    # Energy MUST strictly increase by at least (P_cov - C_max) > 0
    assert energy_infeas > energy_feas + P_cov - C_max


def test_qubo_brute_force_exact_equivalence_small_instance():
    """
    Exhaustively check all 2^M states on a 5-stop, 2-vehicle instance.
    Asserts that the global QUBO minimum is 100% feasible and matches the optimal feasible selection.
    """
    candidates = [
        {"id": 0, "stops": [1, 2, 3], "vehicle_id": "V1", "total_route_cost_inr": 1200.0},
        {"id": 1, "stops": [1, 2],    "vehicle_id": "V1", "total_route_cost_inr": 900.0},
        {"id": 2, "stops": [4, 5],    "vehicle_id": "V1", "total_route_cost_inr": 1100.0},
        {"id": 3, "stops": [4, 5],    "vehicle_id": "V2", "total_route_cost_inr": 1000.0},
        {"id": 4, "stops": [3, 4, 5], "vehicle_id": "V2", "total_route_cost_inr": 1400.0},
        {"id": 5, "stops": [1, 2, 3], "vehicle_id": "V2", "total_route_cost_inr": 1300.0},
    ]
    couplings = [(0, 3, 25.0)]  # coupling between cand 0 and cand 3
    num_stops = 5
    M = len(candidates)

    C_max, P_cov, P_veh = calculate_scale_derived_penalties(candidates)
    Q, const_offset = build_explicit_qubo_matrix(candidates, couplings, P_cov, P_veh, num_stops=num_stops)

    # Brute force search across all 2^6 = 64 states
    best_x = None
    min_energy = float("inf")

    for state_int in range(1 << M):
        x = np.array([(state_int >> k) & 1 for k in range(M)], dtype=np.float64)
        e = evaluate_qubo_energy(Q, const_offset, x)
        if e < min_energy:
            min_energy = e
            best_x = x

    # Expected optimal selection: Cand 0 (V1: [1, 2, 3]) + Cand 3 (V2: [4, 5])
    # Total cost = 1200 + 1000 + coupling(0, 3)=25.0 = 2225.0
    expected_optimal_x = np.array([1, 0, 0, 1, 0, 0], dtype=np.float64)
    expected_min_cost = 2225.0

    assert np.array_equal(best_x, expected_optimal_x), f"Global minimum state {best_x} != expected {expected_optimal_x}"
    assert abs(min_energy - expected_min_cost) < 1e-6, f"Min energy {min_energy} != expected cost {expected_min_cost}"

    # Verify feasibility of best state
    active = [i for i in range(M) if best_x[i] == 1]
    covered = []
    for i in active:
        covered.extend(candidates[i]["stops"])
    assert sorted(covered) == [1, 2, 3, 4, 5], "All stops must be covered exactly once"
    assert len(set(candidates[i]["vehicle_id"] for i in active)) == len(active), "No vehicle conflicts"

