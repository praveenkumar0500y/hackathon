"""
tests/test_optimality.py
========================
Small-Instance Optimality Verification:
On a small-scale VRPTW instance (<=6 stops, 2-3 vehicles), compares:
1. Method A: Exhaustive Brute-Force enumeration over all 2^M states in {0, 1}^M.
2. Method B: HiGHS Exact Integer Linear Programming reference (Branch-and-Cut).
3. Method C: QpiAI Opt / Simulated Annealing QUBO solver.

Verifies:
- Brute Force minimum energy exactly equals HiGHS certified mathematical optimum.
- QUBO simulated annealer achieves 0.00% optimality gap.
- All 3 methods produce 100% identical stop coverage and active vehicle selection.
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
    identify_cascading_delay_couplings,
    calculate_scale_derived_penalties,
    build_explicit_qubo_matrix,
    evaluate_qubo_energy,
    solve_qubo_simulated_annealing,
    solve_exact_reference_milp
)


def test_small_instance_optimality_comparison():
    """
    5 stops (1..5), 3 vehicles (EcoVan, CityTruck, HeavyHauler).
    Compares Brute Force, HiGHS MILP, and QUBO Annealer.
    """
    num_stops = 5
    small_stops = {0: DEPOT, **{s["id"]: s for s in STOPS if 1 <= s["id"] <= num_stops}}
    small_vehs = {
        "V1_EcoVan": VEHICLES["V1_EcoVan"],
        "V2_CityTruck": VEHICLES["V2_CityTruck"],
        "V3_HeavyHauler": VEHICLES["V3_HeavyHauler"]
    }

    # Generate candidate pool with max 2 stops per vehicle
    cands, _ = generate_candidate_route_pool(
        small_stops, EDGES, small_vehs, risk_weight=1.0, max_stops_per_route=2
    )

    # Pick 5 candidate routes per vehicle => M = 15 candidates
    selected_cands = []
    for vid in small_vehs:
        vcands = [c for c in cands if c["vehicle_id"] == vid][:5]
        selected_cands.extend(vcands)

    for idx, c in enumerate(selected_cands):
        c["id"] = idx

    M = len(selected_cands)
    assert M == 15, f"Expected 15 candidates, got {M}"

    # Identify quadratic cascading delay couplings
    couplings = identify_cascading_delay_couplings(selected_cands, small_stops, gamma_cascade=300.0)

    # 1. Method B: HiGHS Exact Reference MILP
    sol_highs = solve_exact_reference_milp(selected_cands, couplings, num_stops=num_stops)
    assert sol_highs["status"] == "OPTIMAL", "HiGHS must solve small instance to optimality"
    highs_opt_cost = sol_highs["objective"]
    assert highs_opt_cost > 0.0

    # 2. Build Explicit QUBO Matrix with Scale-Derived Penalties
    C_max, P_cov, P_veh = calculate_scale_derived_penalties(selected_cands)
    Q, const_offset = build_explicit_qubo_matrix(
        selected_cands, couplings, P_cov, P_veh, num_stops=num_stops
    )

    # 3. Method A: Exhaustive Brute-Force Search over all 2^M states
    min_energy_bf = float("inf")
    best_x_bf = None
    for state_int in range(1 << M):
        x = np.array([(state_int >> k) & 1 for k in range(M)], dtype=np.float64)
        e = evaluate_qubo_energy(Q, const_offset, x)
        if e < min_energy_bf:
            min_energy_bf = e
            best_x_bf = x

    # 4. Method C: QpiAI Simulated Annealing Solver
    sol_anneal = solve_qubo_simulated_annealing(
        Q=Q,
        const_offset=const_offset,
        candidates=selected_cands,
        couplings=couplings,
        num_stops=num_stops,
        num_restarts=16,
        num_sweeps=2500,
        seed=42
    )
    assert sol_anneal["is_feasible"], "Annealer solution must be feasible"
    anneal_cost = sol_anneal["true_cost"]

    # --- THREE-WAY OPTIMALITY & BEHAVIORAL ASSERTIONS ---

    # Assert 1: Brute Force minimum energy matches HiGHS objective exactly
    assert abs(min_energy_bf - highs_opt_cost) < 1e-4, (
        f"QUBO Brute Force min energy {min_energy_bf} != HiGHS MILP optimum {highs_opt_cost}"
    )

    # Assert 2: Annealer achieves 0.00% gap vs HiGHS
    optimality_gap = abs(anneal_cost - highs_opt_cost) / highs_opt_cost * 100.0
    assert optimality_gap < 1e-4, f"Annealer optimality gap {optimality_gap:.4f}% > 0.00%"

    # Assert 3: Feasibility of Brute-Force global optimum
    active_indices_bf = [i for i in range(M) if best_x_bf[i] == 1]
    covered_bf = []
    used_vehs_bf = []
    for i in active_indices_bf:
        covered_bf.extend(selected_cands[i]["stops"])
        used_vehs_bf.append(selected_cands[i]["vehicle_id"])

    assert sorted(covered_bf) == list(range(1, num_stops + 1)), (
        f"Brute force optimum must cover stops 1..{num_stops} exactly once"
    )
    assert len(used_vehs_bf) == len(set(used_vehs_bf)), "Brute force optimum has vehicle conflict"

    # Assert 4: HiGHS and Brute-Force select identical routes
    active_indices_highs = sorted([c["id"] for c in sol_highs["selected_routes"]])
    assert sorted(active_indices_bf) == active_indices_highs, (
        f"Brute force indices {active_indices_bf} != HiGHS indices {active_indices_highs}"
    )

    # Assert 5: Annealer selected routes match HiGHS
    active_indices_anneal = sorted(sol_anneal["selected_indices"])
    assert active_indices_anneal == active_indices_highs, (
        f"Annealer indices {active_indices_anneal} != HiGHS indices {active_indices_highs}"
    )
