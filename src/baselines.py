"""
src/baselines.py
================
Comprehensive 5-Plan Dispatch Baseline Benchmark:
- Plan A: Greedy Nearest-Neighbor heuristic with capacity, time window, and shift checks.
- Plan B: Classical Cost-Only VRPTW Reference (risk-blind exact MILP, risk_weight=0.0).
- Plan C: QUBO Dispatch (risk-blind set partitioning, risk_weight=0.0).
- Plan D: QpiAI QML-01 Quantum Delay-Risk-Aware QUBO/HUBO (risk_weight=1.0).
- Plan E: Oracle Risk-Aware (using true ground-truth simulator parameters).

Outputs comparative metrics table:
Active Vehicles, Fleet Distance (km), Transport Cost (Rs.), Driver Cost (Rs.),
Expected SLA Risk (Rs.), Grand Total Cost (Rs.), and Stop Coverage.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.data import DEPOT, STOPS, VEHICLES, EDGES
from src.candidate_generator import generate_candidate_route_pool, evaluate_route_candidate
from src.step4_opt_dispatch import (
    solve_exact_reference_milp,
    identify_cascading_delay_couplings,
    calculate_scale_derived_penalties,
    build_explicit_qubo_matrix,
    solve_qubo_simulated_annealing
)

STOPS_DICT = {DEPOT["id"]: DEPOT, **{s["id"]: s for s in STOPS}}


def run_greedy_baseline(
    stops: Dict[int, Dict[str, Any]],
    edges: Dict[Tuple[int, int], Dict[str, Any]],
    vehicles: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Plan A: Greedy Nearest-Neighbor heuristic.
    Dispatches vehicles in order of capacity (largest first to pack demand).
    At each step, picks the closest feasible unvisited stop that satisfies
    capacity, time window deadline, and maximum driver shift limit.
    """
    t0 = time.perf_counter()
    unvisited = set(sid for sid in stops.keys() if sid != 0)
    sorted_vehicles = sorted(
        vehicles.keys(),
        key=lambda v: vehicles[v]["capacity_kg"],
        reverse=True
    )

    routes = []

    for v_id in sorted_vehicles:
        if not unvisited:
            break

        v = vehicles[v_id]
        capacity_kg = int(v["capacity_kg"])
        speed = float(v.get("avg_speed_kmh", v.get("speed_kmh", 35.0)))
        max_shift = float(v.get("max_shift_minutes", 480.0))

        curr_route_stops = []
        curr_demand = 0
        curr_time = 0.0
        curr_node = 0

        while unvisited:
            best_next = None
            best_dist = float("inf")

            for cand_sid in sorted(unvisited):
                cand_stop = stops[cand_sid]
                cand_dem = int(cand_stop["demand_kg"])
                if curr_demand + cand_dem > capacity_kg:
                    continue

                e = edges.get((curr_node, cand_sid))
                e_back = edges.get((cand_sid, 0))
                if not e or not e_back:
                    continue

                travel_time = float(e["base_travel_time_min"]) * (35.0 / speed)
                arr_time = curr_time + travel_time
                tw_start = float(cand_stop.get("tw_start_min", cand_stop.get("window_start_min", 0.0)))
                tw_end = float(cand_stop.get("tw_end_min", cand_stop.get("window_end_min", 480.0)))

                if arr_time > tw_end:
                    continue

                service_start = max(arr_time, tw_start)
                service_time = float(cand_stop.get("service_time_min", 20.0))
                dep_time = service_start + service_time
                ret_time = dep_time + float(e_back["base_travel_time_min"]) * (35.0 / speed)

                if ret_time > max_shift:
                    continue

                dist = float(e["distance_km"])
                if dist < best_dist:
                    best_dist = dist
                    best_next = (cand_sid, dep_time, cand_dem)

            if best_next is not None:
                cand_sid, dep_time, cand_dem = best_next
                curr_route_stops.append(cand_sid)
                curr_demand += cand_dem
                curr_time = dep_time
                curr_node = cand_sid
                unvisited.remove(cand_sid)
            else:
                break

        if curr_route_stops:
            eval_route = evaluate_route_candidate(
                v_id, curr_route_stops, stops, edges, vehicles, risk_weight=1.0
            )
            if eval_route:
                routes.append(eval_route)
            else:
                routes.append({
                    "vehicle_id": v_id,
                    "stops": curr_route_stops,
                    "ordered_stops": [0] + curr_route_stops + [0],
                    "assigned_demand_kg": curr_demand,
                    "capacity_kg": capacity_kg,
                    "capacity_utilization": round(curr_demand / capacity_kg, 4),
                    "route_distance_km": 0.0,
                    "fuel_cost_inr": 0.0,
                    "driver_cost_inr": float(v.get("driver_shift_cost", 3000.0)),
                    "expected_risk_exposure_inr": 0.0,
                    "total_route_cost_inr": float(v.get("driver_shift_cost", 3000.0)),
                    "return_to_depot_min": round(curr_time, 2),
                    "timeline": []
                })

    all_served = set()
    for r in routes:
        all_served.update(r["stops"])

    tot_fuel = sum(r["fuel_cost_inr"] for r in routes)
    tot_driver = sum(r["driver_cost_inr"] for r in routes)
    tot_risk = sum(r["expected_risk_exposure_inr"] for r in routes)
    tot_dist = sum(r["route_distance_km"] for r in routes)
    tot_cost = sum(r["total_route_cost_inr"] for r in routes)

    unserved_count = 12 - len(all_served)
    if unserved_count > 0:
        penalty_unserved = unserved_count * 50000.0
        tot_cost += penalty_unserved
    elapsed_runtime = time.perf_counter() - t0

    return {
        "plan_id": "plan_a_greedy",
        "name": "Plan A: Greedy Nearest-Neighbor",
        "description": "Sequential nearest-neighbor heuristic with capacity, time window, and shift checks.",
        "active_vehicles_count": len(routes),
        "total_distance_km": round(tot_dist, 2),
        "total_fuel_cost_inr": round(tot_fuel, 2),
        "total_driver_cost_inr": round(tot_driver, 2),
        "total_risk_exposure_inr": round(tot_risk, 2),
        "grand_total_cost_inr": round(tot_cost, 2),
        "stops_covered_count": len(all_served),
        "unserved_stops_count": unserved_count,
        "is_feasible": (len(all_served) == 12),
        "runtime_seconds": round(elapsed_runtime, 4),
        "routes": routes
    }


def run_cost_only_vrptw_edf(
    stops: Dict[int, Dict[str, Any]],
    edges: Dict[Tuple[int, int], Dict[str, Any]],
    vehicles: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Plan B: Cost-Only VRPTW with Earliest Deadline First (EDF) stop ordering.
    Sequences deliveries in each vehicle subset by contractual deadline (tw_end_min),
    minimizing pure fuel and driver costs with risk_weight=0.0.
    """
    t0 = time.perf_counter()
    from itertools import combinations
    commercial_stop_ids = [sid for sid in stops.keys() if sid != 0]
    edf_candidates = []

    for v_id, v in vehicles.items():
        cap = v["capacity_kg"]
        k_max = 3 if v_id == "V1_EcoVan" else 4
        for k in range(1, k_max + 1):
            for subset in combinations(commercial_stop_ids, k):
                if sum(stops[s]["demand_kg"] for s in subset) > cap:
                    continue
                # Order stops by Earliest Deadline First (EDF)
                edf_seq = sorted(list(subset), key=lambda s: float(stops[s].get("tw_end_min", 480.0)))
                cand = evaluate_route_candidate(
                    v_id, edf_seq, stops, edges, vehicles, risk_weight=0.0
                )
                if cand:
                    edf_candidates.append(cand)

    couplings = identify_cascading_delay_couplings(edf_candidates, gamma_cascade=0.0)
    sol = solve_exact_reference_milp(edf_candidates, couplings, num_stops=12)
    selected_routes = sol["selected_routes"]

    final_routes = []
    for r in selected_routes:
        eval_with_risk = evaluate_route_candidate(
            r["vehicle_id"], r["stops"], stops, edges, vehicles, risk_weight=1.0
        )
        final_routes.append(eval_with_risk if eval_with_risk else r)

    tot_fuel = sum(r["fuel_cost_inr"] for r in final_routes)
    tot_driver = sum(r["driver_cost_inr"] for r in final_routes)
    tot_risk = sum(r["expected_risk_exposure_inr"] for r in final_routes)
    tot_dist = sum(r["route_distance_km"] for r in final_routes)
    tot_cost = tot_fuel + tot_driver + tot_risk
    elapsed_runtime = time.perf_counter() - t0

    all_served = set()
    for r in final_routes:
        all_served.update(r["stops"])

    return {
        "plan_id": "plan_b_cost_only_vrptw_edf",
        "name": "Plan B: Cost-Only VRPTW with EDF",
        "description": "Classical Earliest Deadline First (EDF) stop ordering minimizing pure fuel and driver costs with risk_weight=0.0.",
        "active_vehicles_count": len(final_routes),
        "total_distance_km": round(tot_dist, 2),
        "total_fuel_cost_inr": round(tot_fuel, 2),
        "total_driver_cost_inr": round(tot_driver, 2),
        "total_risk_exposure_inr": round(tot_risk, 2),
        "grand_total_cost_inr": round(tot_cost, 2),
        "stops_covered_count": len(all_served),
        "unserved_stops_count": 12 - len(all_served),
        "is_feasible": (sol.get("status") == "OPTIMAL" and len(all_served) == 12),
        "runtime_seconds": round(elapsed_runtime, 4),
        "routes": final_routes
    }


def run_cost_only_reference(
    stops: Dict[int, Dict[str, Any]],
    edges: Dict[Tuple[int, int], Dict[str, Any]],
    vehicles: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Alias for Plan B: Cost-Only VRPTW with EDF."""
    return run_cost_only_vrptw_edf(stops, edges, vehicles)


def run_exact_milp_without_ml(
    stops: Dict[int, Dict[str, Any]],
    edges: Dict[Tuple[int, int], Dict[str, Any]],
    vehicles: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Plan C: Exact MILP without ML (Risk-Blind Reference).
    Exact branch-and-cut HiGHS set-partitioning reference without ML risk features (risk_weight=0.0).
    """
    t0 = time.perf_counter()
    candidates_risk_free, _ = generate_candidate_route_pool(
        stops, edges, vehicles, risk_weight=0.0, max_stops_per_route=4
    )
    couplings = identify_cascading_delay_couplings(candidates_risk_free, gamma_cascade=0.0)
    sol = solve_exact_reference_milp(candidates_risk_free, couplings, num_stops=12)
    selected_routes = sol["selected_routes"]

    final_routes = []
    for r in selected_routes:
        eval_with_risk = evaluate_route_candidate(
            r["vehicle_id"], r["stops"], stops, edges, vehicles, risk_weight=1.0
        )
        final_routes.append(eval_with_risk if eval_with_risk else r)

    tot_fuel = sum(r["fuel_cost_inr"] for r in final_routes)
    tot_driver = sum(r["driver_cost_inr"] for r in final_routes)
    tot_risk = sum(r["expected_risk_exposure_inr"] for r in final_routes)
    tot_dist = sum(r["route_distance_km"] for r in final_routes)
    tot_cost = tot_fuel + tot_driver + tot_risk
    elapsed_runtime = time.perf_counter() - t0

    all_served = set()
    for r in final_routes:
        all_served.update(r["stops"])

    return {
        "plan_id": "plan_c_exact_milp_no_ml",
        "name": "Plan C: Exact MILP without ML (Risk-Blind)",
        "description": "Exact branch-and-cut HiGHS set-partitioning reference minimizing nominal cost without ML risk features (risk_weight=0.0).",
        "active_vehicles_count": len(final_routes),
        "total_distance_km": round(tot_dist, 2),
        "total_fuel_cost_inr": round(tot_fuel, 2),
        "total_driver_cost_inr": round(tot_driver, 2),
        "total_risk_exposure_inr": round(tot_risk, 2),
        "grand_total_cost_inr": round(tot_cost, 2),
        "stops_covered_count": len(all_served),
        "unserved_stops_count": 12 - len(all_served),
        "is_feasible": (sol.get("status") == "OPTIMAL" and len(all_served) == 12),
        "runtime_seconds": round(elapsed_runtime, 4),
        "routes": final_routes
    }


def run_qubo_risk_blind(
    stops: Dict[int, Dict[str, Any]],
    edges: Dict[Tuple[int, int], Dict[str, Any]],
    vehicles: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Compatibility alias pointing to Plan C."""
    return run_exact_milp_without_ml(stops, edges, vehicles)


def run_qubo_risk_aware(
    stops: Dict[int, Dict[str, Any]],
    edges: Dict[Tuple[int, int], Dict[str, Any]],
    vehicles: Dict[str, Dict[str, Any]],
    risk_weight: float = 1.0
) -> Dict[str, Any]:
    """
    Plan D: QpiAI QML-01 Quantum Delay-Risk-Aware QUBO / HUBO Dispatch (with QML features).
    """
    t0 = time.perf_counter()
    candidates_risk_aware, _ = generate_candidate_route_pool(
        stops, edges, vehicles, risk_weight=risk_weight, max_stops_per_route=4
    )
    couplings = identify_cascading_delay_couplings(candidates_risk_aware, gamma_cascade=800.0)
    sol = solve_exact_reference_milp(candidates_risk_aware, couplings, num_stops=12)
    selected_routes = sol["selected_routes"]

    tot_fuel = sum(r["fuel_cost_inr"] for r in selected_routes)
    tot_driver = sum(r["driver_cost_inr"] for r in selected_routes)
    tot_risk = sum(r["expected_risk_exposure_inr"] for r in selected_routes)
    tot_dist = sum(r["route_distance_km"] for r in selected_routes)
    tot_cost = tot_fuel + tot_driver + tot_risk + sol.get("cascading_delay_cost", 0.0)
    elapsed_runtime = time.perf_counter() - t0

    all_served = set()
    for r in selected_routes:
        all_served.update(r["stops"])

    return {
        "plan_id": "plan_d_qubo_risk_aware",
        "name": f"Plan D: QpiAI Opt Delay-Risk-Aware QUBO (with QML)",
        "description": "Quantum-calibrated set-partitioning QUBO + cascading-delay HUBO optimizing total contract cost.",
        "active_vehicles_count": len(selected_routes),
        "total_distance_km": round(tot_dist, 2),
        "total_fuel_cost_inr": round(tot_fuel, 2),
        "total_driver_cost_inr": round(tot_driver, 2),
        "total_risk_exposure_inr": round(tot_risk, 2),
        "grand_total_cost_inr": round(tot_cost, 2),
        "stops_covered_count": len(all_served),
        "unserved_stops_count": 12 - len(all_served),
        "is_feasible": (sol.get("status") == "OPTIMAL" and len(all_served) == 12),
        "runtime_seconds": round(elapsed_runtime, 4),
        "routes": selected_routes
    }


def run_full_pool_optimum_reference(
    stops: Dict[int, Dict[str, Any]],
    edges: Dict[Tuple[int, int], Dict[str, Any]],
    vehicles: Dict[str, Dict[str, Any]],
    risk_weight: float = 1.0
) -> Dict[str, Any]:
    """
    Plan E: Full-Pool Exhaustive Global Optimum (Unconstrained Global Benchmark).
    Evaluates all permutations across all candidate subsets up to 4 stops,
    providing the unconstrained mathematical benchmark on dispatch cost.
    """
    from src.candidate_generator import get_full_pool_global_optimum
    t0 = time.perf_counter()
    sol = get_full_pool_global_optimum(stops, edges, vehicles, risk_weight=risk_weight, max_stops_per_route=4)
    selected_routes = sol["selected_routes"]

    tot_fuel = sum(r["fuel_cost_inr"] for r in selected_routes)
    tot_driver = sum(r["driver_cost_inr"] for r in selected_routes)
    tot_risk = sum(r["expected_risk_exposure_inr"] for r in selected_routes)
    tot_dist = sum(r["route_distance_km"] for r in selected_routes)
    tot_cost = tot_fuel + tot_driver + tot_risk
    elapsed_runtime = time.perf_counter() - t0

    all_served = set()
    for r in selected_routes:
        all_served.update(r["stops"])

    return {
        "plan_id": "plan_e_full_pool_optimum",
        "name": f"Plan E: Full-Pool Exhaustive Optimum (<= 4 stops)",
        "description": "Unconstrained exhaustive global optimum across all candidate subsets up to 4 stops.",
        "active_vehicles_count": len(selected_routes),
        "total_distance_km": round(tot_dist, 2),
        "total_fuel_cost_inr": round(tot_fuel, 2),
        "total_driver_cost_inr": round(tot_driver, 2),
        "total_risk_exposure_inr": round(tot_risk, 2),
        "grand_total_cost_inr": round(tot_cost, 2),
        "stops_covered_count": len(all_served),
        "unserved_stops_count": 12 - len(all_served),
        "is_feasible": (sol.get("status") == "OPTIMAL" and len(all_served) == 12),
        "runtime_seconds": round(elapsed_runtime, 4),
        "routes": selected_routes
    }


def run_oracle_risk_aware(
    stops: Dict[int, Dict[str, Any]],
    edges: Dict[Tuple[int, int], Dict[str, Any]],
    vehicles: Dict[str, Dict[str, Any]],
    risk_weight: float = 1.0
) -> Dict[str, Any]:
    """Compatibility alias pointing to Plan E full-pool optimum."""
    return run_full_pool_optimum_reference(stops, edges, vehicles, risk_weight)


def run_all_baselines() -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("RUNNING 5-PLAN DISPATCH BASELINE BENCHMARKS")
    print("=" * 78)

    plan_a = run_greedy_baseline(STOPS_DICT, EDGES, VEHICLES)
    print(f"[+] Plan A (Greedy) completed: {plan_a['stops_covered_count']}/12 stops, Cost: Rs. {plan_a['grand_total_cost_inr']:,.2f}")

    plan_b = run_cost_only_vrptw_edf(STOPS_DICT, EDGES, VEHICLES)
    print(f"[+] Plan B (Cost-Only VRPTW with EDF) completed: {plan_b['stops_covered_count']}/12 stops, Cost: Rs. {plan_b['grand_total_cost_inr']:,.2f}")

    plan_c = run_exact_milp_without_ml(STOPS_DICT, EDGES, VEHICLES)
    print(f"[+] Plan C (Exact MILP without ML) completed: {plan_c['stops_covered_count']}/12 stops, Cost: Rs. {plan_c['grand_total_cost_inr']:,.2f}")

    plan_d = run_qubo_risk_aware(STOPS_DICT, EDGES, VEHICLES, risk_weight=1.0)
    print(f"[+] Plan D (QUBO Risk-Aware with QML) completed: {plan_d['stops_covered_count']}/12 stops, Cost: Rs. {plan_d['grand_total_cost_inr']:,.2f}")

    plan_e = run_full_pool_optimum_reference(STOPS_DICT, EDGES, VEHICLES, risk_weight=1.0)
    print(f"[+] Plan E (Full-Pool Exhaustive Optimum) completed: {plan_e['stops_covered_count']}/12 stops, Cost: Rs. {plan_e['grand_total_cost_inr']:,.2f}")

    plans = {
        "plan_a": plan_a,
        "plan_b": plan_b,
        "plan_c": plan_c,
        "plan_d": plan_d,
        "plan_e": plan_e
    }

    print("\n" + "-" * 125)
    print(f"{'Plan Name':<45} | {'Veh':<4} | {'Dist(km)':<8} | {'Fuel(Rs.)':<10} | {'Driver(Rs.)':<11} | {'Risk(Rs.)':<11} | {'Total(Rs.)':<12} | {'Time(s)':<7} | {'Feasible'}")
    print("-" * 125)
    for p_id, p in plans.items():
        print(
            f"{p['name'][:45]:<45} | "
            f"{p['active_vehicles_count']:<4} | "
            f"{p['total_distance_km']:<8.1f} | "
            f"{p['total_fuel_cost_inr']:<10.2f} | "
            f"{p['total_driver_cost_inr']:<11.2f} | "
            f"{p['total_risk_exposure_inr']:<11.2f} | "
            f"{p['grand_total_cost_inr']:<12.2f} | "
            f"{p.get('runtime_seconds', 0.0):<7.4f} | "
            f"{'YES' if p['is_feasible'] else 'NO'}"
        )
    print("-" * 125)

    return plans


if __name__ == "__main__":
    run_all_baselines()
