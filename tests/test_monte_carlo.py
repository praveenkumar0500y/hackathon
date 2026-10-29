"""
tests/test_monte_carlo.py
=========================
Behavioral Unit Tests for Monte Carlo Simulation Engine:
1. Deterministic reproducibility: same seed gives bit-for-bit identical outputs.
2. Seed sensitivity: different seeds give strictly different realization metrics.
3. Central Limit Theorem: increasing N scenarios reduces sample variance proportionally (Var ~ 1/N).
4. Physical limits: realized penalties, durations, and breach rates remain strictly non-negative.
"""

import pytest
import sys
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.data import DEPOT, STOPS, VEHICLES, EDGES
from src.candidate_generator import evaluate_route_candidate
from src.monte_carlo import simulate_single_plan

STOPS_DICT = {DEPOT["id"]: DEPOT, **{s["id"]: s for s in STOPS}}


def test_monte_carlo_reproducibility():
    """Verify that Monte Carlo simulations produce identical outputs given identical seed."""
    route1 = evaluate_route_candidate("V2_CityTruck", [2, 6, 5], STOPS_DICT, EDGES, VEHICLES)
    routes = [route1]

    sim_run_1 = simulate_single_plan(routes, STOPS_DICT, EDGES, VEHICLES, n_scenarios=100, seed=42)
    sim_run_2 = simulate_single_plan(routes, STOPS_DICT, EDGES, VEHICLES, n_scenarios=100, seed=42)

    assert sim_run_1["mean_realized_penalty_inr"] == sim_run_2["mean_realized_penalty_inr"]
    assert sim_run_1["mean_realized_cost_inr"] == sim_run_2["mean_realized_cost_inr"]
    assert sim_run_1["late_delivery_rate_pct"] == sim_run_2["late_delivery_rate_pct"]
    assert sim_run_1["p95_grand_total_cost_inr"] == sim_run_2["p95_grand_total_cost_inr"]
    assert sim_run_1["mean_route_duration_min"] == sim_run_2["mean_route_duration_min"]


def test_different_seeds_produce_different_outputs():
    """Verify that running with different seeds produces strictly different realization metrics."""
    route1 = evaluate_route_candidate("V2_CityTruck", [2, 6, 5], STOPS_DICT, EDGES, VEHICLES)
    routes = [route1]

    sim_seed_a = simulate_single_plan(routes, STOPS_DICT, EDGES, VEHICLES, n_scenarios=100, seed=42)
    sim_seed_b = simulate_single_plan(routes, STOPS_DICT, EDGES, VEHICLES, n_scenarios=100, seed=999)

    # Different seeds must produce different simulated outcomes (non-trivial random sampling)
    diff_penalty = abs(sim_seed_a["mean_realized_penalty_inr"] - sim_seed_b["mean_realized_penalty_inr"])
    diff_duration = abs(sim_seed_a["mean_route_duration_min"] - sim_seed_b["mean_route_duration_min"])
    diff_cost = abs(sim_seed_a["mean_realized_cost_inr"] - sim_seed_b["mean_realized_cost_inr"])

    assert diff_penalty > 0 or diff_duration > 0 or diff_cost > 0, (
        "Different random seeds must produce distinct stochastic outcomes"
    )


def test_monte_carlo_variance_scales_sensibly_with_scenarios():
    """
    Verify Central Limit Theorem behavior:
    The sample variance of the sample mean Var(X_bar_N) scales as sigma^2 / N.
    Evaluating across K independent repetitions at N_1=30 scenarios vs N_2=300 scenarios
    must exhibit a significant, sensible reduction in sample variance (Var_2 << Var_1).
    """
    route1 = evaluate_route_candidate("V2_CityTruck", [2, 6, 5], STOPS_DICT, EDGES, VEHICLES)
    routes = [route1]

    K = 25  # Number of independent batches
    durations_small_n = []
    for s in range(100, 100 + K):
        res = simulate_single_plan(routes, STOPS_DICT, EDGES, VEHICLES, n_scenarios=30, seed=s)
        durations_small_n.append(res["mean_route_duration_min"])

    durations_large_n = []
    for s in range(300, 300 + K):
        res = simulate_single_plan(routes, STOPS_DICT, EDGES, VEHICLES, n_scenarios=300, seed=s)
        durations_large_n.append(res["mean_route_duration_min"])

    var_small = float(np.var(durations_small_n, ddof=1))
    var_large = float(np.var(durations_large_n, ddof=1))

    # Variance with 300 scenarios must be strictly smaller than with 30 scenarios
    assert var_large < var_small, (
        f"Variance failed to decrease with higher scenario count: Var(N=300)={var_large:.4f} >= Var(N=30)={var_small:.4f}"
    )

    # Theoretical ratio is N_1 / N_2 = 30 / 300 = 0.10.
    # Allowing for finite sample variation across K=25 runs, ratio must be < 0.40.
    variance_ratio = var_large / var_small
    assert variance_ratio < 0.40, (
        f"Variance ratio {variance_ratio:.4f} exceeded expected CLT bound (< 0.40)"
    )


def test_physical_limits():
    """Verify that travel times and realized costs never fall below zero."""
    route1 = evaluate_route_candidate("V2_CityTruck", [2, 6], STOPS_DICT, EDGES, VEHICLES)
    routes = [route1]

    sim = simulate_single_plan(routes, STOPS_DICT, EDGES, VEHICLES, n_scenarios=50, seed=42)

    assert sim["mean_realized_penalty_inr"] >= 0.0
    assert sim["mean_route_duration_min"] > 0.0
    assert sim["late_delivery_rate_pct"] >= 0.0
    assert sim["late_delivery_rate_pct"] <= 100.0
