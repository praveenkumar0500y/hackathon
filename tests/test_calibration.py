"""
tests/test_calibration.py
=========================
Behavioral Unit Tests for Quantum Delay-Risk Calibration on Held-Out Data:
- Generates stochastic realization data using strictly disjoint evaluation seeds.
- Verifies that held-out calibration error (MSE, correlation) stays below documented thresholds.
- Verifies that calibration assessment label is programmatically "WELL CALIBRATED".
- Verifies test sensitivity: corrupted or uncalibrated probabilities fail the test.
"""

import pytest
import sys
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.data import DEPOT, STOPS, VEHICLES, EDGES
from src.stochastic_model import (
    LABEL_DATA_SEED,
    TRAIN_DATA_SEEDS,
    CV_SPLIT_SEED,
    MONTE_CARLO_EVAL_SEEDS
)
from src.candidate_generator import evaluate_route_candidate
from src.monte_carlo import simulate_plan_crn, validate_quantum_calibration

STOPS_DICT = {DEPOT["id"]: DEPOT, **{s["id"]: s for s in STOPS}}

# Strictly disjoint evaluation seeds for testing calibration
CALIBRATION_TEST_SEEDS = list(range(5000, 5200))


def test_disjoint_seeds_used_for_calibration():
    """Verify that calibration test seeds do not overlap with any training, labeling, or MC seeds."""
    test_seed_set = set(CALIBRATION_TEST_SEEDS)
    assert {LABEL_DATA_SEED}.isdisjoint(test_seed_set), "Overlap with LABEL_DATA_SEED"
    assert set(TRAIN_DATA_SEEDS).isdisjoint(test_seed_set), "Overlap with TRAIN_DATA_SEEDS"
    assert {CV_SPLIT_SEED}.isdisjoint(test_seed_set), "Overlap with CV_SPLIT_SEED"
    assert set(MONTE_CARLO_EVAL_SEEDS).isdisjoint(test_seed_set), "Overlap with MONTE_CARLO_EVAL_SEEDS"


def test_held_out_quantum_calibration_below_threshold():
    """
    Simulates delivery routes on held-out seeds (5000..5199) and asserts that
    quantum probability predictions achieve calibration MSE <= 0.05 and Pearson r >= 0.60.
    """
    # Build representative routes matching Plan D
    r1 = evaluate_route_candidate("V1_EcoVan", [1, 8], STOPS_DICT, EDGES, VEHICLES, risk_weight=1.0)
    r2 = evaluate_route_candidate("V2_CityTruck", [7, 6, 11], STOPS_DICT, EDGES, VEHICLES, risk_weight=1.0)
    r3 = evaluate_route_candidate("V3_HeavyHauler", [2, 9, 3, 12, 4, 5, 10], STOPS_DICT, EDGES, VEHICLES, risk_weight=1.0)

    # Fallback to smaller routes if combined route exceeds shift
    if not r3:
        r3 = evaluate_route_candidate("V3_HeavyHauler", [2, 9, 3, 12], STOPS_DICT, EDGES, VEHICLES, risk_weight=1.0)
    routes = [r for r in [r1, r2, r3] if r is not None]
    assert len(routes) >= 2, "Must evaluate at least 2 routes"

    # Simulate empirical delivery execution across held-out seeds
    sim_results = simulate_plan_crn(
        routes, STOPS_DICT, EDGES, VEHICLES,
        eval_seeds=CALIBRATION_TEST_SEEDS,
        traffic_multiplier=1.0,
        service_var_multiplier=1.0
    )

    # Validate calibration against quantum predictions
    calib = validate_quantum_calibration(routes, sim_results)

    mse = calib["calibration_mse"]
    corr = calib["pearson_correlation"]
    assessment = calib["assessment"]

    # Documented threshold for WELL CALIBRATED: MSE <= 0.05 and Pearson r >= 0.60
    DOCUMENTED_MAX_MSE = 0.05
    DOCUMENTED_MIN_CORR = 0.60

    assert mse <= DOCUMENTED_MAX_MSE, (
        f"Held-out calibration MSE {mse:.6f} exceeded documented threshold ({DOCUMENTED_MAX_MSE})"
    )
    assert corr >= DOCUMENTED_MIN_CORR, (
        f"Held-out Pearson correlation {corr:.4f} fell below documented threshold ({DOCUMENTED_MIN_CORR})"
    )
    assert "CALIBRATED" in assessment and "UNCALIBRATED" not in assessment and "POORLY" not in assessment, (
        f"Expected calibrated assessment, got {assessment}"
    )


def test_uncalibrated_predictions_rejected():
    """Verify that uncalibrated or inverted predictions correctly trigger test failure."""
    mock_routes = [{
        "timeline": [
            {"stop_id": 1, "name": "Stop 1", "qcloud_sla_breach_prob": 0.01},
            {"stop_id": 2, "name": "Stop 2", "qcloud_sla_breach_prob": 0.02},
            {"stop_id": 3, "name": "Stop 3", "qcloud_sla_breach_prob": 0.95}
        ]
    }]
    # Inverted empirical rates
    mock_sim_inverted = {"stop_empirical_breach_rates": {1: 0.95, 2: 0.90, 3: 0.01}}
    calib_inverted = validate_quantum_calibration(mock_routes, mock_sim_inverted)

    # Inverted predictions must fail calibration threshold
    assert "UNCALIBRATED" in calib_inverted["assessment"] or "POORLY" in calib_inverted["assessment"], (
        f"Inverted predictions must fail calibration threshold, got {calib_inverted['assessment']}"
    )
    assert calib_inverted["calibration_mse"] > 0.05, "Inverted predictions must have high MSE"
