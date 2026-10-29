"""
src/sim_model.py
================
Backwards-compatible bridge pointing to src/stochastic_model.py.
Ensures unified stochastic parameters and seed hygiene across all components.
"""

from src.stochastic_model import (
    LABEL_DATA_SEED,
    TRAIN_DATA_SEEDS,
    CV_SPLIT_SEED,
    INNER_CANDIDATE_SEED,
    MONTE_CARLO_EVAL_SEEDS,
    LEG_BREACH_TOLERANCE_MIN,
    sample_travel_time,
    sample_service_time,
    simulate_leg_realization,
    simulate_route_execution,
    compute_analytic_route_breach_probabilities
)

# Backwards compatibility alias for leg shock event
def simulate_leg_shock_event(
    distance_km: float,
    speed_kmh: float,
    congestion_index: float,
    loading_dock_variance: float,
    weather_risk: float,
    base_service_min: float,
    rng,
    shock_threshold_ratio: float = 1.25,
    tolerance_min: float = LEG_BREACH_TOLERANCE_MIN
) -> int:
    res = simulate_leg_realization(
        distance_km, speed_kmh, congestion_index, loading_dock_variance, weather_risk,
        base_service_min, rng, tolerance_min=tolerance_min
    )
    return res["is_breach"]
