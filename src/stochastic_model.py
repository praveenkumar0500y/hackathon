"""
src/stochastic_model.py
=======================
Unified Stochastic Leg & Route Simulation Model for QML-01 Contract Logistics.

Single source of truth for all stochastic physical processes:
1. Transit Travel Time:
   Sampled from a Lognormal distribution:
     T_ij ~ Lognormal(mu_ij, sigma_ij^2)
   where:
     nominal_time_min = max(0.5, (distance_km / max(speed_kmh, 10.0)) * 60.0 * traffic_multiplier)
     sigma_ij = 0.12 + 0.32 * clip(congestion_index, 0.05, 0.98) + 0.16 * clip(weather_risk, 0.05, 0.95)
     mu_ij = ln(nominal_time_min) - 0.5 * sigma_ij^2

2. Loading Dock Service Time:
   Sampled from a Gamma distribution:
     S_j ~ Gamma(shape=k_j, scale=theta_j)
   where:
     k_j = max(1.8, 5.0 - 3.5 * clip(loading_dock_variance, 0.05, 0.95))
     theta_j = base_service_min / k_j
     E[S_j] = base_service_min

3. Leg Breach Definition (Historical Observations in step2_data_gen.py):
   A leg observation counts as a breach if the simulated duration (travel + service)
   exceeds the planned baseline (base_travel + base_service) by more than LEG_BREACH_TOLERANCE_MIN:
     (simulated_travel + simulated_service) > (base_travel + base_service) + LEG_BREACH_TOLERANCE_MIN
   Default tolerance = 5.0 minutes (operational dispatch buffer).

4. Stop & Route SLA Breach Definition (Monte Carlo & Dispatch Planning):
   A delivery at stop s is late if:
     arrival_time_min > tw_end_min
   Contract SLA penalty is applied if late, plus a marginal late-minute surcharge.
   Expected exposure is:
     E[Penalty_s] = P(arrival_time > tw_end) * sla_penalty_inr
   ensuring planned expected penalty and simulated mean penalty have identical units (INR)
   and identical physical meaning.

Seed Hygiene:
- LABEL_DATA_SEED = 2024 (or range(100, 200) for observation generation)
- CV_SPLIT_SEED = 42 (5-fold stratified cross-validation)
- INNER_CANDIDATE_SEED = 777 (route candidate evaluation)
- MONTE_CARLO_EVAL_SEEDS = range(1000, 3000) (2,000 disjoint held-out scenarios)
"""

import math
from typing import Dict, Any, List, Tuple
import numpy as np

# Strict Seed Hygiene Constants
LABEL_DATA_SEED = 555
TRAIN_DATA_SEEDS = list(range(100, 200))          # 100 seeds for training data generation
CV_SPLIT_SEED = 42                                # 5-fold cross-validation split seed
INNER_CANDIDATE_SEED = 777                        # Seed for route candidate breach evaluation
MONTE_CARLO_EVAL_SEEDS = list(range(1000, 3000))  # 2000 disjoint seeds for Monte Carlo evaluation

# Documented operational buffer tolerance for leg delay breach (in minutes)
LEG_BREACH_TOLERANCE_MIN = 5.0


def sample_travel_time(
    distance_km: float,
    speed_kmh: float,
    congestion_index: float,
    weather_risk: float,
    rng: np.random.RandomState,
    traffic_multiplier: float = 1.0
) -> float:
    """
    Samples transit travel time from a Lognormal distribution.
    Mean is calibrated to nominal travel time * traffic_multiplier.
    Variance scales with congestion_index and weather_risk.
    """
    nominal_time_min = max(0.5, (distance_km / max(speed_kmh, 10.0)) * 60.0 * traffic_multiplier)
    sigma = 0.12 + 0.32 * float(np.clip(congestion_index, 0.05, 0.98)) + 0.16 * float(np.clip(weather_risk, 0.05, 0.95))
    mu = math.log(nominal_time_min) - 0.5 * (sigma ** 2)
    return float(rng.lognormal(mean=mu, sigma=sigma))


def sample_service_time(
    base_service_min: float,
    loading_dock_variance: float,
    rng: np.random.RandomState,
    service_var_multiplier: float = 1.0
) -> float:
    """
    Samples loading dock service time from a Gamma distribution.
    Mean equals base_service_min. Variance scales with loading_dock_variance * service_var_multiplier.
    """
    if base_service_min <= 0:
        return 0.0
    dock_var_eff = float(np.clip(loading_dock_variance * service_var_multiplier, 0.05, 0.98))
    k = max(1.8, 5.0 - 3.5 * dock_var_eff)
    theta = base_service_min / k
    return float(rng.gamma(shape=k, scale=theta))


def simulate_leg_realization(
    distance_km: float,
    speed_kmh: float,
    congestion_index: float,
    loading_dock_variance: float,
    weather_risk: float,
    base_service_min: float,
    rng: np.random.RandomState,
    tolerance_min: float = LEG_BREACH_TOLERANCE_MIN,
    traffic_multiplier: float = 1.0
) -> Dict[str, Any]:
    """
    Simulates a single directed leg realization under the unified stochastic model.
    Returns simulated travel time, service time, total duration, delay excess, and breach indicator.
    """
    nominal_travel = max(0.5, (distance_km / max(speed_kmh, 10.0)) * 60.0 * traffic_multiplier)
    planned_total = nominal_travel + base_service_min

    realized_travel = sample_travel_time(distance_km, speed_kmh, congestion_index, weather_risk, rng, traffic_multiplier)
    realized_service = sample_service_time(base_service_min, loading_dock_variance, rng)
    realized_total = realized_travel + realized_service

    delay_excess = realized_total - planned_total
    is_breach = 1 if delay_excess > tolerance_min else 0

    return {
        "planned_travel_min": round(nominal_travel, 2),
        "planned_total_min": round(planned_total, 2),
        "realized_travel_min": round(realized_travel, 2),
        "realized_service_min": round(realized_service, 2),
        "realized_total_min": round(realized_total, 2),
        "delay_excess_min": round(delay_excess, 2),
        "is_breach": is_breach
    }


def simulate_route_execution(
    ordered_stops: List[int],
    stops_dict: Dict[int, Dict[str, Any]],
    edges_dict: Dict[Tuple[int, int], Dict[str, Any]],
    vehicle_dict: Dict[str, Any],
    rng: np.random.RandomState,
    traffic_multiplier: float = 1.0,
    late_minute_surcharge_inr: float = 50.0,
    overtime_minute_surcharge_inr: float = 25.0,
    service_var_multiplier: float = 1.0
) -> Dict[str, Any]:
    """
    Simulates full route execution from depot (0), visiting stops, returning to depot (0).
    Tracks arrival times against contract time windows [tw_start, tw_end],
    waiting when early, dock service times, and applies contractual SLA penalties for delays.
    """
    speed = float(vehicle_dict.get("avg_speed_kmh", vehicle_dict.get("speed_kmh", 35.0)))
    max_shift = float(vehicle_dict.get("max_shift_minutes", 480.0))
    fuel_rate = float(vehicle_dict.get("fuel_cost_per_km", vehicle_dict.get("cost_per_km", 20.0)))
    driver_cost = float(vehicle_dict.get("driver_shift_cost", 3000.0))

    curr_time = 0.0
    curr_node = 0
    total_distance_km = 0.0

    stop_arrivals = {}
    stop_delays = {}
    stop_is_late = {}
    stop_penalties = {}

    route_stops = [sid for sid in ordered_stops if sid != 0]

    for sid in route_stops:
        s_info = stops_dict[sid]
        e = edges_dict.get((curr_node, sid))

        dist = float(e["distance_km"]) if e else 10.0
        total_distance_km += dist
        cong = float(e.get("congestion_index", 0.5)) if e else 0.5
        dock_var = float(e.get("loading_dock_variance", 0.3)) if e else 0.3
        weath = float(e.get("weather_risk", 0.2)) if e else 0.2
        base_service = float(s_info.get("service_time_min", 20.0))

        # Sample travel time
        trav = sample_travel_time(dist, speed, cong, weath, rng, traffic_multiplier)
        arr = curr_time + trav
        stop_arrivals[sid] = round(arr, 2)

        tw_start = float(s_info.get("tw_start_min", s_info.get("window_start_min", 0.0)))
        tw_end = float(s_info.get("tw_end_min", s_info.get("window_end_min", 480.0)))

        # Delay check against contractual time window
        is_late = arr > tw_end
        stop_is_late[sid] = is_late
        delay_min = max(0.0, arr - tw_end)
        stop_delays[sid] = round(delay_min, 2)

        if is_late:
            contract_penalty = float(s_info.get("sla_penalty_inr", s_info.get("sla_penalty_rs", 5000.0)))
            realized_penalty = contract_penalty + late_minute_surcharge_inr * min(delay_min, 120.0)
            stop_penalties[sid] = round(realized_penalty, 2)
        else:
            stop_penalties[sid] = 0.0

        # Waiting for window start if arrived early
        service_start = max(arr, tw_start)
        serv = sample_service_time(base_service, dock_var, rng, service_var_multiplier=service_var_multiplier)
        curr_time = service_start + serv
        curr_node = sid

    # Return leg to Depot (node 0)
    e_ret = edges_dict.get((curr_node, 0))
    ret_dist = float(e_ret["distance_km"]) if e_ret else 10.0
    total_distance_km += ret_dist
    ret_cong = float(e_ret.get("congestion_index", 0.5)) if e_ret else 0.5
    ret_weath = float(e_ret.get("weather_risk", 0.2)) if e_ret else 0.2
    ret_trav = sample_travel_time(ret_dist, speed, ret_cong, ret_weath, rng, traffic_multiplier)
    return_time = curr_time + ret_trav

    overtime_min = max(0.0, return_time - max_shift)
    overtime_penalty = overtime_min * overtime_minute_surcharge_inr

    total_realized_penalty = sum(stop_penalties.values()) + overtime_penalty
    fuel_cost = total_distance_km * fuel_rate
    total_realized_cost = fuel_cost + driver_cost + total_realized_penalty

    return {
        "ordered_stops": ordered_stops,
        "total_distance_km": round(total_distance_km, 2),
        "total_duration_min": round(return_time, 2),
        "fuel_cost_inr": round(fuel_cost, 2),
        "driver_cost_inr": round(driver_cost, 2),
        "overtime_minutes": round(overtime_min, 2),
        "overtime_penalty_inr": round(overtime_penalty, 2),
        "stop_arrivals": stop_arrivals,
        "stop_delays": stop_delays,
        "stop_is_late": stop_is_late,
        "stop_penalties": stop_penalties,
        "total_realized_penalty_inr": round(total_realized_penalty, 2),
        "total_realized_cost_inr": round(total_realized_cost, 2),
        "breaches_count": sum(1 for v in stop_is_late.values() if v)
    }


def compute_analytic_route_breach_probabilities(
    ordered_stops: List[int],
    stops_dict: Dict[int, Dict[str, Any]],
    edges_dict: Dict[Tuple[int, int], Dict[str, Any]],
    vehicle_dict: Dict[str, Any],
    traffic_multiplier: float = 1.0,
    p_edge_map: Dict[Tuple[int, int], float] = None,
    service_var_multiplier: float = 1.0
) -> Dict[str, Any]:
    """
    Computes per-stop breach probability along a route prefix analytically:
    - Propagates cumulative mean transit times and Gamma service times.
    - Propagates cumulative variance Var(T_l) + Var(S_l).
    - When p_edge_map is provided (from learned Quantum / Classical model), scales
      effective transit variance by (1.0 + 0.8 * p_ij).
    - Evaluates Z = (tw_end - arrival_mean) / sqrt(cumulative_variance).
    - Computes P(late) = 1 - Phi(Z), strictly capturing stop slack and corridor risk.
    - Expected exposure per stop is E[Penalty_s] = P(late) * sla_penalty_inr.
      This ensures planned expected penalty and simulated mean penalty have identical units and meaning.
    """
    speed = float(vehicle_dict.get("avg_speed_kmh", vehicle_dict.get("speed_kmh", 35.0)))
    route_stops = [sid for sid in ordered_stops if sid != 0]

    if not route_stops:
        return {
            "per_stop_breach_prob": {},
            "expected_sla_penalty_inr": 0.0,
            "mean_route_duration_min": 0.0
        }

    t_curr = 0.0
    var_curr = 0.0
    curr_node = 0

    per_stop_prob = {}
    per_stop_arrivals = {}
    tot_exp = 0.0

    for sid in route_stops:
        s_info = stops_dict[sid]
        e = edges_dict.get((curr_node, sid))

        dist = float(e["distance_km"]) if e else 10.0
        cong = float(e.get("congestion_index", 0.5)) if e else 0.5
        dock_var = float(e.get("loading_dock_variance", 0.3)) if e else 0.3
        weath = float(e.get("weather_risk", 0.2)) if e else 0.2
        base_serv = float(s_info.get("service_time_min", 20.0))

        # Nominal travel time and lognormal variance
        nom_trav = (dist / max(speed, 10.0)) * 60.0 * traffic_multiplier
        sig_trav = 0.12 + 0.32 * float(np.clip(cong, 0.05, 0.98)) + 0.16 * float(np.clip(weath, 0.05, 0.95))
        var_trav = (nom_trav ** 2) * (math.exp(sig_trav ** 2) - 1.0)

        # Apply learned quantum / classical model probability scaling if supplied
        if p_edge_map and (curr_node, sid) in p_edge_map:
            p_learned = float(p_edge_map[(curr_node, sid)])
            nom_trav *= (1.0 + 0.35 * p_learned)
            var_trav *= (1.0 + 1.20 * p_learned)

        # Arrival mean and cumulative variance
        arr_mean = t_curr + nom_trav
        var_arr = var_curr + var_trav
        sd_arr = math.sqrt(max(var_arr, 1e-4))

        tw_start = float(s_info.get("tw_start_min", s_info.get("window_start_min", 0.0)))
        tw_end = float(s_info.get("tw_end_min", s_info.get("window_end_min", 480.0)))

        # Standard normal Z-score based on slack
        slack = tw_end - arr_mean
        z = slack / sd_arr
        # P(arrival > tw_end) = 1 - Phi(z)
        p_breach = 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))
        p_breach = float(np.clip(p_breach, 0.0, 1.0))
        per_stop_prob[sid] = round(p_breach, 4)
        per_stop_arrivals[sid] = {
            "mean_arrival_min": round(arr_mean, 2),
            "sd_arrival_min": round(sd_arr, 2),
            "p10_arrival_min": round(max(0.0, arr_mean - 1.2816 * sd_arr), 2),
            "p90_arrival_min": round(arr_mean + 1.2816 * sd_arr, 2),
            "tw_start_min": tw_start,
            "tw_end_min": tw_end
        }

        penalty = float(s_info.get("sla_penalty_inr", s_info.get("sla_penalty_rs", 5000.0)))
        tot_exp += p_breach * penalty

        # Service duration and Gamma variance
        dock_var_eff = float(np.clip(dock_var * service_var_multiplier, 0.05, 0.98))
        k_shape = max(1.8, 5.0 - 3.5 * dock_var_eff)
        var_serv = (base_serv ** 2) / k_shape

        service_start = max(arr_mean, tw_start)
        t_curr = service_start + base_serv
        # Reset or carry variance (waiting at tw_start reduces arrival variance impact)
        if arr_mean < tw_start:
            var_curr = var_serv
        else:
            var_curr = var_arr + var_serv
        curr_node = sid

    # Return leg to depot
    e_ret = edges_dict.get((curr_node, 0))
    ret_dist = float(e_ret["distance_km"]) if e_ret else 10.0
    ret_trav = (ret_dist / max(speed, 10.0)) * 60.0 * traffic_multiplier
    tot_duration = t_curr + ret_trav

    return {
        "per_stop_breach_prob": per_stop_prob,
        "per_stop_arrivals": per_stop_arrivals,
        "expected_sla_penalty_inr": round(tot_exp, 2),
        "mean_route_duration_min": round(tot_duration, 2)
    }


def sample_travel_time_mismatched(
    distance_km: float,
    speed_kmh: float,
    congestion_index: float,
    weather_risk: float,
    rng: np.random.RandomState,
    traffic_multiplier: float = 1.0,
    is_arterial_shock: bool = False
) -> float:
    """
    Mismatched Generative Distribution for Robustness Stress-Testing:
    - Uses heavy-tailed Student-t transit delay (df=3.5) with Pareto extreme spikes.
    - Applies a +40% correlated rush-hour transit shock on arterial congestion legs.
    """
    nom = max(0.5, (distance_km / max(speed_kmh, 10.0)) * 60.0 * traffic_multiplier)
    if is_arterial_shock and congestion_index > 0.45:
        nom *= 1.40  # 40% correlated rush-hour corridor shock

    # Heavy-tailed Student-t perturbation (df=3.5)
    t_shock = float(rng.standard_t(df=3.5))
    scale = 0.22 + 0.35 * float(np.clip(congestion_index, 0.05, 0.98)) + 0.20 * float(np.clip(weather_risk, 0.05, 0.95))
    multiplier = math.exp(scale * np.clip(t_shock, -2.0, 3.5))
    return float(max(0.5, nom * multiplier))


def simulate_route_execution_mismatched(
    ordered_stops: List[int],
    stops_dict: Dict[int, Dict[str, Any]],
    edges_dict: Dict[Tuple[int, int], Dict[str, Any]],
    vehicle_dict: Dict[str, Any],
    rng: np.random.RandomState,
    traffic_multiplier: float = 1.0,
    late_minute_surcharge_inr: float = 50.0,
    overtime_minute_surcharge_inr: float = 25.0,
    arterial_shock_active: bool = True
) -> Dict[str, Any]:
    """
    Simulates route execution under the mismatched heavy-tailed generative distribution.
    """
    speed = float(vehicle_dict.get("avg_speed_kmh", 35.0))
    max_shift = float(vehicle_dict.get("max_shift_minutes", 480.0))
    fuel_rate = float(vehicle_dict.get("fuel_cost_per_km", 20.0))
    driver_cost = float(vehicle_dict.get("driver_shift_cost", 3000.0))

    curr_time = 0.0
    curr_node = 0
    total_dist = 0.0
    stop_arrivals = {}
    stop_delays = {}
    stop_is_late = {}
    stop_penalties = {}

    route_stops = [sid for sid in ordered_stops if sid != 0]

    for sid in route_stops:
        s_info = stops_dict[sid]
        e = edges_dict.get((curr_node, sid))
        dist = float(e["distance_km"]) if e else 10.0
        total_dist += dist
        cong = float(e.get("congestion_index", 0.5)) if e else 0.5
        dock_var = float(e.get("loading_dock_variance", 0.3)) if e else 0.3
        weath = float(e.get("weather_risk", 0.2)) if e else 0.2
        base_serv = float(s_info.get("service_time_min", 20.0))

        trav = sample_travel_time_mismatched(
            dist, speed, cong, weath, rng,
            traffic_multiplier=traffic_multiplier,
            is_arterial_shock=arterial_shock_active
        )
        arr = curr_time + trav
        stop_arrivals[sid] = round(arr, 2)

        tw_start = float(s_info.get("tw_start_min", 0.0))
        tw_end = float(s_info.get("tw_end_min", 480.0))

        is_late = arr > tw_end
        stop_is_late[sid] = is_late
        delay_min = max(0.0, arr - tw_end)
        stop_delays[sid] = round(delay_min, 2)

        if is_late:
            contract_penalty = float(s_info.get("sla_penalty_inr", 5000.0))
            realized_penalty = contract_penalty + late_minute_surcharge_inr * min(delay_min, 120.0)
            stop_penalties[sid] = round(realized_penalty, 2)
        else:
            stop_penalties[sid] = 0.0

        service_start = max(arr, tw_start)
        serv = sample_service_time(base_serv, dock_var, rng, service_var_multiplier=1.2)
        curr_time = service_start + serv
        curr_node = sid

    e_ret = edges_dict.get((curr_node, 0))
    ret_dist = float(e_ret["distance_km"]) if e_ret else 10.0
    total_dist += ret_dist
    ret_cong = float(e_ret.get("congestion_index", 0.5)) if e_ret else 0.5
    ret_weath = float(e_ret.get("weather_risk", 0.2)) if e_ret else 0.2
    ret_trav = sample_travel_time_mismatched(
        ret_dist, speed, ret_cong, ret_weath, rng,
        traffic_multiplier=traffic_multiplier,
        is_arterial_shock=arterial_shock_active
    )
    return_time = curr_time + ret_trav
    overtime_min = max(0.0, return_time - max_shift)
    overtime_penalty = overtime_min * overtime_minute_surcharge_inr
    total_penalty = sum(stop_penalties.values()) + overtime_penalty
    total_cost = total_dist * fuel_rate + driver_cost + total_penalty

    return {
        "ordered_stops": ordered_stops,
        "total_distance_km": round(total_dist, 2),
        "total_duration_min": round(return_time, 2),
        "fuel_cost_inr": round(total_dist * fuel_rate, 2),
        "driver_cost_inr": round(driver_cost, 2),
        "overtime_minutes": round(overtime_min, 2),
        "overtime_penalty_inr": round(overtime_penalty, 2),
        "stop_arrivals": stop_arrivals,
        "stop_delays": stop_delays,
        "stop_is_late": stop_is_late,
        "stop_penalties": stop_penalties,
        "total_realized_penalty_inr": round(total_penalty, 2),
        "total_realized_cost_inr": round(total_cost, 2),
        "breaches_count": sum(1 for v in stop_is_late.values() if v)
    }
