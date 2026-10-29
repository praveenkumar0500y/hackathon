"""
tests/test_cascading_coupling.py
================================
Unit tests for data-derived cascading delay risk couplings in QUBO formulation:
- Verifies extraction and validation of SLA breach probabilities.
- Verifies missing probability raises ValueError for risk-aware candidates.
- Verifies risk-neutral candidates (w=0.0) return zero coupling cost without error.
- Verifies gamma_cascade=0.0 disables couplings completely (returns empty list).
- Verifies same-vehicle candidates are excluded from concurrent interaction.
- Verifies distance and temporal decay thresholds.
- Verifies monotonic risk response (higher breach probability -> higher coupling cost).
- Verifies non-negativity of all coupling terms.
"""

import pytest
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.step4_opt_dispatch import (
    identify_cascading_delay_couplings,
    extract_stop_breach_probability
)

MOCK_STOPS = {
    0: {"id": 0, "name": "Depot", "lat": 12.9716, "lng": 77.5946, "sla_penalty_inr": 0.0},
    1: {"id": 1, "name": "Stop 1", "lat": 12.9720, "lng": 77.5950, "sla_penalty_inr": 5000.0},
    2: {"id": 2, "name": "Stop 2", "lat": 12.9725, "lng": 77.5955, "sla_penalty_inr": 6000.0},
    3: {"id": 3, "name": "Stop 3 Far", "lat": 13.1500, "lng": 77.7000, "sla_penalty_inr": 5000.0},
}


def test_extract_stop_breach_probability():
    """Verify SLA breach probability extraction and missing data validation."""
    item1 = {"stop_id": 1, "p_breach": 0.15}
    cand_aware = {"vehicle_id": "V1", "risk_weight": 1.0}
    assert extract_stop_breach_probability(item1, cand_aware) == 0.15

    item2 = {"stop_id": 1, "qcloud_sla_breach_prob": 0.22}
    assert extract_stop_breach_probability(item2, cand_aware) == 0.22

    item_missing = {"stop_id": 1}
    with pytest.raises(ValueError, match="Missing required SLA breach probability"):
        extract_stop_breach_probability(item_missing, cand_aware)

    cand_neutral = {"vehicle_id": "V1", "risk_weight": 0.0}
    assert extract_stop_breach_probability(item_missing, cand_neutral) == 0.0


def test_gamma_cascade_disables_couplings():
    """Verify gamma_cascade <= 0.0 disables cascading interactions completely."""
    cands = [
        {
            "vehicle_id": "V1",
            "risk_weight": 1.0,
            "timeline": [{"stop_id": 1, "arrival_time_min": 60.0, "p_breach": 0.20}]
        },
        {
            "vehicle_id": "V2",
            "risk_weight": 1.0,
            "timeline": [{"stop_id": 2, "arrival_time_min": 65.0, "p_breach": 0.25}]
        }
    ]

    assert identify_cascading_delay_couplings(cands, stops_dict=MOCK_STOPS, gamma_cascade=0.0) == []
    assert identify_cascading_delay_couplings(cands, stops_dict=MOCK_STOPS, gamma_cascade=-5.0) == []


def test_same_vehicle_exclusion():
    """Verify that routes for the same vehicle cannot form cascading concurrent interactions."""
    cands = [
        {
            "vehicle_id": "V1",
            "risk_weight": 1.0,
            "timeline": [{"stop_id": 1, "arrival_time_min": 60.0, "p_breach": 0.20}]
        },
        {
            "vehicle_id": "V1",
            "risk_weight": 1.0,
            "timeline": [{"stop_id": 2, "arrival_time_min": 65.0, "p_breach": 0.25}]
        }
    ]

    couplings = identify_cascading_delay_couplings(
        cands, stops_dict=MOCK_STOPS, gamma_cascade=800.0, min_coupling_cost=0.01
    )
    assert len(couplings) == 0, "Routes for the same vehicle should never have cascading couplings"


def test_spatial_and_temporal_decay():
    """Verify spatial and temporal distance bounds prune interactions."""
    cand_v1 = {
        "vehicle_id": "V1",
        "risk_weight": 1.0,
        "timeline": [{"stop_id": 1, "arrival_time_min": 60.0, "p_breach": 0.20}]
    }

    cand_v2_far = {
        "vehicle_id": "V2",
        "risk_weight": 1.0,
        "timeline": [{"stop_id": 3, "arrival_time_min": 65.0, "p_breach": 0.20}]
    }
    c_far = identify_cascading_delay_couplings(
        [cand_v1, cand_v2_far], stops_dict=MOCK_STOPS, gamma_cascade=800.0, min_coupling_cost=0.01
    )
    assert len(c_far) == 0, "Stops > 5km apart should have zero coupling"

    cand_v2_late = {
        "vehicle_id": "V2",
        "risk_weight": 1.0,
        "timeline": [{"stop_id": 2, "arrival_time_min": 120.0, "p_breach": 0.20}]
    }
    c_late = identify_cascading_delay_couplings(
        [cand_v1, cand_v2_late], stops_dict=MOCK_STOPS, gamma_cascade=800.0, min_coupling_cost=0.01
    )
    assert len(c_late) == 0, "Stops >= 45 min apart should have zero coupling"

    cand_v2_close = {
        "vehicle_id": "V2",
        "risk_weight": 1.0,
        "timeline": [{"stop_id": 2, "arrival_time_min": 65.0, "p_breach": 0.20}]
    }
    c_close = identify_cascading_delay_couplings(
        [cand_v1, cand_v2_close], stops_dict=MOCK_STOPS, gamma_cascade=800.0, min_coupling_cost=0.01
    )
    assert len(c_close) == 1, "Close stops must produce an interaction coupling"
    assert c_close[0][2] > 0.0, "Coupling cost must be strictly positive"


def test_monotonic_risk_scaling():
    """Verify higher breach probability strictly increases coupling penalty."""
    cand_v1_low = {
        "vehicle_id": "V1",
        "risk_weight": 1.0,
        "timeline": [{"stop_id": 1, "arrival_time_min": 60.0, "p_breach": 0.05}]
    }
    cand_v2_low = {
        "vehicle_id": "V2",
        "risk_weight": 1.0,
        "timeline": [{"stop_id": 2, "arrival_time_min": 65.0, "p_breach": 0.05}]
    }

    cand_v1_high = {
        "vehicle_id": "V1",
        "risk_weight": 1.0,
        "timeline": [{"stop_id": 1, "arrival_time_min": 60.0, "p_breach": 0.40}]
    }
    cand_v2_high = {
        "vehicle_id": "V2",
        "risk_weight": 1.0,
        "timeline": [{"stop_id": 2, "arrival_time_min": 65.0, "p_breach": 0.40}]
    }

    c_low = identify_cascading_delay_couplings(
        [cand_v1_low, cand_v2_low], stops_dict=MOCK_STOPS, gamma_cascade=800.0, min_coupling_cost=0.001
    )
    c_high = identify_cascading_delay_couplings(
        [cand_v1_high, cand_v2_high], stops_dict=MOCK_STOPS, gamma_cascade=800.0, min_coupling_cost=0.001
    )

    assert len(c_low) == 1 and len(c_high) == 1
    assert c_high[0][2] > c_low[0][2], "Higher breach probabilities must strictly increase coupling cost"
