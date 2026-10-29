"""
tests/test_fairness_and_security.py
===================================
Unit tests for Fairness, Security, and Failure Handling:
1. Fairness Evidence: Validates schema, multi-tier evaluations (penalty, distance, demand),
   disparity ratios, and concrete mitigations in evidence/06_fairness_analysis.json.
2. CORS Security: Asserts CORS is restricted to localhost origins, never wildcard "*".
3. Input Validation: Asserts Pydantic boundaries on traffic_severity, service_var, scenarios, and solver_type.
4. Failure Handling: Asserts that exceptions trigger is_stale_cached_result=True and degraded status.
5. Service Health: Asserts /api/health reports connectivity of OSRM, QCloud, and Opt.
"""

import os
import sys
import json
import pytest
from pathlib import Path
from pydantic import ValidationError

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.main import app, DispatchRequest, ALLOWED_ORIGINS
from fastapi.testclient import TestClient

EVIDENCE_DIR = BASE_DIR / "evidence"


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_fairness_evidence_structure_and_disparity():
    """Verify that evidence/06_fairness_analysis.json is populated with real evaluations and mitigations."""
    fair_file = EVIDENCE_DIR / "06_fairness_analysis.json"
    assert fair_file.exists(), f"Missing {fair_file}"

    with open(fair_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check top-level keys
    assert "metadata" in data
    assert "tiers_definition" in data
    assert "evaluations_by_risk_weight" in data
    assert "key_findings" in data
    assert "proposed_mitigations" in data

    evals = data["evaluations_by_risk_weight"]
    assert len(evals) >= 5, f"Expected at least 5 weights tested, got {len(evals)}"

    for ev in evals:
        assert "risk_weight" in ev
        assert "penalty_tiers" in ev
        assert "distance_tiers" in ev
        assert "demand_tiers" in ev
        assert "disparity_summary" in ev
        assert "stop_details" in ev
        assert len(ev["stop_details"]) == 12

    # Check mitigations
    mitigations = data["proposed_mitigations"]
    assert len(mitigations) == 4
    mitigation_ids = [m["mitigation_id"] for m in mitigations]
    assert "M1_MINIMUM_SLACK_FLOOR" in mitigation_ids
    assert "M2_EQUITY_REGULARIZED_QUBO" in mitigation_ids
    assert "M3_POSITION_CAP_FOR_LOW_TIER" in mitigation_ids
    assert "M4_DYNAMIC_SLA_REBATE_CREDITS" in mitigation_ids


def test_security_cors_origins_restricted():
    """Verify that CORS origins are restricted to localhost/127.0.0.1 by default and never wildcard."""
    assert "*" not in ALLOWED_ORIGINS, "Wildcard CORS origin '*' is strictly prohibited"
    for origin in ALLOWED_ORIGINS:
        assert "localhost" in origin or "127.0.0.1" in origin, f"Origin {origin} must be localhost"


def test_security_input_validation_bounds():
    """Verify that DispatchRequest enforces strict boundary validation and rejects invalid payloads."""
    # Valid request
    req = DispatchRequest(traffic_severity=1.2, service_var=1.0, solver_type="qubo", risk_aware=True, risk_weight=1.5, scenarios=250)
    assert req.traffic_severity == 1.2

    # Out-of-bounds traffic severity (< 0.2)
    with pytest.raises(ValidationError):
        DispatchRequest(traffic_severity=0.05)

    # Out-of-bounds traffic severity (> 3.0)
    with pytest.raises(ValidationError):
        DispatchRequest(traffic_severity=5.0)

    # Invalid solver type
    with pytest.raises(ValidationError):
        DispatchRequest(solver_type="arbitrary_untrusted_solver")

    # Out-of-bounds scenarios (< 50)
    with pytest.raises(ValidationError):
        DispatchRequest(scenarios=10)


def test_health_endpoint_reports_status(client):
    """Verify /api/health reports connectivity and configuration of OSRM, QCloud, and Opt."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "services" in data
    assert "osrm_routing" in data["services"]
    assert "qpiai_quantum" in data["services"]
    assert "qpiai_opt" in data["services"]


def test_fairness_api_endpoint(client):
    """Verify /api/fairness exposes the full multi-tier fairness data."""
    resp = client.get("/api/fairness")
    assert resp.status_code == 200
    data = resp.json()
    assert "evaluations_by_risk_weight" in data
    assert "proposed_mitigations" in data


def test_failure_handling_flags_stale_cached_result(monkeypatch, client):
    """Verify that backend exceptions trigger is_stale_cached_result=True and degraded status."""
    import app.main as main_module

    # Force run_fleet_dispatch to raise an exception simulating unreachable solver or OSRM
    def mock_broken_dispatch(**kwargs):
        raise ConnectionError("QpiAI Platform unreachable (503 Service Unavailable)")

    monkeypatch.setattr(main_module, "run_fleet_dispatch", mock_broken_dispatch)

    payload = {
        "traffic_severity": 1.0,
        "service_var": 1.0,
        "solver_type": "qubo",
        "risk_aware": True,
        "risk_weight": 1.0,
        "scenarios": 200
    }
    resp = client.post("/api/dispatch", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data.get("is_stale_cached_result") is True, "Failed dispatch must flag is_stale_cached_result=True"
    assert data.get("status") == "degraded_stale_cached_fallback"
    assert "ConnectionError" in data.get("summary", {}).get("error_note", "")
    assert data.get("service_health", {}).get("status") == "degraded"
