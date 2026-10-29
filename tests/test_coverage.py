"""
tests/test_coverage.py
======================
Unit tests for stop coverage and vehicle uniqueness in the dispatch solution:
- All 12 commercial stops must be served exactly once (coverage rate = 1.0).
- Zero duplicate stop assignments across vehicles.
- Each vehicle operates at most one candidate route.
"""

import pytest
import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

EVIDENCE_DIR = BASE_DIR / "evidence"
SOLUTION_JSON = EVIDENCE_DIR / "03_qpiai_opt_dispatch_solution.json"


def test_solution_coverage_and_uniqueness():
    """Verify that evidence/03_qpiai_opt_dispatch_solution.json satisfies exact set partitioning."""
    assert SOLUTION_JSON.exists(), f"Evidence solution file missing: {SOLUTION_JSON}"
    
    with open(SOLUTION_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    routes = data.get("routes", [])
    assert len(routes) > 0, "Routes list cannot be empty"
    
    covered_stops = []
    used_vehicles = []
    
    for r in routes:
        vid = r["vehicle_id"]
        used_vehicles.append(vid)
        for sid in r["stops"]:
            covered_stops.append(sid)
            
    # 1. Exactly 12 commercial stops covered
    unique_stops = set(covered_stops)
    expected_stops = set(range(1, 13))
    assert unique_stops == expected_stops, f"Expected stops {expected_stops}, got {unique_stops}"
    
    # 2. Zero duplicate visits
    assert len(covered_stops) == 12, f"Duplicate stops detected! Total stops visited: {len(covered_stops)}"
    
    # 3. Vehicle conflict: at most one route per vehicle
    assert len(used_vehicles) == len(set(used_vehicles)), f"Duplicate vehicle assigned: {used_vehicles}"
    assert len(used_vehicles) <= 4, f"Cannot exceed fleet size of 4: {len(used_vehicles)}"
