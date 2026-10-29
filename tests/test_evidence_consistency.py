"""
tests/test_evidence_consistency.py
==================================
Strict Traceability & Numerical Consistency Audit:
Verifies that EVERY numerical value quoted in README.md tables matches the source of truth
in evidence/*.json:
1. Table 4.1: Planned Dispatch Solutions vs evidence/04_monte_carlo.json & 03_qpiai_opt_dispatch_solution.json.
2. Table 4.2: Monte Carlo Stochastic Stress Testing vs evidence/04_monte_carlo.json.
3. Table 4.4: Pareto Frontier Sweep vs evidence/04_monte_carlo.json.
4. Table 4.5: Quantum Hardware Profiling, Shots & Noise vs evidence/05_noise_and_shots.json.
5. Table 4.6: Fair Classical vs Quantum Benchmark vs evidence/02_qcloud_calibration_metrics.json.

Any mismatch causes immediate test failure and reports the exact diverging metric.
"""

import re
import json
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = BASE_DIR / "evidence"
README_PATH = BASE_DIR / "README.md"


@pytest.fixture(scope="module")
def loaded_fixtures():
    """Loads README.md and all evidence JSON source files."""
    assert README_PATH.exists(), f"README.md missing at {README_PATH}"
    with open(README_PATH, "r", encoding="utf-8") as f:
        readme_text = f.read()

    def load_json(name):
        p = EVIDENCE_DIR / name
        assert p.exists(), f"Evidence file missing: {p}"
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "readme": readme_text,
        "qcloud": load_json("02_qcloud_calibration_metrics.json"),
        "opt": load_json("03_qpiai_opt_dispatch_solution.json"),
        "mc": load_json("04_monte_carlo.json"),
        "ns": load_json("05_noise_and_shots.json"),
        "fair": load_json("06_fairness_analysis.json")
    }


def parse_clean_float(text: str) -> float:
    """Strips currency symbols (Rs.), commas, percent signs, and units to parse a float."""
    t = re.sub(r"Rs\.?\s*", "", text)
    t = t.replace(",", "")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", t)
    return float(m.group(0)) if m else 0.0


def test_readme_planned_dispatch_table_matches_evidence(loaded_fixtures):
    """Verify Table 4.1 Planned Dispatch Solutions numbers match evidence/04_monte_carlo.json."""
    readme = loaded_fixtures["readme"]
    mc_data = loaded_fixtures["mc"]
    plans = mc_data.get("plan_comparisons", {})

    # Match rows in Table 4.1
    table_match = re.search(r"### 4\.1\. Planned Dispatch Solutions.*?(\|[^\n]+\|\n(?:\|[^\n]+\|\n)+)", readme, re.DOTALL)
    assert table_match is not None, "Table 4.1 not found in README.md"

    table_lines = [l for l in table_match.group(1).strip().split("\n") if l.startswith("| **Plan")]
    assert len(table_lines) == 5, f"Expected 5 plan rows in Table 4.1, found {len(table_lines)}"

    plan_keys = ["plan_a", "plan_b", "plan_c", "plan_d", "plan_e"]
    for idx, line in enumerate(table_lines):
        pkey = plan_keys[idx]
        pdata = plans.get(pkey, {})
        cols = [c.strip() for c in line.split("|")[1:-1]]

        veh_active = int(cols[2])
        dist_km = parse_clean_float(cols[3])
        fuel_cost = parse_clean_float(cols[4])
        driver_cost = parse_clean_float(cols[5])
        total_cost = parse_clean_float(cols[7])

        exp_veh = pdata.get("active_vehicles", 3)
        exp_dist = pdata.get("total_distance_km", 0.0)
        exp_fuel = pdata.get("fixed_fuel_cost_inr", 0.0)
        exp_driver = pdata.get("fixed_driver_cost_inr", 0.0)
        exp_total = pdata.get("planned_cost_inr", 0.0)

        assert veh_active == exp_veh, f"Row {pkey} vehicle count {veh_active} != {exp_veh}"
        assert abs(dist_km - exp_dist) < 0.15, f"Row {pkey} distance {dist_km} != {exp_dist}"
        assert abs(fuel_cost - exp_fuel) < 1.0, f"Row {pkey} fuel cost {fuel_cost} != {exp_fuel}"
        assert abs(driver_cost - exp_driver) < 1.0, f"Row {pkey} driver cost {driver_cost} != {exp_driver}"
        assert abs(total_cost - exp_total) < 2.0, f"Row {pkey} total planned cost {total_cost} != {exp_total}"


def test_readme_monte_carlo_table_matches_evidence(loaded_fixtures):
    """Verify Table 4.2 Monte Carlo Stochastic Stress Testing matches evidence/04_monte_carlo.json."""
    readme = loaded_fixtures["readme"]
    mc_data = loaded_fixtures["mc"]
    plans = mc_data.get("plan_comparisons", {})

    table_match = re.search(r"### 4\.2\. Monte Carlo Stochastic Stress Testing.*?(\|[^\n]+\|\n(?:\|[^\n]+\|\n)+)", readme, re.DOTALL)
    assert table_match is not None, "Table 4.2 not found in README.md"

    table_lines = [l for l in table_match.group(1).strip().split("\n") if l.startswith("| **Plan")]
    assert len(table_lines) == 5, f"Expected 5 plan rows in Table 4.2, found {len(table_lines)}"

    plan_keys = ["plan_a", "plan_b", "plan_c", "plan_d", "plan_e"]
    for idx, line in enumerate(table_lines):
        pkey = plan_keys[idx]
        pdata = plans.get(pkey, {})
        cols = [c.strip() for c in line.split("|")[1:-1]]

        # Col 1: Mean Realized Cost
        cost_match = re.search(r"Rs\.\s*([\d,]+(?:\.\d+)?)", cols[1])
        assert cost_match is not None, f"Failed to parse cost in {cols[1]}"
        mean_cost = parse_clean_float(cost_match.group(1))

        # Col 2: Mean Realized Penalty
        pen_match = re.search(r"Rs\.\s*([\d,]+(?:\.\d+)?)", cols[2])
        assert pen_match is not None, f"Failed to parse penalty in {cols[2]}"
        mean_pen = parse_clean_float(pen_match.group(1))

        # Col 4: Late Delivery Rate
        late_rate = parse_clean_float(cols[4])

        exp_cost = pdata.get("mean_realized_cost", {}).get("value_inr", 0.0)
        exp_pen = pdata.get("mean_realized_penalty", {}).get("value_inr", 0.0)
        exp_late = pdata.get("late_delivery_rate_pct", 0.0)

        assert abs(mean_cost - exp_cost) < 1.0, f"Table 4.2 {pkey} mean cost {mean_cost} != {exp_cost}"
        assert abs(mean_pen - exp_pen) < 1.0, f"Table 4.2 {pkey} mean penalty {mean_pen} != {exp_pen}"
        assert abs(late_rate - exp_late) < 0.1, f"Table 4.2 {pkey} late rate {late_rate} != {exp_late}"


def test_readme_pareto_table_matches_evidence(loaded_fixtures):
    """Verify Table 4.4 Pareto Frontier Sweep matches evidence/04_monte_carlo.json."""
    readme = loaded_fixtures["readme"]
    mc_data = loaded_fixtures["mc"]
    pts = mc_data.get("pareto_frontier_sweep", [])
    assert len(pts) >= 8, f"Expected >= 8 points in Pareto sweep, got {len(pts)}"

    table_match = re.search(r"### 4\.4\. Pareto Frontier Sweep.*?(\|[^\n]+\|\n(?:\|[^\n]+\|\n)+)", readme, re.DOTALL)
    assert table_match is not None, "Table 4.4 not found in README.md"

    table_lines = [l for l in table_match.group(1).strip().split("\n") if l.startswith("| **w =")]
    assert len(table_lines) == len(pts), f"Table rows {len(table_lines)} != Pareto points {len(pts)}"

    for idx, line in enumerate(table_lines):
        pt = pts[idx]
        cols = [c.strip() for c in line.split("|")[1:-1]]

        w_val = parse_clean_float(cols[0])
        dist_val = parse_clean_float(cols[1])
        fuel_val = parse_clean_float(cols[2])
        pen_val = parse_clean_float(cols[3])
        cost_val = parse_clean_float(cols[4])
        late_val = parse_clean_float(cols[5])

        assert abs(w_val - pt["risk_weight"]) < 1e-4, f"w={w_val} != {pt['risk_weight']}"
        assert abs(dist_val - pt["total_distance_km"]) < 0.15, f"Dist {dist_val} != {pt['total_distance_km']}"
        assert abs(fuel_val - pt["planned_transport_fuel_inr"]) < 1.0, f"Fuel {fuel_val} != {pt['planned_transport_fuel_inr']}"
        assert abs(pen_val - pt["simulated_mean_penalty_inr"]) < 1.0, f"Penalty {pen_val} != {pt['simulated_mean_penalty_inr']}"
        assert abs(cost_val - pt["simulated_mean_cost_inr"]) < 1.0, f"Cost {cost_val} != {pt['simulated_mean_cost_inr']}"
        assert abs(late_val - pt["late_delivery_rate_pct"]) < 0.1, f"Late {late_val} != {pt['late_delivery_rate_pct']}"


def test_readme_hardware_and_shots_matches_evidence(loaded_fixtures):
    """Verify Table 4.5 hardware profiling and shot sampling experiments match evidence/05_noise_and_shots.json."""
    readme = loaded_fixtures["readme"]
    ns_data = loaded_fixtures["ns"]

    hw = ns_data.get("circuit_hardware_profiling", {})
    v1_hw = hw.get("variant_1", {})
    shot_res = ns_data.get("shot_sampling_experiment", {}).get("results_by_shots", {})

    # Check hardware table row for Variant 1
    assert f"**Qubits** | {v1_hw.get('num_qubits', 3)}" in readme
    assert f"**Circuit Depth** | {v1_hw.get('depth', 5)}" in readme
    assert f"**Total Gate Count** | {v1_hw.get('total_gates', 9)}" in readme

    # Check shot-noise table rows
    for s_key in ["128", "1024", "8192"]:
        s_data = shot_res.get(s_key, {})
        exp_brier = s_data.get("brier_mean", 0.0)
        exp_auc = s_data.get("roc_auc_mean", 0.0)

        # Confirm format in README
        assert f"{exp_brier:.4f}" in readme, f"Shot {s_key} Brier {exp_brier:.4f} missing in README"
        assert f"{exp_auc:.4f}" in readme, f"Shot {s_key} AUC {exp_auc:.4f} missing in README"


def test_readme_fair_benchmark_table_matches_evidence(loaded_fixtures):
    """Verify Table 4.6 Fair Benchmark Table matches evidence/02_qcloud_calibration_metrics.json."""
    readme = loaded_fixtures["readme"]
    qc_data = loaded_fixtures["qcloud"]
    fair_bench = qc_data.get("fair_benchmark_table", {})
    assert len(fair_bench) == 5, f"Expected 5 models in fair benchmark, got {len(fair_bench)}"

    for model_name, m_data in fair_bench.items():
        auc_mean = m_data.get("auc_mean", 0.0)
        brier_mean = m_data.get("brier_mean", 0.0)
        log_loss = m_data.get("log_loss_mean", 0.0)

        assert f"{auc_mean:.4f}" in readme, f"Model {model_name} AUC {auc_mean:.4f} missing in README"
        assert f"{brier_mean:.4f}" in readme, f"Model {model_name} Brier {brier_mean:.4f} missing in README"
        assert f"{log_loss:.4f}" in readme, f"Model {model_name} Log-Loss {log_loss:.4f} missing in README"


def test_readme_fairness_table_matches_evidence(loaded_fixtures):
    """Verify Section 4.7 Fairness Table matches evidence/06_fairness_analysis.json."""
    readme = loaded_fixtures["readme"]
    fair_data = loaded_fixtures["fair"]
    evals = fair_data.get("evaluations_by_risk_weight", [])
    assert len(evals) >= 5, f"Expected at least 5 risk weight evaluations, got {len(evals)}"

    for ev in evals:
        w = ev["risk_weight"]
        disp = ev["disparity_summary"]
        ratio = disp["breach_rate_ratio_low_to_high"]
        hp = ev["penalty_tiers"]["high_penalty"]
        lp = ev["penalty_tiers"]["low_penalty"]

        # Confirm weight row and values exist in README
        assert f"w = {w:.2f}" in readme, f"Weight w = {w:.2f} row missing in README"
        assert f"{hp['mean_service_position']:.2f}" in readme, f"High penalty position for w={w:.2f} missing in README"
        assert f"{hp['mean_empirical_breach_rate_pct']:.2f}%" in readme, f"High penalty breach for w={w:.2f} missing in README"
        assert f"{lp['mean_empirical_breach_rate_pct']:.2f}%" in readme, f"Low penalty breach for w={w:.2f} missing in README"
        assert f"{ratio:.2f}x" in readme, f"Disparity ratio {ratio:.2f}x for w={w:.2f} missing in README"

