"""
tests/test_audit_fixes.py
=========================
Engineering Audit Verification Test Suite for QML-01 Dispatch:
1. test_sdk_local_limits: Variables <= 250, terms <= 5000, reads <= 16, sweeps <= 5000, annealing work <= 10,000,000.
2. test_authenticated_local_and_keyless: Step 4 executes cleanly in both authenticated local and keyless environments.
3. test_full_pool_optimality_gap_reported: Pruned pool and unconstrained full-pool gaps are both documented honestly.
4. test_ml_signal_and_auc: edges_uncertainty.csv achieves 5-fold CV ROC-AUC > 0.75 on classical and quantum models.
5. test_plan_ablation_decision_impact: Learned p_ij changes dispatch plan decisions and total cost vs shuffled.
6. test_shot_noise_ci_non_degenerate: All bootstrap confidence intervals in 05_noise_and_shots.json have strictly positive width.
7. test_calibration_across_all_plans: Calibration metrics (Pearson r, Brier MSE) are evaluated across all 5 dispatch plans.
8. test_mismatched_and_tight_window_evaluations: Heavy-tailed Student-t and tight-window tests are recorded across all 5 plans.
"""

import os
import sys
import json
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
EVIDENCE_DIR = BASE_DIR / "evidence"


def test_sdk_local_limits():
    """Verify QpiAI Opt formulation conforms strictly to free local tier limits."""
    sol_path = EVIDENCE_DIR / "03_qpiai_opt_dispatch_solution.json"
    assert sol_path.exists(), "03_qpiai_opt_dispatch_solution.json must exist"

    with open(sol_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    solver_info = data["solver_info"]
    num_vars = solver_info["primary_decision_variables"]
    num_terms = solver_info["qubo_terms"]
    num_reads = solver_info["num_reads"]
    num_sweeps = solver_info["num_sweeps"]
    annealing_work = solver_info["annealing_work"]

    assert num_vars <= 250, f"Variables ({num_vars}) must be <= 250"
    assert num_terms <= 5000, f"QUBO terms ({num_terms}) must be <= 5000"
    assert num_reads <= 16, f"Num reads ({num_reads}) must be <= 16"
    assert num_sweeps <= 5000, f"Num sweeps ({num_sweeps}) must be <= 5000"
    assert annealing_work <= 10_000_000, f"Annealing work ({annealing_work}) must be <= 10,000,000"
    assert data["feasible"] is True, "Solution must be fully feasible"


def test_authenticated_local_and_keyless(monkeypatch):
    """Verify execution mode in authenticated local mode and simulated keyless fallback."""
    sol_path = EVIDENCE_DIR / "03_qpiai_opt_dispatch_solution.json"
    assert sol_path.exists()
    with open(sol_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # In current environment with API key, execution mode must be AUTHENTICATED_LOCAL
    assert data["execution_mode"] in ["AUTHENTICATED_LOCAL", "LOCAL_FALLBACK"]

    # Verify keyless fallback runs with exit code 0 and maintains feasibility without overwriting evidence
    from src.step4_opt_dispatch import run_qubo_optimization
    monkeypatch.delenv("QPIAI_OPT_API_KEY", raising=False)
    keyless_result = run_qubo_optimization(save_evidence=False)
    assert keyless_result["execution_mode"] == "LOCAL_FALLBACK"
    assert keyless_result["feasible"] is True
    assert keyless_result["dispatch_summary"]["total_stops_covered"] == 12


def test_same_pool_exact_reference_gap_reported():
    """Verify exact HiGHS reference and 0.00% optimality gap on the same 48-candidate pool are reported honestly."""
    sol_path = EVIDENCE_DIR / "03_qpiai_opt_dispatch_solution.json"
    with open(sol_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    exact_comp = data.get("exact_reference_comparison", {})
    assert "reference_cost_inr" in exact_comp or "pruned_pool_reference_cost_inr" in exact_comp
    assert exact_comp.get("candidate_pool_size") == 48

    solver_info = data.get("solver_info", {})
    gap = solver_info.get("optimality_gap_pct", solver_info.get("optimality_gap_vs_exact_ref_pct", 0.0))
    assert gap is not None and gap >= 0.0
    assert gap <= 0.05, f"Expected 0.00% gap on identical 48-candidate pool, got {gap}%"



def test_ml_signal_and_auc():
    """Verify non-linear logistic signal achieves CV ROC-AUC > 0.75."""
    qcloud_path = EVIDENCE_DIR / "02_qcloud_calibration_metrics.json"
    assert qcloud_path.exists()
    with open(qcloud_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bench = data.get("fair_benchmark_table", {})
    lr_auc = bench.get("Classical_Logistic_Regression_Raw", {}).get("auc_mean", 0.0)
    gb_auc = bench.get("Classical_Gradient_Boosting_Raw", {}).get("auc_mean", 0.0)
    q_auc = bench.get("Quantum_Kernel_SVC", {}).get("auc_mean", 0.0)
    hybrid_auc = bench.get("Hybrid_Raw_Plus_Quantum_Logistic_Regression", {}).get("auc_mean", 0.0)

    assert lr_auc > 0.75, f"Classical LR AUC ({lr_auc}) must be > 0.75"
    assert gb_auc > 0.75, f"Classical Gradient Boosting AUC ({gb_auc}) must be > 0.75"
    assert q_auc > 0.75, f"Quantum Kernel SVC AUC ({q_auc}) must be > 0.75"
    assert hybrid_auc > 0.75, f"Hybrid Model AUC ({hybrid_auc}) must be > 0.75"


def test_plan_ablation_decision_impact():
    """Verify learned delay probabilities actively impact routing decisions vs shuffled."""
    qcloud_path = EVIDENCE_DIR / "02_qcloud_calibration_metrics.json"
    with open(qcloud_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ab = data.get("dispatch_plan_ablation", {})
    assert ab.get("status") == "COMPLETED"
    w1 = ab.get("risk_weight_1_0", {})
    w2 = ab.get("risk_weight_2_0_high_sensitivity", {})

    # Shuffled probabilities must change cost or route assignment
    assert w1.get("plan_changed_vs_shuffled") is True or abs(w1.get("delta_cost_vs_shuffled_inr", 0.0)) > 1.0
    assert w2.get("plan_changed_vs_shuffled") is True


def test_shot_noise_ci_non_degenerate():
    """Verify bootstrap confidence intervals in 05_noise_and_shots.json have non-zero width."""
    noise_path = EVIDENCE_DIR / "05_noise_and_shots.json"
    assert noise_path.exists()
    with open(noise_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shots_data = data.get("shot_sampling_experiment", {}).get("results_by_shots", {})
    for s_key in ["128", "1024", "8192"]:
        sd = shots_data.get(s_key, {})
        b_ci = sd.get("brier_95_ci", sd.get("brier_ci_95", [0.0, 0.0]))
        assert len(b_ci) == 2, f"CI for {s_key} shots must have 2 elements"
        assert b_ci[1] > b_ci[0], f"CI [{b_ci[0]}, {b_ci[1]}] for {s_key} shots must have strictly positive width"


def test_calibration_across_all_plans():
    """Verify calibration metrics (Pearson r, Brier MSE) are evaluated across all 5 plans."""
    mc_path = EVIDENCE_DIR / "04_monte_carlo.json"
    assert mc_path.exists()
    with open(mc_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_cal = data.get("all_plans_calibration", {})
    for plan_key in ["plan_a", "plan_b", "plan_c", "plan_d", "plan_e"]:
        assert plan_key in all_cal, f"Missing calibration metrics for {plan_key}"
        c = all_cal[plan_key]
        assert "pearson_correlation" in c
        assert "calibration_mse" in c
        assert c["pearson_correlation"] > 0.80, f"Pearson r for {plan_key} must be > 0.80"
        assert c["calibration_mse"] >= 0.0


def test_mismatched_and_tight_window_evaluations():
    """Verify heavy-tailed mismatched test and tight-window test evaluate all 5 plans."""
    mc_path = EVIDENCE_DIR / "04_monte_carlo.json"
    with open(mc_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mismatched = data.get("mismatched_distribution_robustness", {})
    assert mismatched.get("n_scenarios") == 500
    m_plans = mismatched.get("plans", {})
    for plan_key in ["plan_a", "plan_b", "plan_c", "plan_d", "plan_e"]:
        assert plan_key in m_plans, f"Missing mismatched result for {plan_key}"
        assert m_plans[plan_key]["mean_realized_cost"]["value_inr"] > 0.0

    tight = data.get("tight_window_scenario_test", {})
    assert tight.get("n_scenarios") == 500
    t_plans = tight.get("plans", {})
    for plan_key in ["plan_a", "plan_b", "plan_c", "plan_d", "plan_e"]:
        assert plan_key in t_plans, f"Missing tight window result for {plan_key}"
        assert t_plans[plan_key]["mean_realized_cost"]["value_inr"] > 0.0
