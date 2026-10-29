"""
tests/test_honesty_and_pii.py
==============================
Unit tests for Scientific Rigor and Honesty:
1. Zero PII: Asserts no personal names or emails in README.md or evidence JSONs.
2. Disjoint Seed Hygiene: Asserts disjoint seed sets between training, CV, candidate generation, and evaluation.
3. Live Mode Guard: When API keys are empty, asserts execution_mode is LOCAL_* and fallback reason is logged.
4. Cascading Delay Coupling: Asserts all qualifying interactions are included in HiGHS reference solving.
5. Calibration Assessment: Asserts calibration label logic correctly categorizes MSE and correlation.
"""

import os
import sys
import json
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.stochastic_model import (
    LABEL_DATA_SEED,
    TRAIN_DATA_SEEDS,
    CV_SPLIT_SEED,
    INNER_CANDIDATE_SEED,
    MONTE_CARLO_EVAL_SEEDS
)
from app.data import DEPOT, STOPS, VEHICLES, EDGES
from src.candidate_generator import generate_candidate_route_pool
from src.step4_opt_dispatch import identify_cascading_delay_couplings, solve_exact_reference_milp

EVIDENCE_DIR = BASE_DIR / "evidence"


def test_zero_pii_in_evidence_and_readme():
    """Verify that no personal identifying information (PII) exists in evidence files or README."""
    forbidden_terms = ["praveen", "ronad", "@gmail.com", "praveenkumarmr45"]

    files_to_check = [
        BASE_DIR / "README.md",
        EVIDENCE_DIR / "02_qcloud_calibration_metrics.json",
        EVIDENCE_DIR / "03_qpiai_opt_dispatch_solution.json",
        EVIDENCE_DIR / "04_monte_carlo.json",
        EVIDENCE_DIR / "05_noise_and_shots.json",
        EVIDENCE_DIR / "06_fairness_analysis.json",
        EVIDENCE_DIR / "CLAIMS.md"
    ]

    for fpath in files_to_check:
        if fpath.exists():
            content = fpath.read_text(encoding="utf-8").lower()
            for term in forbidden_terms:
                assert term not in content, f"PII leak detected! Found '{term}' in {fpath.name}"


def test_client_info_stores_only_plan_tier_and_status():
    """Verify that client_info in evidence/03 strictly contains only plan_tier and status (zero PII/org leaks)."""
    opt_file = EVIDENCE_DIR / "03_qpiai_opt_dispatch_solution.json"
    assert opt_file.exists(), f"Missing {opt_file}"
    with open(opt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    client_info = data.get("client_info")
    assert client_info is not None, "client_info must be populated in evidence/03"
    assert isinstance(client_info, dict), "client_info must be a dictionary"
    allowed_keys = {"plan_tier", "status"}
    assert set(client_info.keys()) == allowed_keys, f"client_info contains unauthorized keys: {set(client_info.keys()) - allowed_keys}"
    assert "name" not in client_info and "email" not in client_info and "organization_id" not in client_info


def test_disjoint_seed_hygiene():
    """Verify that training, CV, candidate generation, and evaluation seeds have zero overlap."""
    label_set = {LABEL_DATA_SEED}
    train_set = set(TRAIN_DATA_SEEDS)
    cv_set = {CV_SPLIT_SEED}
    inner_set = {INNER_CANDIDATE_SEED}
    eval_set = set(MONTE_CARLO_EVAL_SEEDS)

    assert label_set.isdisjoint(eval_set), "Dedicated LABEL seed must not overlap with evaluation seeds"
    assert train_set.isdisjoint(eval_set), "Training seeds must not overlap with evaluation seeds"
    assert cv_set.isdisjoint(eval_set), "CV split seed must not be in evaluation set"
    assert inner_set.isdisjoint(eval_set), "Candidate evaluation seed must not be in evaluation set"
    assert train_set.isdisjoint(inner_set), "Training seeds must not overlap with candidate seeds"
    assert len(eval_set) >= 1000, "Must have at least 1000 held-out evaluation scenarios"


def test_live_mode_guard_and_honest_fallback():
    """Verify that execution modes are strictly honest (AUTHENTICATED_LOCAL or LOCAL_*), never claiming LIVE_*."""
    q_file = EVIDENCE_DIR / "02_qcloud_calibration_metrics.json"
    opt_file = EVIDENCE_DIR / "03_qpiai_opt_dispatch_solution.json"

    if q_file.exists():
        with open(q_file, "r", encoding="utf-8") as f:
            q_data = json.load(f)
        q_exec = q_data.get("execution_mode")
        assert not str(q_exec).startswith("LIVE_"), "Must never label as LIVE_* (real cloud jobs are not being made)"
        if not os.getenv("QPIAI_QUANTUM_API_KEY"):
            assert q_exec in ["LOCAL_SIMULATOR", "LOCAL_FALLBACK"], "Must report LOCAL_* when key is unset"
            assert q_data.get("fallback_reason") is not None, "Fallback reason must be populated"
        else:
            assert q_exec in ["AUTHENTICATED_LOCAL", "LOCAL_SIMULATOR"], "Must report AUTHENTICATED_LOCAL or LOCAL_SIMULATOR"

    if opt_file.exists():
        with open(opt_file, "r", encoding="utf-8") as f:
            opt_data = json.load(f)
        opt_exec = opt_data.get("execution_mode")
        assert not str(opt_exec).startswith("LIVE_"), "Must never label as LIVE_* (real cloud jobs are not being made)"
        if not os.getenv("QPIAI_OPT_API_KEY"):
            assert opt_exec in ["LOCAL_FALLBACK"], "Must report LOCAL_FALLBACK when key is unset"
            assert opt_data.get("fallback_reason") is not None, "Fallback reason must be populated"
        else:
            assert opt_exec in ["AUTHENTICATED_LOCAL", "LOCAL_FALLBACK"], "Must report AUTHENTICATED_LOCAL or LOCAL_FALLBACK"


def test_cascading_couplings_included_in_exact_reference():
    """Verify that cascading delay couplings are identified and properly evaluated in exact reference."""
    stops_dict = {DEPOT["id"]: DEPOT, **{s["id"]: s for s in STOPS}}
    candidates, _ = generate_candidate_route_pool(stops_dict, EDGES, VEHICLES, risk_weight=1.0)
    couplings = identify_cascading_delay_couplings(candidates, gamma_cascade=800.0)

    assert len(couplings) > 0, "Should identify non-zero qualifying cascading interactions"
    for i, j, cost in couplings[:10]:
        assert i < j, "Coupling indices must be ordered i < j"
        assert cost > 0, "Coupling interaction cost must be positive"

    # Solve exact reference
    sol = solve_exact_reference_milp(candidates, couplings, num_stops=12)
    assert sol["status"] == "OPTIMAL", "Exact HiGHS reference must find optimal solution"
    assert sol["objective"] is not None
    assert sol["objective"] > 0
    assert len(sol["selected_routes"]) in [3, 4], "Should select 3 or 4 routes to cover 12 stops"


def test_calibration_programmatic_labels():
    """Verify that calibration labels are programmatically assigned based on metrics, not hardcoded."""
    from src.monte_carlo import validate_quantum_calibration

    # Mock perfect predictions
    mock_routes = [{
        "timeline": [
            {"stop_id": 1, "name": "Stop 1", "qcloud_sla_breach_prob": 0.10},
            {"stop_id": 2, "name": "Stop 2", "qcloud_sla_breach_prob": 0.80}
        ]
    }]
    mock_sim = {"stop_empirical_breach_rates": {1: 0.11, 2: 0.79}}
    res_good = validate_quantum_calibration(mock_routes, mock_sim)
    assert res_good["assessment"] == "WELL CALIBRATED"

    # Mock uncorrelated predictions
    mock_sim_bad = {"stop_empirical_breach_rates": {1: 0.90, 2: 0.10}}
    res_bad = validate_quantum_calibration(mock_routes, mock_sim_bad)
    assert res_bad["assessment"] in ["POORLY CALIBRATED", "MODERATELY CALIBRATED"]


def test_noise_and_shots_evidence_structure():
    """Verify that evidence/05_noise_and_shots.json contains valid hardware profiling, shot convergence, and noise tests."""
    fpath = EVIDENCE_DIR / "05_noise_and_shots.json"
    assert fpath.exists(), "05_noise_and_shots.json must exist"

    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Hardware Profiling
    hw = data.get("circuit_hardware_profiling", {})
    v1 = hw.get("variant_1", {})
    v2 = hw.get("variant_2", {})
    assert v1.get("num_qubits") == 3, "Variant 1 must have 3 qubits"
    assert v1.get("depth") == 5, "Variant 1 depth must be 5"
    assert v1.get("total_gates") == 9, "Variant 1 must have 9 gates"
    assert v2.get("num_qubits") == 3, "Variant 2 must have 3 qubits"
    assert v2.get("depth") == 10, "Variant 2 depth must be 10"
    assert v2.get("total_gates") == 18, "Variant 2 must have 18 gates"

    # 2. Shot Convergence
    shot_data = data.get("shot_sampling_experiment", {})
    results_by_shots = shot_data.get("results_by_shots", {})
    assert "128" in results_by_shots and "1024" in results_by_shots and "8192" in results_by_shots
    assert results_by_shots["128"]["n_repeats"] == 20
    assert results_by_shots["1024"]["n_repeats"] == 20
    assert results_by_shots["8192"]["n_repeats"] == 20

    # 1/sqrt(S) variance reduction check across leg standard deviations
    std_128 = results_by_shots["128"]["mean_p_ij_std_across_legs"]
    std_1024 = results_by_shots["1024"]["mean_p_ij_std_across_legs"]
    std_8192 = results_by_shots["8192"]["mean_p_ij_std_across_legs"]
    assert std_128 > std_1024 > std_8192, "Standard deviation must decrease strictly as shot count increases"

    # 3. Noise Emulation
    noise_data = data.get("noise_emulation_experiment", {})
    assert noise_data.get("model_name") == "EMULATED_DEPOLARIZING_AND_READOUT"
    res_noise = noise_data.get("results_by_noise_level", {})
    assert "eps_0.00" in res_noise
    assert "eps_0.01" in res_noise
    assert "eps_0.03" in res_noise
    assert "eps_0.05" in res_noise
    assert "eps_0.10" in res_noise
    # Brier should remain reasonable (between 0.20 and 0.25)
    for k, v in res_noise.items():
        assert 0.20 <= v["brier_score"] <= 0.25
        assert 0.50 <= v["roc_auc"] <= 0.65


def test_fair_benchmark_and_quantum_advantage_assessment():
    """Verify that fair benchmark table, ablation study, and honest quantum advantage assessment exist."""
    fpath = EVIDENCE_DIR / "02_qcloud_calibration_metrics.json"
    assert fpath.exists(), "02_qcloud_calibration_metrics.json must exist"

    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Fair Benchmark Table
    assert "fair_benchmark_table" in data, "Must contain fair_benchmark_table"
    table = data["fair_benchmark_table"]
    expected_models = [
        "Classical_Logistic_Regression_Raw",
        "Classical_Gradient_Boosting_Raw",
        "Quantum_Features_Logistic_Regression",
        "Hybrid_Raw_Plus_Quantum_Logistic_Regression",
        "Quantum_Kernel_SVC"
    ]
    for m in expected_models:
        assert m in table, f"Model {m} must be present in fair_benchmark_table"
        entry = table[m]
        for metric in ["brier_mean", "brier_std", "auc_mean", "auc_std", "log_loss_mean", "log_loss_std", "ece_mean", "ece_std", "n_folds_evaluated"]:
            assert metric in entry, f"Metric {metric} missing for {m}"
        assert entry["n_folds_evaluated"] >= 5, f"{m} must be evaluated on at least 5 folds"
        assert 0.0 <= entry["auc_mean"] <= 1.0, f"AUC must be in [0, 1] for {m}"
        assert 0.0 <= entry["brier_mean"] <= 1.0, f"Brier must be in [0, 1] for {m}"

    # 2. Ablation Study
    assert "ablation_study" in data, "Must contain ablation_study"
    ablation = data["ablation_study"]
    expected_ablations = [
        "With_Entanglement_scale_0.5",
        "With_Entanglement_scale_1.0",
        "With_Entanglement_scale_2.0",
        "Without_Entanglement_scale_0.5",
        "Without_Entanglement_scale_1.0",
        "Without_Entanglement_scale_2.0"
    ]
    for a in expected_ablations:
        assert a in ablation, f"Ablation condition {a} must be present"
        entry = ablation[a]
        assert "has_entanglement" in entry
        assert "angle_scaling" in entry
        assert "auc_mean" in entry
        assert "brier_mean" in entry

    # 3. Quantum Advantage Assessment
    assert "quantum_advantage_assessment" in data, "Must contain quantum_advantage_assessment"
    q_adv = data["quantum_advantage_assessment"]
    assert "quantum_beat_classical" in q_adv, "Must contain quantum_beat_classical flag"
    assert isinstance(q_adv["quantum_beat_classical"], bool)
    assert "best_classical_baseline" in q_adv
    assert "best_standalone_quantum_model" in q_adv
    assert "plain_language_verdict" in q_adv
    assert "limitations" in q_adv
    assert len(q_adv["limitations"]) >= 3, "Must document at least 3 physical / statistical limitations"

    # Verify no misleading claims
    verdict = q_adv["plain_language_verdict"].lower()
    assert "no empirical quantum advantage observed" in verdict or "no quantum advantage" in verdict, (
        "Verdict must honestly state that no quantum advantage was observed when standalone quantum lags classical"
    )
    assert not data.get("conclusion", "").lower().startswith("quantum features outperform classical"), (
        "Old misleading conclusion sentence must not be present"
    )


