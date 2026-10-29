"""
run_all.py
==========
Master Reproduction Pipeline for QUANT-A-MAZE 3.0 (QML Track, Bengaluru):
"QML-01 Delay-Risk-Aware Fleet Dispatch for Contract Logistics"

Executes end-to-end:
1. Data Integrity & Bengaluru Geometry Verification
2. Quantum Delay-Risk Prediction & Platt Calibration (qpiai-quantum)
3. Set-Partitioning QUBO + Cascading-Delay HUBO Optimization (qpiai-opt)
4. 4-Plan Baseline Dispatch Benchmarks (Greedy, Cost-Only VRPTW, QUBO Blind, QUBO Risk-Aware)
5. Monte Carlo Stochastic Stress Testing (N=1000 scenarios) & Pareto Frontier Sweep
6. Full Unit Test Suite Verification (pytest)
7. Final Evidence Audit & Honesty Scorecard
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

EVIDENCE_DIR = BASE_DIR / "evidence"


def print_banner(text: str):
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)


def run_pipeline():
    total_start = time.perf_counter()
    print_banner("QUANT-A-MAZE 3.0: QML-01 DELAY-RISK-AWARE FLEET DISPATCH PIPELINE")
    print(f"[*] Workspace Root: {BASE_DIR}")
    print(f"[*] Evidence Directory: {EVIDENCE_DIR}")

    # Step 1: Generate & Verify Dataset
    print_banner("STAGE 1: GENERATING & VERIFYING BENGALURU LOGISTICS DATASET")
    from src.step2_data_gen import generate_logistics_data
    generate_logistics_data()
    from app.data import DEPOT, STOPS, VEHICLES, EDGES
    print(f"[+] Loaded Depot: {DEPOT['name']} at ({DEPOT['lat']}, {DEPOT['lng']})")
    print(f"[+] Loaded Commercial Stops: {len(STOPS)} stops (Total Demand: {sum(s['demand_kg'] for s in STOPS)} kg)")
    print(f"[+] Loaded Fleet: {len(VEHICLES)} heterogeneous vehicles ({', '.join(VEHICLES.keys())})")
    print(f"[+] Loaded Network Edges: {len(EDGES)} directed legs with congestion & uncertainty")

    # Step 2: Quantum Risk Modeling (Step 3)
    print_banner("STAGE 2: QPIAI QUANTUM DELAY-RISK PREDICTION & CALIBRATION")
    t0 = time.perf_counter()
    from src.step3_qcloud_risk import run_quantum_risk_calibration
    q_metrics = run_quantum_risk_calibration()
    t_quantum = time.perf_counter() - t0
    print(f"[+] Quantum Stage Completed in {t_quantum:.2f}s | Mode: {q_metrics.get('execution_mode')}")

    # Step 3: QUBO Dispatch Optimization (Step 4)
    print_banner("STAGE 3: QPIAI OPT SET-PARTITIONING QUBO & HUBO DISPATCH")
    t0 = time.perf_counter()
    from src.step4_opt_dispatch import run_qubo_optimization
    opt_sol = run_qubo_optimization(risk_weight=1.0)
    t_opt = time.perf_counter() - t0
    print(f"[+] Optimization Stage Completed in {t_opt:.2f}s | Mode: {opt_sol.get('execution_mode')}")

    # Step 4: Monte Carlo & Baselines (Step 5)
    print_banner("STAGE 4: 4-PLAN BASELINES & MONTE CARLO STRESS TESTING (N=1000)")
    t0 = time.perf_counter()
    from src.monte_carlo import run_full_monte_carlo_analysis
    mc_results = run_full_monte_carlo_analysis()
    t_mc = time.perf_counter() - t0
    print(f"[+] Monte Carlo & Baselines Completed in {t_mc:.2f}s")

    # Step 5: Fairness & Demographic Bias Analysis
    print_banner("STAGE 5: ALGORITHMIC FAIRNESS & BIAS EVALUATION (MULTI-TIER AUDIT)")
    t0 = time.perf_counter()
    from src.fairness_analysis import run_fairness_analysis
    fair_results = run_fairness_analysis()
    t_fair = time.perf_counter() - t0
    print(f"[+] Fairness Evaluation Completed in {t_fair:.2f}s")

    # Step 6: Re-render README and CLAIMS.md from Fresh Evidence
    print_banner("STAGE 6: RENDERING README.MD AND EVIDENCE/CLAIMS.MD")
    from scripts.render_readme import render_readme
    render_readme()

    # Step 7: Run Pytest Test Suite
    print_banner("STAGE 7: RUNNING PYTEST UNIT TEST SUITE")
    res = subprocess.run([sys.executable, "-m", "pytest", "-v", "tests/"], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("[-] Warning: Some unit tests failed!")
    else:
        print("[+] All unit tests PASSED successfully!")

    # Step 8: Evidence Audit
    print_banner("STAGE 8: GENERATED EVIDENCE AUDIT")
    expected_files = [
        EVIDENCE_DIR / "02_qcloud_calibration_metrics.json",
        EVIDENCE_DIR / "03_qpiai_opt_dispatch_solution.json",
        EVIDENCE_DIR / "04_monte_carlo.json",
        EVIDENCE_DIR / "04_pareto_frontier.png",
        EVIDENCE_DIR / "05_noise_and_shots.json",
        EVIDENCE_DIR / "06_fairness_analysis.json",
        EVIDENCE_DIR / "CLAIMS.md"
    ]
    all_present = True
    for ef in expected_files:
        if ef.exists():
            size_kb = ef.stat().st_size / 1024.0
            print(f"[OK] {ef.name:<36} ({size_kb:.1f} KB)")
        else:
            print(f"[MISSING] {ef.name}")
            all_present = False

    total_time = time.perf_counter() - total_start
    print_banner("FINAL EXECUTION SCORECARD")
    print(f"Total Pipeline Runtime: {total_time:.2f} seconds")
    print(f"Quantum Execution Mode: {q_metrics.get('execution_mode')}")
    print(f"Optimization Execution Mode: {opt_sol.get('execution_mode')}")
    print(f"Dispatch Optimality Gap vs Exact Ref: {opt_sol.get('solver_info', {}).get('optimality_gap_vs_exact_ref_pct', 0.0):.2f}%")
    print(f"All Evidence Artifacts Verified: {'YES' if all_present else 'NO'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_pipeline()
