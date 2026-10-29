# CLAIMS.md: Traceability & Verification Audit
> Project: QML-01 Delay-Risk-Aware Fleet Dispatch for Contract Logistics  
> QUANT-A-MAZE 3.0 Hackathon (QML Track, Bengaluru)

This document maps every headline claim made in the pitch, dashboard, and README directly to machine-readable keys in the `evidence/` directory.

---

## 1. Primary Scientific Claims

| # | Claim | Status | Traceable Evidence Key | Numerical Value (with 95% CI) |
| :-: | :--- | :---: | :--- | :--- |
| **C1** | **Plan D achieves lower realized cost than Risk-Blind MILP (Plan C)** | **VERIFIED** | `evidence/04_monte_carlo.json` -> `plan_comparisons` -> `plan_c` & `plan_d` | Plan C: Rs. 19,108.84 vs Plan D: Rs. 18,803.34 (Rs. 305.50 daily savings; $\Delta_{D-B} = \text{Rs. } 36.25$, 95% CI: [-52.37, 126.77]) |
| **C2** | **Plan D slashes late delivery rate to lowest across all baselines** | **VERIFIED** | `evidence/04_monte_carlo.json` -> `plan_comparisons` -> `late_delivery_rate_pct` | Plan A: 7.80% | Plan C: 0.55% | Plan B: 0.22% $\to$ Plan D: **0.16%** (lowest breach rate) |
| **C3** | **Quantum Delay Model is Rank-Consistent & Well Calibrated** | **VERIFIED** | `evidence/04_monte_carlo.json` -> `quantum_calibration_validation` -> `pearson_correlation`, `calibration_mse` | Pearson $r = 0.9973$, $MSE = 0.000019$, Assessment: `WELL CALIBRATED` |
| **C4** | **QpiAI Opt QUBO achieves certified pruned optimum & honest global gap disclosed** | **VERIFIED** | `evidence/03_qpiai_opt_dispatch_solution.json` -> `solver_info` | Pruned pool gap: 0.00%, 4-stop full pool gap: 0.62%, 5-stop global gap: 10.60% |
| **C5** | **Zero Hard Constraint Violations** | **VERIFIED** | `evidence/03_qpiai_opt_dispatch_solution.json` -> `feasibility_verification` -> `is_fully_feasible` | 12/12 stops covered, 0 duplicates, all vehicle capacities and shift limits respected |
| **C6** | **Fair Classical vs Quantum Benchmark (Honest Advantage Assessment)** | **VERIFIED** | `evidence/02_qcloud_calibration_metrics.json` -> `quantum_advantage_assessment` -> `plain_language_verdict` | Classical LR: AUC 0.7935 vs Quantum Standalone: AUC 0.7766 (Delta = -0.0169). Zero false advantage claimed. |
| **C7** | **Zero Personal Identifying Information (PII)** | **VERIFIED** | Entire repository audit | Zero personal names or email addresses in README or evidence files |
| **C8** | **Shot-Noise Scaling Obeys Theoretical $1/\sqrt{S}$ Convergence** | **VERIFIED** | `evidence/05_noise_and_shots.json` -> `shot_sampling_experiment` -> `results_by_shots` | Mean leg $p_{ij}$ std: $0.0075 \to 0.0027 \to 0.0010$ across $S \in [128, 1024, 8192]$ (ratio $\approx 2.8 \approx \sqrt{8}$) |
| **C9** | **Variational Quantum Circuit Resilient to Hardware Decoherence** | **VERIFIED** | `evidence/05_noise_and_shots.json` -> `noise_emulation_experiment` -> `results_by_noise_level` | Brier degrades by only $+0.09\%$ under severe 10% readout bit-flip and 20% depolarizing rate |
| **C10** | **Real QCloud Job Submission & Honest Fallback Protocol Verified** | **VERIFIED** | `evidence/02_qcloud_calibration_metrics.json` -> `cloud_submission_attempted`, `cloud_submission_error` | Submission attempted via `JobManager.submit_circuit_job`; server error recorded; zero false LIVE claims |
| **C11** | **Algorithmic Fairness Disparity Quantified** | **VERIFIED** | `evidence/06_fairness_analysis.json` -> `disparity_summary` | Disparity ratio 1.1x at w=1.0, rising to low-tier position shift (3.33 vs 2.20) at w=3.0; 4 concrete mitigations specified |
| **C12** | **Threat Model, Zero PII & Origin-Restricted CORS** | **VERIFIED** | Codebase & `evidence/` audit | CORS restricted to localhost development origins; zero API keys logged; all inputs validated via Pydantic bounds |
| **C13** | **100% Offline Capability & Dynamic Evidence Integration** | **VERIFIED** | `static/vendor/`, `static/index.html`, `/api/evidence` | Zero remote CDN links; in-browser Babel eliminated; local React/ReactDOM/Leaflet/Tailwind bundles; plain canvas fallback; execution_mode badges & Evidence catalog |

---

## 2. Methodology & Honesty Audit

- **Disjoint Seed Hygiene**:
  - Training / edge shock labeling: `range(100, 200)`
  - Cross-validation split: `42`
  - Candidate evaluation: `777`
  - Monte Carlo held-out evaluation: `range(1000, 3000)` (2,000 independent seeds with Common Random Numbers)
  - Fairness multi-tier evaluation: `range(4000, 4200)` (200 independent seeds with Common Random Numbers)
- **Execution Mode Transparency**:
  - When `QPIAI_QUANTUM_API_KEY` is absent or subscription inactive: Honest `LOCAL_SIMULATOR` with exact reason logged.
  - When `QPIAI_OPT_API_KEY` is absent: Honest `LOCAL_FALLBACK` with exact reason logged.
- **Security & Privacy Enforcement**:
  - `ALLOWED_ORIGINS` restricts CORS to authorized local dashboard ports by default.
  - API keys are loaded into memory and never logged, printed, or serialized to artifacts.
  - Client session objects (`client_info`) store only functional `plan_tier` and `status`.
- **Physical Generative Grounding**:
  - Exact same lognormal travel times and gamma dock service times used across dataset generation, route candidate evaluation, and Monte Carlo validation.
- **Synthetic Data Disclosure**:
  - All operational data (stops, demands, delivery windows, congestion indices, dock variances, and historical SLA breach labels) is **100% synthetically generated** via `src/stochastic_model.py`. Zero proprietary carrier or customer PII is included.
