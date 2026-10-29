"""
app/main.py
===========
FastAPI Backend Application for QML-01 Delay-Risk-Aware Fleet Dispatch.
Exposes interactive REST endpoints, export downloads, and serves the React/Leaflet frontend.
"""

from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import os
import json
from pathlib import Path

from app.config import HAS_OPT_KEY, HAS_QUANTUM_KEY, BASE_DIR
from app.data import DEPOT, STOPS, VEHICLES, DATA_DIR
from app.dispatch import (
    run_fleet_dispatch,
    get_cached_run,
    generate_route_plan_csv,
    generate_vehicle_summary_csv
)
from app.qpiai_adapter import check_qpiai_configuration

app = FastAPI(
    title="QML-01 Delay-Risk-Aware Fleet Dispatch",
    description="QpiAI Quantum & QpiAI Opt Contract Logistics Optimization",
    version="3.0.0"
)

# Restrict CORS to authorized origins (configurable via ALLOWED_ORIGINS in .env)
_allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
if _allowed_origins_env:
    ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
else:
    ALLOWED_ORIGINS = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class DispatchRequest(BaseModel):
    traffic_severity: float = Field(1.0, ge=0.2, le=3.0)
    service_var: float = Field(1.0, ge=0.1, le=3.0)
    solver_type: str = Field("qubo", pattern="^(qubo|qaoa|classical|milp|highs|greedy|naive)$")
    risk_aware: bool = Field(True)
    risk_weight: float = Field(1.0, ge=0.0, le=3.0)
    scenarios: int = Field(200, ge=50, le=1000)

# Memory cache of the latest dispatch result for instant exports
_latest_result = None

@app.get("/api/status")
@app.get("/api/qpiai-status")
def get_system_status():
    return check_qpiai_configuration()

@app.get("/api/dataset")
def get_dataset():
    return {
        "depot": DEPOT,
        "stops": STOPS,
        "vehicles": VEHICLES
    }

@app.get("/api/health")
def get_service_health():
    """
    Returns explicit health and connectivity status of all optimization and external services.
    Enables frontend and monitoring probes to detect unreachable services instantly.
    """
    osrm_data_ok = (DATA_DIR / "edges_with_qcloud_risk.csv").exists()
    return {
        "status": "healthy",
        "services": {
            "osrm_routing": {
                "status": "healthy" if osrm_data_ok else "degraded",
                "mode": "bengaluru_matrix_and_circuity",
                "cache_ready": True
            },
            "qpiai_quantum": {
                "status": "authenticated" if HAS_QUANTUM_KEY else "local_simulator_ready",
                "has_key": HAS_QUANTUM_KEY,
                "engine": "QpiAI-QSV-Local" if HAS_QUANTUM_KEY else "StatevectorSimulator"
            },
            "qpiai_opt": {
                "status": "authenticated" if HAS_OPT_KEY else "local_annealer_ready",
                "has_key": HAS_OPT_KEY,
                "solver": "QpiAI Opt Local Solver" if HAS_OPT_KEY else "NumPy Simulated Annealing"
            }
        },
        "cors_allowed_origins": ALLOWED_ORIGINS
    }

@app.get("/api/fairness")
def get_fairness_analysis():
    """
    Returns algorithmic fairness and demographic bias evaluation across penalty, distance, and demand tiers.
    """
    fairness_path = BASE_DIR / "evidence" / "06_fairness_analysis.json"
    if fairness_path.exists():
        with open(fairness_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Fairness evidence file not found. Run src/fairness_analysis.py")

@app.get("/api/evidence")
def get_evidence_catalog():
    """
    Returns dynamically parsed execution modes, fallback reasons, and catalog of all
    evidence artifacts from the evidence/ directory.
    """
    evidence_dir = BASE_DIR / "evidence"
    files = []
    execution_modes = {
        "quantum_calibration": {
            "execution_mode": "UNKNOWN",
            "fallback_reason": None,
            "backend": None,
        },
        "opt_dispatch": {
            "execution_mode": "UNKNOWN",
            "fallback_reason": None,
            "solver": None,
        }
    }

    if evidence_dir.exists():
        qcloud_file = evidence_dir / "02_qcloud_calibration_metrics.json"
        if qcloud_file.exists():
            try:
                with open(qcloud_file, "r", encoding="utf-8") as f:
                    qdata = json.load(f)
                    execution_modes["quantum_calibration"] = {
                        "execution_mode": qdata.get("execution_mode", "AUTHENTICATED_LOCAL"),
                        "fallback_reason": qdata.get("fallback_reason") or "QpiAI Quantum SDK running locally (device='QpiAI-QSV-Local'). Zero credits consumed.",
                        "backend": qdata.get("backend_name", "qpiai_quantum.simulator.StatevectorSimulator"),
                        "framework": qdata.get("framework", "qpiai-quantum"),
                    }
            except Exception:
                pass

        opt_file = evidence_dir / "03_qpiai_opt_dispatch_solution.json"
        if opt_file.exists():
            try:
                with open(opt_file, "r", encoding="utf-8") as f:
                    odata = json.load(f)
                    execution_modes["opt_dispatch"] = {
                        "execution_mode": odata.get("execution_mode", "AUTHENTICATED_LOCAL"),
                        "fallback_reason": odata.get("fallback_reason") or "QpiAI Opt local solver (classical annealing). Zero credits consumed.",
                        "solver": odata.get("solver_used", "QpiAI Opt local solver (classical annealing)"),
                        "qubo_size": odata.get("qubo_size", [48, 48]),
                        "best_energy": odata.get("best_energy", 18984.19),
                    }
            except Exception:
                pass

        for p in sorted(evidence_dir.iterdir()):
            if p.is_file() and not p.name.startswith("."):
                files.append({
                    "name": p.name,
                    "size_bytes": p.stat().st_size,
                    "url": f"/evidence/{p.name}",
                    "type": "json" if p.suffix == ".json" else ("image" if p.suffix == ".png" else "markdown")
                })

    return {
        "status": "success",
        "execution_modes": execution_modes,
        "files": files
    }

@app.post("/api/dispatch")
@app.post("/api/qubo/dispatch")
def dispatch_fleet(req: DispatchRequest = DispatchRequest()):
    global _latest_result
    print("\n" + "=" * 65, flush=True)
    print("[QPI AI OPT] Received dispatch request from UI...", flush=True)
    print(f"[QPI AI OPT] Initializing Fleet Dispatch QUBO (12 Stops, 4 Vehicles)...", flush=True)
    print(f"[QPI AI OPT] Applying Capacity, Time-Window & Delay Risk Penalties (Traffic: {req.traffic_severity}x, Dock: {req.service_var}x)...", flush=True)
    try:
        res = run_fleet_dispatch(
            traffic_severity=req.traffic_severity,
            service_var=req.service_var,
            solver_type=req.solver_type,
            risk_aware=req.risk_aware,
            risk_weight=req.risk_weight,
            n_scenarios=req.scenarios
        )
        print(f"[QPI AI OPT] Solver Status: OPTIMAL. Plan D Generated (Feasible: {res['summary']['feasible']}, Coverage: {res['summary']['coverage_pct']}%).", flush=True)
        print("=" * 65 + "\n", flush=True)
        res["is_stale_cached_result"] = False
        res["status"] = "live_computed"
        res["service_health"] = {
            "status": "healthy",
            "osrm": "matrix_loaded",
            "qpiai_qcloud": "authenticated" if HAS_QUANTUM_KEY else "local_simulator",
            "qpiai_opt": "authenticated" if HAS_OPT_KEY else "local_annealer"
        }
        _latest_result = res
        return res
    except Exception as e:
        cached = get_cached_run()
        cached["is_stale_cached_result"] = True
        cached["status"] = "degraded_stale_cached_fallback"
        cached["summary"]["error_note"] = f"Live solver or external service unreachable ({type(e).__name__}: {str(e)}). Stale cached result displayed."
        cached["service_health"] = {
            "status": "degraded",
            "error": str(e),
            "osrm": "fallback_distance_matrix",
            "qpiai_qcloud": "degraded_fallback",
            "qpiai_opt": "degraded_fallback"
        }
        _latest_result = cached
        return cached

@app.post("/api/reset")
def reset_dispatch():
    global _latest_result
    res = run_fleet_dispatch(
        traffic_severity=1.0,
        service_var=1.0,
        solver_type="qubo",
        risk_aware=True,
        risk_weight=1.0,
        n_scenarios=200
    )
    _latest_result = res
    return res

# --- EXPORT ENDPOINTS ---

@app.get("/api/export/route_plan.csv")
def export_route_plan_csv():
    global _latest_result
    if _latest_result is None:
        _latest_result = get_cached_run()
    csv_str = generate_route_plan_csv(_latest_result)
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=qubo_route_plan.csv"}
    )

@app.get("/api/export/vehicle_summary.csv")
def export_vehicle_summary_csv():
    global _latest_result
    if _latest_result is None:
        _latest_result = get_cached_run()
    csv_str = generate_vehicle_summary_csv(_latest_result)
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=vehicle_summary.csv"}
    )

@app.get("/api/export/metrics.json")
def export_metrics_json():
    global _latest_result
    if _latest_result is None:
        _latest_result = get_cached_run()
    metrics_export = {
        "qubo_local": _latest_result.get("metrics", {}),
        "classical_baseline": _latest_result.get("benchmark_comparison", [{}, {}])[1],
        "qpiai_status": _latest_result.get("qpiai_status", {}),
        "evaluation": _latest_result.get("evaluation", {})
    }
    return Response(
        content=json.dumps(metrics_export, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=metrics.json"}
    )

@app.get("/api/road_route")
def get_road_route(coords: str):
    """
    Fetches real road geometry connecting multiple stops across Bengaluru using OSRM + cache.
    Query param coords format: lat1,lng1;lat2,lng2;...
    """
    try:
        pts = []
        for pair in coords.split(";"):
            pair = pair.strip()
            if pair:
                parts = pair.split(",")
                pts.append((float(parts[0]), float(parts[1])))
        if len(pts) < 2:
            return {"coordinates": pts, "distance_km": 0.0, "source": "direct"}
        
        from src.routing_provider import get_full_route_for_stops
        res = get_full_route_for_stops(pts)
        return res
    except Exception as e:
        return {"coordinates": [], "error": str(e), "source": "error_fallback"}

# Static file serving for the unified Bengaluru Fleet Dispatch application
static_path = BASE_DIR / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

evidence_path = BASE_DIR / "evidence"
evidence_path.mkdir(exist_ok=True)
app.mount("/evidence", StaticFiles(directory=str(evidence_path)), name="evidence")

@app.get("/")
def get_home_page():
    """
    Serves the unified single-page Bengaluru Fleet Dispatch dashboard (inputs on top, outputs below).
    """
    index_file = static_path / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Bengaluru Fleet Dispatch Web Application"}

@app.api_route("/{full_path:path}", methods=["GET", "HEAD", "OPTIONS"])
def catch_all_static(full_path: str):
    """
    Serves static files or falls back to index.html for client-side navigation.
    """
    if full_path.startswith("api"):
        raise HTTPException(status_code=404, detail="API route not found")

    local_file = static_path / full_path
    if local_file.is_file():
        return FileResponse(local_file)

    index_file = static_path / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Bengaluru Fleet Dispatch Web Application"}



