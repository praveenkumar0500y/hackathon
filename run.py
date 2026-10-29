"""
run.py
======
Entrypoint script to launch the QML-01 Delay-Risk-Aware Fleet Dispatch Web Application.
Serves the FastAPI REST endpoints and the interactive Bengaluru Leaflet dispatch dashboard on port 8000.
"""

import uvicorn
from app.config import BASE_DIR

if __name__ == "__main__":
    print("=" * 75)
    print("  QML-01 Delay-Risk-Aware Fleet Dispatch System")
    print("  Quantum SDK: QpiAI Quantum | Optimization SDK: QpiAI Opt")
    print("  Web Dashboard & Dispatch App: http://localhost:8000")
    print("  Interactive API Docs (Swagger): http://localhost:8000/docs")
    print("=" * 75)

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )


