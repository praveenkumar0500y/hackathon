import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

QPIAI_QUANTUM_API_KEY = os.getenv("QPIAI_QUANTUM_API_KEY", "").strip()
QPIAI_OPT_API_KEY = os.getenv("QPIAI_OPT_API_KEY", "").strip()

HAS_QUANTUM_KEY = bool(QPIAI_QUANTUM_API_KEY)
HAS_OPT_KEY = bool(QPIAI_OPT_API_KEY)

CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "last_dispatch_run.json"
