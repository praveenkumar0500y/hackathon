"""
scripts/build_frontend.py
=========================
Unified offline frontend build pipeline:
1. Verifies/downloads vendor assets (React, ReactDOM, Leaflet) into static/vendor/
2. Precompiles src/app_source.jsx into static/app.js using offline Babel
3. Compiles Tailwind CSS into static/vendor/tailwind.min.css
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def main():
    print("[*] STEP 1: Verifying offline vendor assets...")
    from scripts.download_vendor_assets import download_assets
    download_assets()

    print("\n[*] STEP 2: Precompiling JSX to plain JavaScript (static/app.js)...")
    build_js = BASE_DIR / "scripts" / "build_frontend.js"
    res = subprocess.run(["node", str(build_js)], cwd=str(BASE_DIR), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] Babel build failed:\n{res.stderr}")
        sys.exit(1)
    print(res.stdout.strip())

    print("\n[*] STEP 3: Compiling Tailwind CSS (static/vendor/tailwind.min.css)...")
    res_css = subprocess.run(
        ["npx", "--yes", "tailwindcss@3", "-i", "static/input.css", "-o", "static/vendor/tailwind.min.css", "--minify"],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        shell=True
    )
    if res_css.returncode != 0:
        print(f"[ERROR] Tailwind build failed:\n{res_css.stderr}")
        sys.exit(1)
    print(res_css.stdout.strip())
    print("[SUCCESS] Offline frontend build complete.")

if __name__ == "__main__":
    main()
