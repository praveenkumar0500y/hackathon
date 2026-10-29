"""
scripts/download_vendor_assets.py
=================================
Downloads all third-party vendor libraries and assets required for 100% offline
operation into static/vendor/:
- React 18 & ReactDOM 18 production UMD builds
- Leaflet 1.9.4 JavaScript & CSS
- Leaflet marker icons, retina icons, shadows, and layer PNG images
"""

import os
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
VENDOR_DIR = BASE_DIR / "static" / "vendor"
IMAGES_DIR = VENDOR_DIR / "images"

VENDOR_URLS = {
    # React 18 production UMD
    VENDOR_DIR / "react.production.min.js": "https://unpkg.com/react@18.2.0/umd/react.production.min.js",
    VENDOR_DIR / "react-dom.production.min.js": "https://unpkg.com/react-dom@18.2.0/umd/react-dom.production.min.js",

    # Leaflet 1.9.4
    VENDOR_DIR / "leaflet.js": "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js",
    VENDOR_DIR / "leaflet.css": "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css",

    # Leaflet Marker & Layer Images
    IMAGES_DIR / "marker-icon.png": "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
    IMAGES_DIR / "marker-icon-2x.png": "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
    IMAGES_DIR / "marker-shadow.png": "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
    IMAGES_DIR / "layers.png": "https://unpkg.com/leaflet@1.9.4/dist/images/layers.png",
    IMAGES_DIR / "layers-2x.png": "https://unpkg.com/leaflet@1.9.4/dist/images/layers-2x.png",
}

def download_assets():
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[*] Downloading vendor assets to {VENDOR_DIR}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for target_path, url in VENDOR_URLS.items():
        if target_path.exists() and target_path.stat().st_size > 0:
            print(f" [EXISTS] {target_path.name} ({target_path.stat().st_size} bytes)")
            continue

        print(f" [DOWNLOADING] {url} -> {target_path.relative_to(BASE_DIR)}")
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            target_path.write_bytes(data)
            print(f"  [OK] Saved {len(data)} bytes")

    # Also copy images to static/vendor/leaflet/images/ for max compatibility
    leaflet_images = VENDOR_DIR / "leaflet" / "images"
    leaflet_images.mkdir(parents=True, exist_ok=True)
    for img_file in IMAGES_DIR.glob("*.png"):
        (leaflet_images / img_file.name).write_bytes(img_file.read_bytes())

    print("[SUCCESS] All vendor assets verified and downloaded.")

if __name__ == "__main__":
    download_assets()
