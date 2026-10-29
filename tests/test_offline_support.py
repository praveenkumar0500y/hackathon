"""
tests/test_offline_support.py
=============================
Unit tests for 100% offline capability of the RouteRisk Bengaluru web app:
1. Vendor asset presence: React, ReactDOM, Leaflet (JS, CSS, marker images), and prebuilt Tailwind CSS.
2. Zero external CDN dependencies in static/index.html (no unpkg, no cdn.tailwindcss).
3. Elimination of in-browser Babel (no @babel/standalone script, no type="text/babel").
4. Precompiled static/app.js validity and inclusion of offline canvas & evidence panel.
5. Dynamic /api/evidence endpoint returning execution modes (LOCAL_SIMULATOR/LOCAL_FALLBACK/LIVE_*) and fallback reasons.
6. Direct static /evidence/{filename} artifact serving.
"""

import re
import sys
import json
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from app.main import app
STATIC_DIR = BASE_DIR / "static"
VENDOR_DIR = STATIC_DIR / "vendor"
IMAGES_DIR = VENDOR_DIR / "images"
EVIDENCE_DIR = BASE_DIR / "evidence"


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_vendor_assets_exist_and_non_empty():
    """Verify all required vendor assets exist in static/vendor/ with non-zero size."""
    expected_assets = [
        (VENDOR_DIR / "react.production.min.js", 5_000),
        (VENDOR_DIR / "react-dom.production.min.js", 50_000),
        (VENDOR_DIR / "leaflet.js", 50_000),
        (VENDOR_DIR / "leaflet.css", 5_000),
        (VENDOR_DIR / "tailwind.min.css", 10_000),
        (IMAGES_DIR / "marker-icon.png", 500),
        (IMAGES_DIR / "marker-icon-2x.png", 1_000),
        (IMAGES_DIR / "marker-shadow.png", 300),
        (IMAGES_DIR / "layers.png", 300),
        (IMAGES_DIR / "layers-2x.png", 500),
    ]

    for asset_path, min_bytes in expected_assets:
        assert asset_path.exists(), f"Missing offline vendor asset: {asset_path.relative_to(BASE_DIR)}"
        assert asset_path.stat().st_size >= min_bytes, (
            f"Asset {asset_path.name} is undersized ({asset_path.stat().st_size} bytes, expected >= {min_bytes})"
        )


def test_zero_external_cdn_in_index_html():
    """Verify that static/index.html does not load scripts or styles from remote CDNs."""
    index_file = STATIC_DIR / "index.html"
    assert index_file.exists(), "Missing static/index.html"

    content = index_file.read_text(encoding="utf-8")

    # Prohibited remote CDN domains
    prohibited_cdns = [
        "unpkg.com",
        "cdn.tailwindcss.com",
        "cdnjs.cloudflare.com",
        "jsdelivr.net",
    ]

    for cdn in prohibited_cdns:
        assert cdn not in content, f"Found forbidden remote CDN '{cdn}' in static/index.html"

    # All script src and link href must be local /static/ paths
    script_sources = re.findall(r'<script\s+[^>]*src=["\']([^"\']+)["\']', content)
    for src in script_sources:
        assert src.startswith("/static/"), f"Script source '{src}' must point to local /static/"

    link_hrefs = re.findall(r'<link\s+[^>]*rel=["\']stylesheet["\'][^>]*href=["\']([^"\']+)["\']', content)
    for href in link_hrefs:
        assert href.startswith("/static/"), f"Stylesheet href '{href}' must point to local /static/"


def test_no_in_browser_babel_in_index_html():
    """Verify that in-browser Babel standalone runtime and text/babel tags are completely absent."""
    index_file = STATIC_DIR / "index.html"
    content = index_file.read_text(encoding="utf-8")

    assert "babel.min.js" not in content, "Found @babel/standalone in static/index.html"
    assert 'type="text/babel"' not in content, "Found type='text/babel' in static/index.html"
    assert "type='text/babel'" not in content, "Found type='text/babel' in static/index.html"


def test_precompiled_app_js_valid_and_features():
    """Verify static/app.js is precompiled, non-empty, and contains offline canvas and evidence panel."""
    app_js = STATIC_DIR / "app.js"
    assert app_js.exists(), "Missing static/app.js"
    assert app_js.stat().st_size >= 25_000, f"static/app.js is too small ({app_js.stat().st_size} bytes)"

    content = app_js.read_text(encoding="utf-8")

    # Verify offline plain canvas fallback component is present
    assert "OfflineCanvasFallback" in content, "Missing OfflineCanvasFallback in static/app.js"
    assert "getContext(\"2d\")" in content or "getContext('2d')" in content, "Missing canvas context logic in static/app.js"

    # Verify evidence catalog panel is present
    assert "EvidencePanel" in content, "Missing EvidencePanel in static/app.js"

    # Verify dynamic execution modes integration
    assert "execution_modes" in content, "Missing execution_modes integration in static/app.js"


def test_api_evidence_endpoint_and_modes(client):
    """Verify /api/evidence dynamically reads and returns execution_mode badges and evidence files."""
    res = client.get("/api/evidence")
    assert res.status_code == 200, f"Expected 200 from /api/evidence, got {res.status_code}"

    data = res.json()
    assert "execution_modes" in data
    assert "files" in data

    modes = data["execution_modes"]
    assert "quantum_calibration" in modes
    assert "opt_dispatch" in modes

    q_mode = modes["quantum_calibration"]
    assert q_mode["execution_mode"] in ["AUTHENTICATED_LOCAL", "LOCAL_SIMULATOR", "LOCAL_FALLBACK", "LIVE_QUANTUM"]
    assert isinstance(q_mode["fallback_reason"], str) and len(q_mode["fallback_reason"]) > 0

    opt_mode = modes["opt_dispatch"]
    assert opt_mode["execution_mode"] in ["AUTHENTICATED_LOCAL", "LOCAL_SIMULATOR", "LOCAL_FALLBACK", "LIVE_OPT", "LIVE_QPIAI_OPT"]
    if opt_mode["execution_mode"] == "LOCAL_FALLBACK":
        assert isinstance(opt_mode["fallback_reason"], str) and len(opt_mode["fallback_reason"]) > 0

    # Check evidence file list
    filenames = [f["name"] for f in data["files"]]
    expected_files = [
        "02_qcloud_calibration_metrics.json",
        "03_qpiai_opt_dispatch_solution.json",
        "04_monte_carlo.json",
        "05_noise_and_shots.json",
        "06_fairness_analysis.json",
        "04_pareto_frontier.png",
        "CLAIMS.md",
    ]
    for ef in expected_files:
        assert ef in filenames, f"Expected {ef} in /api/evidence file catalog"


def test_evidence_static_route_serving(client):
    """Verify static /evidence/ route directly serves JSON, PNG, and Markdown artifacts."""
    # Test JSON serving
    r_json = client.get("/evidence/02_qcloud_calibration_metrics.json")
    assert r_json.status_code == 200
    parsed = r_json.json()
    assert "execution_mode" in parsed

    # Test PNG serving
    r_png = client.get("/evidence/04_pareto_frontier.png")
    assert r_png.status_code == 200
    assert r_png.headers.get("content-type", "").startswith("image/png")

    # Test Markdown serving
    r_md = client.get("/evidence/CLAIMS.md")
    assert r_md.status_code == 200
    assert len(r_md.text) > 0
