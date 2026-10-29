"""
src/routing_provider.py
=======================
Road routing integration using public OpenStreetMap OSRM API with local file caching.
Fetches actual road-network geometry across Bengaluru, Karnataka, India.
"""

import os
import json
import urllib.request
import urllib.error
import math
from typing import List, Tuple, Dict, Any

OSRM_DISCLAIMER = (
    "OSRM estimates provide basic geometric road durations and do not include live Bengaluru congestion."
)

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "osrm_cache.json")

def _load_cache() -> Dict[str, Any]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_cache(cache_data: Dict[str, Any]):
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write OSRM cache: {e}")

def get_cache_stats() -> Dict[str, Any]:
    cache = _load_cache()
    return {
        "cached_segments_count": len(cache),
        "cache_file": CACHE_FILE,
        "exists": os.path.exists(CACHE_FILE)
    }

def clear_cache():
    if os.path.exists(CACHE_FILE):
        try:
            os.remove(CACHE_FILE)
        except Exception:
            pass

def _haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_osrm_route_segment(
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    timeout_sec: float = 4.0
) -> Dict[str, Any]:
    cache = _load_cache()
    key = f"{start_lat:.5f},{start_lng:.5f}->{end_lat:.5f},{end_lng:.5f}"
    
    if key in cache:
        res = dict(cache[key])
        res["source"] = "osrm_cache"
        return res

    url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{start_lng:.6f},{start_lat:.6f};{end_lng:.6f},{end_lat:.6f}"
        f"?overview=full&geometries=geojson"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BengaluruFleetDispatch/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("code") == "Ok" and data.get("routes"):
                    route = data["routes"][0]
                    raw_coords = route.get("geometry", {}).get("coordinates", [])
                    folium_coords = [[pt[1], pt[0]] for pt in raw_coords]
                    distance_km = route.get("distance", 0.0) / 1000.0
                    duration_min = route.get("duration", 0.0) / 60.0
                    
                    result = {
                        "coordinates": folium_coords,
                        "distance_km": round(distance_km, 3),
                        "duration_min": round(duration_min, 1),
                        "source": "osrm_live",
                        "disclaimer": OSRM_DISCLAIMER
                    }
                    cache[key] = result
                    _save_cache(cache)
                    return result
    except Exception:
        pass

    # Straight-line fallback with intermediate points
    dist_km = _haversine_distance_km(start_lat, start_lng, end_lat, end_lng)
    duration_min = (dist_km / 30.0) * 60.0
    fallback_coords = [
        [start_lat + (end_lat - start_lat) * (i / 10.0), start_lng + (end_lng - start_lng) * (i / 10.0)]
        for i in range(11)
    ]
    return {
        "coordinates": fallback_coords,
        "distance_km": round(dist_km, 3),
        "duration_min": round(duration_min, 1),
        "source": "straight_line_fallback",
        "error": "OSRM service unreachable or timed out; displaying geometric interpolation.",
        "disclaimer": OSRM_DISCLAIMER
    }

def get_full_route_for_stops(
    ordered_coords: List[Tuple[float, float]],
    timeout_sec: float = 4.0
) -> Dict[str, Any]:
    if len(ordered_coords) < 2:
        return {
            "coordinates": ordered_coords,
            "distance_km": 0.0,
            "duration_min": 0.0,
            "segments": [],
            "source": "none",
            "disclaimer": OSRM_DISCLAIMER
        }

    all_coords = []
    total_dist = 0.0
    total_dur = 0.0
    segments = []
    has_fallback = False

    for i in range(len(ordered_coords) - 1):
        s_lat, s_lng = ordered_coords[i]
        e_lat, e_lng = ordered_coords[i+1]
        
        seg = get_osrm_route_segment(s_lat, s_lng, e_lat, e_lng, timeout_sec=timeout_sec)
        segments.append(seg)
        total_dist += seg["distance_km"]
        total_dur += seg["duration_min"]
        
        if seg["source"] == "straight_line_fallback":
            has_fallback = True

        if not all_coords:
            all_coords.extend(seg["coordinates"])
        else:
            all_coords.extend(seg["coordinates"][1:])

    return {
        "coordinates": all_coords,
        "distance_km": round(total_dist, 2),
        "duration_min": round(total_dur, 1),
        "segments": segments,
        "source": "fallback_mixed" if has_fallback else "osrm",
        "disclaimer": OSRM_DISCLAIMER
    }
