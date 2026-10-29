"""
src/map_view.py
===============
Interactive OpenStreetMap Folium visualization for Bengaluru Fleet Dispatch.
Renders depot, numbered delivery stops, road network polylines (via OSRM),
and delay-risk indicators with full attribution and interactive popups.
"""

import folium
from folium import plugins
import pandas as pd
from typing import Dict, Any, List, Optional
from src.routing_provider import get_full_route_for_stops, OSRM_DISCLAIMER
from src.geo_utils import BENGALURU_DEPOT_LAT, BENGALURU_DEPOT_LNG, get_stop_risk_badge

# Vehicle color mapping
VEHICLE_COLORS = {
    "V1_EcoVan": "#2563EB",       # Blue
    "V2_CityTruck": "#059669",    # Emerald Green
    "V3_HeavyHauler": "#D97706",  # Amber
    "V4_MaxFreight": "#7C3AED",   # Purple
    "Vehicle_A": "#2563EB",
    "Vehicle_B": "#059669",
    "default": "#4B5563"          # Slate Gray
}

def create_bengaluru_dispatch_map(
    stops_df: pd.DataFrame,
    vehicles_df: pd.DataFrame,
    solution: Optional[Dict[str, Any]] = None,
    selected_vehicle: str = "All Vehicles",
    risk_filter: str = "All Risk Levels",
    show_routes: bool = True
) -> folium.Map:
    """
    Creates an interactive Folium map centered on Bengaluru (12.9716°N, 77.5946°E).
    Uses standard OpenStreetMap tiles with attribution.
    """
    # Initialize Folium Map
    bengaluru_map = folium.Map(
        location=[BENGALURU_DEPOT_LAT, BENGALURU_DEPOT_LNG],
        zoom_start=11,
        tiles="OpenStreetMap",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    )

    # Depot data
    depot_row = stops_df[stops_df["stop_id"] == 0]
    depot_lat = float(depot_row["latitude"].values[0]) if not depot_row.empty else BENGALURU_DEPOT_LAT
    depot_lng = float(depot_row["longitude"].values[0]) if not depot_row.empty else BENGALURU_DEPOT_LNG
    depot_name = str(depot_row["name"].values[0]) if not depot_row.empty else "Bengaluru Central Depot"
    is_depot_sim = bool(depot_row["is_simulated"].values[0]) if not depot_row.empty and "is_simulated" in depot_row.columns else True

    # 1. Distinct Central Depot Marker
    depot_popup_html = f"""
    <div style="font-family: sans-serif; width: 230px; line-height: 1.4;">
        <h4 style="margin: 0 0 6px 0; color: #1E293B; border-bottom: 2px solid #3B82F6; padding-bottom: 4px;">
            🏢 {depot_name}
        </h4>
        <p style="margin: 3px 0; font-size: 12px;"><strong>Node ID:</strong> 0 (Central Hub)</p>
        <p style="margin: 3px 0; font-size: 12px;"><strong>GPS:</strong> {depot_lat:.5f}°N, {depot_lng:.5f}°E</p>
        <p style="margin: 3px 0; font-size: 12px;"><strong>Status:</strong> {'Simulated Depot Reference' if is_depot_sim else 'Verified GPS'}</p>
        <p style="margin: 3px 0; font-size: 12px; color: #475569;">Fleet origin & return terminal for all delivery vehicles.</p>
    </div>
    """
    folium.Marker(
        location=[depot_lat, depot_lng],
        tooltip="🏢 Central Fleet Depot (Bengaluru)",
        popup=folium.Popup(depot_popup_html, max_width=300),
        icon=folium.Icon(color="black", icon="home", prefix="fa")
    ).add_to(bengaluru_map)

    # Add a glowing circle indicator around Depot
    folium.Circle(
        location=[depot_lat, depot_lng],
        radius=650,
        color="#1E293B",
        weight=2,
        fill=True,
        fill_color="#3B82F6",
        fill_opacity=0.15
    ).add_to(bengaluru_map)

    # Map stop metadata from solution timelines
    stop_solution_meta = {}
    if solution and "routes" in solution:
        for route in solution["routes"]:
            v_id = route.get("vehicle_id", "Unknown")
            for order_idx, item in enumerate(route.get("timeline", [])):
                s_id = item.get("stop_id")
                stop_solution_meta[s_id] = {
                    "assigned_vehicle": v_id,
                    "stop_order": order_idx + 1,
                    "arrival_min": item.get("arrival_time_min", 0.0),
                    "sla_window": item.get("sla_time_window", [0.0, 0.0]),
                    "breach_prob": item.get("qcloud_sla_breach_prob", 0.0),
                    "penalty_inr": item.get("contractual_penalty_inr", 0.0),
                    "exposure_inr": item.get("expected_financial_exposure_inr", 0.0)
                }

    # 2. Add Road Network Routes if enabled & available
    route_summary_list = []
    if show_routes and solution and "routes" in solution:
        for route in solution["routes"]:
            v_id = route.get("vehicle_id", "Unknown")
            if selected_vehicle != "All Vehicles" and v_id != selected_vehicle:
                continue

            v_color = VEHICLE_COLORS.get(v_id, VEHICLE_COLORS["default"])
            ordered_stops = route.get("ordered_stops", [])
            
            # Extract coordinates for this sequence
            coords_seq = []
            for s_id in ordered_stops:
                matched = stops_df[stops_df["stop_id"] == s_id]
                if not matched.empty:
                    coords_seq.append((float(matched["latitude"].values[0]), float(matched["longitude"].values[0])))

            if len(coords_seq) >= 2:
                # Query road network polyline from OSRM provider
                route_geo = get_full_route_for_stops(coords_seq)
                polyline_points = route_geo["coordinates"]
                road_dist_km = route_geo["distance_km"]
                road_dur_min = route_geo["duration_min"]
                route_source = route_geo["source"]

                route_summary_list.append({
                    "vehicle_id": v_id,
                    "road_dist_km": road_dist_km,
                    "road_dur_min": road_dur_min,
                    "stops_count": len(ordered_stops) - 2,
                    "source": route_source
                })

                # Route popup
                route_popup_html = f"""
                <div style="font-family: sans-serif; width: 240px;">
                    <h4 style="margin: 0 0 5px 0; color: {v_color};">🚚 Route: {v_id}</h4>
                    <p style="margin: 2px 0; font-size: 12px;"><strong>Stops:</strong> {len(ordered_stops) - 2} deliveries</p>
                    <p style="margin: 2px 0; font-size: 12px;"><strong>Road Distance:</strong> {road_dist_km:.2f} km</p>
                    <p style="margin: 2px 0; font-size: 12px;"><strong>Travel Duration:</strong> {road_dur_min:.1f} mins</p>
                    <p style="margin: 2px 0; font-size: 12px;"><strong>Routing Source:</strong> {route_source}</p>
                    <p style="margin: 4px 0 0 0; font-size: 10px; color: #64748B;"><em>{OSRM_DISCLAIMER}</em></p>
                </div>
                """

                folium.PolyLine(
                    locations=polyline_points,
                    color=v_color,
                    weight=5,
                    opacity=0.85,
                    tooltip=f"Route for {v_id} ({road_dist_km:.1f} km via road network)",
                    popup=folium.Popup(route_popup_html, max_width=280)
                ).add_to(bengaluru_map)

    # 3. Add Delivery Stop Markers
    delivery_stops = stops_df[stops_df["stop_id"] != 0]

    for _, row in delivery_stops.iterrows():
        s_id = int(row["stop_id"])
        s_name = str(row["name"])
        lat = float(row["latitude"])
        lng = float(row["longitude"])
        demand = int(row.get("demand_kg", 0))
        tw_start = int(row.get("tw_start_min", 0))
        tw_end = int(row.get("tw_end_min", 0))
        penalty = float(row.get("sla_penalty_inr", 0.0))
        is_sim = bool(row.get("is_simulated", True))

        # Check solution metadata for this stop
        meta = stop_solution_meta.get(s_id, {})
        assigned_v = meta.get("assigned_vehicle", "Unassigned")
        stop_order = meta.get("stop_order", "-")
        arrival_min = meta.get("arrival_min", None)
        breach_prob = meta.get("breach_prob", 0.0)
        exposure = meta.get("exposure_inr", 0.0)
        risk_label, risk_color = get_stop_risk_badge(breach_prob) if arrival_min is not None else ("Unassigned", "#64748B")

        # Apply Filters
        if selected_vehicle != "All Vehicles" and assigned_v != selected_vehicle:
            continue

        if risk_filter != "All Risk Levels":
            if risk_filter == "High Risk (P >= 65%)" and breach_prob < 0.65:
                continue
            elif risk_filter == "Moderate Risk (45% <= P < 65%)" and not (0.45 <= breach_prob < 0.65):
                continue
            elif risk_filter == "Low Risk (P < 45%)" and (breach_prob >= 0.45 or arrival_min is None):
                continue

        # Coordinate status tag
        coord_tag = """<span style="background-color: #FEF3C7; color: #92400E; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: bold;">Simulated Data</span>""" if is_sim else """<span style="background-color: #D1FAE5; color: #065F46; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: bold;">Verified GPS</span>"""

        v_color = VEHICLE_COLORS.get(assigned_v, "#3B82F6")

        stop_popup_html = f"""
        <div style="font-family: sans-serif; width: 250px; line-height: 1.4;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <h4 style="margin: 0; color: #1E293B;">Stop #{s_id}: {s_name}</h4>
            </div>
            <div style="margin-bottom: 6px;">{coord_tag}</div>
            <div style="background-color: #F8FAFC; border-left: 3px solid {risk_color}; padding: 6px 8px; margin-bottom: 6px; border-radius: 2px;">
                <p style="margin: 2px 0; font-size: 12px;"><strong>Assigned Vehicle:</strong> <span style="color: {v_color}; font-weight: bold;">{assigned_v}</span> (Stop Order: #{stop_order})</p>
                <p style="margin: 2px 0; font-size: 12px;"><strong>Demand Cargo:</strong> {demand} kg</p>
                <p style="margin: 2px 0; font-size: 12px;"><strong>SLA Delivery Window:</strong> {tw_start} - {tw_end} min</p>
                <p style="margin: 2px 0; font-size: 12px;"><strong>Predicted Arrival ETA:</strong> {'{:.1f}'.format(arrival_min) + ' min' if arrival_min is not None else 'Pending dispatch'}</p>
            </div>
            <div style="border-top: 1px solid #E2E8F0; padding-top: 6px;">
                <p style="margin: 2px 0; font-size: 12px;"><strong>QCloud Late Prob P(late):</strong> <span style="color: {risk_color}; font-weight: bold;">{breach_prob*100:.1f}% ({risk_label})</span></p>
                <p style="margin: 2px 0; font-size: 12px;"><strong>Contractual Penalty:</strong> ₹{penalty:,.0f}</p>
                <p style="margin: 2px 0; font-size: 12px;"><strong>Financial Risk Exposure:</strong> <span style="color: #DC2626; font-weight: bold;">₹{exposure:,.2f}</span></p>
            </div>
            <p style="margin: 6px 0 0 0; font-size: 10px; color: #94A3B8;">GPS: {lat:.5f}°N, {lng:.5f}°E</p>
        </div>
        """

        # Choose marker color based on risk
        icon_color = "red" if breach_prob >= 0.65 else ("orange" if breach_prob >= 0.45 else "green")
        if arrival_min is None:
            icon_color = "blue"

        # Display stop marker with sequence badge
        marker_icon = folium.Icon(color=icon_color, icon="shopping-cart", prefix="fa")
        
        folium.Marker(
            location=[lat, lng],
            tooltip=f"Stop #{s_id}: {s_name} [{assigned_v} Order #{stop_order}] - {risk_label}",
            popup=folium.Popup(stop_popup_html, max_width=300),
            icon=marker_icon
        ).add_to(bengaluru_map)

    # Add OpenStreetMap Attribution notice in corner
    disclaimer_html = f"""
    <div style="position: fixed; 
                bottom: 12px; left: 12px; width: 340px; 
                background-color: rgba(255, 255, 255, 0.94); 
                padding: 8px 12px; border-radius: 6px; 
                box-shadow: 0 2px 6px rgba(0,0,0,0.25); 
                z-index: 9999; font-family: sans-serif; font-size: 11px; line-height: 1.3;">
        <strong style="color: #0F172A;">Bengaluru Logistics Road Network (OSM)</strong><br/>
        <span style="color: #475569;">Road geometry powered by OSRM API.</span><br/>
        <span style="color: #DC2626; font-size: 10px;">Notice: {OSRM_DISCLAIMER}</span>
    </div>
    """
    bengaluru_map.get_root().html.add_child(folium.Element(disclaimer_html))

    return bengaluru_map
