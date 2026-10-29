"""
app.py
======
Bengaluru Fleet Dispatch Map - Streamlit Application
Interactive OpenStreetMap interface for Delay-Risk-Aware Contract Logistics.
Integrates QpiAI Quantum calibrated risk models and QpiAI Opt QUBO dispatch.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from streamlit_folium import st_folium

from src.geo_utils import (
    load_stops,
    load_vehicles,
    update_stop_coordinates,
    load_dispatch_solution,
    BENGALURU_DEPOT_LAT,
    BENGALURU_DEPOT_LNG
)
from src.routing_provider import OSRM_DISCLAIMER, get_cache_stats
from src.map_view import create_bengaluru_dispatch_map, VEHICLE_COLORS

# Set page configuration
st.set_page_config(
    page_title="Bengaluru Fleet Dispatch Map",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-num {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E293B;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .notice-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 1rem;
        font-size: 0.9rem;
        color: #92400E;
    }
    .osrm-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 1rem;
        font-size: 0.85rem;
        color: #1E293B;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">🚚 Bengaluru Fleet Dispatch Map</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Delay-Risk-Aware Fleet Dispatch for Contract Logistics | Quantum-calibrated P(late) & QUBO Optimization</div>',
    unsafe_allow_html=True
)

# Load data
stops_df, coord_status = load_stops()
vehicles_df = load_vehicles()
solution = load_dispatch_solution()

# Check for session state initialization
if "show_routes" not in st.session_state:
    st.session_state["show_routes"] = True

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🛠️ Dispatch & Filter Controls")

# Dispatch plan state toggle
dispatch_mode = st.sidebar.radio(
    "🗺️ Dispatch Display Mode:",
    options=["Optimized Routes (QpiAI Opt)", "Unassigned Stops Only (Pre-Optimization)"],
    index=0
)
show_routes = (dispatch_mode == "Optimized Routes (QpiAI Opt)")

# Vehicle Filter
vehicle_options = ["All Vehicles"] + list(vehicles_df["vehicle_id"].unique())
selected_vehicle = st.sidebar.selectbox("🚚 Filter by Vehicle:", vehicle_options, index=0)

# Risk Level Filter
risk_options = [
    "All Risk Levels",
    "High Risk (P >= 65%)",
    "Moderate Risk (45% <= P < 65%)",
    "Low Risk (P < 45%)"
]
selected_risk = st.sidebar.selectbox("⚠️ Filter by Delay Risk:", risk_options, index=0)

# Re-optimize or reload plan
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Optimization Actions")
if st.sidebar.button("🔄 Reload QpiAI Dispatch Plan", use_container_width=True):
    st.session_state["show_routes"] = True
    st.toast("Dispatch plan reloaded from QpiAI Opt solution evidence.")

cache_stats = get_cache_stats()
st.sidebar.markdown(f"**OSRM Route Cache:** `{cache_stats['cached_segments_count']} road segments cached`")

# --- TOP LEVEL SUMMARY METRICS ---
delivery_stops_count = len(stops_df[stops_df["stop_id"] != 0])
assigned_vehicles_count = len(solution.get("routes", [])) if solution else len(vehicles_df)

# Calculate at-risk stops and total exposure
high_risk_stops_count = 0
total_financial_exposure = 0.0
total_fleet_distance = 0.0

if solution and "routes" in solution:
    for r in solution["routes"]:
        total_fleet_distance += r.get("route_distance_km", 0.0)
        for t in r.get("timeline", []):
            p = t.get("qcloud_sla_breach_prob", 0.0)
            exp = t.get("expected_financial_exposure_inr", 0.0)
            total_financial_exposure += exp
            if p >= 0.50:
                high_risk_stops_count += 1

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total Deliveries", value=f"{delivery_stops_count} stops", delta="Commercial Nodes")
with col2:
    st.metric(label="Assigned Vehicles", value=f"{assigned_vehicles_count} active", delta="Heterogeneous Fleet")
with col3:
    st.metric(label="Deliveries at Risk (P ≥ 50%)", value=f"{high_risk_stops_count} / {delivery_stops_count}", delta=f"₹{total_financial_exposure:,.0f} Exposure", delta_color="inverse")
with col4:
    st.metric(label="Fleet Road Distance", value=f"{total_fleet_distance:.1f} km", delta="Bengaluru Road Network")

# --- COORDINATES & ROUTING NOTICES ---
if coord_status["is_all_simulated"]:
    st.markdown(
        f"""
        <div class="notice-box">
            <strong>⚠️ Coordinate Status:</strong> <code>data/stops.csv</code> contains <code>(x, y)</code> relative kilometer offsets 
            from Bengaluru Depot (12.9716°N, 77.5946°E). All 12 delivery stop locations are currently labeled as 
            <strong>[Simulated Data]</strong>. Real GPS coordinates can be entered in the Coordinate Management expander below.
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    f"""
    <div class="osrm-box">
        <strong>🌐 Road Network Routing:</strong> Powered by public OpenStreetMap OSRM routing service.
        <em>Notice: {OSRM_DISCLAIMER}</em>
    </div>
    """,
    unsafe_allow_html=True
)

# --- MAP VISUALIZATION ---
st.markdown("### 🗺️ Interactive Bengaluru Logistics Map")

# Build map
folium_map = create_bengaluru_dispatch_map(
    stops_df=stops_df,
    vehicles_df=vehicles_df,
    solution=solution if show_routes else None,
    selected_vehicle=selected_vehicle,
    risk_filter=selected_risk,
    show_routes=show_routes
)

# Render interactive map
st_folium(
    folium_map,
    width="100%",
    height=540,
    returned_objects=[]
)

# --- LOWER PANELS: ROUTE & DELIVERY DETAILS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Vehicle Route Timelines",
    "📊 Delivery Risk Matrix",
    "📍 Coordinate Management (Edit GPS)",
    "⚙️ Quantum & QUBO Solver Specs"
])

with tab1:
    st.subheader("🚚 Planned Vehicle Routes & Stop Sequences")
    if solution and "routes" in solution and show_routes:
        for route in solution["routes"]:
            v_id = route.get("vehicle_id")
            if selected_vehicle != "All Vehicles" and v_id != selected_vehicle:
                continue
                
            color = VEHICLE_COLORS.get(v_id, "#3B82F6")
            ordered_stops = route.get("ordered_stops", [])
            capacity = route.get("capacity_kg", 0)
            demand = route.get("assigned_demand_kg", 0)
            util = route.get("capacity_utilization", 0.0) * 100
            dist = route.get("route_distance_km", 0.0)
            fuel_cost = route.get("fuel_cost_inr", 0.0)
            driver_cost = route.get("driver_cost_inr", 0.0)
            exposure = route.get("expected_risk_exposure_inr", 0.0)
            total_cost = route.get("total_route_cost_inr", 0.0)

            with st.expander(f"🚚 **{v_id}** — Route: `{' → '.join(str(s) for s in ordered_stops)}` | Total Cost: ₹{total_cost:,.2f}", expanded=True):
                c_a, c_b, c_c, c_d = st.columns(4)
                c_a.metric("Capacity Load", f"{demand} / {capacity} kg", f"{util:.1f}% Utilized")
                c_b.metric("Route Distance", f"{dist:.2f} km", "Road Geometry")
                c_c.metric("Transport & Labor", f"₹{(fuel_cost + driver_cost):,.2f}", f"Fuel ₹{fuel_cost:,.0f} | Labor ₹{driver_cost:,.0f}")
                c_d.metric("Expected SLA Risk", f"₹{exposure:,.2f}", "Delay Exposure", delta_color="inverse")

                timeline_data = []
                for item in route.get("timeline", []):
                    p_late = item.get("qcloud_sla_breach_prob", 0.0)
                    risk_cat = "🔴 High Risk" if p_late >= 0.65 else ("🟡 Moderate Risk" if p_late >= 0.45 else "🟢 Low Risk")
                    timeline_data.append({
                        "Stop ID": item.get("stop_id"),
                        "Location Name": item.get("name"),
                        "Arrival ETA (min)": f"{item.get('arrival_time_min', 0.0):.1f}",
                        "Service Start (min)": f"{item.get('service_start_min', 0.0):.1f}",
                        "SLA Window (min)": f"[{item.get('sla_time_window', [0,0])[0]:.0f}, {item.get('sla_time_window', [0,0])[1]:.0f}]",
                        "P(late) Probability": f"{p_late*100:.1f}%",
                        "Risk Category": risk_cat,
                        "SLA Penalty (₹)": f"₹{item.get('contractual_penalty_inr', 0.0):,.0f}",
                        "Expected Exposure (₹)": f"₹{item.get('expected_financial_exposure_inr', 0.0):,.2f}"
                    })
                st.dataframe(pd.DataFrame(timeline_data), use_container_width=True)
    else:
        st.info("Optimizer has not run or 'Unassigned Stops' view is active. Toggle to 'Optimized Routes' to inspect vehicle dispatch timelines.")

with tab2:
    st.subheader("📊 Delivery Risk & SLA Compliance Matrix")
    matrix_rows = []
    
    # Map solution data
    meta_map = {}
    if solution and "routes" in solution:
        for r in solution["routes"]:
            v_id = r.get("vehicle_id")
            for t in r.get("timeline", []):
                meta_map[t.get("stop_id")] = {
                    "assigned_vehicle": v_id,
                    "arrival_min": t.get("arrival_time_min"),
                    "p_late": t.get("qcloud_sla_breach_prob"),
                    "exposure": t.get("expected_financial_exposure_inr")
                }
                
    for _, row in stops_df[stops_df["stop_id"] != 0].iterrows():
        s_id = int(row["stop_id"])
        meta = meta_map.get(s_id, {})
        p = meta.get("p_late", np.nan)
        exposure = meta.get("exposure", np.nan)
        matrix_rows.append({
            "Stop ID": s_id,
            "Name": row["name"],
            "Demand (kg)": row["demand_kg"],
            "Time Window": f"[{row['tw_start_min']} - {row['tw_end_min']} min]",
            "Service Time": f"{row['service_time_min']} min",
            "Contract Penalty (₹)": f"₹{row['sla_penalty_inr']:,.0f}",
            "Assigned Vehicle": meta.get("assigned_vehicle", "Unassigned"),
            "Arrival ETA": f"{meta.get('arrival_min', 0.0):.1f} min" if not pd.isna(meta.get("arrival_min")) else "-",
            "P(late)": f"{p*100:.1f}%" if not pd.isna(p) else "-",
            "Financial Exposure (₹)": f"₹{exposure:,.2f}" if not pd.isna(exposure) else "-",
            "Coordinate Source": row["coord_status"]
        })
    st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True)

with tab3:
    st.subheader("📍 Delivery Coordinates Management")
    st.markdown("""
    According to logistics requirements, records without explicit GPS coordinates are derived from relative $(x, y)$ km 
    offsets and labeled as **[Simulated Data]**. Use the form below to enter and verify real GPS coordinates for any delivery stop.
    """)
    
    stops_list = stops_df[stops_df["stop_id"] != 0]
    stop_selector = st.selectbox(
        "Select Stop to Inspect / Update Coordinates:",
        options=stops_list["stop_id"].tolist(),
        format_func=lambda sid: f"Stop #{sid}: {stops_list[stops_list['stop_id'] == sid]['name'].values[0]}"
    )
    
    current_row = stops_df[stops_df["stop_id"] == stop_selector].iloc[0]
    
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        new_lat = st.number_input(
            "Latitude (°N):",
            value=float(current_row["latitude"]),
            format="%.6f",
            min_value=12.0,
            max_value=14.0
        )
    with col_e2:
        new_lng = st.number_input(
            "Longitude (°E):",
            value=float(current_row["longitude"]),
            format="%.6f",
            min_value=76.0,
            max_value=79.0
        )
    with col_e3:
        st.write("")
        st.write("")
        if st.button("💾 Save Coordinates to data/stops.csv", use_container_width=True):
            success = update_stop_coordinates(stop_selector, new_lat, new_lng)
            if success:
                st.success(f"Successfully updated coordinates for Stop #{stop_selector} ({new_lat:.5f}°N, {new_lng:.5f}°E) in `data/stops.csv`!")
                st.rerun()
            else:
                st.error("Failed to update coordinates.")

    st.markdown("#### Current Stops Coordinate Table")
    coords_view = stops_df[["stop_id", "name", "latitude", "longitude", "coord_status"]]
    st.dataframe(coords_view, use_container_width=True)

with tab4:
    st.subheader("⚙️ QpiAI Quantum & QUBO Optimizer Details")
    if solution:
        sol_info = solution.get("solver_info", {})
        dispatch_sum = solution.get("dispatch_summary", {})
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Solver Formulation & Engine")
            st.write(f"- **Framework:** `{solution.get('optimization_framework')}`")
            st.write(f"- **Engine:** `{sol_info.get('solver_engine')}`")
            st.write(f"- **Formulation:** `{sol_info.get('problem_formulation')}`")
            st.write(f"- **QUBO Decision Variables:** `{sol_info.get('qubo_variables_count')}` variables")
            st.write(f"- **Quadratic Couplings:** `{sol_info.get('qubo_quadratic_couplings_count')}` terms")
            st.write(f"- **Runtime:** `{sol_info.get('solver_runtime_seconds'):.4f}` seconds")
            st.write(f"- **Status:** `{sol_info.get('solution_status')}`")
        with c2:
            st.markdown("#### Cost Objective Decomposition")
            st.write(f"- **Transport Fuel Cost:** `₹{dispatch_sum.get('total_transport_cost_inr', 0.0):,.2f}`")
            st.write(f"- **Driver Labor Shift Cost:** `₹{dispatch_sum.get('total_driver_cost_inr', 0.0):,.2f}`")
            st.write(f"- **Total Expected SLA Penalty Exposure:** `₹{dispatch_sum.get('total_expected_sla_risk_inr', 0.0):,.2f}`")
            st.write(f"- **Grand Total Objective Energy:** `₹{dispatch_sum.get('grand_total_dispatch_cost_inr', 0.0):,.2f}`")
            st.write(f"- **Stops Coverage Rate:** `{dispatch_sum.get('stop_coverage_rate', 1.0)*100:.1f}%` ({dispatch_sum.get('total_stops_covered')}/12 stops)")
