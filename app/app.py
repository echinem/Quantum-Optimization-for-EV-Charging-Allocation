import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import pathlib
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
from datetime import datetime
from qaoa_solver import qaoa_solver
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

DATA_DIR = pathlib.Path("../data")
EV_FILE = DATA_DIR/ "evs.csv"
STATION_FILE = DATA_DIR /"stations.csv"
CIRCUIT_IMAGE = DATA_DIR / "qaoa_circuit.png"

def load_result(method: str):
    path = DATA_DIR / f"result_{method}.json"
    if not path.exists():
        st.warning(f"No result file for method `{method}`. Run the optimizer first.")
        return None
    with open(path) as f:
        return json.load(f)

def load_assignments(result_json):
    """Convert the dict keys back into (ev, station, slot) tuples."""
    assignment = {}
    for key, val in result_json["assignment"].items():
        parsed = eval(key)
        if isinstance(parsed, tuple) and len(parsed) == 3:
            # anneal/dwave/qaoa style
            ev, st, sl = parsed
            assignment[ev] = (st, sl)
        else:
            # greedy style: key is just EV index, value is (station, slot)
            ev = parsed
            st, sl = val
            assignment[ev] = (st, sl)
    return assignment

@st.cache_data
def load_data():
    evs = pd.read_csv(EV_FILE, parse_dates=["arrival_time"])
    stations = pd.read_csv(STATION_FILE)
    return evs, stations

evs, stations = load_data()
st.set_page_config(
    page_title="EV Charging Optimizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("EV-Charging Optimiser")
st.sidebar.markdown("---")
method_options = ["greedy", "anneal", "dwave", "qaoa"]

#Add unique keys to avoid duplicate element IDs
method_a = st.sidebar.selectbox(
    "Method A (left)",
    method_options,
    index=0,
    key="sidebar_method_a"
)

method_b = st.sidebar.selectbox(
    "Method B (right)",
    method_options,
    index=2,
    key="sidebar_method_b"
)
st.sidebar.markdown("---")
st.sidebar.info("Select 'qaoa' to view the quantum circuit visualization")

res_a = load_result(method_a)
res_b = load_result(method_b)

if not (res_a and res_b):
    st.stop()

assign_a = load_assignments(res_a)
assign_b = load_assignments(res_b)
st.title("Quantum EV Charging Optimization Dashboard")
st.markdown("Comparing classical and quantum approaches to EV charging station assignment")
st.markdown("---")
if method_a == "qaoa" or method_b == "qaoa":
    st.markdown("## QAOA Quantum Circuit")
    
    # Check if circuit image exists
    if CIRCUIT_IMAGE.exists():
        try:
            # Load and display the circuit image
            circuit_img = Image.open(CIRCUIT_IMAGE)
            
            # Create columns for better layout
            col_img1, col_img2, col_img3 = st.columns([1, 10, 1])
            
            with col_img2:
                st.image(circuit_img, 
                        caption="QAOA Circuit: The quantum circuit used for optimization",
                        width='stretch')
            
            # Add expandable explanation
            with st.expander("Understanding the QAOA Circuit"):
                st.markdown("""
                ### Circuit Structure
                
                The QAOA (Quantum Approximate Optimization Algorithm) circuit consists of:
                
                🟥 H (Red boxes, left): Creates quantum superposition - tries all 2^40 possibilities at once\n
                🟦 RZ (Blue boxes, "Cost"): Encodes your optimization (travel + wait + power) using parameter γ\n
                ⚫ ⊕ (Blue circles with dots): Creates entanglement - enforces "each EV at ONE station" constraint\n
                🟪 RX (Purple boxes, "Mix"): Explores different solutions using parameter β\n
                📊 Gray boxes (right): Measurements - collapses quantum state to get the answer\n
                | Gray dashed lines |: Just visual separators (Cost0, Mix0, Cost1, Mix1)\n
                The Two Layers:\n
                    Layer 0 (γ₀, β₀): First optimization attempt\n
                    Layer 1 (γ₁, β₁): Refinement with new parameters\n
                Numbers:\n
                    40 qubits total (5 EVs × 2 stations × 4 slots)\n
                    8 qubits shown (pattern repeats)\n
                    2 QAOA layers for better quality\n
                    4 parameters to optimize: γ₀, β₀, γ₁, β₁\n
                """)
            
            st.markdown("---")
            
        except Exception as e:
            st.error(f"Error loading circuit image: {e}")
            st.info("The circuit image file exists but could not be displayed.")
    else:
        st.warning("QAOA circuit image not found!")
        st.info("""
        **To generate the circuit image:**
        1. Run the QAOA optimizer: `python src/run_optimization.py --method qaoa`
        2. The circuit will be automatically saved as an image
        3. Refresh this page to see the visualization
        """)
        
        st.info("Run QAOA optimization to generate circuit visualization")
    
    st.markdown("---")

def compute_metrics(assignments):
    total_wait = 0.0  # minutes
    total_distance = 0.0  # km
    for ev_idx, (st_idx, slot) in assignments.items():
        ev = evs.iloc[ev_idx]
        stn = stations.iloc[st_idx]
        # travel distance
        dist = np.sqrt((ev.lat-stn.lat)**2 + (ev.lon-stn.lon)**2) * 111  # roughly km/deg
        total_distance += dist
        # waiting time (slot start - arrival)
        slot_start_min = slot * 15   # assuming 15‑min slots
        arrival_min = (ev.arrival_time - evs["arrival_time"].min()).total_seconds()/60
        wait = max(0, slot_start_min - arrival_min)
        total_wait += wait
    return {"total_wait_min": total_wait, "total_distance_km": total_distance,
            "energy": res_a["energy"] if assignments is assign_a else res_b["energy"]}

metrics_a = compute_metrics(assign_a)
metrics_b = compute_metrics(assign_b)

st.markdown("## Results Comparison")

col1, col2 = st.columns(2)
import openrouteservice
from openrouteservice import convert

def make_map(assignments, title, ors_api_key="eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjdiMDIyYmJiMzU3NjRkYzZiYjFmMmEzYmFjM2MzZDM2IiwiaCI6Im11cm11cjY0In0="):
    client = openrouteservice.Client(key=ors_api_key)

    point_rows = []
    route_rows = []
    label_rows = []
    for ev_idx, (st_idx, slot) in assignments.items():
        ev = evs.iloc[ev_idx]
        stn = stations.iloc[st_idx]
        point_rows.append({
            "type": "EV",
            "id": ev_idx,
            "lat": ev.lat,
            "lon": ev.lon,
            "lat_formatted": f"{ev.lat:.5f}",  
            "lon_formatted": f"{ev.lon:.5f}",
            "color": [0, 0, 255],
            "size": 30
        })
        label_rows.append({
            "text": f"{ev_idx}",
            "lat": ev.lat,
            "lon": ev.lon,
            "color": [255, 255, 255, 255],  # White text
            "size": 11,
            "background_color": [0, 100, 255, 230]  # Blue background
        })
        point_rows.append({
            "type": "Station",
            "id": st_idx,
            "lat": stn.lat,
            "lon": stn.lon,
            "lat_formatted": f"{stn.lat:.5f}",  
            "lon_formatted": f"{stn.lon:.5f}",
            "color": [255, 0, 0],
            "size": 60
        })
        label_rows.append({
            "text": f"{st_idx}",
            "lat": stn.lat,
            "lon": stn.lon,
            "color": [255, 255, 255, 255],  # White text
            "size": 13,
            "background_color": [255, 60, 60, 230]  # Red background
        })
        try:
            route = client.directions(
                coordinates=[(ev.lon, ev.lat), (stn.lon, stn.lat)],
                profile='driving-car',
                format='geojson'
            )

            coords = route['features'][0]['geometry']['coordinates']

            route_rows.append({
                "path": coords,
                "color": [50, 200, 50, 180],
                "width": 4
            })

        except Exception as e:
            st.warning(f"Route failed for EV {ev_idx} → Station {st_idx}: {e}")


    df_points = pd.DataFrame(point_rows)
    df_routes = pd.DataFrame(route_rows)
    df_labels = pd.DataFrame(label_rows)

    view_state = pdk.ViewState(
        latitude=df_points["lat"].mean(),
        longitude=df_points["lon"].mean(),
        zoom=12,
        pitch=0
    )

    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_points,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius="size",
        pickable=True,
        auto_highlight=True,
        opacity=0.85
    )
    route_layer = pdk.Layer(
        "PathLayer",
        data=df_routes,
        get_path="path",
        get_color="color",
        get_width="width",
        width_scale=1,
        width_min_pixels=2,
        pickable=False
    )
    label_layer = pdk.Layer(
            "TextLayer",
            data=df_labels,
            get_position="[lon, lat]",
            get_text="text",
            get_color="color",
            get_size="size",
            get_alignment_baseline="'center'",
            get_text_anchor="'middle'",
            billboard=True,
            font_family="'Segoe UI', 'Arial', sans-serif",
            font_weight="bold",
            background=True,
            get_background_color="background_color",
            background_padding=[4, 2, 4, 2]
        )
    deck = pdk.Deck(layers=[point_layer,route_layer,label_layer], initial_view_state=view_state, tooltip={ "html": """
                <div style="font-family: Arial; background: rgba(0,0,0,0.8); 
                            color: white; padding: 10px; border-radius: 5px;">
                    <b style="font-size: 14px;">{type} {id}</b><br/>
                    <hr style="margin: 5px 0; border-color: rgba(255,255,255,0.3);">
                    <span style="font-size: 11px;">
                        Lat: {lat_formatted}<br/>
                        Lon: {lon_formatted}
                    </span>
                </div>
            """,
            "style": {
                "fontSize": "12px"
            }})
    st.subheader(title)
    st.pydeck_chart(deck)

with col1:
    make_map(assign_a, f"Method **{method_a.upper()}**")
    
    # Metrics in a nice card-like format
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Total Waiting", f"{metrics_a['total_wait_min']:.1f} min")
    with metric_col2:
        st.metric("Travel Distance", f"{metrics_a['total_distance_km']:.1f} km")
    with metric_col3:
        energy_val = metrics_a['energy']
        if isinstance(energy_val, (int, float)):
            st.metric("QUBO Energy", f"{energy_val:.2f}")
        else:
            st.metric("QUBO Energy", str(energy_val))

with col2:
    make_map(assign_b, f"Method **{method_b.upper()}**")
    
    # Metrics in a nice card-like format
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Total Waiting", f"{metrics_b['total_wait_min']:.1f} min")
    with metric_col2:
        st.metric("Travel Distance", f"{metrics_b['total_distance_km']:.1f} km")
    with metric_col3:
        energy_val = metrics_b['energy']
        if isinstance(energy_val, (int, float)):
            st.metric("QUBO Energy", f"{energy_val:.2f}")
        else:
            st.metric("QUBO Energy", str(energy_val))
st.markdown("---")
st.markdown("##Performance Metrics Comparison")

comparison_df = pd.DataFrame({
    'Metric': ['Total Waiting Time', 'Total Distance', 'QUBO Energy'],
    method_a.upper(): [
        f"{metrics_a['total_wait_min']:.1f} min",
        f"{metrics_a['total_distance_km']:.1f} km",
        f"{metrics_a['energy']:.2f}" if isinstance(metrics_a['energy'], (int, float)) else "N/A"
    ],
    method_b.upper(): [
        f"{metrics_b['total_wait_min']:.1f} min",
        f"{metrics_b['total_distance_km']:.1f} km",
        f"{metrics_b['energy']:.2f}" if isinstance(metrics_b['energy'], (int, float)) else "N/A"
    ]
})

st.table(comparison_df)

# Winner determination
st.markdown("### Winner Analysis")
winner_cols = st.columns(3)

with winner_cols[0]:
    if metrics_a['total_wait_min'] < metrics_b['total_wait_min']:
        st.success(f"{method_a.upper()} has lower waiting time")
    elif metrics_a['total_wait_min'] > metrics_b['total_wait_min']:
        st.info(f"{method_b.upper()} has lower waiting time")
    else:
        st.info("Equal waiting time")

with winner_cols[1]:
    if metrics_a['total_distance_km'] < metrics_b['total_distance_km']:
        st.success(f"{method_a.upper()} has shorter distance")
    elif metrics_a['total_distance_km'] > metrics_b['total_distance_km']:
        st.info(f"{method_b.upper()} has shorter distance")
    else:
        st.info("Equal distance")

with winner_cols[2]:
    if isinstance(metrics_a['energy'], (int, float)) and isinstance(metrics_b['energy'], (int, float)):
        if metrics_a['energy'] < metrics_b['energy']:
            st.success(f"{method_a.upper()} has lower energy")
        elif metrics_a['energy'] > metrics_b['energy']:
            st.info(f"{method_b.upper()} has lower energy")
        else:
            st.info("Equal energy")

st.markdown("---")
with st.expander("Show Raw Assignment Tables"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"### {method_a.upper()} Assignments")
        st.dataframe(
            pd.DataFrame.from_dict(assign_a, orient='index', columns=["Station", "Slot"])
            .reset_index()
            .rename(columns={"index":"EV"}),
            width='stretch'
        )
    
    with col2:
        st.write(f"### {method_b.upper()} Assignments")
        st.dataframe(
            pd.DataFrame.from_dict(assign_b, orient='index', columns=["Station", "Slot"])
            .reset_index()
            .rename(columns={"index":"EV"}),
            width='stretch'
        )

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Quantum Computing for EV Optimization</p>
</div>
""", unsafe_allow_html=True)
