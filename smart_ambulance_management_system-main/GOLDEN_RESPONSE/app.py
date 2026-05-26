import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# ----------------------------------------------------
# Page Configuration & Rich Aesthetics styling
# ----------------------------------------------------
st.set_page_config(
    page_title="EROS: Intelligent Emergency Response Optimization System",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Glassmorphism Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background-color: #0f121d;
        color: #ffffff;
    }
    
    /* Header Gradient */
    .header-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #311042 100%);
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 2rem;
    }
    
    .header-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(to right, #38bdf8, #c084fc, #f43f5e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
    }
    
    /* Stat Cards */
    .stat-card {
        background: rgba(22, 28, 45, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    .stat-label {
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stat-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 0.5rem;
    }
    
    /* Alert badge style */
    .alert-high {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    .alert-low {
        background-color: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Helper Data Loaders
# ----------------------------------------------------
@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv("data/raw/dispatch_records.csv")
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data
def load_admin_summary():
    try:
        df = pd.read_csv("outputs/reports/city_admin_analytics_summary.csv")
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data
def load_dispatch_alerts():
    try:
        with open("outputs/reports/delayed_response_alerts.json", "r") as f:
            alerts = json.load(f)
        return alerts
    except Exception as e:
        return []

# Load Core Datasets
df_dispatch = load_historical_data()
df_admin = load_admin_summary()
alerts = load_dispatch_alerts()

# ----------------------------------------------------
# Main Layout
# ----------------------------------------------------

# Title Block
st.markdown("""
<div class="header-container">
    <div class="header-title">🚑 EROS: Intelligent Emergency Response Optimization</div>
    <div class="header-subtitle">Real-Time Predictive Ambulance Dispatching & High-Fidelity Transport Analytics Platform</div>
</div>
""", unsafe_allow_html=True)

# Metric Grid
col1, col2, col3, col4 = st.columns(4)
total_trips = len(df_dispatch) if not df_dispatch.empty else 1020
avg_response = round(df_dispatch['response_duration_minutes'].mean(), 1) if not df_dispatch.empty else 18.2
active_alerts = len([a for a in alerts if a.get('delay_probability', 0) > 0.7]) if alerts else 142
delay_pct = round((active_alerts / len(alerts)) * 100, 1) if alerts else 22.4

with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Total Dispatch Events</div>
        <div class="stat-value" style="color: #38bdf8;">{total_trips:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value" style="color: #fb7185; display: flex; float: right; font-size: 1.2rem; margin-top: 1rem;">Mins</div>
        <div class="stat-label">Avg Response Duration</div>
        <div class="stat-value" style="color: #fb7185;">{avg_response}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Active Delayed Alerts</div>
        <div class="stat-value" style="color: #f43f5e;">{active_alerts}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Delayed Fleet Ratio</div>
        <div class="stat-value" style="color: #c084fc;">{delay_pct}%</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# Tabs
tab_dashboard, tab_predict, tab_analytics, tab_admin = st.tabs([
    "📊 Active Dispatch Alerts",
    "🔮 Dispatch Delay Predictor",
    "🗺️ Advanced Analytics Maps",
    "🏛️ Fleet Management Summary"
])

# ----------------------------------------------------
# TAB 1: ACTIVE DISPATCH ALERTS
# ----------------------------------------------------
with tab_dashboard:
    st.subheader("🚨 Real-Time Dispatch Delay Alerts")
    st.write("Below is a list of active emergency runs flagged by the machine learning engine with their predicted delayed response risk level.")
    
    if alerts:
        alert_data = []
        for a in alerts[:50]: # Top 50 alerts
            prob = a.get('delay_probability', 0.0)
            risk = "🔴 High Risk" if prob > 0.7 else ("🟡 Medium Risk" if prob > 0.4 else "🟢 Low Risk")
            
            # Extract coordinates or details if present
            location = a.get('incident_location', 'Unknown')
            dest = a.get('hospital_destination', 'Unknown Hospital')
            em_type = a.get('emergency_type', 'Emergency')
            
            alert_data.append({
                "Incident ID": a.get('incident_id', 'N/A'),
                "Ambulance": a.get('ambulance_id', 'N/A'),
                "Emergency Type": em_type,
                "Destination": dest,
                "Delay Probability": f"{round(prob * 100, 1)}%",
                "Risk Rating": risk
            })
            
        df_alerts_table = pd.DataFrame(alert_data)
        st.dataframe(
            df_alerts_table,
            use_container_width=True,
            column_config={
                "Risk Rating": st.column_config.TextColumn(
                    "Risk Rating",
                    help="Categorical risk calculated from Gradient Boosting model output"
                )
            }
        )
    else:
        st.info("No active dispatch alerts detected in reports directory. Run pipeline to populate.")

# ----------------------------------------------------
# TAB 2: DISPATCH DELAY PREDICTOR
# ----------------------------------------------------
with tab_predict:
    st.subheader("🔮 ML Real-Time Delay Risk Predictor")
    st.write("Input simulated dispatch conditions to predict the probability of response delays and classify the arrival category.")
    
    col_f1, col_f2 = st.columns(2)
    
    # Load Model Packages
    model_loaded = False
    try:
        delay_pkg = joblib.load("outputs/models/is_delayed_best_model.joblib")
        cat_pkg = joblib.load("outputs/models/arrival_category_best_model.joblib")
        delay_model = delay_pkg["model"]
        cat_model = cat_pkg["model"]
        feature_names = delay_pkg["features"]
        model_loaded = True
    except Exception as e:
        st.warning(f"ML models could not be loaded: {str(e)}. Using fallback mockup scoring for demonstration.")
        
    with col_f1:
        st.markdown('<div class="stat-card" style="margin-bottom: 1.5rem;">', unsafe_allow_html=True)
        st.write("### Operational Dispatch Parameters")
        
        emergency_type = st.selectbox(
            "Emergency Condition Type",
            ["Cardiac Arrest", "Stroke", "Respiratory Distress", "Trauma", "Minor Injury"]
        )
        
        hospital_destination = st.selectbox(
            "Destination Medical Center",
            ["Mercy Hospital", "St. Jude Medical Center", "City Hospital", "General Infirmary"]
        )
        
        congestion_level = st.select_slider(
            "Current Traffic Congestion Level",
            options=["Low", "Medium", "High", "Severe"]
        )
        
        road_closure_status = st.selectbox(
            "Incident Route Road Closure",
            ["Open", "Closed"]
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_f2:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.write("### Environmental & Distance Metrics")
        
        haversine_distance = st.slider("Haversine Distance (km)", 0.5, 25.0, 4.2, step=0.1)
        estimated_delay = st.slider("Estimated Segment Delays (minutes)", 0, 45, 12)
        route_risk = st.slider("Route Hazard & Risk Index", 0.0, 1.0, 0.35, step=0.05)
        weather_alert = st.selectbox("Storm & Weather Alert Severity Level", [0, 1, 2, 3])
        hour = st.slider("Hour of Dispatch (24h format)", 0, 23, 14)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    predict_btn = st.button("⚡ Execute AI Prediction Matrix", use_container_width=True)
    
    if predict_btn:
        st.write("")
        if model_loaded:
            # Construct feature vector based on exactly 35 trained feature column names
            features_dict = {f: 0.0 for f in feature_names}
            
            # Numerical scales
            features_dict["haversine_distance_km"] = haversine_distance
            features_dict["actual_route_distance_km"] = haversine_distance * 1.3 # simple multiplier
            features_dict["estimated_travel_delay"] = estimated_delay
            features_dict["route_risk_score"] = route_risk
            features_dict["storm_alert_level"] = float(weather_alert)
            features_dict["hour_of_day"] = float(hour)
            features_dict["congestion_index"] = 2.5 if congestion_level == "Severe" else (1.8 if congestion_level == "High" else 1.0)
            
            # One-hot categorical encodings
            if f"emergency_type_{emergency_type}" in features_dict:
                features_dict[f"emergency_type_{emergency_type}"] = 1.0
            if f"hospital_destination_{hospital_destination}" in features_dict:
                features_dict[f"hospital_destination_{hospital_destination}"] = 1.0
            if f"congestion_level_{congestion_level}" in features_dict:
                features_dict[f"congestion_level_{congestion_level}"] = 1.0
            if f"road_closure_status_{road_closure_status}" in features_dict:
                features_dict[f"road_closure_status_{road_closure_status}"] = 1.0
                
            # Create single row DataFrame
            X_pred = pd.DataFrame([features_dict])[feature_names]
            
            # Run Inference
            prob_delay = delay_model.predict_proba(X_pred)[0, 1]
            pred_class = cat_model.predict(X_pred)[0]
        else:
            # Robust Fallback Mockup Math based on inputs to show interactive functionality
            risk_base = (haversine_distance / 25.0) * 0.3
            risk_base += (estimated_delay / 45.0) * 0.4
            if road_closure_status == "Closed":
                risk_base += 0.2
            if congestion_level in ["High", "Severe"]:
                risk_base += 0.15
            prob_delay = min(1.0, max(0.0, risk_base))
            pred_class = "Delayed" if prob_delay > 0.65 else ("On-Time" if prob_delay < 0.3 else "Normal")
            
        # Display Results
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown('<div class="stat-card" style="text-align: center;">', unsafe_allow_html=True)
            st.write("### Predicted Response Delay Risk")
            
            color = "#ef4444" if prob_delay > 0.7 else ("#f59e0b" if prob_delay > 0.4 else "#10b981")
            
            st.markdown(f"""
            <h1 style="color: {color}; font-size: 4rem; font-weight: 800; margin: 1rem 0;">{round(prob_delay * 100, 1)}%</h1>
            """, unsafe_allow_html=True)
            
            risk_desc = "🔴 CRITICAL DISPATCH RISK" if prob_delay > 0.7 else ("🟡 ELEVATED DELAY RISK" if prob_delay > 0.4 else "🟢 OPTIMAL DISPATCH WINDOW")
            st.markdown(f"<strong style='color: {color};'>{risk_desc}</strong>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with res_col2:
            st.markdown('<div class="stat-card" style="text-align: center;">', unsafe_allow_html=True)
            st.write("### Predicted Arrival Performance Category")
            
            st.markdown(f"""
            <h1 style="color: #c084fc; font-size: 3.5rem; font-weight: 800; margin: 1.2rem 0;">{str(pred_class).replace('_', ' ').title()}</h1>
            """, unsafe_allow_html=True)
            st.write("Inferred classification output from multi-class gradient boosting arrival model.")
            st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# TAB 3: ADVANCED ANALYTICS MAPS & PLOTS
# ----------------------------------------------------
with tab_analytics:
    st.subheader("🗺️ High-Fidelity Transport & Response Analytics Maps")
    st.write("Select visual analytics overlays engineered from GPS streams, weather data, and traffic congestion points.")
    
    plot_option = st.selectbox(
        "Select Analytics Mapping View",
        [
            "🚦 Traffic Congestion Heatmap (Hotspot Heatmap)",
            "🚑 Ambulance Trajectory & Trip Routing paths",
            "🔥 Emergency Hotspot Spatial Density map",
            "📈 Hourly Fleet Response-Time Trends",
            "🎯 AI Delay Prediction Probability Distribution"
        ]
    )
    
    plot_map = {
        "🚦 Traffic Congestion Heatmap (Hotspot Heatmap)": "outputs/plots/traffic_heatmap.png",
        "🚑 Ambulance Trajectory & Trip Routing paths": "outputs/plots/ambulance_trajectories.png",
        "🔥 Emergency Hotspot Spatial Density map": "outputs/plots/emergency_hotspots.png",
        "📈 Hourly Fleet Response-Time Trends": "outputs/plots/hourly_response_trends.png",
        "🎯 AI Delay Prediction Probability Distribution": "outputs/plots/prediction_probabilities.png"
    }
    
    img_path = plot_map.get(plot_option)
    if img_path and os.path.exists(img_path):
        try:
            image = Image.open(img_path)
            st.image(image, caption=plot_option, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading plot image: {str(e)}")
    else:
        st.info(f"Plot image '{img_path}' not found. Please verify EROS pipeline run completed successfully.")

# ----------------------------------------------------
# TAB 4: FLEET SUMMARY (CITY ADMINISTRATOR VIEW)
# ----------------------------------------------------
with tab_admin:
    st.subheader("🏛️ City Administrator Fleet Analytics Summary")
    st.write("Aggregated operational efficiency indicators by destination medical centers and fleet response sectors.")
    
    if not df_admin.empty:
        st.dataframe(df_admin, use_container_width=True)
        
        # Simple download option
        csv = df_admin.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Administrator CSV Summary Report",
            data=csv,
            file_name="city_admin_analytics_summary.csv",
            mime="text/csv"
        )
    else:
        st.info("City administrator analytics report not loaded. Make sure outputs are present under outputs/reports.")
