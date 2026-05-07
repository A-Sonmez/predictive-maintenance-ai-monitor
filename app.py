import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sensor_sim import generate_sensor_data
import time
import altair as alt

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Edge Monitoring",
    page_icon="🚨",
    layout="wide"
)

# Load Model with Caching for Performance and Stability
@st.cache_resource
def load_model():
    try:
        return joblib.load('anomaly_model.pkl')
    except:
        return None

model = load_model()

if model is None:
    st.error("❌ Model not found! Please run 'monitor.py' first to generate 'anomaly_model.pkl'.")
    st.stop()

# --- 2. HEADER AND SIDEBAR ---
st.title("🚨 Real-Time AI Machine Health Monitoring")
st.markdown("---")

st.sidebar.header("⚙️ Control Panel")
# Streamflow speed optimization
speed = st.sidebar.slider("Data Stream Speed (Seconds)", 0.01, 1.0, 0.1)

# Initialize Session State for Data History
if 'data_history' not in st.session_state:
    st.session_state.data_history = []

# --- 3. LAYOUT DESIGN ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Real-Time Metrics")
    metric_container = st.empty()

with col2:
    st.subheader("📈 Live Analysis Chart")
    chart_container = st.empty()

st.subheader("📋 System Logs & Records")
table_container = st.empty()

# --- 4. MAIN EXECUTION LOOP ---
while True:
    # Data extraction and AI inference
    raw_data = generate_sensor_data()
    features = np.array([[raw_data['temperature'], raw_data['vibration']]])
    
    # AI Prediction: 1 = Normal, -1 = Anomaly
    prediction = model.predict(features)[0]
    raw_data['status'] = "Normal" if prediction == 1 else "ANOMALY"
    
    # Update History (Keep last 50 records)
    st.session_state.data_history.append(raw_data)
    if len(st.session_state.data_history) > 50:
        st.session_state.data_history.pop(0)
    
    df = pd.DataFrame(st.session_state.data_history)

    # A. Update Metrics
    with metric_container.container():
        m1, m2 = st.columns(2)
        m1.metric("Temperature (°C)", f"{raw_data['temperature']}", 
                  delta="DANGER" if prediction == -1 else None, delta_color="inverse")
        m2.metric("Vibration Score", f"{raw_data['vibration']}", 
                  delta="CRITICAL" if prediction == -1 else None, delta_color="inverse")
        
        if prediction == -1:
            st.error(f"🚨 ANOMALY DETECTED: {raw_data['machine_id']} requires immediate inspection!")
        else:
            st.success("✅ System Status: Healthy")

    # B. Update Visualization
    with chart_container.container():
        line_chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X('timestamp:N', title='Time Axis'),
            y=alt.Y('temperature:Q', title='Temperature Value', scale=alt.Scale(domain=[50, 110])),
            color=alt.condition(
                alt.datum.status == 'ANOMALY',
                alt.value('red'),
                alt.value('#1f77b4')
            )
        ).properties(height=300)
        
        st.altair_chart(line_chart, use_container_width=True)

    # C. Update Data Table
    table_container.dataframe(df.iloc[::-1], use_container_width=True, hide_index=True)

    # Execution delay
    time.sleep(speed)