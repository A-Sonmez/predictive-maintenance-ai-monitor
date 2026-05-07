import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib # For saving and loading the model
from sensor_sim import generate_sensor_data # Data generator module
import time

# --- 1. MODEL TRAINING (With Simulated Normal Data) ---
def train_initial_model():
    """Learns the normal operating patterns of the machinery."""
    print("🤖 AI is learning the machine's normal operating patterns...")
    
    # Generate 500 samples of "healthy" machine data
    normal_data = []
    while len(normal_data) < 500:
        d = generate_sensor_data()
        # Ensure we only learn from normal ranges during training
        if d['temperature'] < 80 and d['vibration'] < 0.8:
            normal_data.append([d['temperature'], d['vibration']])
    
    X = np.array(normal_data)
    
    # Isolation Forest: Excellent for detecting outliers/anomalies
    # Contamination defines the expected proportion of outliers in the data
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)
    
    # Save the model to avoid retraining on every execution
    joblib.dump(model, 'anomaly_model.pkl')
    print("✅ Model trained and saved as 'anomaly_model.pkl'.")
    return model

# --- 2. LIVE MONITORING LOOP (Backend) ---
def start_monitoring():
    """Starts the real-time background monitoring process."""
    try:
        model = joblib.load('anomaly_model.pkl')
    except:
        model = train_initial_model()

    print("\n🔍 Live Monitoring Started... (Watching for anomalies)")
    print("-" * 60)

    while True:
        data = generate_sensor_data()
        features = np.array([[data['temperature'], data['vibration']]])
        
        # Prediction logic: 1 = Normal, -1 = Anomaly (Outlier)
        prediction = model.predict(features)[0]
        
        status = "✅ NORMAL"
        if prediction == -1:
            status = "🚨 ANOMALY DETECTED!"
            # Log triggering or alert logic can be integrated here
        
        print(f"[{data['timestamp']}] {data['machine_id']} | Temp: {data['temperature']}°C | Vib: {data['vibration']} | Status: {status}")
        
        time.sleep(1) # Real-time flow simulation

if __name__ == "__main__":
    start_monitoring()