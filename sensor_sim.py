import pandas as pd
import numpy as np
import time
from datetime import datetime

def generate_sensor_data():
    """
    Simulates real-time industrial sensor data for machine monitoring.
    Generates normal operating values with a 5% chance of creating anomalies.
    """
    # Normal operating data simulation
    # Mean temperature ~70°C, Mean vibration ~0.5
    temp = np.random.normal(70, 2, 1)[0]
    vibration = np.random.normal(0.5, 0.1, 1)[0]
    
    # Generate anomaly (failure) with 5% probability
    if np.random.random() > 0.95:
        temp += np.random.uniform(15, 30)       # Sudden overheating
        vibration += np.random.uniform(0.5, 1.5) # Severe vibration/shaking
        
    return {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "temperature": round(temp, 2),
        "vibration": round(vibration, 2),
        "machine_id": "CNC-01"
    }

# For testing purposes:
if __name__ == "__main__":
    print("🚀 Sensor data stream started (Press Ctrl+C to stop)...")
    try:
        for _ in range(10):
            print(generate_sensor_data())
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStream stopped by user.")