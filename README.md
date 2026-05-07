<img width="1920" height="872" alt="Kayıt 2026-05-07 111954 (1)" src="https://github.com/user-attachments/assets/e4b5bf56-490f-4c24-92e0-88cce9d26667" />

# 🚨 AI-Powered Predictive Maintenance System

### 🛠 Project Overview
This project is a real-time monitoring solution designed for industrial environments to predict machine failures. Using **Machine Learning (Isolation Forest)**, the system analyzes live sensor telemetry to identify behavioral anomalies before they lead to costly downtime.

### 🌟 Key Features
- **Real-Time Anomaly Detection:** Leverages Scikit-learn to spot outliers in temperature and vibration data.
- **Interactive Dashboard:** Built with **Streamlit** and **Altair** for high-performance visual data streaming.
- **Edge Simulation Engine:** A custom-built data generator that mimics real-world CNC machine behaviors and failure patterns.
- **Modern UI/UX:** Clean, dark-mode optimized interface with instant danger alerts.

### 🚀 Tech Stack
- **Language:** Python 3.x
- **AI/ML:** Scikit-learn (Isolation Forest), NumPy
- **Dashboard:** Streamlit
- **Visualization:** Altair (Declarative Statistical Visualization)
- **Deployment Ready:** Clean code structure for local or cloud environments

### 📊 How It Works
1. **Model Training:** Run `monitor.py` to train the AI on "healthy" machine patterns.
2. **Streaming:** The `sensor_sim.py` generates synthetic live data.
3. **Monitoring:** Launch the dashboard using `streamlit run app.py` to watch the AI detect anomalies in real-time.
