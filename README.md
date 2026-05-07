# 🚨 AI-Powered Predictive Maintenance System

<img width="1920" height="872" alt="Project Dashboard" src="https://github.com/user-attachments/assets/e4b5bf56-490f-4c24-92e0-88cce9d26667" />

### 🛠 Project Overview
This project is a real-time monitoring solution designed for industrial environments to predict machine failures. Using **Machine Learning (Isolation Forest)**, the system analyzes live sensor telemetry to identify behavioral anomalies before they lead to costly downtime.

### 🌟 Key Features
- **Real-Time Anomaly Detection:** Leverages Scikit-learn to spot outliers in temperature and vibration data.
- **Interactive Dashboard:** Built with **Streamlit** and **Altair** for high-performance visual data streaming.
- **Dockerized Infrastructure:** Fully containerized for easy deployment and consistent performance across environments.
- **Edge Simulation Engine:** A custom-built data generator that mimics real-world CNC machine behaviors and failure patterns.
- **Modern UI/UX:** Clean, dark-mode optimized interface with instant danger alerts.

### 🚀 Tech Stack
- **Language:** Python 3.9+
- **AI/ML:** Scikit-learn (Isolation Forest), NumPy, Joblib
- **Containerization:** Docker
- **Dashboard:** Streamlit
- **Visualization:** Altair

### 🐳 Running with Docker (Recommended)
This project is fully dockerized. Follow these steps to build and run the system:

1. **Build the Image:**
   docker build -t predictive-maintenance-ai .

2. **Run the Container:**
   docker run -d -p 8501:8501 --name ai_monitor predictive-maintenance-ai

3. **Initial Model Training:**
   Trigger the AI training inside the container:
   docker exec -it ai_monitor python monitor.py

4. **Access the Dashboard:**
   Open http://localhost:8501 in your browser.

### 📊 Local Setup (Without Docker)
1. **Install Dependencies:** pip install -r requirements.txt
2. **Model Training:** Run python monitor.py to train the AI on "healthy" machine patterns.
3. **Monitoring:** Launch the dashboard using streamlit run app.py to watch the AI detect anomalies in real-time.
