import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sensor_sim import generate_sensor_data
import time
import altair as alt

# --- 1. SAYFA AYARLARI ---
st.set_page_config(
    page_title="AI Edge Monitoring",
    page_icon="🚨",
    layout="wide"
)

# Modeli Önbelleğe Alarak Yükle (Hız ve Stabilite Sağlar)
@st.cache_resource
def load_model():
    try:
        return joblib.load('anomaly_model.pkl')
    except:
        return None

model = load_model()

if model is None:
    st.error("❌ Model bulunamadı! Lütfen önce 'monitor.py' dosyasını çalıştırarak 'anomaly_model.pkl' dosyasını oluşturun.")
    st.stop()

# --- 2. BAŞLIK VE SIDEBAR ---
st.title("🚨 Real-Time AI Machine Health Monitoring")
st.markdown("---")

st.sidebar.header("⚙️ Kontrol Paneli")
# Hız ayarını biraz daha optimize ettik
speed = st.sidebar.slider("Veri Akış Hızı (Saniye)", 0.01, 1.0, 0.1)

# Veri geçmişi için session state
if 'data_history' not in st.session_state:
    st.session_state.data_history = []

# --- 3. GÖRSEL YERLEŞİM ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Metrikler")
    metric_container = st.empty()

with col2:
    st.subheader("📈 Canlı Analiz Grafiği")
    chart_container = st.empty()

st.subheader("📋 Sistem Kayıtları")
table_container = st.empty()

# --- 4. ANA DÖNGÜ ---
while True:
    # Veri al ve tahmin et
    raw_data = generate_sensor_data()
    features = np.array([[raw_data['temperature'], raw_data['vibration']]])
    
    # AI Tahmini: 1 = Normal, -1 = Anomali
    prediction = model.predict(features)[0]
    raw_data['status'] = "Normal" if prediction == 1 else "ANOMALİ"
    
    # Listeye ekle ve son 50 kaydı tut
    st.session_state.data_history.append(raw_data)
    if len(st.session_state.data_history) > 50:
        st.session_state.data_history.pop(0)
    
    df = pd.DataFrame(st.session_state.data_history)

    # A. Metrikleri Güncelle
    with metric_container.container():
        m1, m2 = st.columns(2)
        m1.metric("Isı (°C)", f"{raw_data['temperature']}", 
                  delta="TEHLİKE" if prediction == -1 else None, delta_color="inverse")
        m2.metric("Titreşim", f"{raw_data['vibration']}", 
                  delta="KRİTİK" if prediction == -1 else None, delta_color="inverse")
        
        if prediction == -1:
            st.error(f"🚨 ANOMALİ: {raw_data['machine_id']} kontrol edilmeli!")
        else:
            st.success("✅ Sistem Sağlıklı")

    # B. Grafiği Güncelle (Hata veren width="container" yerine use_container_width kullanıldı)
    with chart_container.container():
        line_chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X('timestamp:N', title='Zaman'),
            y=alt.Y('temperature:Q', title='Sıcaklık', scale=alt.Scale(domain=[50, 110])),
            color=alt.condition(
                alt.datum.status == 'ANOMALİ',
                alt.value('red'),
                alt.value('#1f77b4')
            )
        ).properties(height=300)
        
        st.altair_chart(line_chart, use_container_width=True)

    # C. Tabloyu Güncelle
    table_container.dataframe(df.iloc[::-1], use_container_width=True, hide_index=True)

    # Hız ayarı
    time.sleep(speed)   