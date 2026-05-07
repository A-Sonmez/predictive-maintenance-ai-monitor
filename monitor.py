import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib # Modeli kaydetmek için
from sensor_sim import generate_sensor_data # Bir önceki yazdığın dosya
import time

# --- 1. MODEL EĞİTİMİ (Simüle Edilmiş Normal Veri İle) ---
def train_initial_model():
    print("🤖 Yapay zeka makinenin normal çalışma düzenini öğreniyor...")
    
    # Makinenin "sağlıklı" olduğu 500 adet örnek veri üretelim
    normal_data = []
    for _ in range(500):
        d = generate_sensor_data()
        # Eğitim sırasında anomali istemiyoruz, sadece normali öğrensin
        if d['temperature'] < 80 and d['vibration'] < 0.8:
            normal_data.append([d['temperature'], d['vibration']])
    
    X = np.array(normal_data)
    
    # Isolation Forest: Aykırı değerleri bulmak için birebir
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)
    
    # Modeli kaydedelim ki her seferinde eğitmek zorunda kalmayalım
    joblib.dump(model, 'anomaly_model.pkl')
    print("✅ Model eğitildi ve 'anomaly_model.pkl' olarak kaydedildi.")
    return model

# --- 2. CANLI İZLEME DÖNGÜSÜ ---
def start_monitoring():
    try:
        model = joblib.load('anomaly_model.pkl')
    except:
        model = train_initial_model()

    print("\n🔍 Canlı İzleme Başlatıldı... (Anomali bekleniyor)")
    print("-" * 50)

    while True:
        data = generate_sensor_data()
        features = np.array([[data['temperature'], data['vibration']]])
        
        # Tahmin yap: 1 = Normal, -1 = Anomali (Hata)
        prediction = model.predict(features)[0]
        
        status = "✅ NORMAL"
        if prediction == -1:
            status = "🚨 ANOMALİ TESPİT EDİLDİ!"
            # Burada mail atma veya log tutma kodu tetiklenebilir
        
        print(f"[{data['timestamp']}] {data['machine_id']} | Isı: {data['temperature']}°C | Titreşim: {data['vibration']} | Durum: {status}")
        
        time.sleep(1) # Gerçek zamanlı akış simülasyonu

if __name__ == "__main__":
    start_monitoring()