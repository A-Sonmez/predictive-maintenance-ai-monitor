import pandas as pd
import numpy as np
import time
from datetime import datetime

def generate_sensor_data():
    # Normal çalışma verileri
    temp = np.random.normal(70, 2, 1)[0]  # 70 derece civarı
    vibration = np.random.normal(0.5, 0.1, 1)[0] # Düşük titreşim
    
    # %5 ihtimalle anomali (arıza) oluştur
    if np.random.random() > 0.95:
        temp += np.random.uniform(15, 30) # Ani ısınma
        vibration += np.random.uniform(0.5, 1.5) # Sert sarsıntı
        
    return {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "temperature": round(temp, 2),
        "vibration": round(vibration, 2),
        "machine_id": "CNC-01"
    }

# Test için:
if __name__ == "__main__":
    print("Sensör verisi akışı başlatıldı (Durdurmak için Ctrl+C)...")
    for _ in range(10):
        print(generate_sensor_data())
        time.sleep(1)