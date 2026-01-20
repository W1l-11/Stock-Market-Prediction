import pandas as pd
import numpy as np
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
import joblib # Untuk menyimpan model agar tidak perlu training ulang
from datetime import datetime, timedelta

def generate_signal():
    # 1. Konfigurasi
    TICKER = "BBRI.JK"
    # Ambil data 60 hari terakhir saja untuk efisiensi
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"--- Fetching Latest Data for {TICKER} ---")
    loader = DataLoader(TICKER, start_date, end_date)
    raw_data = loader.get_data_with_macro()
    
    # 2. Feature Engineering
    fe = FeatureEngineer(raw_data)
    processed_data = fe.add_technical_indicators().add_macro_features().add_lags().get_final_df()
    
    # Ambil baris terakhir (data terbaru hari ini)
    latest_features = processed_data.tail(1)
    
    # 3. Load Model & Features List
    # Asumsikan kita sudah menyimpan model sebelumnya di main.py menggunakan joblib
    try:
        model_data = joblib.load("quant_model_bbri.pkl")
        model = model_data['model']
        feature_cols = model_data['features']
        entry_threshold = model_data['entry_threshold']
    except:
        return "Error: Model file tidak ditemukan. Jalankan training di main.py dulu."

    # 4. Prediksi
    X = latest_features[feature_cols]
    prob = model.predict_proba(X)[0, 1]
    
    # 5. Output Keputusan
    print("\n" + "="*30)
    print(f"TRADING SIGNAL DASHBOARD")
    print(f"Ticker    : {TICKER}")
    print(f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Confidence: {prob:.4f}")
    print(f"Threshold : {entry_threshold:.4f}")
    print("="*30)

    if prob > entry_threshold:
        return "DECISION: >>> BUY / STRONG HOLD <<<"
    elif prob > 0.48:
        return "DECISION: >>> NEUTRAL / WAIT <<<"
    else:
        return "DECISION: >>> SELL / STAY IN CASH <<<"

if __name__ == "__main__":
    # Simpan model dulu di main.py (tambahkan baris ini di main.py Anda)
    # joblib.dump({'model': qm.model, 'features': features, 'entry_threshold': entry_threshold}, "quant_model_bbri.pkl")
    
    signal = generate_signal()
    print(signal)
    print("="*30)