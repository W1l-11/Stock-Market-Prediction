import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import joblib
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from model_trainer import QuantModel
from backtester import VectorizedBacktester

def main():
    # Setup
    TICKER = "BBRI.JK" # Coba bank, sangat sensitif thd makro
    START = "2015-01-01"
    END = "2023-12-31"
    
    # 1. Load Data dengan Makro
    print("--- 1. Data Loading (Multi-Asset) ---")
    loader = DataLoader(TICKER, START, END)
    # Gunakan fungsi baru
    raw_data = loader.get_data_with_macro()
    
    # 2. Feature Engineering dengan Makro
    print("--- 2. Feature Engineering ---")
    fe = FeatureEngineer(raw_data)
    # Chain method baru: add_macro_features
    processed_data = fe.add_technical_indicators().add_macro_features().add_lags().add_target().get_final_df()
    
    # Pilih Fitur
    exclude_cols = ['Adj Close', 'High', 'Low', 'Open', 'Volume', 'Log_Return', 
                    'Target_Next_Day_Return', 'Target_Direction', 
                    'USDIDR', 'SPX', 'OIL'] # Exclude harga mentah makro juga
    features = [c for c in processed_data.columns if c not in exclude_cols]
    print(f"Total Fitur: {len(features)}")
    
# 3. Training & Tuning
    qm = QuantModel(processed_data, features)
    X_train, y_train, X_test, y_test, test_df = qm.prepare_split()
    
    qm.optimize_hyperparameters(X_train, y_train, X_test, y_test, n_trials=20)
    qm.train_final_model(X_train, y_train)
    
    # 4. Evaluasi & Backtest
    preds, probs = qm.evaluate(X_test, y_test)
    
    # Hitung entry_threshold di sini agar bisa disimpan
    entry_threshold = np.percentile(probs, 80)
    if entry_threshold < 0.5:
        entry_threshold = 0.505
        
    # --- BAGIAN PENYIMPANAN MODEL (PERBAIKAN DI SINI) ---
    model_export = {
        'model': qm.model,
        'features': features,
        'entry_threshold': entry_threshold
    }
    joblib.dump(model_export, "quant_model_bbri.pkl")
    print(f"\n[SUCCESS] Model disimpan dengan Threshold: {entry_threshold:.4f}")
    # ----------------------------------------------------

    # Jalankan backtest seperti biasa
    bt = VectorizedBacktester(test_df, probs)
    bt.run_smart_execution(entry_threshold=entry_threshold, exit_threshold=np.percentile(probs, 40))
    bt.plot_results(TICKER)

if __name__ == "__main__":
    main()

