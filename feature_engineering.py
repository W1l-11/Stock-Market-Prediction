import pandas as pd
import numpy as np
import ta

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        # Deteksi kolom harga dinamis untuk mencegah KeyError
        self.price_col = next((c for c in ['Adj Close', 'Close', 'close'] if c in self.df.columns), self.df.columns[0])

    def add_technical_indicators(self):
        self.df['RSI'] = ta.momentum.rsi(self.df[self.price_col], window=14)
        macd = ta.trend.MACD(self.df[self.price_col])
        self.df['MACD'] = macd.macd()
        self.df['MACD_Signal'] = macd.macd_signal()
        
        indicator_bb = ta.volatility.BollingerBands(self.df[self.price_col], window=20, window_dev=2)
        self.df['BB_High'] = indicator_bb.bollinger_hband()
        self.df['BB_Low'] = indicator_bb.bollinger_lband()
        self.df['BB_Mid'] = indicator_bb.bollinger_mavg()
        
        # ATR sangat penting untuk Dashboard Trading Plan
        self.df['ATR'] = ta.volatility.average_true_range(self.df['High'], self.df['Low'], self.df[self.price_col], window=14)
        self.df['Log_Return'] = np.log(self.df[self.price_col] / self.df[self.price_col].shift(1))
        return self

    def add_lags(self, lags=[1, 2, 3, 5]):
        for lag in lags:
            self.df[f'Return_Lag_{lag}'] = self.df['Log_Return'].shift(lag)
            self.df[f'RSI_Lag_{lag}'] = self.df['RSI'].shift(lag)
        return self

    def add_macro_features(self, model_features):
        """
        Solusi LightGBMError: Memastikan semua kolom yang diminta model tersedia.
        Jika kolom makro tidak ada di data hourly, isi dengan 0 atau nilai terakhir.
        """
        for col in model_features:
            if col not in self.df.columns:
                self.df[col] = 0.0 # Mengisi fitur yang hilang agar jumlah kolom pas (23)
        return self

    def get_final_df(self):
        self.df.dropna(inplace=True)
        return self.df