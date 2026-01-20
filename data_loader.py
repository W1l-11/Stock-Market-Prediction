import yfinance as yf
import pandas as pd
import os

class DataLoader:
    def __init__(self, ticker, start_date, end_date, cache_dir="data"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = cache_dir
        
        # PENTING: Daftar Ticker Makro Ekonomi
        # IDR=X : Kurs USD ke Rupiah
        # ^GSPC : S&P 500 (Indeks Saham AS - Sentimen Global)
        # CL=F  : Crude Oil Futures (Indikator Energi/Komoditas)
        self.macro_tickers = {
            'USDIDR': 'IDR=X',
            'SPX': '^GSPC',
            'OIL': 'CL=F'
        }
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _download_ticker(self, symbol, name):
        """Helper function untuk download satu ticker"""
        file_path = os.path.join(self.cache_dir, f"{name}_{self.start_date}_{self.end_date}.parquet")
        
        if os.path.exists(file_path):
            return pd.read_parquet(file_path)
            
        print(f"Downloading {name} ({symbol})...")
        df = yf.download(symbol, start=self.start_date, end=self.end_date, auto_adjust=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        df = df[[target_col]].copy() # Kita cuma butuh harga penutup untuk data makro
        df.rename(columns={target_col: name}, inplace=True)
        
        # Simpan cache
        df.to_parquet(file_path)
        return df

    def get_data_with_macro(self):
        # 1. Ambil Data Saham Utama (Lengkap dengan OHLCV)
        main_file = os.path.join(self.cache_dir, f"{self.ticker}_{self.start_date}_{self.end_date}_MAIN.parquet")
        
        # Logika download utama (sama seperti sebelumnya, tapi kita simpan OHLCV)
        if os.path.exists(main_file):
            main_df = pd.read_parquet(main_file)
        else:
            print(f"Downloading Main Ticker {self.ticker}...")
            main_df = yf.download(self.ticker, start=self.start_date, end=self.end_date, auto_adjust=False)
            if isinstance(main_df.columns, pd.MultiIndex):
                main_df.columns = main_df.columns.get_level_values(0)
            
            target_col = 'Adj Close' if 'Adj Close' in main_df.columns else 'Close'
            main_df = main_df[[target_col, 'Volume', 'High', 'Low', 'Open']].copy()
            main_df.rename(columns={target_col: 'Adj Close'}, inplace=True)
            main_df.to_parquet(main_file)

        # 2. Ambil Data Makro dan Gabungkan (Merge)
        for name, symbol in self.macro_tickers.items():
            macro_df = self._download_ticker(symbol, name)
            # Gabungkan berdasarkan tanggal (Index)
            main_df = main_df.join(macro_df, how='left')
        
        # 3. Handle Missing Values pada Data Makro
        # Karena libur pasar US beda dengan Indo, akan ada NaN. Kita ffill.
        main_df.ffill(inplace=True)
        main_df.dropna(inplace=True)
        
        return main_df

if __name__ == "__main__":
    loader = DataLoader("BBCA.JK", "2018-01-01", "2026-01-12")
    df = loader.get_data_with_macro()
    print(df.head())
    print("Kolom tersedia:", df.columns.tolist())