import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class VectorizedBacktester:
    def __init__(self, test_df, probabilities, tc=0.002):
        """
        probabilities: Raw probability output dari model (0.0 sampai 1.0)
        tc: Transaction Cost (default 0.2% per trade, buy+sell = 0.4% roundtrip risk)
        """
        self.df = test_df.copy()
        self.df['probs'] = probabilities
        self.tc = tc

    def run_smart_execution(self, entry_threshold=0.55, exit_threshold=0.45):
        # Inisialisasi posisi (0 = Cash, 1 = Full Invested)
        self.df['position'] = 0
        
        # Logika Loop (Perlu loop karena keputusan hari ini tergantung posisi kemarin)
        # Di Python loop lambat, tapi untuk backtest harian ini sangat cepat (negligible)
        current_position = 0
        positions = []
        
        for i in range(len(self.df)):
            prob = self.df['probs'].iloc[i]
            
            # Logika Hysteresis
            if current_position == 0: # Sedang pegang Cash
                if prob > entry_threshold: # Hanya beli jika sangat yakin
                    current_position = 1
            elif current_position == 1: # Sedang pegang Saham
                if prob < exit_threshold: # Hanya jual jika sinyal memburuk
                    current_position = 0
            
            positions.append(current_position)
            
        self.df['position'] = positions
        
        # Shift posisi ke depan 1 hari 
        # (Keputusan hari ini berdasarkan data penutupan, dieksekusi besok open/close)
        # Di backtest sederhana, kita anggap kita dapat return besoknya.
        self.df['strategy_position'] = self.df['position'].shift(1).fillna(0)
        
        # Hitung Returns
        self.df['benchmark_returns'] = self.df['Log_Return']
        self.df['strategy_returns'] = self.df['strategy_position'] * self.df['Log_Return']
        
        # Hitung Biaya Transaksi
        # Trades terjadi jika posisi hari ini beda dengan kemarin
        self.df['trades'] = self.df['strategy_position'].diff().fillna(0).abs()
        self.df['strategy_returns_net'] = self.df['strategy_returns'] - (self.df['trades'] * self.tc)
        
        # Equity Curve
        self.df['benchmark_equity'] = self.df['benchmark_returns'].cumsum().apply(np.exp)
        self.df['strategy_equity'] = self.df['strategy_returns_net'].cumsum().apply(np.exp)
        
        return self.df

    def plot_results(self, ticker_name):
        plt.figure(figsize=(12, 6))
        
        # Plot Benchmark
        plt.plot(self.df.index, self.df['benchmark_equity'], 
                 label=f'Benchmark (Buy & Hold)', color='grey', alpha=0.5)
        
        # Plot Strategy
        plt.plot(self.df.index, self.df['strategy_equity'], 
                 label='Smart Strategy (With Threshold)', color='blue', linewidth=2)
        
        # Visualisasi Posisi (Kapan kita masuk pasar?)
        # Kita arsir area hijau saat kita punya saham
        ymin, ymax = plt.ylim()
        plt.fill_between(self.df.index, ymin, ymax, 
                         where=(self.df['strategy_position'] == 1), 
                         color='green', alpha=0.1, label='In Market (Holding)')

        plt.title(f'Institutional Backtest: {ticker_name}')
        plt.ylabel('Growth of 1 Unit Currency')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.show()

        # Statistik
        total_ret = (self.df['strategy_equity'].iloc[-1] - 1) * 100
        n_trades = self.df['trades'].sum()
        print(f"Total Return Strategy: {total_ret:.2f}%")
        print(f"Jumlah Transaksi: {int(n_trades)}")