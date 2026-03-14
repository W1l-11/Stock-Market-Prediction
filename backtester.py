import warnings
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

IDX_BUY_TC      = 0.0015   # 0.15% broker commission 
IDX_SELL_TC     = 0.0025   # 0.25% broker commission  + stamp duty 
IDX_T_PLUS      = 2        # Settlement delay in trading days (T+2 on IDX)
IDX_LOT_SIZE    = 100      # 1 lot = 100 shares
ANNUAL_PERIODS  = 252      # Trading days per year

def volatility_target_size(
    realized_vol_series: pd.Series,
    target_annual_vol:   float = 0.15,
    max_position:        float = 1.0,
    min_position:        float = 0.0,
    lookback:            int   = 20,
) -> pd.Series:
    clean_vol = realized_vol_series.replace(0, np.nan).fillna(method="ffill")
    sizes     = (target_annual_vol / clean_vol).clip(lower=min_position, upper=max_position)
    return sizes.fillna(0.0)

class VectorizedBacktester:
    def __init__(
        self,
        test_df:        pd.DataFrame,
        probabilities:  np.ndarray,
        target_vol:     float = 0.15,       # 15% annualised vol target
        buy_tc:         float = IDX_BUY_TC,
        sell_tc:        float = IDX_SELL_TC,
        settlement_days: int  = IDX_T_PLUS,
    ):

        if len(probabilities) != len(test_df):
            raise ValueError("probabilities length must match test_df length.")

        self.df              = test_df.copy()
        self.df["probs"]     = probabilities
        self.target_vol      = target_vol
        self.buy_tc          = buy_tc
        self.sell_tc         = sell_tc
        self.settlement_days = settlement_days
        self._results        = None   # Populated by run()

    def run(
        self,
        entry_threshold:  float = 0.55,
        exit_threshold:   float = 0.45,
        vol_col:          str   = "Realized_Vol_20",
        vol_fallback_window: int = 20,
    ) -> pd.DataFrame:
        df = self.df

        # Realized volatility
        if vol_col in df.columns:
            realized_vol = df[vol_col]
        else:
            realized_vol = (
                df["Log_Return"]
                .rolling(vol_fallback_window)
                .std()
                .mul(np.sqrt(ANNUAL_PERIODS))
                .fillna(method="bfill")
            )
            print(f"  [WARN] '{vol_col}' not found; computing realized vol from Log_Return.")

        vol_sizes = volatility_target_size(realized_vol, self.target_vol)
        df["vol_target_size"] = vol_sizes

        n             = len(df)
        raw_signal    = np.zeros(n)   # 1 = want to be in, 0 = want cash
        current_state = 0             # 0 = cash, 1 = invested
        settlement_free_at = -1       # Bar index when sold cash becomes available again

        for i in range(n):
            prob = df["probs"].iloc[i]

            if current_state == 0:
                can_trade = (i >= settlement_free_at)
                if can_trade and prob > entry_threshold:
                    current_state = 1

            elif current_state == 1:
                if prob < exit_threshold:
                    current_state = 0
                    settlement_free_at = i + self.settlement_days

            raw_signal[i] = current_state

        df["raw_signal"] = raw_signal

        df["target_position"] = df["raw_signal"] * df["vol_target_size"]

        df["execution_position"] = df["target_position"].shift(1).fillna(0.0)

        pos_changes = df["execution_position"].diff().fillna(0.0)
        buys        = pos_changes.clip(lower=0)     # Positive change = buy
        sells       = pos_changes.clip(upper=0).abs()  # Negative change = sell
        tc_costs    = buys * self.buy_tc + sells * self.sell_tc

        # Returns
        df["benchmark_returns"]       = df["Log_Return"]
        df["strategy_returns_gross"]  = df["execution_position"] * df["Log_Return"]
        df["strategy_returns_net"]    = df["strategy_returns_gross"] - tc_costs

        # Equity curves
        df["benchmark_equity"]  = df["benchmark_returns"].cumsum().apply(np.exp)
        df["strategy_equity"]   = df["strategy_returns_net"].cumsum().apply(np.exp)

        self._results = df
        return df

    def compute_metrics(self) -> dict:
        if self._results is None:
            raise RuntimeError("Call run() before compute_metrics().")

        df  = self._results
        ret = df["strategy_returns_net"].values
        eq  = df["strategy_equity"].values
        bm  = df["benchmark_equity"].values
        bm_ret = df["benchmark_returns"].values

        # Return
        total_ret     = (eq[-1] - 1) * 100
        bm_total_ret  = (bm[-1] - 1) * 100
        n_years       = len(ret) / ANNUAL_PERIODS
        ann_ret       = ((eq[-1]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

        # Sharpe & Sortino
        ann_ret_dec   = ann_ret / 100
        ann_vol       = ret.std() * np.sqrt(ANNUAL_PERIODS)
        sharpe        = ann_ret_dec / ann_vol if ann_vol > 0 else 0
        downside      = ret[ret < 0]
        down_vol      = downside.std() * np.sqrt(ANNUAL_PERIODS) if len(downside) > 0 else 1e-9
        sortino       = ann_ret_dec / down_vol

        # Drawdown
        rolling_max   = np.maximum.accumulate(eq)
        drawdowns     = (eq - rolling_max) / rolling_max
        max_dd        = float(-drawdowns.min()) * 100
        calmar        = ann_ret / max_dd if max_dd > 0 else 0

        # Trade stats
        in_market     = df["execution_position"].values
        pct_in_market = (in_market > 0).mean() * 100

        trade_rets    = ret[ret != 0]
        winners       = trade_rets[trade_rets > 0]
        losers        = trade_rets[trade_rets < 0]
        win_rate      = len(winners) / max(len(trade_rets), 1) * 100
        avg_win       = winners.mean() * 100 if len(winners) > 0 else 0
        avg_loss      = losers.mean()  * 100 if len(losers)  > 0 else 0
        profit_factor = abs(winners.sum() / losers.sum()) if losers.sum() != 0 else np.inf

        n_trades      = int(df["execution_position"].diff().abs().gt(0).sum())
        avg_position  = df.loc[df["execution_position"] > 0, "execution_position"].mean()

        metrics = {
            # Return
            "Total Return Strategy (%)":    round(total_ret, 2),
            "Total Return Benchmark (%)":   round(bm_total_ret, 2),
            "Alpha (%)":                    round(total_ret - bm_total_ret, 2),
            "Annualised Return (%)":        round(ann_ret, 2),
            # Risk-adjusted
            "Sharpe Ratio":                 round(sharpe, 4),
            "Sortino Ratio":                round(sortino, 4),
            "Max Drawdown (%)":             round(max_dd, 2),
            "Calmar Ratio":                 round(calmar, 4),
            "Annualised Volatility (%)":    round(ann_vol * 100, 2),
            # Trade analysis
            "Win Rate (%)":                 round(win_rate, 2),
            "Avg Win (%)":                  round(avg_win, 4),
            "Avg Loss (%)":                 round(avg_loss, 4),
            "Profit Factor":                round(profit_factor, 4),
            "Number of Trades":             n_trades,
            "% Time in Market":             round(pct_in_market, 2),
            "Avg Position Size":            round(avg_position, 4) if not np.isnan(avg_position) else 0,
        }

        return metrics

    def print_metrics(self) -> None:
        metrics = self.compute_metrics()
        print("\n" + "=" * 55)
        print("  BACKTEST RESULTS — SUMMARY")
        print("=" * 55)
        sections = [
            ("RETURNS", ["Total Return Strategy (%)", "Total Return Benchmark (%)",
                         "Alpha (%)", "Annualised Return (%)"]),
            ("RISK-ADJUSTED", ["Sharpe Ratio", "Sortino Ratio", "Max Drawdown (%)",
                                "Calmar Ratio", "Annualised Volatility (%)"]),
            ("TRADE ANALYSIS", ["Win Rate (%)", "Avg Win (%)", "Avg Loss (%)",
                                 "Profit Factor", "Number of Trades",
                                 "% Time in Market", "Avg Position Size"]),
        ]
        for section_name, keys in sections:
            print(f"\n  --- {section_name} ---")
            for k in keys:
                if k in metrics:
                    print(f"  {k:<40} {metrics[k]}")
        print("=" * 55 + "\n")

    def plot_results(self, ticker_name: str = "Strategy") -> None:
        if self._results is None:
            raise RuntimeError("Call run() before plot_results().")

        df  = self._results
        fig, axes = plt.subplots(4, 1, figsize=(14, 16),
                                 gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1.5]})
        fig.suptitle(f"Institutional Backtest: {ticker_name}", fontsize=14, fontweight="bold")

        # Equity curves
        ax1 = axes[0]
        ax1.plot(df.index, df["benchmark_equity"], label="Benchmark (Buy & Hold)",
                 color="#94a3b8", linewidth=1.5, alpha=0.8)
        ax1.plot(df.index, df["strategy_equity"], label="Vol-Target Strategy",
                 color="#3b82f6", linewidth=2)
        ax1.fill_between(df.index, 1, df["strategy_equity"],
                         where=df["strategy_equity"] >= 1,
                         alpha=0.08, color="#3b82f6")
        ax1.set_ylabel("NAV (start = 1.0)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.2)

        # Position size
        ax2 = axes[1]
        ax2.fill_between(df.index, 0, df["execution_position"],
                         color="#10b981", alpha=0.4, label="Position size (vol-targeted)")
        ax2.set_ylabel("Position Size")
        ax2.set_ylim(-0.05, 1.15)
        ax2.axhline(1.0, color="#64748b", linestyle="--", linewidth=0.8, alpha=0.5)
        ax2.legend(loc="upper left", fontsize=9)
        ax2.grid(True, alpha=0.2)

        # Drawdown
        ax3 = axes[2]
        rolling_max = df["strategy_equity"].cummax()
        drawdown    = (df["strategy_equity"] - rolling_max) / rolling_max * 100
        ax3.fill_between(df.index, drawdown, 0, color="#ef4444", alpha=0.5)
        ax3.set_ylabel("Drawdown (%)")
        ax3.grid(True, alpha=0.2)

        # Rolling 60-day Sharpe
        ax4 = axes[3]
        rolling_sharpe = (
            df["strategy_returns_net"]
            .rolling(60)
            .apply(lambda r: r.mean() / r.std() * np.sqrt(ANNUAL_PERIODS)
                   if r.std() > 0 else 0, raw=True)
        )
        ax4.plot(df.index, rolling_sharpe, color="#8b5cf6", linewidth=1.2)
        ax4.axhline(0, color="#64748b", linestyle="--", linewidth=0.8)
        ax4.axhline(1, color="#10b981", linestyle=":", linewidth=0.8, alpha=0.7)
        ax4.set_ylabel("Rolling Sharpe (60d)")
        ax4.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig("backtest_results.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("  Chart saved to backtest_results.png")


if __name__ == "__main__":
    np.random.seed(42)
    n   = 500
    idx = pd.date_range("2022-01-01", periods=n, freq="B")

    log_ret = np.random.randn(n) * 0.012
    df = pd.DataFrame({
        "Log_Return":      log_ret,
        "Realized_Vol_20": np.abs(np.random.randn(n) * 0.05 + 0.20),
    }, index=idx)

    probs = np.clip(np.random.randn(n) * 0.15 + 0.52, 0, 1)

    bt = VectorizedBacktester(df, probs, target_vol=0.15)
    bt.run(entry_threshold=0.55, exit_threshold=0.45)
    bt.print_metrics()
    bt.plot_results("BBRI.JK Test")