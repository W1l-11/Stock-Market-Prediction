import numpy as np
import pandas as pd
import ta

DEFAULT_LAGS         = [1, 2, 3, 5, 10]   # Return and RSI lags
RSI_WINDOW           = 14
BB_WINDOW            = 20
ATR_WINDOW           = 14
MACD_SLOW            = 26
MACD_FAST            = 12
MACD_SIGNAL          = 9
VOL_WINDOW           = 20    # Realized volatility lookback (trading days)

class FeatureEngineer:
    """
    Usage (training)
    ----------------
        fe = FeatureEngineer(raw_df)
        processed = (fe
                     .add_technical_indicators()
                     .add_volume_features()
                     .add_macro_features()
                     .add_lags()
                     .add_target()
                     .get_df())
        # Then split FIRST, dropna AFTER inside each split
        train = processed.iloc[:split_idx].dropna()
        test  = processed.iloc[split_idx:].dropna()

    Usage (live inference)
    ----------------------
        fe = FeatureEngineer(live_df)
        processed = (fe
                     .add_technical_indicators()
                     .add_volume_features()
                     .add_macro_features()
                     .add_lags()
                     .get_df()
                     .dropna())
        latest = processed[model_features].tail(1)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        self.price_col = next(
            (c for c in ["Close", "Adj Close", "close"] if c in self.df.columns),
            self.df.columns[0],
        )

        for col in ["High", "Low", "Volume"]:
            if col not in self.df.columns:
                raise KeyError(
                    f"FeatureEngineer requires '{col}' column. "
                    f"Available: {self.df.columns.tolist()}"
                )

    def add_technical_indicators(self) -> "FeatureEngineer":
        p = self.df[self.price_col]

        # RSI
        self.df["RSI"] = ta.momentum.rsi(p, window=RSI_WINDOW)

        # MACD histogram 
        macd_obj = ta.trend.MACD(p, window_slow=MACD_SLOW,
                                     window_fast=MACD_FAST,
                                     window_sign=MACD_SIGNAL)
        self.df["MACD_Hist"] = macd_obj.macd_diff()  

        # Bollinger Bands — normalized
        bb = ta.volatility.BollingerBands(p, window=BB_WINDOW, window_dev=2)
        self.df["BB_PctB"]  = bb.bollinger_pband()   
        self.df["BB_Width"] = bb.bollinger_wband()   

        # ATR as percentage of price 
        atr_abs = ta.volatility.average_true_range(
            self.df["High"], self.df["Low"], p, window=ATR_WINDOW
        )
        self.df["ATR_Pct"] = atr_abs / p            

        # Log return 
        self.df["Log_Return"] = np.log(p / p.shift(1))

        # Realized volatility 
        self.df["Realized_Vol_20"] = (
            self.df["Log_Return"]
            .rolling(VOL_WINDOW)
            .std()
            * np.sqrt(252)
        )

        return self

    def add_volume_features(self) -> "FeatureEngineer":
        vol = self.df["Volume"].astype(float)
        roll_mean = vol.rolling(VOL_WINDOW).mean()
        roll_std  = vol.rolling(VOL_WINDOW).std()

        self.df["Volume_ZScore"] = (vol - roll_mean) / roll_std.replace(0, np.nan)

        obv = ta.volume.OnBalanceVolumeIndicator(
            close=self.df[self.price_col], volume=vol
        )
        self.df["OBV_Change"] = obv.on_balance_volume().diff()

        return self

    def add_macro_features(self) -> "FeatureEngineer":
        macro_return_cols = ["USDIDR_Ret", "SPX_Ret", "OIL_Ret", "JKSE_Ret"]

        for col in macro_return_cols:
            if col not in self.df.columns:
                raise KeyError(
                    f"Macro column '{col}' is missing from DataFrame. "
                    f"Use DataLoader.get_historical_data() or .get_live_data() "
                    f"which fetches and joins macro data automatically. "
                    f"DO NOT zero-fill macro columns — this is train/serve skew."
                )

        if "USDIDR" in self.df.columns:
            self.df["USDIDR_Mom5"] = (
                np.log(self.df["USDIDR"] / self.df["USDIDR"].shift(5))
            )

        if "SPX" in self.df.columns:
            self.df["SPX_Regime"] = (
                self.df["SPX"] > self.df["SPX"].rolling(50).mean()
            ).astype(float)

        return self

    def add_lags(self, lags: list = None) -> "FeatureEngineer":
        lags = lags or DEFAULT_LAGS

        if "Log_Return" not in self.df.columns:
            raise RuntimeError("Call add_technical_indicators() before add_lags().")

        for lag in lags:
            self.df[f"Return_Lag_{lag}"] = self.df["Log_Return"].shift(lag)

        if "RSI" in self.df.columns:
            for lag in lags:
                self.df[f"RSI_Lag_{lag}"] = self.df["RSI"].shift(lag)

        return self

    def add_target(
        self,
        forward_days: int = 1,
        threshold_pct: float = 0.0,
    ) -> "FeatureEngineer":
        if "Log_Return" not in self.df.columns:
            raise RuntimeError("Call add_technical_indicators() before add_target().")

        forward_return = self.df["Log_Return"].shift(-forward_days)
        self.df["Target_Return"]    = forward_return
        self.df["Target_Direction"] = (forward_return > threshold_pct).astype(int)

        pos_rate = self.df["Target_Direction"].mean()
        print(f"  [Target] Forward days={forward_days} | "
              f"Positive class rate: {pos_rate:.2%} "
              f"({'balanced' if 0.4 < pos_rate < 0.6 else 'IMBALANCED — consider threshold_pct'})")
        return self

    def get_df(self) -> pd.DataFrame:
        return self.df

    def get_feature_names(self, exclude_cols: list = None) -> list:
        always_exclude = {
            "Close", "Adj Close", "Open", "High", "Low", "Volume",
            "Log_Return", "Target_Return", "Target_Direction",
            "USDIDR", "SPX", "OIL", "JKSE",   # Raw macro levels
        }
        if exclude_cols:
            always_exclude.update(exclude_cols)

        return [c for c in self.df.columns if c not in always_exclude]

if __name__ == "__main__":
    import yfinance as yf
    raw = yf.download("BBRI.JK", start="2020-01-01", end="2023-12-31",
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    for col in ["USDIDR_Ret", "SPX_Ret", "OIL_Ret", "JKSE_Ret",
                "USDIDR", "SPX", "OIL", "JKSE"]:
        raw[col] = 0.0

    fe = FeatureEngineer(raw)
    df = (fe
          .add_technical_indicators()
          .add_volume_features()
          .add_macro_features()
          .add_lags()
          .add_target()
          .get_df())

    features = fe.get_feature_names()
    print(f"Total features: {len(features)}")
    print(features)
    print(df[features].tail(3))