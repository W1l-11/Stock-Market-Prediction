import hashlib
import json
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

# IDX trading calendar: Mon–Fri, 09:00–16:00 WIB
MACRO_TICKERS = {
    "USDIDR": "IDR=X",   
    "SPX":    "^GSPC",  
    "OIL":    "CL=F",   
    "JKSE":   "^JKSE",   # IHSG — domestic market regime signal
}

CACHE_TTL_DAYS   = 7     
MAX_RETRIES      = 3  
RETRY_BACKOFF_S  = 2   

def _param_hash(ticker: str, start: str, end: str) -> str:
    payload = json.dumps({"t": ticker, "s": start, "e": end}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:8]


def _is_cache_valid(file_path: str, ttl_days: int = CACHE_TTL_DAYS) -> bool:
    if not os.path.exists(file_path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
    return datetime.now() - mtime < timedelta(days=ttl_days)


def _download_with_retry(symbol: str, start: str, end: str,
                         auto_adjust: bool = True, max_retries: int = MAX_RETRIES) -> pd.DataFrame:

    for attempt in range(max_retries):
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                auto_adjust=auto_adjust,  
                progress=False,
                threads=False,           
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty:
                raise ValueError(f"yfinance returned empty DataFrame for {symbol}")

            df.index = pd.to_datetime(df.index).normalize()
            df.index.name = "Date"

            return df

        except Exception as exc:
            wait = RETRY_BACKOFF_S * (2 ** attempt)
            print(f"  [WARN] Attempt {attempt+1}/{max_retries} failed for {symbol}: {exc}. "
                  f"Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"[FATAL] Could not download {symbol} after {max_retries} retries.")


def _flatten_ohlcv(df: pd.DataFrame, price_col_name: str = "Close") -> pd.DataFrame:
    rename_map = {}
    if "Adj Close" in df.columns:
        rename_map["Adj Close"] = price_col_name
    elif "Close" in df.columns and price_col_name != "Close":
        rename_map["Close"] = price_col_name

    df = df.rename(columns=rename_map)

    required = [price_col_name, "Open", "High", "Low", "Volume"]
    available = [c for c in required if c in df.columns]
    return df[available].copy()

class DataLoader:
    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        cache_dir: str = "data",
        macro_tickers: dict = None,
    ):
        self.ticker      = ticker
        self.start_date  = start_date
        self.end_date    = end_date
        self.cache_dir   = cache_dir
        self.macro_tickers = macro_tickers or MACRO_TICKERS

        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, name: str, suffix: str = "") -> str:
        h = _param_hash(name, self.start_date, self.end_date)
        fname = f"{name}_{self.start_date}_{self.end_date}_{h}{suffix}.parquet"
        return os.path.join(self.cache_dir, fname)

    def _load_or_download_macro(self, name: str, symbol: str) -> pd.DataFrame:
        path = self._cache_path(name, suffix="_macro")

        if _is_cache_valid(path):
            return pd.read_parquet(path)

        print(f"  Downloading macro: {name} ({symbol}) ...")
        df = _download_with_retry(symbol, self.start_date, self.end_date)

        close_col = "Close" if "Close" in df.columns else df.columns[0]
        df = df[[close_col]].rename(columns={close_col: name})

        df[f"{name}_Ret"] = df[name].pct_change()

        df.to_parquet(path)
        return df

    def _load_or_download_equity(self) -> pd.DataFrame:
        path = self._cache_path(self.ticker, suffix="_MAIN")

        if _is_cache_valid(path):
            return pd.read_parquet(path)

        print(f"  Downloading equity: {self.ticker} ...")
        df = _download_with_retry(self.ticker, self.start_date, self.end_date)
        df = _flatten_ohlcv(df, price_col_name="Close")

        zero_vol_pct = (df["Volume"] == 0).mean() * 100
        if zero_vol_pct > 5:
            print(f"  [WARN] {self.ticker}: {zero_vol_pct:.1f}% of bars have zero volume. "
                  "Check for trading halts or bad data.")

        df.to_parquet(path)
        return df

    def get_historical_data(self) -> pd.DataFrame:
        print(f"--- DataLoader: {self.ticker} [{self.start_date} → {self.end_date}] ---")

        main_df = self._load_or_download_equity()

        for name, symbol in self.macro_tickers.items():
            macro_df = self._load_or_download_macro(name, symbol)
            main_df = main_df.join(macro_df, how="left")

        main_df = main_df.ffill().bfill()
        main_df.dropna(inplace=True)

        print(f"  Final shape: {main_df.shape} | "
              f"From {main_df.index[0].date()} to {main_df.index[-1].date()}")
        return main_df

    def get_live_data(self, lookback_days: int = 90) -> pd.DataFrame:
        end   = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        print(f"--- DataLoader (LIVE): {self.ticker} [{start} → {end}] ---")

        # Equity — fresh, no cache
        equity_df = _download_with_retry(self.ticker, start, end)
        equity_df = _flatten_ohlcv(equity_df, price_col_name="Close")

        # Macro — fresh, no cache
        for name, symbol in self.macro_tickers.items():
            macro_raw = _download_with_retry(symbol, start, end)
            close_col = "Close" if "Close" in macro_raw.columns else macro_raw.columns[0]
            macro_series = macro_raw[[close_col]].rename(columns={close_col: name})
            macro_series[f"{name}_Ret"] = macro_series[name].pct_change()
            equity_df = equity_df.join(macro_series, how="left")

        equity_df = equity_df.ffill().bfill()
        equity_df.dropna(inplace=True)

        print(f"  Live data shape: {equity_df.shape}")
        return equity_df

if __name__ == "__main__":
    loader = DataLoader("BBRI.JK", "2020-01-01", "2026-2-28")
    df = loader.get_historical_data()
    print(df.tail(3))
    print("Columns:", df.columns.tolist())