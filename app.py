import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import yfinance as yf
from feature_engineering import FeatureEngineer
from datetime import datetime, timedelta

st.set_page_config(page_title="IndoQuant Pro Dashboard", layout="wide")

st.markdown("""
    <style>
    /* Latar belakang aplikasi utama */
    .stApp { background-color: #0e1117; }
    
    [data-testid="stMetric"] {
        background-color: #1f2937 !important; 
        padding: 20px !important;
        border-radius: 12px !important;
        border: 1px solid #374151 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2) !important;
    }

    /* Memaksa warna teks LABEL (Judul Metrik) agar terlihat */
    [data-testid="stMetricLabel"] {
        color: #9ca3af !important; /* Abu-abu terang */
        font-size: 1rem !important;
    }

    /* Memaksa warna teks VALUE (Angka Utama) agar putih terang */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Kartu Portfolio Custom */
    .portfolio-card { 
        background-color: #1f2937; 
        padding: 25px; 
        border-radius: 15px; 
        border-left: 5px solid #3b82f6;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_quant_model():
    try:
        return joblib.load("quant_model_bbri.pkl")
    except:
        return None

def main():
    st.sidebar.title("🚀 IndoQuant Menu")
    page = st.sidebar.radio("Pilih Halaman:", ["Market Analysis", "My Portfolio"])
    
    model_data = load_quant_model()
    if not model_data:
        st.error("Model tidak ditemukan! Pastikan 'quant_model_bbri.pkl' tersedia.")
        return

    # --- 2. ROBUST DATA INGESTION ---
    with st.spinner('Sinkronisasi Data Hourly...'):
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=30) 
        
        # Download data
        df_raw = yf.download("BBRI.JK", start=start_dt, end=end_dt, interval="1h")
        
        # SOLUSI KEYERROR: Flatten MultiIndex jika ada
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)
            
        # Feature Engineering
        fe = FeatureEngineer(df_raw)
        df_processed = fe.add_technical_indicators().add_lags()
        
        # SOLUSI LIGHTGBM ERROR: Sinkronisasi Fitur (23 Kolom)
        model_features = model_data['features']
        # Tambahkan kolom makro yang hilang sebagai dummy (0.0) agar total fitur pas 23
        for col in model_features:
            if col not in df_processed.df.columns:
                df_processed.df[col] = 0.0
        
        df = df_processed.get_final_df()

        # Deteksi Kolom Harga Dinamis
        price_col = next((c for c in ['Adj Close', 'Close', 'close'] if c in df.columns), df.columns[0])
        current_price = float(df[price_col].iloc[-1])
        
        # Prediksi AI dengan urutan kolom yang SAMA dengan saat training
        latest_features = df[model_features].tail(1)
        
        # Prediksi Probabilitas
        prob = model_data['model'].predict_proba(latest_features)[0, 1]
        threshold = model_data['entry_threshold']

    # --- HALAMAN 1: MARKET ANALYSIS ---
    if page == "Market Analysis":
        st.title("🏦 BBRI Hourly Intelligence")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Harga Saat Ini", f"Rp {current_price:,.0f}")
        c2.metric("AI Confidence", f"{prob:.2%}", delta=f"{(prob-threshold):.2%}")
        c3.metric("Threshold", f"{threshold:.2%}")
        
        is_buy = prob > threshold
        sig_color = "green" if is_buy else "red"
        c4.markdown(f"### Signal: :{sig_color}[{'BUY / HOLD' if is_buy else 'STAY IN CASH'}]")

        st.divider()

        st.subheader("📊 Hourly Candlestick")
        fig = go.Figure(data=[go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df[price_col], name='BBRI 1H'
        )])
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

    # --- HALAMAN 2: MY PORTOFOLIO ---
    elif page == "My Portfolio":
        st.title("💼 Live Portfolio Performance")
        
        with st.expander("⚙️ Pengaturan Investasi"):
            col_in1, col_in2 = st.columns(2)
            avg_p = col_in1.number_input("Harga Beli Rata-rata", value=3635.44)
            lots = col_in2.number_input("Jumlah Lot (1 Lot = 100 Lembar)", value=1, step=1)
            total_shares = lots * 100
        
        invested_capital = avg_p * total_shares
        current_equity = current_price * total_shares
        total_pnl = current_equity - invested_capital
        gain_pct = (total_pnl / invested_capital * 100) if invested_capital > 0 else 0
        
        st.markdown(f"""
            <div class="portfolio-card">
                <p style='margin-bottom:0px; color:#9ca3af; font-size:1.1rem;'>Total Equity Value</p>
                <h1 style='margin-top:0px; font-size:3rem; color:#ffffff;'>Rp {current_equity:,.0f}</h1>
            </div>
        """, unsafe_allow_html=True)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Invested Capital", f"Rp {invested_capital:,.0f}")
        m2.metric("Unrealized P&L", f"Rp {total_pnl:,.0f}", f"{gain_pct:+.2f}%")
        m3.metric("Gain Percentage", f"{gain_pct:.2f}%")

        st.divider()
        st.subheader("📈 Real-time Portfolio Trend (Last 24h)")
        equity_curve = df[price_col].tail(24) * total_shares
        st.line_chart(equity_curve)

if __name__ == "__main__":
    main()