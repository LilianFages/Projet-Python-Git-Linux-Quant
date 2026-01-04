# app/common/market_aux_data.py
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz

# Liste d'indices mondiaux pour le bandeau
GLOBAL_INDICES = {
    "^GSPC": "S&P 500",
    "^FCHI": "CAC 40",
    "^GDAXI": "DAX",
    "^FTSE": "FTSE 100",
    "^N225": "Nikkei 225",
    "BTC-USD": "Bitcoin",
    "EURUSD=X": "EUR/USD"
}

@st.cache_data(ttl=300, show_spinner=False)
def get_global_ticker_data():
    """Récupère les variations pour le bandeau défilant (2 derniers closes valides par ticker)."""
    tickers = list(GLOBAL_INDICES.keys())
    data = []

    try:
        df = yf.download(tickers, period="10d", progress=False)["Close"]
        # df : colonnes = tickers (en général). Si un seul ticker, df peut être une Series.
        if isinstance(df, pd.Series):
            df = df.to_frame()

        for t in tickers:
            if t not in df.columns:
                continue

            s = df[t].dropna()
            if len(s) < 2:
                continue

            price = float(s.iloc[-1])
            prev = float(s.iloc[-2])

            # sécurité
            if prev == 0:
                continue

            change = (price - prev) / prev

            data.append({
                "name": GLOBAL_INDICES[t],
                "price": price,
                "change": float(change),
            })

    except Exception:
        return []

    return data

def get_world_clocks():
    """Récupère l'heure des places boursières."""
    zones = {
        "NY": "America/New_York",
        "London": "Europe/London",
        "Paris": "Europe/Paris",
        "Tokyo": "Asia/Tokyo",
        "HK": "Asia/Hong_Kong"
    }
    clocks = {}
    now_utc = datetime.now(pytz.utc)
    for city, zone in zones.items():
        try:
            tz = pytz.timezone(zone)
            clocks[city] = now_utc.astimezone(tz).strftime("%H:%M")
        except:
            clocks[city] = "--:--"
    return clocks