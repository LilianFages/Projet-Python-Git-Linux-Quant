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

@st.cache_data(ttl=3600, show_spinner=False)
def get_global_ticker_data():
    """Récupère les variations pour le bandeau défilant."""
    tickers = list(GLOBAL_INDICES.keys())
    data = []
    try:
        # Téléchargement rapide des 2 derniers jours
        df = yf.download(tickers, period="2d", progress=False)['Close']
        if len(df) < 2: return []

        today = df.iloc[-1]
        yesterday = df.iloc[-2]

        for t in tickers:
            # Gestion robuste si un ticker manque
            if t in today and t in yesterday:
                price = today[t]
                prev = yesterday[t]
                # Vérification NaN
                if pd.isna(price) or pd.isna(prev): continue
                
                change = (price - prev) / prev
                data.append({
                    "name": GLOBAL_INDICES[t],
                    "price": price,
                    "change": change
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