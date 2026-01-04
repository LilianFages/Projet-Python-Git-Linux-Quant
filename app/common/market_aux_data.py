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

@st.cache_data(ttl=300, show_spinner=False)  # 5 min, cohérent avec ton objectif
def get_global_ticker_data():
    """Récupère les variations pour le bandeau défilant (robuste aux NaN/week-ends)."""
    tickers = list(GLOBAL_INDICES.keys())
    data = []

    try:
        raw = yf.download(
            tickers,
            period="7d",          # plus large pour garantir 2 points valides
            interval="1d",
            progress=False,
            auto_adjust=False,
            group_by="column"
        )
        if raw is None or raw.empty:
            return []

        # yfinance renvoie souvent un MultiIndex (Open/High/Low/Close, ticker)
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            # cas atypique : une seule série / structure différente
            close = raw["Close"] if "Close" in raw.columns else raw

        for t in tickers:
            if isinstance(close, pd.DataFrame):
                if t not in close.columns:
                    continue
                s = close[t].dropna()
            else:
                # Series: uniquement possible si un seul ticker
                s = close.dropna()

            if len(s) < 2:
                continue

            price = float(s.iloc[-1])
            prev = float(s.iloc[-2])
            if prev == 0:
                continue

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