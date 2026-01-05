from __future__ import annotations

from datetime import datetime
from typing import Union

import pandas as pd
import yfinance as yf
import requests


# Mapping Yahoo -> Binance pour les cryptos que tu utilises
BINANCE_SYMBOLS = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "BNB-USD": "BNBUSDT",
    "SOL-USD": "SOLUSDT",
    "XRP-USD": "XRPUSDT",
    "ADA-USD": "ADAUSDT",
    "DOGE-USD": "DOGEUSDT",
    "LTC-USD": "LTCUSDT",
}


def _now_paris_naive() -> pd.Timestamp:
    """Timestamp 'now' en Europe/Paris, tz-naïf."""
    return pd.Timestamp.now(tz="Europe/Paris").tz_localize(None)


def _to_naive_paris(ts: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    """Timestamp tz-naïf en Europe/Paris."""
    t = pd.to_datetime(ts)
    if getattr(t, "tz", None) is not None:
        t = t.tz_convert("Europe/Paris").tz_localize(None)
    return t


def _is_intraday(interval: str) -> bool:
    return interval.endswith("m") or interval.endswith("h") or interval in {"60m"}


def _yf_period_for_intraday(span: pd.Timedelta, interval: str) -> str:
    """
    Choix d'un `period=` robuste pour yfinance en intraday.
    On évite start/end (source d'ambiguïtés timezone et de fenêtres refusées).
    """
    if interval == "1m":
        return "7d"

    if span <= pd.Timedelta(days=1):
        return "1d"
    if span <= pd.Timedelta(days=5):
        return "5d"
    if span <= pd.Timedelta(days=7):
        return "7d"
    if span <= pd.Timedelta(days=30):
        return "1mo"

    return "60d"


def _normalize_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    # Aplatir MultiIndex si jamais
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normaliser noms de colonnes
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    cols_lower = {c: c.lower() for c in df.columns}
    df = df.rename(columns=cols_lower)

    if "close" not in df.columns:
        return pd.DataFrame()

    if "volume" not in df.columns:
        df["volume"] = 0.0

    # Index datetime
    idx = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[~idx.isna()].copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[~df.index.isna()]

    # Convertir vers Europe/Paris naïf
    # yfinance intraday est souvent tz-aware ; daily souvent tz-naïf.
    if getattr(df.index, "tz", None) is None:
        # Par défaut (et pour rester compatible avec ton existant), on suppose UTC
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index = df.index.tz_convert("Europe/Paris").tz_localize(None)
    df.index.name = "date"

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# --- Helper : fetch depuis Binance pour les cryptos ------------------------


def _fetch_ohlcv_binance(
    binance_symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Télécharge les données OHLCV depuis l'API publique de Binance
    et les renvoie dans le même format que yfinance :

    - index : DatetimeIndex naïf en Europe/Paris
    - colonnes : open, high, low, close, volume
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    if start_dt.tzinfo is None:
        start_utc = start_dt.tz_localize("Europe/Paris").tz_convert("UTC")
    else:
        start_utc = start_dt.tz_convert("UTC")

    if end_dt.tzinfo is None:
        end_utc = end_dt.tz_localize("Europe/Paris").tz_convert("UTC")
    else:
        end_utc = end_dt.tz_convert("UTC")

    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)

    interval_map = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "60m": "1h",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "12h": "12h",
        "1d": "1d",
        "1wk": "1w",
        "1mo": "1M",
    }

    if interval not in interval_map:
        raise ValueError(f"Interval {interval} non supporté pour Binance.")

    bin_interval = interval_map[interval]
    url = "https://api.binance.com/api/v3/klines"

    all_rows = []
    cur_start = start_ms

    while cur_start < end_ms:
        params = {
            "symbol": binance_symbol,
            "interval": bin_interval,
            "startTime": cur_start,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        all_rows.extend(data)

        last_open_time = data[-1][0]
        next_start = last_open_time + 1
        if next_start <= cur_start:
            break
        cur_start = next_start

        if len(all_rows) > 10000:
            break
        if len(data) < 1000:
            break

    if not all_rows:
        raise ValueError(f"Aucune donnée retournée par Binance pour {binance_symbol}.")

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(all_rows, columns=cols)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["date"] = (
        pd.to_datetime(df["open_time"], unit="ms", utc=True)
        .dt.tz_convert("Europe/Paris")
        .dt.tz_localize(None)
    )
    df = df.set_index("date")
    df = df[["open", "high", "low", "close", "volume"]]

    # Filtre final sur fenêtre demandée (naïf Europe/Paris)
    start_naive = _to_naive_paris(start)
    end_naive = _to_naive_paris(end)
    df = df[(df.index >= start_naive) & (df.index <= end_naive)]
    df.index.name = "date"

    return df


# --- Fonction principale : Yahoo + Binance ---------------------------------


def fetch_ohlcv(
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Télécharge les données OHLCV :

    - cryptos connues → Binance
    - tout le reste → yfinance

    Sortie :
    - index DatetimeIndex naïf Europe/Paris
    - colonnes : open, high, low, close, volume (au minimum close)
    """
    # --- Crypto : Binance ---
    if symbol in BINANCE_SYMBOLS:
        return _fetch_ohlcv_binance(BINANCE_SYMBOLS[symbol], start, end, interval)

    now_paris = _now_paris_naive()

    start_naive = _to_naive_paris(start)
    end_naive = _to_naive_paris(end)

    # --------- GARDE-FOUS ANTI-FUTUR (clé pour ta page 2) ----------
    if start_naive > now_paris:
        start_naive = now_paris
    if end_naive > now_paris:
        end_naive = now_paris

    # Si l'appel upstream a inversé / décalé les dates, on sécurise sans planter.
    if start_naive > end_naive:
        # fallback minimal : 1 jour de données avant end
        start_naive = end_naive - pd.Timedelta(days=1)
    # ---------------------------------------------------------------

    intraday = _is_intraday(interval)

    # --- Yahoo intraday : period= (robuste), puis slice ---
    if intraday:
        span = max(end_naive - start_naive, pd.Timedelta(minutes=5))
        period = _yf_period_for_intraday(span, interval)

        raw = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            prepost=False,
            threads=False,
        )
        df = _normalize_yf_df(raw)

        if df.empty:
            raise ValueError(
                f"Aucune donnée retournée pour {symbol} (intraday {interval}, period={period})."
            )

        sliced = df.loc[(df.index >= start_naive) & (df.index <= end_naive)]
        if not sliced.empty:
            return sliced

        # Marché fermé / fenêtre hors cotation : renvoyer du récent plutôt que vide
        cutoff = df.index.max() - pd.Timedelta(days=1)
        out = df.loc[df.index >= cutoff]
        return out if not out.empty else df.tail(500)

    # --- Yahoo daily : start/end (end exclusif), avec cap anti-futur ---
    # NB: yfinance end est EXCLUSIF. On met +1 jour pour inclure le jour end_naive.
    start_date = start_naive.normalize().date()
    end_exclusive = (end_naive.normalize() + pd.Timedelta(days=1)).date()

    # cap dur : ne jamais dépasser (au plus) demain en "exclusive"
    # (si end_naive = aujourd'hui, demain exclusive est normal)
    max_end_exclusive = (now_paris.normalize() + pd.Timedelta(days=1)).date()
    if end_exclusive > max_end_exclusive:
        end_exclusive = max_end_exclusive

    # sécurité : fenêtre au moins 1 jour
    if end_exclusive <= start_date:
        end_exclusive = (start_date + pd.Timedelta(days=1)).date()
        if end_exclusive > max_end_exclusive:
            end_exclusive = max_end_exclusive

    raw = yf.download(
        symbol,
        start=start_date,
        end=end_exclusive,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    df = _normalize_yf_df(raw)

    if df.empty:
        raise ValueError(f"Aucune donnée retournée pour {symbol} (daily 1d).")

    # Slice final exact sur fenêtre demandée
    sliced = df.loc[(df.index >= start_naive.normalize()) & (df.index <= end_naive)]
    if not sliced.empty:
        return sliced

    # fallback : renvoyer le df non slicé (ex: dernière bougie pas encore publiée)
    return df
