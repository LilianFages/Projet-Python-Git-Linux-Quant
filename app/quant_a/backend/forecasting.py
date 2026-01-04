# app/quant_a/backend/forecasting.py
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def _infer_future_freq(dt_index: pd.DatetimeIndex) -> str:
    """
    Détermine une fréquence future raisonnable :
    - Si l'historique contient des week-ends => 'D'
    - Sinon => 'B' (jours ouvrés)
    """
    if len(dt_index) == 0:
        return "B"

    # On regarde les ~90 derniers points pour décider
    tail = dt_index[-min(len(dt_index), 90):]
    has_weekend = any(d.weekday() >= 5 for d in tail)  # 5=Sat, 6=Sun
    return "D" if has_weekend else "B"


def generate_forecast(series: pd.Series, steps: int = 30) -> pd.DataFrame:
    """
    Génère une prévision ARIMA simple sur 'steps' périodes.
    Retourne un DataFrame indexé par dates futures avec :
      - forecast
      - lower_conf
      - upper_conf

    Amélioration vs version initiale :
      - ARIMA sur log(niveau) + drift (trend) pour éviter des prévisions trop plates
      - fréquence future inférée ('B' vs 'D') selon présence week-end
    """
    if series is None:
        return pd.DataFrame()

    # Nettoyage basique
    clean = series.dropna().astype(float)
    if clean.empty or len(clean) < 20:
        # Pas assez de données pour un ARIMA fiable
        return pd.DataFrame()

    # Index datetime propre
    clean = clean.copy()
    clean.index = pd.to_datetime(clean.index)
    clean = clean[~clean.index.duplicated(keep="last")].sort_index()

    # La série doit être strictement positive pour log
    # equity_curve l'est normalement; si ce n'est pas le cas, fallback simple
    if (clean <= 0).any():
        return pd.DataFrame()

    # Log-transform pour stabilité
    y = np.log(clean)

    try:
        # ARIMA(1,1,1) avec drift (trend='t' -> tendance/ drift)
        # En ARIMA avec d=1, 't' correspond à un drift dans le niveau.
        model = ARIMA(y, order=(1, 1, 1), trend="t")
        model_fit = model.fit()

        forecast_res = model_fit.get_forecast(steps=steps)
        sf = forecast_res.summary_frame(alpha=0.05)  # 95% CI

        # Index futur
        last_date = clean.index[-1]
        future_freq = _infer_future_freq(clean.index)
        future_dates = pd.date_range(start=last_date, periods=steps + 1, freq=future_freq)[1:]
        sf.index = future_dates

        # La summary_frame renvoie la moyenne et les CI dans l'espace log
        # On repasse en niveau via exp
        # Colonnes typiques: mean, mean_ci_lower, mean_ci_upper
        out = pd.DataFrame(index=future_dates)
        out["forecast"] = np.exp(sf["mean"])
        out["lower_conf"] = np.exp(sf["mean_ci_lower"])
        out["upper_conf"] = np.exp(sf["mean_ci_upper"])

        # Sécurité : éviter des bornes inversées dues aux arrondis
        out["lower_conf"] = np.minimum(out["lower_conf"], out["upper_conf"])
        out["upper_conf"] = np.maximum(out["lower_conf"], out["upper_conf"])

        return out[["forecast", "lower_conf", "upper_conf"]]

    except Exception as e:
        print(f"Erreur ARIMA: {e}")
        return pd.DataFrame()
