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
    tail = dt_index[-min(len(dt_index), 90):]
    has_weekend = any(d.weekday() >= 5 for d in tail)
    return "D" if has_weekend else "B"


def _clean_series(series: pd.Series) -> pd.Series:
    """Nettoie la série : float, index datetime, tri, dédup."""
    s = series.dropna().astype(float).copy()
    if s.empty:
        return s
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _forecast_level_arima_log_drift(clean: pd.Series, steps: int, alpha: float) -> pd.DataFrame:
    """
    Fallback : ARIMA(1,1,1) sur log(niveau) avec drift.
    """
    if (clean <= 0).any() or len(clean) < 20:
        return pd.DataFrame()

    y = np.log(clean)

    model = ARIMA(y, order=(1, 1, 1), trend="t")  # drift
    fit = model.fit()

    res = fit.get_forecast(steps=steps)
    sf = res.summary_frame(alpha=alpha)

    last_date = clean.index[-1]
    freq = _infer_future_freq(clean.index)
    future_dates = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]
    sf.index = future_dates

    out = pd.DataFrame(index=future_dates)
    out["forecast"] = np.exp(sf["mean"])
    out["lower_conf"] = np.exp(sf["mean_ci_lower"])
    out["upper_conf"] = np.exp(sf["mean_ci_upper"])
    return out[["forecast", "lower_conf", "upper_conf"]]


def _forecast_from_log_returns(clean: pd.Series, steps: int, alpha: float) -> pd.DataFrame:
    """
    Méthode principale (niveau 2) :
    1) r_t = diff(log(level))  (rendements log)
    2) ARMA via ARIMA(order=(1,0,1), trend='c') sur r_t
    3) Reconstruction :
       log(level_{t+h}) = log(level_t) + cumsum(r_hat)
       IC : approx via sqrt(cumsum(se_mean^2)) sur le cumul
    """
    if (clean <= 0).any():
        return pd.DataFrame()

    log_level = np.log(clean)
    log_ret = log_level.diff().dropna()

    # Il faut suffisamment de points pour estimer un modèle
    if len(log_ret) < 30:
        return pd.DataFrame()

    # ARMA sur rendements log (stationnaires), trend='c' => moyenne non nulle
    model = ARIMA(log_ret, order=(1, 0, 1), trend="c")
    fit = model.fit()

    fc = fit.get_forecast(steps=steps)
    sf = fc.summary_frame(alpha=alpha)

    # On veut la moyenne + l'erreur-type de la moyenne (mean_se)
    # summary_frame contient typiquement: mean, mean_se, mean_ci_lower, mean_ci_upper
    if "mean" not in sf.columns:
        return pd.DataFrame()

    # Erreur-type sur la moyenne des rendements prévus
    if "mean_se" in sf.columns:
        se = sf["mean_se"].to_numpy(dtype=float)
    else:
        # fallback : approx via CI
        # (upper-lower)/(2*z) où z ~ 1.96 à 95%
        z = 1.96
        se = ((sf["mean_ci_upper"] - sf["mean_ci_lower"]) / (2 * z)).to_numpy(dtype=float)

    mean_ret = sf["mean"].to_numpy(dtype=float)

    # Reconstruction du log-niveau
    last_log = float(log_level.iloc[-1])
    cum_mean = np.cumsum(mean_ret)

    # IC cumulatif (approx indépendance)
    z = 1.96  # pour alpha=0.05; si tu changes alpha, z n'est plus exact, mais c'est ok pour UI
    cum_se = np.sqrt(np.cumsum(se**2))

    future_log_forecast = last_log + cum_mean
    future_log_lower = future_log_forecast - z * cum_se
    future_log_upper = future_log_forecast + z * cum_se

    last_date = clean.index[-1]
    freq = _infer_future_freq(clean.index)
    future_dates = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]

    out = pd.DataFrame(index=future_dates)
    out["forecast"] = np.exp(future_log_forecast)
    out["lower_conf"] = np.exp(future_log_lower)
    out["upper_conf"] = np.exp(future_log_upper)

    # Sécurité bornes
    out["lower_conf"] = np.minimum(out["lower_conf"], out["upper_conf"])
    out["upper_conf"] = np.maximum(out["lower_conf"], out["upper_conf"])

    return out[["forecast", "lower_conf", "upper_conf"]]


def generate_forecast(series: pd.Series, steps: int = 30) -> pd.DataFrame:
    """
    Génère une prévision sur 'steps' jours.
    Retourne un DataFrame avec [forecast, lower_conf, upper_conf].

    Stratégie :
    - Méthode principale : ARMA sur rendements log + reconstruction du niveau
    - Fallback : ARIMA(1,1,1) sur log(niveau) avec drift
    """
    if series is None:
        return pd.DataFrame()

    clean = _clean_series(series)
    if clean.empty:
        return pd.DataFrame()

    # On protège contre steps aberrants
    steps = int(steps)
    if steps <= 0:
        return pd.DataFrame()

    alpha = 0.05

    try:
        # 1) Méthode rendements (préférée)
        out = _forecast_from_log_returns(clean, steps=steps, alpha=alpha)
        if out is not None and not out.empty:
            return out

        # 2) Fallback niveau log + drift
        out = _forecast_level_arima_log_drift(clean, steps=steps, alpha=alpha)
        if out is not None and not out.empty:
            return out

        return pd.DataFrame()

    except Exception as e:
        print(f"Erreur forecast: {e}")
        return pd.DataFrame()
