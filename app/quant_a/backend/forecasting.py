# app/quant_a/backend/forecasting.py
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def generate_forecast(series: pd.Series, steps: int = 30) -> pd.DataFrame:
    """
    Génère une prévision ARIMA simple sur 'steps' jours.
    Retourne un DataFrame avec [forecast, lower_conf, upper_conf].
    """
    # Nettoyage basique : pas de NaNs/Infs
    clean_series = series.dropna()
    if clean_series.empty:
        return pd.DataFrame()

    try:
        # Modèle ARIMA simple (1,1,1) pour la rapidité et robustesse générale
        # On pourrait optimiser les p,d,q avec auto_arima mais c'est lent pour une UI web
        model = ARIMA(clean_series, order=(1, 1, 1))
        model_fit = model.fit()

        forecast_res = model_fit.get_forecast(steps=steps)
        forecast_df = forecast_res.summary_frame(alpha=0.05)  # 95% conf

        # Création des dates futures (business days si possible, sinon days)
        last_date = clean_series.index[-1]
        future_dates = pd.date_range(start=last_date, periods=steps + 1, freq='B')[1:]
        
        forecast_df.index = future_dates
        forecast_df = forecast_df.rename(columns={
            'mean': 'forecast', 
            'mean_ci_lower': 'lower_conf', 
            'mean_ci_upper': 'upper_conf'
        })
        
        return forecast_df[['forecast', 'lower_conf', 'upper_conf']]

    except Exception as e:
        print(f"Erreur ARIMA: {e}")
        return pd.DataFrame()