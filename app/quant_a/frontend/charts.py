# app/quant_a/frontend/charts.py

import altair as alt
import pandas as pd
from app.quant_a.backend.strategies import StrategyResult
from app.common.market_time import build_compressed_intraday_df, MARKET_HOURS
from app.common.config import commodity_intraday_ok
import numpy as np


def make_price_chart(
    df: pd.DataFrame,
    selected_period: str,
    asset_class: str,
    equity_index: str | None,
    symbol: str,
    interval: str,
) -> alt.Chart | None:
    """
    Construit le graphique de prix principal en gérant :
      - 5 jours / 1 mois en temps de marché compressé (bar_index)
      - toutes les autres périodes en dates réelles
    """

    if df is None or df.empty:
        return None

    # Aplatir MultiIndex éventuel
    df_base = df.copy()
    if isinstance(df_base.columns, pd.MultiIndex):
        df_base.columns = df_base.columns.get_level_values(0)

    if "close" not in df_base.columns:
        return None

    # -----------------------------------
    # CAS 1 : 5 jours -> temps compressé
    # -----------------------------------
    if (
        selected_period == "5 jours"
        and asset_class in ("Actions", "ETF", "Indices")
        and equity_index in MARKET_HOURS
    ):
        df_plot = build_compressed_intraday_df(df, equity_index, freq="15min")
        if df_plot.empty or "close" not in df_plot.columns:
            return None

        if "date" not in df_plot.columns:
            df_plot = df_plot.reset_index().rename(columns={"index": "date"})
        if "bar_index" not in df_plot.columns:
            df_plot = df_plot.reset_index(drop=True)
            df_plot["bar_index"] = range(len(df_plot))

        y_min = float(df_plot["close"].min())
        y_max = float(df_plot["close"].max())
        if not pd.notna(y_min) or not pd.notna(y_max):
            return None
        padding = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

        x_enc = alt.X(
            "bar_index:Q",
            title=None,
            axis=alt.Axis(grid=False),
        )
        y_enc = alt.Y(
            "close:Q",
            title="Prix",
            scale=alt.Scale(domain=[y_min - padding, y_max + padding]),
            axis=alt.Axis(grid=True),
        )

        chart = (
            alt.Chart(df_plot)
            .mark_line()
            .encode(
                x=x_enc,
                y=y_enc,
                tooltip=[
                    alt.Tooltip(
                        "date:T",
                        title="Date/heure réelle",
                        format="%d/%m/%Y %H:%M",
                    ),
                    alt.Tooltip("close:Q", title="Clôture", format=",.2f"),
                ],
            )
            .interactive()
        )
        return chart

    # -----------------------------------
    # CAS 2 : 1 mois -> temps compressé
    # -----------------------------------
    if (
        selected_period == "1 mois"
        and (
            (asset_class in ("Actions", "ETF", "Indices") and equity_index in MARKET_HOURS)
            or (asset_class == "Forex")
            or (
                asset_class == "Matières premières"
                and commodity_intraday_ok(symbol)
            )
        )
    ):
        if asset_class in ("Actions", "ETF", "Indices"):
            market_key = equity_index
        elif asset_class == "Forex":
            market_key = "FOREX"
        else:
            market_key = "COMMODITIES"

        df_plot = build_compressed_intraday_df(df, market_key, freq="30min")
        if df_plot.empty or "close" not in df_plot.columns:
            return None

        if "date" not in df_plot.columns:
            df_plot = df_plot.reset_index().rename(columns={"index": "date"})
        if "bar_index" not in df_plot.columns:
            df_plot = df_plot.reset_index(drop=True)
            df_plot["bar_index"] = range(len(df_plot))

        y_min = float(df_plot["close"].min())
        y_max = float(df_plot["close"].max())
        if not pd.notna(y_min) or not pd.notna(y_max):
            return None
        padding = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

        x_enc = alt.X(
            "bar_index:Q",
            title=None,
            axis=alt.Axis(grid=False),
        )
        y_enc = alt.Y(
            "close:Q",
            title="Prix",
            scale=alt.Scale(domain=[y_min - padding, y_max + padding]),
            axis=alt.Axis(grid=True),
        )

        chart = (
            alt.Chart(df_plot)
            .mark_line()
            .encode(
                x=x_enc,
                y=y_enc,
                tooltip=[
                    alt.Tooltip(
                        "date:T",
                        title="Date/heure réelle",
                        format="%d/%m/%Y %H:%M",
                    ),
                    alt.Tooltip("close:Q", title="Clôture", format=",.2f"),
                ],
            )
            .interactive()
        )
        return chart

    # -----------------------------------
    # CAS 3 : toutes les autres périodes
    # -----------------------------------
    df_plot = df_base.reset_index().sort_values("date")

    y_min = float(df_base["close"].min())
    y_max = float(df_base["close"].max())
    if not pd.notna(y_min) or not pd.notna(y_max):
        return None
    padding = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

    # Axe X en fonction de la période
    if selected_period == "1 jour":
        x_enc = alt.X(
            "date:T",
            title="Heure",
            axis=alt.Axis(format="%H:%M", labelAngle=0, tickCount=24),
        )
    elif selected_period == "5 jours":
        x_enc = alt.X(
            "date:T",
            title="Date / heure",
            axis=alt.Axis(format="%d/%m %Hh", labelAngle=45, tickCount=10),
        )
    elif selected_period == "1 mois":
        x_enc = alt.X(
            "date:T",
            title="Date",
            axis=alt.Axis(format="%d/%m", labelAngle=45, tickCount=15),
        )
    elif selected_period == "6 mois":
        x_enc = alt.X(
            "date:T",
            title="Date",
            axis=alt.Axis(format="%b %d", labelAngle=0, tickCount=12),
        )
    elif selected_period in ("Année écoulée", "1 année"):
        x_enc = alt.X(
            "date:T",
            title="Mois",
            axis=alt.Axis(format="%b", labelAngle=0, tickCount=12),
        )
    elif selected_period == "5 années":
        x_enc = alt.X(
            "date:T",
            title="Année",
            axis=alt.Axis(format="%Y", labelAngle=0, tickCount=6),
        )
    else:
        x_enc = alt.X(
            "date:T",
            title="Année",
            axis=alt.Axis(format="%Y", labelAngle=0, tickCount=10),
        )

    # Tooltip date selon la période
    if selected_period in ("1 jour", "5 jours", "1 mois"):
        date_tooltip = alt.Tooltip(
            "date:T",
            title="Date/heure",
            format="%d/%m/%Y %H:%M",
        )
    else:
        date_tooltip = alt.Tooltip(
            "date:T",
            title="Date",
            format="%d/%m/%Y",
        )

    chart = (
        alt.Chart(df_plot)
        .mark_line()
        .encode(
            x=x_enc,
            y=alt.Y(
                "close:Q",
                title="Prix",
                scale=alt.Scale(domain=[y_min - padding, y_max + padding]),
            ),
            tooltip=[
                date_tooltip,
                alt.Tooltip("close:Q", title="Clôture", format=",.2f"),
            ],
        )
        .interactive()
    )

    return chart


def make_strategy_comparison_chart(
    df: pd.DataFrame,
    strategy_result: StrategyResult,
    selected_period: str,
) -> alt.Chart:
    """
    Graphique 2 : Performance normalisée (Base 1.0)
    CORRIGÉ : Utilise l'alignement par Index pour éviter le crash de longueur.
    """
    # Sécurités de base
    if df is None or df.empty or strategy_result is None:
        return alt.Chart(pd.DataFrame({"Date": [], "valeur": []})).mark_line()

    # 1. Alignement intelligent via Pandas
    # On crée un container vide indexé sur les dates du DataFrame principal
    data = pd.DataFrame(index=df.index)
    
    # On injecte les séries. Pandas aligne les dates. 
    # Si une date manque dans la stratégie, il met NaN (pas de crash).
    data["Actif"] = strategy_result.benchmark
    data["Stratégie"] = strategy_result.equity_curve
    
    # 2. Nettoyage des NaN (ex: début du backtest)
    data = data.dropna()

    if data.empty:
         return alt.Chart(pd.DataFrame({"Date": [], "valeur": []})).mark_line()

    # 3. Normalisation (Base 1.0 sur la première valeur valide)
    data["Actif (normalisé)"] = data["Actif"] / data["Actif"].iloc[0]
    data["Stratégie (normalisé)"] = data["Stratégie"] / data["Stratégie"].iloc[0]
    
    # 4. Formatage pour Altair
    # Reset index et on force le nom de la colonne date en 'Date'
    plot_df = data[["Actif (normalisé)", "Stratégie (normalisé)"]].reset_index()
    plot_df = plot_df.rename(columns={plot_df.columns[0]: 'Date'})
    
    # Passage en format long (Melt)
    source = plot_df.melt('Date', var_name="Série", value_name="valeur")
    
    # --- Axe X et tooltips (Ton code original conservé) ---
    if selected_period in ("1 jour", "5 jours", "1 mois"):
        x_axis = alt.X("Date:T", title="Date / heure", axis=alt.Axis(format="%d/%m %H:%M", labelAngle=45))
        date_tooltip = alt.Tooltip("Date:T", title="Date/heure", format="%d/%m/%Y %H:%M")
    else:
        x_axis = alt.X("Date:T", title="Date", axis=alt.Axis(format="%d/%m/%Y", labelAngle=0))
        date_tooltip = alt.Tooltip("Date:T", title="Date", format="%d/%m/%Y")

    chart = (
        alt.Chart(source)
        .mark_line()
        .encode(
            x=x_axis,
            y=alt.Y("valeur:Q", title="Performance normalisée (base 1.0)"),
            color=alt.Color("Série:N", title="Série", scale=alt.Scale(domain=['Actif (normalisé)', 'Stratégie (normalisé)'], range=['#5DADE2', '#2E86C1'])),
            tooltip=[
                date_tooltip,
                alt.Tooltip("Série:N", title="Série"),
                alt.Tooltip("valeur:Q", title="Valeur normalisée", format=",.2f"),
            ],
        )
        .properties(height=350)
        .interactive()
    )

    return chart

def make_strategy_value_chart(
    df: pd.DataFrame,
    strategy_result: StrategyResult,
    selected_period: str,
) -> alt.Chart:
    """
    Graphique 1 : Valeur du portefeuille ($)
    CORRIGÉ : Utilise l'alignement par Index.
    """
    # Sécurités de base
    if df is None or df.empty or strategy_result is None:
        return alt.Chart(pd.DataFrame({"Date": [], "valeur": []})).mark_line()

    # 1. Alignement intelligent via Pandas
    data = pd.DataFrame(index=df.index)
    data["Buy & Hold"] = strategy_result.benchmark
    data["Stratégie"] = strategy_result.equity_curve
    
    # 2. Nettoyage
    data = data.dropna()
    
    if data.empty:
        return alt.Chart(pd.DataFrame({"Date": [], "valeur": []})).mark_line()

    # 3. Formatage pour Altair
    plot_df = data.reset_index()
    plot_df = plot_df.rename(columns={plot_df.columns[0]: 'Date'})
    
    source = plot_df.melt('Date', var_name="Série", value_name="valeur")

    # --- Axe X et tooltips (Ton code original conservé) ---
    if selected_period in ("1 jour", "5 jours", "1 mois"):
        x_axis = alt.X("Date:T", title="Date / heure", axis=alt.Axis(format="%d/%m %H:%M", labelAngle=45))
        date_tooltip = alt.Tooltip("Date:T", title="Date/heure", format="%d/%m/%Y %H:%M")
    else:
        x_axis = alt.X("Date:T", title="Date", axis=alt.Axis(format="%d/%m/%Y", labelAngle=0))
        date_tooltip = alt.Tooltip("Date:T", title="Date", format="%d/%m/%Y")

    chart = (
        alt.Chart(source)
        .mark_line()
        .encode(
            x=x_axis,
            y=alt.Y("valeur:Q", title="Valeur du portefeuille"),
            color=alt.Color("Série:N", title="Série", scale=alt.Scale(domain=['Buy & Hold', 'Stratégie'], range=['#5DADE2', '#2E86C1'])),
            tooltip=[
                date_tooltip,
                alt.Tooltip("Série:N", title="Série"),
                alt.Tooltip("valeur:Q", title="Valeur", format=",.2f"),
            ],
        )
        .properties(height=400)
        .interactive()
    )

    return chart

def make_returns_distribution_chart(equity_curve: pd.Series):
    """
    Histogramme des rendements journaliers
    """
    returns = equity_curve.pct_change().dropna()
    df_rets = pd.DataFrame({'Rendement': returns})

    chart = alt.Chart(df_rets).mark_bar(opacity=0.7, color='#29b5e8').encode(
        alt.X('Rendement', bin=alt.Bin(maxbins=50), title='Rendement Journalier'),
        alt.Y('count()', title='Fréquence'),
        tooltip=['count()']
    ).properties(
        title="Distribution des rendements journaliers",
        height=250
    )
    return chart

def make_drawdown_chart(equity_curve: pd.Series):
    """
    Graphique "Underwater" (Drawdown au cours du temps)
    """
    # Calcul du drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    
    df_dd = drawdown.reset_index()
    df_dd.columns = ['Date', 'Drawdown']

    chart = alt.Chart(df_dd).mark_area(color='red', opacity=0.3, line={'color':'darkred'}).encode(
        x='Date:T',
        y=alt.Y('Drawdown', axis=alt.Axis(format='%')),
        tooltip=[alt.Tooltip('Date', format='%Y-%m-%d'), alt.Tooltip('Drawdown', format='.2%')]
    ).properties(
        title="Underwater Plot (Drawdowns)",
        height=200
    )
    return chart

def make_forecast_chart(historical_series: pd.Series, forecast_df: pd.DataFrame):
    """
    Combine l'historique récent et la prévision
    """
    # On ne garde que les 90 derniers jours d'historique pour que le graphe soit lisible
    lookback = min(len(historical_series), 252)
    recent_history = historical_series.iloc[-lookback:].reset_index()
    
    recent_history.columns = ['Date', 'Prix']
    recent_history['Type'] = 'Historique'

    # Préparation forecast
    fcast = forecast_df.reset_index().rename(columns={'index': 'Date', 'forecast': 'Prix'})
    fcast['Type'] = 'Prévision'

    # --- CORRECTION DU TITRE DE L'AXE Y ICI ---
    # On définit l'axe Y sur le graph de base
    base_hist = alt.Chart(recent_history).mark_line(color='white').encode(
        x='Date:T',
        # On force le titre "Valeur du Portefeuille"
        y=alt.Y('Prix', scale=alt.Scale(zero=False), title="Valeur du Portefeuille")
    )

    # Chart Prévision (Ligne pointillée)
    line_fcast = alt.Chart(fcast).mark_line(strokeDash=[5, 5], color='#FFA500').encode(
        x='Date:T',
        y='Prix'
    )

    # Chart Intervalle de confiance (Zone)
    band_fcast = alt.Chart(forecast_df.reset_index().rename(columns={'index': 'Date'})).mark_area(opacity=0.2, color='#FFA500').encode(
        x='Date:T',
        y='lower_conf',
        y2='upper_conf'
    )

    return (base_hist + band_fcast + line_fcast).properties(
        title="Prévision ARIMA",
        height=300
    )