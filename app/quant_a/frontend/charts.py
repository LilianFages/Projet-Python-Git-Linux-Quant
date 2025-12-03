# app/quant_a/frontend/charts.py

import altair as alt
import pandas as pd
from app.quant_a.backend.strategies import StrategyResult
from app.common.market_time import build_compressed_intraday_df, MARKET_HOURS
from app.common.config import commodity_intraday_ok
import numpy as np


import altair as alt
import pandas as pd
# Assurez-vous d'avoir vos imports habituels et les fonctions 
# build_compressed_intraday_df, commodity_intraday_ok, MARKET_HOURS définies.

def make_price_chart(
    df: pd.DataFrame,
    selected_period: str,
    asset_class: str,
    equity_index: str | None,
    symbol: str,
    interval: str,
    chart_style: str = "Ligne",
) -> alt.Chart | None:
    """
    Construit le graphique de prix (Ligne ou Bougies) avec Volume optionnel en superposition.
    """

    if df is None or df.empty:
        return None

    # Aplatir MultiIndex éventuel
    df_base = df.copy()
    if isinstance(df_base.columns, pd.MultiIndex):
        df_base.columns = df_base.columns.get_level_values(0)

    # Vérification colonnes minimales
    if "close" not in df_base.columns:
        return None
    
    # Couleurs
    COLOR_UP = "#00C805"
    COLOR_DOWN = "#FF333A"
    COLOR_WICK = "#888888"

    # --- NOUVEAU : Définition du paramètre interactif (Checkbox) ---
    # Crée un paramètre 'toggle_volume' qui est True par défaut,
    # et le lie à une case à cocher HTML.
    volume_toggle_param = alt.param(
        name="toggle_volume",
        value=True,
        bind=alt.binding_checkbox(name="Afficher le Volume ")
    )

    # --- NOUVEAU : Condition d'opacité basée sur la checkbox ---
    # Si 'toggle_volume' est vrai, opacité = 0.3, sinon 0.
    volume_opacity_condition = alt.condition(
        volume_toggle_param,
        alt.value(0.3),
        alt.value(0)
    )
    # ------------------------------------------------------------

    # =========================================================
    # CAS 1 : 5 jours -> temps compressé
    # =========================================================
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

        # --- Ticks et Axe X ---
        df_ticks = df_plot.copy()
        df_ticks['date_str'] = df_ticks['date'].dt.strftime('%d/%m')
        ticks_df = df_ticks.drop_duplicates(subset=['date_str'], keep='first')
        
        tick_indices = ticks_df['bar_index'].tolist()
        tick_labels = ticks_df['date_str'].tolist()

        label_expr = " : ".join([f"datum.value == {int(i)} ? '{l}'" for i, l in zip(tick_indices, tick_labels)])
        if label_expr:
            label_expr += " : ''"
        
        x_axis_custom = alt.Axis(values=tick_indices, labelExpr=label_expr, title="Date", grid=False)

        # --- BRANCHE : BOUGIES + VOLUME ---
        if chart_style == "Bougies" and {"open", "high", "low"}.issubset(df_plot.columns):
            y_min = float(df_plot["low"].min())
            y_max = float(df_plot["high"].max())
            padding = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
            
            base = alt.Chart(df_plot).encode(x=alt.X("bar_index:Q", axis=x_axis_custom))

            rule = base.mark_rule(color=COLOR_WICK).encode(
                y=alt.Y("low:Q", scale=alt.Scale(domain=[y_min - padding, y_max + padding]), axis=alt.Axis(title="Prix", grid=True)),
                y2=alt.Y2("high:Q")
            )
            bar = base.mark_bar().encode(
                y=alt.Y("open:Q"),
                y2=alt.Y2("close:Q"),
                color=alt.condition("datum.open < datum.close", alt.value(COLOR_UP), alt.value(COLOR_DOWN)),
                tooltip=[
                    alt.Tooltip("date:T", title="Date", format="%d/%m %H:%M"),
                    alt.Tooltip("open:Q", title="Ouv", format=",.2f"),
                    alt.Tooltip("high:Q", title="Haut", format=",.2f"),
                    alt.Tooltip("low:Q", title="Bas", format=",.2f"),
                    alt.Tooltip("close:Q", title="Clôt", format=",.2f"),
                    alt.Tooltip("volume:Q", title="Volume", format=",.0f") if "volume" in df_plot.columns else alt.Tooltip("close:Q", title="Volume", format=",.0f")
                ]
            )
            price_chart = rule + bar

            # Graphique de Volume (Superposé en bas)
            if "volume" in df_plot.columns and df_plot["volume"].sum() > 0:
                vol_max = float(df_plot["volume"].max())
                # ON UTILISE volume_opacity_condition ICI AU LIEU DE opacity=0.3
                volume_chart = base.mark_bar().encode(
                    opacity=volume_opacity_condition, # <--- Modifié ici
                    y=alt.Y("volume:Q", axis=None, scale=alt.Scale(domain=[0, vol_max * 5])),
                    color=alt.condition("datum.open < datum.close", alt.value(COLOR_UP), alt.value(COLOR_DOWN)),
                    tooltip=[alt.Tooltip("volume:Q", title="Volume", format=",.0f")]
                )
                # ON AJOUTE LE PARAMÈTRE AU GRAPHIQUE FINAL
                return alt.layer(volume_chart, price_chart).resolve_scale(y='independent').add_params(volume_toggle_param).interactive() # <--- add_params ici
            
            return price_chart.interactive()

        # --- BRANCHE : LIGNE ---
        else:
            y_min = float(df_plot["close"].min())
            y_max = float(df_plot["close"].max())
            if not pd.notna(y_min) or not pd.notna(y_max): return None
            padding = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

            x_enc = alt.X("bar_index:Q", axis=x_axis_custom)
            y_enc = alt.Y("close:Q", title="Prix", scale=alt.Scale(domain=[y_min - padding, y_max + padding]), axis=alt.Axis(grid=True))

            return alt.Chart(df_plot).mark_line().encode(
                x=x_enc, y=y_enc,
                tooltip=[
                    alt.Tooltip("date:T", title="Date/heure réelle", format="%d/%m/%Y %H:%M"),
                    alt.Tooltip("close:Q", title="Clôture", format=",.2f"),
                ]
            ).interactive()

    # =========================================================
    # CAS 2 : 1 mois -> temps compressé
    # =========================================================
    if (
        selected_period == "1 mois"
        and (
            (asset_class in ("Actions", "ETF", "Indices") and equity_index in MARKET_HOURS)
            or (asset_class == "Forex")
            or (asset_class == "Matières premières" and commodity_intraday_ok(symbol))
        )
    ):
        if asset_class in ("Actions", "ETF", "Indices"): market_key = equity_index
        elif asset_class == "Forex": market_key = "FOREX"
        else: market_key = "COMMODITIES"

        df_plot = build_compressed_intraday_df(df, market_key, freq="30min")
        if df_plot.empty or "close" not in df_plot.columns:
            return None

        if "date" not in df_plot.columns:
            df_plot = df_plot.reset_index().rename(columns={"index": "date"})
        if "bar_index" not in df_plot.columns:
            df_plot = df_plot.reset_index(drop=True)
            df_plot["bar_index"] = range(len(df_plot))

        # --- Ticks et Axe X ---
        df_ticks = df_plot.copy()
        df_ticks['date_str'] = df_ticks['date'].dt.strftime('%d/%m')
        ticks_df = df_ticks.drop_duplicates(subset=['date_str'], keep='first')
        
        step = max(1, len(ticks_df) // 6)
        ticks_df = ticks_df.iloc[::step]

        tick_indices = ticks_df['bar_index'].tolist()
        tick_labels = ticks_df['date_str'].tolist()

        label_expr = " : ".join([f"datum.value == {int(i)} ? '{l}'" for i, l in zip(tick_indices, tick_labels)])
        if label_expr:
            label_expr += " : ''"

        x_axis_custom = alt.Axis(values=tick_indices, labelExpr=label_expr, title="Date", grid=False)

        # --- BRANCHE : BOUGIES + VOLUME ---
        if chart_style == "Bougies" and {"open", "high", "low"}.issubset(df_plot.columns):
            y_min = float(df_plot["low"].min())
            y_max = float(df_plot["high"].max())
            padding = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
            
            base = alt.Chart(df_plot).encode(x=alt.X("bar_index:Q", axis=x_axis_custom))
            
            rule = base.mark_rule(color=COLOR_WICK).encode(
                y=alt.Y("low:Q", scale=alt.Scale(domain=[y_min - padding, y_max + padding]), axis=alt.Axis(title="Prix", grid=True)),
                y2=alt.Y2("high:Q")
            )
            bar = base.mark_bar().encode(
                y=alt.Y("open:Q"),
                y2=alt.Y2("close:Q"),
                color=alt.condition("datum.open < datum.close", alt.value(COLOR_UP), alt.value(COLOR_DOWN)),
                tooltip=[
                    alt.Tooltip("date:T", title="Date", format="%d/%m %H:%M"),
                    alt.Tooltip("open:Q", title="Ouv", format=",.2f"),
                    alt.Tooltip("high:Q", title="Haut", format=",.2f"),
                    alt.Tooltip("low:Q", title="Bas", format=",.2f"),
                    alt.Tooltip("close:Q", title="Clôt", format=",.2f"),
                    alt.Tooltip("volume:Q", title="Volume", format=",.0f") if "volume" in df_plot.columns else alt.Tooltip("close:Q", title="Vol", format=",.0f")
                ]
            )
            price_chart = rule + bar

            if "volume" in df_plot.columns and df_plot["volume"].sum() > 0:
                vol_max = float(df_plot["volume"].max())
                # ON UTILISE volume_opacity_condition ICI
                volume_chart = base.mark_bar().encode(
                    opacity=volume_opacity_condition, # <--- Modifié ici
                    y=alt.Y("volume:Q", axis=None, scale=alt.Scale(domain=[0, vol_max * 5])),
                    color=alt.condition("datum.open < datum.close", alt.value(COLOR_UP), alt.value(COLOR_DOWN)),
                    tooltip=[alt.Tooltip("volume:Q", title="Volume", format=",.0f")]
                )
                # ON AJOUTE LE PARAMÈTRE
                return alt.layer(volume_chart, price_chart).resolve_scale(y='independent').add_params(volume_toggle_param).interactive() # <--- add_params ici

            return price_chart.interactive()

        # --- BRANCHE : LIGNE ---
        else:
            y_min = float(df_plot["close"].min())
            y_max = float(df_plot["close"].max())
            if not pd.notna(y_min) or not pd.notna(y_max): return None
            padding = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

            x_enc = alt.X("bar_index:Q", axis=x_axis_custom)
            y_enc = alt.Y("close:Q", title="Prix", scale=alt.Scale(domain=[y_min - padding, y_max + padding]), axis=alt.Axis(grid=True))

            return alt.Chart(df_plot).mark_line().encode(
                x=x_enc, y=y_enc,
                tooltip=[
                    alt.Tooltip("date:T", title="Date/heure réelle", format="%d/%m/%Y %H:%M"),
                    alt.Tooltip("close:Q", title="Clôture", format=",.2f"),
                ]
            ).interactive()

    # =========================================================
    # CAS 3 : toutes les autres périodes
    # =========================================================
    df_plot = df_base.reset_index().sort_values("date")

    # Définition de l'axe X (partagée pour assurer la même échelle temporelle)
    if selected_period == "1 jour":
        x_enc = alt.X("date:T", title="Heure", axis=alt.Axis(format="%H:%M", labelAngle=0, tickCount=24))
        tooltip_fmt = "%d/%m/%Y %H:%M"
    elif selected_period == "5 jours":
        x_enc = alt.X("date:T", title="Date / heure", axis=alt.Axis(format="%d/%m %Hh", labelAngle=45, tickCount=10))
        tooltip_fmt = "%d/%m/%Y %H:%M"
    elif selected_period == "1 mois":
        x_enc = alt.X("date:T", title="Date", axis=alt.Axis(format="%d/%m", labelAngle=45, tickCount=15))
        tooltip_fmt = "%d/%m/%Y %H:%M"
    elif selected_period == "6 mois":
        x_enc = alt.X("date:T", title="Date", axis=alt.Axis(format="%b %d", labelAngle=0, tickCount=12))
        tooltip_fmt = "%d/%m/%Y"
    elif selected_period in ("Année écoulée", "1 année"):
        x_enc = alt.X("date:T", title="Mois", axis=alt.Axis(format="%b", labelAngle=0, tickCount=12))
        tooltip_fmt = "%d/%m/%Y"
    elif selected_period == "5 années":
        x_enc = alt.X("date:T", title="Année", axis=alt.Axis(format="%Y", labelAngle=0, tickCount=6))
        tooltip_fmt = "%d/%m/%Y"
    else:
        x_enc = alt.X("date:T", title="Année", axis=alt.Axis(format="%Y", labelAngle=0, tickCount=10))
        tooltip_fmt = "%d/%m/%Y"

    # --- BRANCHE : BOUGIES + VOLUME ---
    if chart_style == "Bougies" and {"open", "high", "low"}.issubset(df_plot.columns):
        y_min = float(df_plot["low"].min())
        y_max = float(df_plot["high"].max())
        padding = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

        base = alt.Chart(df_plot).encode(x=x_enc)
        
        rule = base.mark_rule(color=COLOR_WICK).encode(
            y=alt.Y("low:Q", scale=alt.Scale(domain=[y_min - padding, y_max + padding]), axis=alt.Axis(title="Prix")),
            y2=alt.Y2("high:Q")
        )
        bar = base.mark_bar().encode(
            y=alt.Y("open:Q"),
            y2=alt.Y2("close:Q"),
            color=alt.condition("datum.open < datum.close", alt.value(COLOR_UP), alt.value(COLOR_DOWN)),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format=tooltip_fmt),
                alt.Tooltip("open:Q", title="Ouv", format=",.2f"),
                alt.Tooltip("high:Q", title="Haut", format=",.2f"),
                alt.Tooltip("low:Q", title="Bas", format=",.2f"),
                alt.Tooltip("close:Q", title="Clôt", format=",.2f"),
                alt.Tooltip("volume:Q", title="Volume", format=",.0f") if "volume" in df_plot.columns else alt.Tooltip("close:Q", title="Vol", format=",.0f")
            ]
        )
        price_chart = rule + bar

        if "volume" in df_plot.columns and df_plot["volume"].sum() > 0:
            vol_max = float(df_plot["volume"].max())
            # ON UTILISE volume_opacity_condition ICI
            volume_chart = base.mark_bar().encode(
                opacity=volume_opacity_condition, # <--- Modifié ici
                y=alt.Y("volume:Q", axis=None, scale=alt.Scale(domain=[0, vol_max * 5])),
                color=alt.condition("datum.open < datum.close", alt.value(COLOR_UP), alt.value(COLOR_DOWN)),
                tooltip=[alt.Tooltip("volume:Q", title="Volume", format=",.0f")]
            )
            # ON AJOUTE LE PARAMÈTRE
            return alt.layer(volume_chart, price_chart).resolve_scale(y='independent').add_params(volume_toggle_param).interactive() # <--- add_params ici

        return price_chart.interactive()

    # --- BRANCHE : LIGNE (Ton code exact) ---
    else:
        y_min = float(df_base["close"].min())
        y_max = float(df_base["close"].max())
        if not pd.notna(y_min) or not pd.notna(y_max): return None
        padding = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

        if selected_period in ("1 jour", "5 jours", "1 mois"):
            date_tooltip = alt.Tooltip("date:T", title="Date/heure", format="%d/%m/%Y %H:%M")
        else:
            date_tooltip = alt.Tooltip("date:T", title="Date", format="%d/%m/%Y")

        return alt.Chart(df_plot).mark_line().encode(
            x=x_enc,
            y=alt.Y("close:Q", title="Prix", scale=alt.Scale(domain=[y_min - padding, y_max + padding])),
            tooltip=[
                date_tooltip,
                alt.Tooltip("close:Q", title="Clôture", format=",.2f"),
            ],
        ).interactive()


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

def make_seasonality_heatmap(df: pd.DataFrame, return_col: str = "strategy_return") -> alt.Chart | None:
    """
    Génère une Heatmap des rendements mensuels (Année vs Mois).
    Inclus un nettoyage automatique des années partielles vides (artefacts de début).
    """
    if df is None or df.empty:
        return None

    df_copy = df.copy()
    
    if "date" not in df_copy.columns and isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy = df_copy.reset_index().rename(columns={"index": "date"})
    elif "date" not in df_copy.columns:
        return None

    # 1. Calcul des rendements mensuels
    df_copy["ret_factor"] = 1 + df_copy[return_col]
    
    monthly_df = (
        df_copy.set_index("date")["ret_factor"]
        .resample("ME")  # 'ME' pour pandas récent, ou 'M'
        .prod() - 1
    ).reset_index()
    
    monthly_df.columns = ["date", "monthly_return"]
    
    monthly_df["year"] = monthly_df["date"].dt.year
    monthly_df["month"] = monthly_df["date"].dt.strftime('%b') 
    
    # --- AJOUT : NETTOYAGE ARTEFACT DÉBUT ---
    # Si la première année a seulement 1 mois de données ET que le rendement est 0% (ou proche), on l'enlève.
    years = monthly_df['year'].unique()
    if len(years) > 1:
        first_year = min(years)
        first_year_data = monthly_df[monthly_df['year'] == first_year]
        
        # Condition : 1 seul mois enregistré et rendement nul
        if len(first_year_data) <= 1 and abs(first_year_data.iloc[0]['monthly_return']) < 0.0001:
            monthly_df = monthly_df[monthly_df['year'] != first_year]
    # ----------------------------------------

    months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # 3. Graphique Altair
    base = alt.Chart(monthly_df).transform_filter(
        alt.datum.month != None
    ).encode(
        x=alt.X("month:N", sort=months_order, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("year:O", title=None)
    )

    heatmap = base.mark_rect().encode(
        color=alt.Color(
            "monthly_return:Q",
            title="Rendement",
            scale=alt.Scale(scheme="redyellowgreen", domainMid=0),
            legend=None
        ),
        tooltip=[
            alt.Tooltip("year:O", title="Année"),
            alt.Tooltip("month:N", title="Mois"),
            alt.Tooltip("monthly_return:Q", title="Rendement", format=".2%")
        ]
    )

    text = base.mark_text(baseline="middle").encode(
        text=alt.Text("monthly_return:Q", format=".1%"),
        color=alt.value("black")
    )

    return (heatmap + text)


def make_rolling_stats_chart(
    df: pd.DataFrame, 
    strategy_col: str = "strategy_return", 
    benchmark_col: str = "benchmark_return",
    window_days: int = 126
) -> alt.Chart | None:
    """
    Affiche le Beta et la Corrélation glissants sur une fenêtre donnée (ex: 126 jours = 6 mois).
    """
    if df is None or df.empty:
        return None
    
    # Sécurisation des colonnes requises
    if strategy_col not in df.columns or benchmark_col not in df.columns:
        return None
        
    df_calc = df.copy()

    # --- CORRECTION : Gestion de l'index Date ---
    if "date" not in df_calc.columns and isinstance(df_calc.index, pd.DatetimeIndex):
        df_calc = df_calc.reset_index()
        if "date" not in df_calc.columns and "index" in df_calc.columns:
            df_calc = df_calc.rename(columns={"index": "date"})
            
    if "date" not in df_calc.columns:
        return None
    # ---------------------------------------------

    df_calc = df_calc.sort_values("date")
    
    # 1. Calculs Glissants (Rolling)
    df_rolling = df_calc.set_index('date')
    
    rolling_window = df_rolling.rolling(window=window_days)
    
    rolling_corr = rolling_window[strategy_col].corr(df_rolling[benchmark_col])
    
    rolling_cov = rolling_window[strategy_col].cov(df_rolling[benchmark_col])
    rolling_var = rolling_window[benchmark_col].var()
    rolling_beta = rolling_cov / rolling_var
    
    # Reconstitution du DataFrame pour le plot Altair
    df_plot = pd.DataFrame({
        "date": rolling_corr.index,
        "rolling_corr": rolling_corr.values,
        "rolling_beta": rolling_beta.values
    }).dropna() # On enlève les NaN du début de période
    
    # Transformation format "Long"
    df_long = df_plot.melt(
        id_vars=["date"], 
        value_vars=["rolling_corr", "rolling_beta"], 
        var_name="Metric", 
        value_name="Value"
    )
    
    # Dictionnaire pour renommer proprement dans la légende
    label_map = {
        "rolling_corr": f"Corrélation ({window_days}j)", 
        "rolling_beta": f"Beta ({window_days}j)"
    }
    df_long["MetricLabel"] = df_long["Metric"].map(label_map)

    # 2. Graphique
    chart = alt.Chart(df_long).mark_line().encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("Value:Q", title="Valeur"),
        # CORRECTION : On place la légende en bas pour ne pas gêner le titre
        color=alt.Color("MetricLabel:N", title=None, legend=alt.Legend(orient="bottom")),
        strokeDash=alt.condition(
            alt.datum.Metric == 'rolling_beta',
            alt.value([4, 2]),  # Beta en pointillés
            alt.value([0])      # Corrélation ligne pleine
        ),
        tooltip=[
            alt.Tooltip("date:T", format="%d/%m/%Y"),
            alt.Tooltip("MetricLabel:N", title="Indicateur"),
            alt.Tooltip("Value:Q", format=".2f")
        ]
    ) # CORRECTION : Retrait du titre Altair (.properties(title=...))
    
    # Ligne zéro et un
    rule_zero = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='gray', opacity=0.5).encode(y='y')
    rule_one = alt.Chart(pd.DataFrame({'y': [1]})).mark_rule(color='gray', strokeDash=[2,2], opacity=0.3).encode(y='y')

    return (chart + rule_zero + rule_one).interactive()