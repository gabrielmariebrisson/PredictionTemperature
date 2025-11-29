"""
Multi-City Weather Prediction System - Streamlit Version
Predicts temperature for Paris and Silicon Valley
Using Meteostat, trained models, and comparing with OpenWeatherMap forecasts
"""

import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Imports from src modules
from src.config import City, CITIES, LANGUAGES, WINDOW_SIZE, FORECAST_HORIZON, MODELS_BASE_PATH
from src.data_loader import collect_historical_data, preprocess_data, get_openweather_forecast
from src.model_service import load_model_info, prepare_scalers, predict_7day_forecast
from src.utils import gettext

# Page configuration
st.set_page_config(
    page_title= "Multi-City Weather Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #2980b9, #6dd5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .city-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}

# Initialisation de la langue
if 'language' not in st.session_state:
    st.session_state.language = 'fr'

# Sélecteur de langue
lang = st.sidebar.selectbox(
    "🌐 Language / Langue",
    options=list(LANGUAGES.keys()),
    format_func=lambda x: LANGUAGES[x],
    index=list(LANGUAGES.keys()).index(st.session_state.language)
)

st.session_state.language = lang

# Bouton de redirection
st.markdown(
    f"""
    <a href="https://gabriel.mariebrisson.fr" target="_blank" style="text-decoration:none;">
    <div style="
    display: inline-block;
    background: linear-gradient(135deg, #6A11CB 0%, #2575FC 100% );
    color: white;
    padding: 12px 25px;
    border-radius: 30px;
    text-align: center;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    box-shadow: 0 4px 15px rgba(37, 117, 252, 0.3);
    transition: all 0.3s ease;
    text-transform: uppercase;
    letter-spacing: 1px;
    border: 2px solid transparent;
    position: relative;
    overflow: hidden;
    ">
    {gettext("Retour", lang)}
    <span style="
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255,255,255,0.2);
    transform: scaleX(0);
    transform-origin: right;
    transition: transform 0.3s ease;
    z-index: 1;
    "></span>
    </div>
    </a>
    """,
    unsafe_allow_html=True
)


# Main App
st.markdown(gettext('<h1 class="main-header"> Multi-City Weather Prediction System</h1>', lang), unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header(gettext("⚙️ Configuration", lang))

    selected_cities = st.multiselect(
        gettext("Select Cities", lang),
        options=list(CITIES.keys()),
        default=list(CITIES.keys()),
        format_func=lambda x: f"{CITIES[x].emoji} {CITIES[x].name}"
    )

    st.divider()

    show_historical = st.checkbox(gettext("Show Historical Data", lang), value=False)
    show_comparison = st.checkbox(gettext("Compare with OpenWeather", lang), value=True)

    st.divider()

    st.info(gettext("""
    **About this app:**
    - Uses Conv1D + LSTM models
    - Trained on 10 years of data
    - Predicts 7 days ahead
    - Compares with OpenWeatherMap
    """, lang))

    if st.button(gettext("🔄 Refresh Predictions", lang), type="primary"):
        st.cache_data.clear()
        st.rerun()

# Main content
if not selected_cities:
    st.warning(gettext("Please select at least one city from the sidebar.", lang))
    st.stop()

# Create tabs for each city
tabs = st.tabs([f"{CITIES[city].emoji} {CITIES[city].name}" for city in selected_cities])

for tab, city_key in zip(tabs, selected_cities):
    with tab:
        city = CITIES[city_key]

        st.markdown(gettext(f"""
        <div class="city-card">
            <h2>{city.emoji} {city.name}</h2>
            <p>📍 Coordinates: {city.lat}°N, {city.lon}°E</p>
            <p>🕐 Timezone: {city.timezone}</p>
        </div>
        """, lang), unsafe_allow_html=True)

        # Data loading
        with st.spinner(gettext(f"Loading data for {city.name}...", lang)):
            try:
                # Collect and preprocess data
                df_raw = collect_historical_data(city, years_back=10)
                if df_raw is None:
                    st.error(gettext(f"No data available for {city.name}", lang))
                    continue

                df = preprocess_data(df_raw)
                if df is None:
                    st.error(gettext(f"Error preprocessing data for {city.name}", lang))
                    continue

                # Charger le modèle et les infos
                model, model_info = load_model_info(city_key)
                if model_info is None:
                    st.error(gettext(f"⚠️ Missing feature info for {city.name}.", lang))
                    continue

                # S'assurer que le DataFrame contient les bonnes colonnes
                expected_features = model_info['feature_cols']
                for col in expected_features:
                    if col not in df.columns:
                        df[col] = 0  # colonne manquante = 0 par défaut

                # Préparer les scalers avec les features attendues
                scalers = prepare_scalers(df, expected_features=expected_features)
                recent_data = df[expected_features].tail(WINDOW_SIZE).values

                # Vérifier que le modèle existe
                model_path = MODELS_BASE_PATH / f'{city_key}_model.keras'
                if model_path.exists():
                    # Make prediction
                    model_pred = predict_7day_forecast(model, recent_data, scalers)

                    # Get OpenWeather forecast
                    ow_forecast = get_openweather_forecast(city) if show_comparison else None

                    # Display predictions
                    st.subheader(gettext("📊 7-Day Temperature Forecast", lang))

                    # Prepare data for visualization
                    today = datetime.now().date()
                    dates = [today + timedelta(days=i) for i in range(len(model_pred))]

                    # Create dataframe for display
                    pred_df = pd.DataFrame({
                        gettext('Date', lang): dates,
                        gettext('Avg Temp', lang): model_pred[:, 0],
                        gettext('Min Temp', lang): model_pred[:, 1],
                        gettext('Max Temp', lang): model_pred[:, 2]
                    })

                    if ow_forecast:
                        # Make sure lengths match
                        ow_forecast = ow_forecast[:len(dates)]
                        if len(ow_forecast) < len(dates):
                            # pad missing days with NaN
                            missing = len(dates) - len(ow_forecast)
                            for i in range(missing):
                                ow_forecast.append({'temp_avg': np.nan, 'temp_min': np.nan, 'temp_max': np.nan})

                        pred_df['OW Avg'] = [f['temp_avg'] for f in ow_forecast]
                        pred_df['OW Min'] = [f['temp_min'] for f in ow_forecast]
                        pred_df['OW Max'] = [f['temp_max'] for f in ow_forecast]
                        pred_df['Δ Avg'] = pred_df[gettext('Avg Temp', lang)] - pred_df['OW Avg']

                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(gettext("Today's Avg", lang), f"{model_pred[0, 0]:.1f}°C")
                    with col2:
                        st.metric(gettext("Today's Min", lang), f"{model_pred[0, 1]:.1f}°C")
                    with col3:
                        st.metric(gettext("Today's Max", lang), f"{model_pred[0, 2]:.1f}°C")
                    with col4:
                        if ow_forecast:
                            mae = np.mean(np.abs(pred_df['Δ Avg']))
                            st.metric(gettext("MAE vs OW", lang), f"{mae:.2f}°C")

                    # Interactive chart
                    fig = go.Figure()

                    # Model predictions
                    fig.add_trace(go.Scatter(
                        x=pred_df[gettext('Date', lang)], y=pred_df[gettext('Avg Temp', lang)],
                        name='Model Avg', mode='lines+markers',
                        line=dict(color='#667eea', width=3),
                        marker=dict(size=8)
                    ))

                    fig.add_trace(go.Scatter(
                        x=pred_df[gettext('Date', lang)], y=pred_df[gettext('Max Temp', lang)],
                        name='Model Max', mode='lines',
                        line=dict(color='#f093fb', width=2, dash='dash')
                    ))

                    fig.add_trace(go.Scatter(
                        x=pred_df[gettext('Date', lang)], y=pred_df[gettext('Min Temp', lang)],
                        name='Model Min', mode='lines',
                        line=dict(color='#4facfe', width=2, dash='dash')
                    ))

                    # OpenWeather comparison
                    if ow_forecast:
                        fig.add_trace(go.Scatter(
                            x=pred_df[gettext('Date', lang)], y=pred_df['OW Avg'],
                            name='OpenWeather Avg', mode='lines+markers',
                            line=dict(color='#ff6b6b', width=2),
                            marker=dict(size=6, symbol='x')
                        ))

                    fig.update_layout(
                        title=f"Temperature Forecast - {city.name}",
                        xaxis_title="Date",
                        yaxis_title="Temperature (°C)",
                        hovermode='x unified',
                        height=500,
                        template='plotly_white'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Data table
                    st.subheader(gettext("📋 Detailed Forecast", lang))
                    st.dataframe(
                        pred_df.style.format({
                            gettext('Avg Temp', lang): '{:.1f}°C',
                            gettext('Min Temp', lang): '{:.1f}°C',
                            gettext('Max Temp', lang): '{:.1f}°C',
                            'OW Avg': '{:.1f}°C',
                            'OW Min': '{:.1f}°C',
                            'OW Max': '{:.1f}°C',
                            'Δ Avg': '{:.1f}°C'
                        }),
                        use_container_width=True
                    )

                    # Historical data
                    if show_historical:
                        st.subheader(gettext("📈 Historical Temperature Trends", lang))

                        hist_df = df[['temp_avg', 'temp_min', 'temp_max']].tail(365)

                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Scatter(
                            x=hist_df.index, y=hist_df['temp_avg'],
                            name='Average', mode='lines',
                            line=dict(color='#667eea', width=2)
                        ))

                        fig_hist.update_layout(
                            title=gettext("Last 365 Days - Temperature History", lang),
                            xaxis_title=gettext("Date", lang),
                            yaxis_title=gettext("Temperature (°C)", lang),
                            height=400,
                            template='plotly_white'
                        )

                        st.plotly_chart(fig_hist, use_container_width=True)

                        # Statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(gettext("**📊 Historical Statistics (Last Year)**", lang))
                            stats = hist_df['temp_avg'].describe()
                            st.write(gettext(f"- Mean: {stats['mean']:.1f}°C", lang))
                            st.write(gettext(f"- Std: {stats['std']:.1f}°C", lang))
                            st.write(gettext(f"- Min: {stats['min']:.1f}°C", lang))
                            st.write(gettext(f"- Max: {stats['max']:.1f}°C", lang))

                        with col2:
                            st.markdown(gettext("**📅 Monthly Averages**", lang))
                            monthly = df[['temp_avg']].groupby(df.index.month).mean()
                            for month, temp in monthly.iterrows():
                                month_name = datetime(2000, month, 1).strftime('%B')
                                st.write(f"- {month_name}: {temp['temp_avg']:.1f}°C")

                else:
                    st.error(gettext(f"Model not found for {city.name}. Please train the model first.", lang))
                    st.info(gettext(f"Expected path: {model_path}", lang))

            except Exception as e:
                st.error(gettext(f"Error processing {city.name}: {str(e)}", lang))
                st.exception(e)

# Section Présentation
st.header(gettext("Présentation", lang))
st.markdown(gettext(
    """
    Ce projet vise à prédire la méteo sur différente ville en fonction de l'historique et des paramètres tel que la pression, le taux de pluis etc ... Les séries temporelles sont un domaine complexe, que l'on retrouve dans la météo, nénamoins le cas d'application est diverse et varié tel que :

    **Applications potentielles :**
    - **Finance :** Prévision des prix des actions, analyse des tendances du marché.
    - **Santé :** Surveillance des signes vitaux, prédiction des épidémies.
    - **Agriculture :** Prévision des températures, prévision des plantations.
    - **Énergie :** Prévision de la demande énergétique, gestion des réseaux électriques.
    - **Transport :** Prévision du trafic, gestion des flottes de véhicules.
    - **Commerce de détail :** Prévision des ventes, gestion des stocks.
    Pour cela, nous avons utilisé l'api Meteostat qui nous permet de récupérer les données météorologiques historiques de différentes villes. Nous avons choisi Paris et la Silicon Valley pour leur contraste climatique. Nous avons fait le dernier entrainement sur les données de 8 octobre 2025, plus les données sont récentes plus la précision est bonne.
    Les données collectées incluent des paramètres tels que la température moyenne, minimale et maximale, les précipitations, la vitesse du vent, la pression atmosphérique, etc. Ces données sont ensuite nettoyées et standardisées pour garantir une qualité optimale avant l'entraînement du modèle.
    """, lang
))

# Section Architecture du Modèle
st.header(gettext("Architecture du Modèle", lang))
st.markdown(gettext(
    """
    Pour prédire des séries temporelles multivariées, il est crucial de capturer à la fois les motifs locaux et les dépendances séquentielles dans les données. Notre modèle combine des couches convolutionnelles et LSTM pour atteindre cet objectif.

    Le modèle se compose de plusieurs blocs :

    - **Couche Conv1D :** Cette couche extrait les motifs locaux dans les séquences temporelles grâce à des filtres convolutifs. Elle permet au modèle de détecter des tendances ou des motifs répétitifs sur de petites fenêtres de temps.

    - **Couches LSTM :** Trois couches LSTM sont empilées pour capturer les dépendances temporelles à long terme. Les deux premières couches renvoient des séquences complètes (`return_sequences=True`) afin que les informations temporelles puissent être transmises aux couches suivantes. La troisième couche renvoie uniquement le dernier état caché (`return_sequences=False`), qui résume l'information séquentielle.

    - **Couches Dense :** Après l'extraction des motifs et des dépendances séquentielles, les couches denses transforment la représentation en sorties prédictives. Des couches `Dropout` sont intercalées pour réduire le surapprentissage et améliorer la généralisation.

    Le modèle produit une sortie structurée pour plusieurs pas de temps (`forecast_horizon`) et pour plusieurs variables cibles (`n_targets`). La compilation utilise la **loss Huber** adaptée aux valeurs aberrantes, l'optimiseur **Adam** avec un taux d'apprentissage de 0.001, et suit les métriques **MAE** et **MSE** pour évaluer les performances.

    Les hyperparamètres clés incluent : la taille de la fenêtre (`window_size`), le nombre de filtres et de neurones dans les couches LSTM et Dense, le taux de dropout et le nombre de pas de temps prédits (`forecast_horizon`).
    """, lang
))
st.image("./templates/assets/température/model_architecture.png",
         caption="Structure du modèle de prédiction de météo",
         width=800)  # largeur en pixels


# Section Résultats
st.header(gettext("Résultats", lang))
st.markdown(gettext(
    """
    Les tests montrent que le modèle prédit correctement les températures journalières pour différentes localisations.

    📍 **Paris**
    MAE : 2.84°C
    Le modèle suit globalement les valeurs observées, avec de petits écarts pour certaines journées.

    📍 **Silicon Valley**
    MAE : 0.98°C
    Les prédictions sont proches des valeurs réelles, montrant une bonne précision du modèle.

    Ces résultats confirment que le modèle généralise bien et que les techniques de régularisation, comme le dropout, permettent de limiter le surajustement.
    """, lang
))


# Section Coût et Maintenance
st.header(gettext("Coût de Développement", lang))
st.markdown(gettext(
    """
    Le modèle a été entraîné sur une machine Linux avec les caractéristiques suivantes :

    - **Processeur :** AMD Ryzen 5 3500X 6 cœurs, fréquence max 4,12 GHz
    - **RAM :** 15 Go

    L'entraînement a été extrêmement rapide, prenant moins de **2 secondes** pour traiter **10 ans de données** d'une seule ville.

    Ces performances montrent que le modèle est très léger et efficace, capable de générer des prédictions rapides tout en restant précis.

    **Analyse des coûts :** L'usage de ressources limitées rend ce modèle économique et facilement déployable sur des machines standards.

    **Perspectives d'amélioration :** Il serait possible d'étendre le modèle à plusieurs villes sur le même modéle ou d'intégrer des données supplémentaires sans augmenter significativement le temps de calcul.
    """, lang
))


# Footer
st.markdown(gettext(
    """
    ---
    Développé par [Gabriel Marie-Brisson](https://gabriel.mariebrisson.fr )
    """, lang
))

