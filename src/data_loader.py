"""
Data loading and preprocessing module
Handles historical data collection from Meteostat and OpenWeatherMap API
"""

import streamlit as st
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from meteostat import Point, Daily

from src.config import City, OPENWEATHER_API_KEY


@st.cache_data(ttl=3600)
def collect_historical_data(city: City, years_back: int = 10):
    """Collect historical weather data from Meteostat"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back*365)

    location = Point(city.lat, city.lon)
    data = Daily(location, start_date, end_date)
    df = data.fetch()

    if df.empty:
        return None

    df['city'] = city.name
    return df


def create_time_features(df):
    """Create time-based features"""
    df = df.copy()
    df['day_of_year'] = df.index.dayofyear
    df['month'] = df.index.month
    df['day_of_month'] = df.index.day
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df


def preprocess_data(df):
    """Clean and standardize dataset"""
    df = df.copy()

    column_mapping = {
        'tavg': 'temp_avg', 'tmin': 'temp_min', 'tmax': 'temp_max',
        'prcp': 'precipitation', 'snow': 'snowfall', 'wdir': 'wind_direction',
        'wspd': 'wind_speed', 'wpgt': 'wind_gust', 'pres': 'pressure', 'tsun': 'sunshine'
    }
    df = df.rename(columns=column_mapping)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    df = create_time_features(df)

    if not {'temp_avg', 'temp_min', 'temp_max'}.issubset(df.columns):
        return None

    df = df[(df['temp_avg'] != 0) | (df['temp_min'] != 0) | (df['temp_max'] != 0)]
    return df


@st.cache_data(ttl=3600)
def get_openweather_forecast(city: City, days: int = 8):
    """Get forecast from OpenWeatherMap"""
    url = "http://api.openweathermap.org/data/2.5/forecast"
    params = {
        'lat': city.lat,
        'lon': city.lon,
        'appid': OPENWEATHER_API_KEY,
        'units': 'metric'
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        daily_data = {}
        for item in data['list']:
            date = datetime.fromtimestamp(item['dt']).date()
            if date not in daily_data:
                daily_data[date] = {'temps': [], 'temp_min': [], 'temp_max': []}
            daily_data[date]['temps'].append(item['main']['temp'])
            daily_data[date]['temp_min'].append(item['main']['temp_min'])
            daily_data[date]['temp_max'].append(item['main']['temp_max'])

        forecast = []
        for date in sorted(daily_data.keys())[:days]:
            forecast.append({
                'date': date,
                'temp_avg': np.mean(daily_data[date]['temps']),
                'temp_min': np.min(daily_data[date]['temp_min']),
                'temp_max': np.max(daily_data[date]['temp_max'])
            })

        return forecast
    except Exception as e:
        st.error(f"Error fetching OpenWeatherMap data: {e}")
        return None

