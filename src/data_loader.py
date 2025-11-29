"""
Data loading and preprocessing module
Handles historical data collection from Meteostat and OpenWeatherMap API
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from meteostat import Daily, Point

from src.config import City, OPENWEATHER_API_KEY

# Configure logger
logger = logging.getLogger(__name__)


@st.cache_data(ttl=3600, persist=True)
def collect_historical_data(city: City, years_back: int = 10) -> Optional[pd.DataFrame]:
    """
    Collect historical weather data from Meteostat.
    
    Args:
        city: City object with coordinates
        years_back: Number of years of historical data to collect
        
    Returns:
        DataFrame with historical weather data or None if data is empty
        
    Raises:
        ValueError: If years_back is not positive
    """
    if years_back <= 0:
        raise ValueError(f"years_back must be positive, got {years_back}")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)

    try:
        location = Point(city.lat, city.lon)
        data = Daily(location, start_date, end_date)
        df = data.fetch()

        if df.empty:
            logger.warning(f"No data available for {city.name} from {start_date} to {end_date}")
            return None

        df['city'] = city.name
        logger.info(f"Successfully collected {len(df)} records for {city.name}")
        return df
    except Exception as e:
        logger.error(f"Error collecting historical data for {city.name}: {e}")
        raise


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from datetime index.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with added time-based features
    """
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


def preprocess_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Clean and standardize dataset.
    
    Args:
        df: Raw DataFrame from Meteostat
        
    Returns:
        Preprocessed DataFrame or None if required columns are missing
        
    Raises:
        ValueError: If input DataFrame is empty
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
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

    required_cols = {'temp_avg', 'temp_min', 'temp_max'}
    if not required_cols.issubset(df.columns):
        logger.error(f"Missing required columns: {required_cols - set(df.columns)}")
        return None

    df = df[(df['temp_avg'] != 0) | (df['temp_min'] != 0) | (df['temp_max'] != 0)]
    logger.debug(f"Preprocessed {len(df)} records")
    return df


@st.cache_data(ttl=3600)
def get_openweather_forecast(
    city: City, 
    days: int = 8
) -> Optional[List[Dict[str, float]]]:
    """
    Get forecast from OpenWeatherMap API.
    
    Args:
        city: City object with coordinates
        days: Number of days to forecast (max 8)
        
    Returns:
        List of forecast dictionaries with date, temp_avg, temp_min, temp_max
        or None if API call fails
        
    Raises:
        ValueError: If days is not between 1 and 8
        requests.RequestException: If API request fails
    """
    if not (1 <= days <= 8):
        raise ValueError(f"days must be between 1 and 8, got {days}")
    
    # OPENWEATHER_API_KEY is validated at startup in config.py
    url = "http://api.openweathermap.org/data/2.5/forecast"
    params = {
        'lat': city.lat,
        'lon': city.lon,
        'appid': OPENWEATHER_API_KEY,
        'units': 'metric'
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'list' not in data:
            logger.error(f"Unexpected API response structure: {list(data.keys())}")
            return None

        daily_data: Dict[date, Dict[str, List[float]]] = {}
        for item in data['list']:
            date_obj = datetime.fromtimestamp(item['dt']).date()
            if date_obj not in daily_data:
                daily_data[date_obj] = {'temps': [], 'temp_min': [], 'temp_max': []}
            daily_data[date_obj]['temps'].append(item['main']['temp'])
            daily_data[date_obj]['temp_min'].append(item['main']['temp_min'])
            daily_data[date_obj]['temp_max'].append(item['main']['temp_max'])

        forecast = []
        for date_obj in sorted(daily_data.keys())[:days]:
            forecast.append({
                'date': date_obj,
                'temp_avg': float(np.mean(daily_data[date_obj]['temps'])),
                'temp_min': float(np.min(daily_data[date_obj]['temp_min'])),
                'temp_max': float(np.max(daily_data[date_obj]['temp_max']))
            })

        logger.info(f"Successfully fetched {len(forecast)} days forecast for {city.name}")
        return forecast
        
    except requests.Timeout:
        logger.error(f"Timeout while fetching OpenWeatherMap data for {city.name}")
        raise
    except requests.RequestException as e:
        logger.error(f"Request error while fetching OpenWeatherMap data: {e}")
        raise
    except (KeyError, ValueError) as e:
        logger.error(f"Error parsing OpenWeatherMap response: {e}")
        raise ValueError(f"Invalid API response format: {e}") from e

