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
    """Create cyclical time-based features from datetime index.
    
    Extracts temporal features from the DataFrame's datetime index and creates
    cyclical encoding using sine/cosine transformations. This helps the model
    capture seasonal patterns in weather data.
    
    Args:
        df: DataFrame with a DatetimeIndex. The index must be datetime type
            to extract temporal information.
    
    Returns:
        DataFrame with added columns:
            - 'day_of_year': Day of year (1-366)
            - 'month': Month (1-12)
            - 'day_of_month': Day of month (1-31)
            - 'day_of_week': Day of week (0=Monday, 6=Sunday)
            - 'day_of_year_sin': Sine encoding of day of year (range: [-1, 1])
            - 'day_of_year_cos': Cosine encoding of day of year (range: [-1, 1])
            - 'month_sin': Sine encoding of month (range: [-1, 1])
            - 'month_cos': Cosine encoding of month (range: [-1, 1])
    
    Note:
        The original DataFrame is not modified. A copy is returned with
        additional columns.
    
    Example:
        >>> df = pd.DataFrame({'temp': [10, 15, 20]}, 
        ...                   index=pd.date_range('2024-01-01', periods=3))
        >>> df_features = create_time_features(df)
        >>> 'day_of_year_sin' in df_features.columns
        True
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
    """Clean, standardize, and enrich weather dataset from Meteostat.
    
    This function performs comprehensive data preprocessing including column
    renaming, missing value handling, time feature creation, and data filtering.
    The output is ready for model training or prediction.
    
    Args:
        df: Raw DataFrame from Meteostat API with columns: tavg, tmin, tmax,
            prcp, wspd, pres, etc. Must have a DatetimeIndex.
    
    Returns:
        Preprocessed DataFrame with:
            - Renamed columns (tavg -> temp_avg, etc.)
            - Missing values filled with 0
            - Time-based features added
            - Rows with all-zero temperatures filtered out
        Returns None if required temperature columns (temp_avg, temp_min,
        temp_max) are missing after renaming.
    
    Raises:
        ValueError: If the input DataFrame is empty.
    
    Note:
        The function modifies a copy of the input DataFrame. Original data
        remains unchanged. Rows where all temperature values are zero are
        removed as they represent invalid data points.
    
    Example:
        >>> raw_df = collect_historical_data(city, years_back=10)
        >>> processed_df = preprocess_data(raw_df)
        >>> if processed_df is not None:
        ...     print(f"Processed {len(processed_df)} valid records")
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
    """Fetch weather forecast from OpenWeatherMap API.
    
    Retrieves hourly forecasts for the specified city and aggregates them into
    daily summaries. Results are cached for 1 hour to reduce API calls.
    
    Args:
        city: City object containing latitude, longitude, and name. Used to
            query the OpenWeatherMap API for the specific location.
        days: Number of forecast days to return (1-8). The API provides up to
            5 days of hourly forecasts, which are aggregated into daily values.
            Default is 8 to match the model's forecast horizon.
    
    Returns:
        List of dictionaries, each containing:
            - 'date': date object for the forecast day
            - 'temp_avg': Average temperature in Celsius (mean of hourly temps)
            - 'temp_min': Minimum temperature in Celsius
            - 'temp_max': Maximum temperature in Celsius
        Returns None if the API response is invalid or missing expected data.
    
    Raises:
        ValueError: If days parameter is not between 1 and 8, or if the API
            response has an unexpected structure.
        requests.Timeout: If the API request times out after 10 seconds.
        requests.RequestException: If the HTTP request fails (network error,
            invalid API key, etc.).
    
    Note:
        Requires OPENWEATHER_API_KEY to be set in environment variables.
        The function is cached by Streamlit for 1 hour (3600 seconds) to
        minimize API usage.
    
    Example:
        >>> forecast = get_openweather_forecast(city, days=7)
        >>> if forecast:
        ...     print(f"Forecast for {forecast[0]['date']}: {forecast[0]['temp_avg']}°C")
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

