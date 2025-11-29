"""
Pytest configuration and fixtures
"""

import os
import pytest
from datetime import datetime, timedelta

# Set a dummy API key for testing BEFORE importing config
# This prevents the ValueError from being raised during import
os.environ['OPENWEATHER_API_KEY'] = os.environ.get('OPENWEATHER_API_KEY', 'test_api_key_for_testing')

from src.config import City


@pytest.fixture
def test_city() -> City:
    """Fixture providing a test City instance."""
    return City(
        name="Test City",
        lat=48.8566,
        lon=2.3522,
        timezone="Europe/Paris",
        emoji="🏙️"
    )


@pytest.fixture
def sample_meteostat_dataframe():
    """Fixture providing a sample DataFrame mimicking Meteostat data structure."""
    import pandas as pd
    import numpy as np
    
    # Create date range for 30 days
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    
    # Create DataFrame with Meteostat column names
    df = pd.DataFrame({
        'tavg': np.random.uniform(10, 25, 30),  # Average temperature
        'tmin': np.random.uniform(5, 20, 30),   # Min temperature
        'tmax': np.random.uniform(15, 30, 30),  # Max temperature
        'prcp': np.random.uniform(0, 10, 30),    # Precipitation
        'wspd': np.random.uniform(0, 20, 30),   # Wind speed
        'pres': np.random.uniform(1000, 1020, 30),  # Pressure
    }, index=dates)
    
    return df


@pytest.fixture
def preprocessed_dataframe():
    """Fixture providing a preprocessed DataFrame with time features."""
    import pandas as pd
    import numpy as np
    
    # Create date range for 30 days
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    
    # Create DataFrame with preprocessed column names
    df = pd.DataFrame({
        'temp_avg': np.random.uniform(10, 25, 30),
        'temp_min': np.random.uniform(5, 20, 30),
        'temp_max': np.random.uniform(15, 30, 30),
        'precipitation': np.random.uniform(0, 10, 30),
        'wind_speed': np.random.uniform(0, 20, 30),
        'pressure': np.random.uniform(1000, 1020, 30),
    }, index=dates)
    
    return df

