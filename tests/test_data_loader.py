"""
Unit tests for data_loader module
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from src.data_loader import create_time_features, preprocess_data


class TestCreateTimeFeatures:
    """Test suite for create_time_features function."""
    
    def test_creates_time_features(self, sample_meteostat_dataframe):
        """Test that time features are correctly created."""
        df = create_time_features(sample_meteostat_dataframe)
        
        # Check that new columns are added
        assert 'day_of_year' in df.columns
        assert 'month' in df.columns
        assert 'day_of_month' in df.columns
        assert 'day_of_week' in df.columns
        assert 'day_of_year_sin' in df.columns
        assert 'day_of_year_cos' in df.columns
        assert 'month_sin' in df.columns
        assert 'month_cos' in df.columns
    
    def test_time_features_values(self, sample_meteostat_dataframe):
        """Test that time features have correct values."""
        df = create_time_features(sample_meteostat_dataframe)
        
        # Check day_of_year is between 1 and 366
        assert df['day_of_year'].min() >= 1
        assert df['day_of_year'].max() <= 366
        
        # Check month is between 1 and 12
        assert df['month'].min() >= 1
        assert df['month'].max() <= 12
        
        # Check day_of_month is between 1 and 31
        assert df['day_of_month'].min() >= 1
        assert df['day_of_month'].max() <= 31
        
        # Check day_of_week is between 0 and 6
        assert df['day_of_week'].min() >= 0
        assert df['day_of_week'].max() <= 6
    
    def test_sin_cos_features_range(self, sample_meteostat_dataframe):
        """Test that sin/cos features are in correct range [-1, 1]."""
        df = create_time_features(sample_meteostat_dataframe)
        
        # Check sin/cos values are in range [-1, 1]
        assert df['day_of_year_sin'].min() >= -1.0
        assert df['day_of_year_sin'].max() <= 1.0
        assert df['day_of_year_cos'].min() >= -1.0
        assert df['day_of_year_cos'].max() <= 1.0
        assert df['month_sin'].min() >= -1.0
        assert df['month_sin'].max() <= 1.0
        assert df['month_cos'].min() >= -1.0
        assert df['month_cos'].max() <= 1.0
    
    def test_does_not_modify_original_dataframe(self, sample_meteostat_dataframe):
        """Test that original DataFrame is not modified."""
        original_columns = set(sample_meteostat_dataframe.columns)
        df_result = create_time_features(sample_meteostat_dataframe)
        
        # Original DataFrame should not have new columns
        assert set(sample_meteostat_dataframe.columns) == original_columns
        # Result DataFrame should have new columns
        assert len(df_result.columns) > len(original_columns)
    
    def test_handles_datetime_index(self):
        """Test that function works with datetime index."""
        dates = pd.date_range(start='2024-06-15', periods=10, freq='D')
        df = pd.DataFrame({'value': range(10)}, index=dates)
        
        result = create_time_features(df)
        
        assert 'day_of_year' in result.columns
        assert len(result) == 10


class TestPreprocessData:
    """Test suite for preprocess_data function."""
    
    def test_renames_columns_correctly(self, sample_meteostat_dataframe):
        """Test that Meteostat columns are renamed correctly."""
        df = preprocess_data(sample_meteostat_dataframe)
        
        # Check renamed columns exist
        assert 'temp_avg' in df.columns
        assert 'temp_min' in df.columns
        assert 'temp_max' in df.columns
        assert 'precipitation' in df.columns
        
        # Check original columns are gone
        assert 'tavg' not in df.columns
        assert 'tmin' not in df.columns
        assert 'tmax' not in df.columns
        assert 'prcp' not in df.columns
    
    def test_adds_time_features(self, sample_meteostat_dataframe):
        """Test that time features are added during preprocessing."""
        df = preprocess_data(sample_meteostat_dataframe)
        
        # Check time features are present
        assert 'day_of_year' in df.columns
        assert 'month_sin' in df.columns
        assert 'month_cos' in df.columns
    
    def test_fills_nan_values(self):
        """Test that NaN values are filled with 0."""
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'tavg': [10, np.nan, 15, 20, np.nan],
            'tmin': [5, 8, np.nan, 15, 18],
            'tmax': [15, 18, 22, np.nan, 25],
            'prcp': [0, 5, np.nan, 10, 0],
        }, index=dates)
        
        result = preprocess_data(df)
        
        # Check no NaN values remain in numeric columns
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert result[numeric_cols].isna().sum().sum() == 0
    
    def test_filters_zero_temperatures(self):
        """Test that rows with all zero temperatures are filtered out."""
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'tavg': [10, 0, 15, 0, 20],
            'tmin': [5, 0, 10, 0, 15],
            'tmax': [15, 0, 20, 0, 25],
            'prcp': [0, 0, 5, 0, 10],
        }, index=dates)
        
        result = preprocess_data(df)
        
        # Should keep rows where at least one temp is non-zero
        # Rows 0, 2, 4 should remain (indices 0, 2, 4)
        assert len(result) >= 3
        # All remaining rows should have at least one non-zero temperature
        assert ((result['temp_avg'] != 0) | 
                (result['temp_min'] != 0) | 
                (result['temp_max'] != 0)).all()
    
    def test_returns_none_if_missing_required_columns(self):
        """Test that None is returned if required columns are missing."""
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'prcp': [0, 5, 10, 15, 20],
            'wspd': [5, 10, 15, 20, 25],
        }, index=dates)
        
        result = preprocess_data(df)
        
        assert result is None
    
    def test_raises_error_on_empty_dataframe(self):
        """Test that ValueError is raised for empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            preprocess_data(df)
    
    def test_preserves_dataframe_index(self, sample_meteostat_dataframe):
        """Test that DataFrame index is preserved."""
        original_index = sample_meteostat_dataframe.index
        result = preprocess_data(sample_meteostat_dataframe)
        
        if result is not None:
            # Index should be preserved (may be filtered)
            assert len(result.index) <= len(original_index)
            # All result indices should be in original
            assert result.index.isin(original_index).all()

