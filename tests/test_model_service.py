"""
Unit tests for model_service module
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, mock_open
import pickle

from src.config import WINDOW_SIZE, FORECAST_HORIZON
from src.model_service import load_model_info, prepare_scalers, predict_7day_forecast


class TestPrepareScalers:
    """Test suite for prepare_scalers function."""
    
    def test_prepares_scalers_with_expected_features(self, preprocessed_dataframe):
        """Test that scalers are prepared correctly with expected features."""
        expected_features = ['temp_avg', 'temp_min', 'temp_max', 'precipitation', 'wind_speed']
        
        scalers = prepare_scalers(preprocessed_dataframe, expected_features=expected_features)
        
        assert 'X_scaler' in scalers
        assert 'y_scaler' in scalers
        assert 'feature_cols' in scalers
        assert 'target_cols' in scalers
        assert scalers['feature_cols'] == expected_features
        assert 'temp_avg' in scalers['target_cols']
        assert 'temp_min' in scalers['target_cols']
        assert 'temp_max' in scalers['target_cols']
    
    def test_adds_missing_columns_with_zeros(self, preprocessed_dataframe):
        """Test that missing expected features are added with zeros."""
        expected_features = ['temp_avg', 'temp_min', 'temp_max', 'missing_feature']
        
        scalers = prepare_scalers(preprocessed_dataframe, expected_features=expected_features)
        
        assert 'missing_feature' in scalers['feature_cols']
        assert 'missing_feature' in preprocessed_dataframe.columns
        assert (preprocessed_dataframe['missing_feature'] == 0).all()
    
    def test_raises_error_on_empty_dataframe(self):
        """Test that ValueError is raised for empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="DataFrame is empty"):
            prepare_scalers(df)
    
    def test_raises_error_if_no_temp_columns(self):
        """Test that ValueError is raised if no temperature columns exist."""
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'precipitation': [0, 5, 10, 15, 20],
            'wind_speed': [5, 10, 15, 20, 25],
        }, index=dates)
        
        with pytest.raises(ValueError, match="No temperature columns"):
            prepare_scalers(df)
    
    def test_auto_detects_features_without_expected_list(self, preprocessed_dataframe):
        """Test that features are auto-detected when expected_features is None."""
        scalers = prepare_scalers(preprocessed_dataframe, expected_features=None)
        
        assert 'feature_cols' in scalers
        assert len(scalers['feature_cols']) > 0
        # Should include all numeric columns
        numeric_cols = preprocessed_dataframe.select_dtypes(include=[np.number]).columns
        assert len(scalers['feature_cols']) == len(numeric_cols)


class TestPredict7DayForecast:
    """Test suite for predict_7day_forecast function."""
    
    @pytest.fixture
    def mock_model(self):
        """Fixture providing a mock TensorFlow model."""
        model = MagicMock()
        # Mock predict to return shape [1, FORECAST_HORIZON, 3] (batch, days, targets)
        mock_output = np.random.randn(FORECAST_HORIZON, 3).astype(np.float32)
        model.predict.return_value = [mock_output]
        return model
    
    @pytest.fixture
    def mock_scalers(self):
        """Fixture providing mock scalers."""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        X_scaler = StandardScaler()
        y_scaler = MinMaxScaler()
        
        # Fit with dummy data
        dummy_X = np.random.randn(100, 5).astype(np.float32)
        dummy_y = np.random.randn(100, 3).astype(np.float32)
        X_scaler.fit(dummy_X)
        y_scaler.fit(dummy_y)
        
        return {
            'X_scaler': X_scaler,
            'y_scaler': y_scaler,
            'feature_cols': ['feat1', 'feat2', 'feat3', 'feat4', 'feat5'],
            'target_cols': ['temp_avg', 'temp_min', 'temp_max']
        }
    
    @pytest.fixture
    def sample_recent_data(self):
        """Fixture providing sample recent data with correct shape."""
        # Create data with more than WINDOW_SIZE samples
        n_samples = WINDOW_SIZE + 10
        n_features = 5
        return np.random.randn(n_samples, n_features).astype(np.float32)
    
    def test_returns_correct_shape(self, mock_model, mock_scalers, sample_recent_data):
        """Test that prediction returns correct shape (7 days, 3 variables)."""
        result = predict_7day_forecast(mock_model, sample_recent_data, mock_scalers)
        
        assert result.shape == (FORECAST_HORIZON, 3)
        assert result.dtype == np.float32
    
    def test_uses_last_window_size_samples(self, mock_model, mock_scalers, sample_recent_data):
        """Test that only last WINDOW_SIZE samples are used."""
        result = predict_7day_forecast(mock_model, sample_recent_data, mock_scalers)
        
        # Verify model.predict was called
        assert mock_model.predict.called
        # Check that input shape is correct [1, WINDOW_SIZE, n_features]
        call_args = mock_model.predict.call_args
        input_shape = call_args[0][0].shape
        assert input_shape[0] == 1  # batch size
        assert input_shape[1] == WINDOW_SIZE
        assert input_shape[2] == sample_recent_data.shape[1]
    
    def test_raises_error_if_insufficient_data(self, mock_model, mock_scalers):
        """Test that ValueError is raised if data has fewer than WINDOW_SIZE samples."""
        insufficient_data = np.random.randn(WINDOW_SIZE - 1, 5).astype(np.float32)
        
        with pytest.raises(ValueError, match=f"at least {WINDOW_SIZE} samples"):
            predict_7day_forecast(mock_model, insufficient_data, mock_scalers)
    
    def test_raises_error_if_missing_scalers(self, mock_model, sample_recent_data):
        """Test that ValueError is raised if scalers are missing."""
        incomplete_scalers = {'X_scaler': MagicMock()}
        
        with pytest.raises(ValueError, match="must contain 'X_scaler' and 'y_scaler'"):
            predict_7day_forecast(mock_model, sample_recent_data, incomplete_scalers)
    
    def test_applies_scalers_correctly(self, mock_model, mock_scalers, sample_recent_data):
        """Test that scalers are applied correctly."""
        result = predict_7day_forecast(mock_model, sample_recent_data, mock_scalers)
        
        # Verify result shape is correct (scalers were applied)
        assert result.shape == (FORECAST_HORIZON, 3)
        # Verify result contains valid float values
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()
    
    def test_handles_model_prediction_errors(self, mock_model, mock_scalers, sample_recent_data):
        """Test that model prediction errors are handled correctly."""
        # Make model.predict raise an exception
        mock_model.predict.side_effect = Exception("Model prediction failed")
        
        with pytest.raises(ValueError, match="Prediction failed"):
            predict_7day_forecast(mock_model, sample_recent_data, mock_scalers)


class TestLoadModelInfo:
    """Test suite for load_model_info function."""
    
    @patch('src.model_service.tf.keras.models.load_model')
    @patch('src.model_service.Path')
    def test_loads_model_when_files_exist(self, mock_path_class, mock_load_model):
        """Test that model and info are loaded when files exist."""
        from pathlib import Path
        
        # Setup mock paths
        mock_model_path = MagicMock(spec=Path)
        mock_model_path.exists.return_value = True
        mock_model_path.__str__ = lambda x: "test_city_model.keras"
        
        mock_info_path = MagicMock(spec=Path)
        mock_info_path.exists.return_value = True
        
        # Make Path division return appropriate mock
        def path_div_side_effect(path, other):
            if 'info' in str(other):
                return mock_info_path
            return mock_model_path
        
        mock_base_path = MagicMock(spec=Path)
        mock_base_path.__truediv__ = MagicMock(side_effect=path_div_side_effect)
        
        # Patch MODELS_BASE_PATH
        with patch('src.model_service.MODELS_BASE_PATH', mock_base_path):
            mock_model = MagicMock()
            mock_load_model.return_value = mock_model
            
            mock_info = {'feature_cols': ['feat1', 'feat2'], 'target_cols': ['temp_avg']}
            
            with patch('builtins.open', mock_open(read_data=pickle.dumps(mock_info))):
                with patch('pickle.load', return_value=mock_info):
                    model, info = load_model_info('test_city')
            
            assert model is not None
            assert info is not None
            assert info == mock_info
    
    @patch('src.model_service.Path')
    def test_returns_none_when_model_not_found(self, mock_path_class):
        """Test that (None, None) is returned when model file doesn't exist."""
        from pathlib import Path
        
        mock_model_path = MagicMock(spec=Path)
        mock_model_path.exists.return_value = False
        
        mock_base_path = MagicMock(spec=Path)
        mock_base_path.__truediv__ = MagicMock(return_value=mock_model_path)
        
        with patch('src.model_service.MODELS_BASE_PATH', mock_base_path):
            model, info = load_model_info('test_city')
        
        assert model is None
        assert info is None
    
    @patch('src.model_service.tf.keras.models.load_model')
    @patch('src.model_service.Path')
    def test_raises_error_on_corrupted_model(self, mock_path_class, mock_load_model):
        """Test that ValueError is raised when model file is corrupted."""
        from pathlib import Path
        
        mock_model_path = MagicMock(spec=Path)
        mock_model_path.exists.return_value = True
        mock_model_path.__str__ = lambda x: "test_city_model.keras"
        
        mock_base_path = MagicMock(spec=Path)
        mock_base_path.__truediv__ = MagicMock(return_value=mock_model_path)
        
        with patch('src.model_service.MODELS_BASE_PATH', mock_base_path):
            mock_load_model.side_effect = Exception("Corrupted model file")
            
            with pytest.raises(ValueError, match="Failed to load model"):
                load_model_info('test_city')

