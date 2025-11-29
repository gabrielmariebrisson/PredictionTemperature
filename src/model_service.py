"""
Model service module
Handles TensorFlow model loading, scaling, and prediction
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import MODELS_BASE_PATH, WINDOW_SIZE

# Configure logger
logger = logging.getLogger(__name__)


def load_model_info(city_key: str) -> Tuple[Optional[tf.keras.Model], Optional[Dict]]:
    """
    Load model and its feature information.
    
    Args:
        city_key: City identifier (e.g., 'paris', 'silicon_valley')
        
    Returns:
        Tuple of (model, info_dict) or (None, None) if model not found
        info_dict contains 'feature_cols' and 'target_cols'
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model file is corrupted
    """
    model_path = os.path.join(MODELS_BASE_PATH, f'{city_key}_model.keras')
    info_path = os.path.join(MODELS_BASE_PATH, f'{city_key}_info.pkl')

    if not os.path.exists(model_path):
        logger.warning(f"Model not found at {model_path}")
        return None, None

    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Successfully loaded model for {city_key}")
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise ValueError(f"Failed to load model for {city_key}: {e}") from e

    # Load feature info if available
    info = None
    if os.path.exists(info_path):
        try:
            with open(info_path, 'rb') as f:
                info = pickle.load(f)
            logger.info(f"Successfully loaded model info for {city_key}")
        except Exception as e:
            logger.warning(f"Error loading model info from {info_path}: {e}")
            # Continue without info - model can still be used
    else:
        logger.warning(f"Model info file not found at {info_path} for {city_key}")

    return model, info


def prepare_scalers(
    df: pd.DataFrame, 
    expected_features: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Prepare scalers for features and targets.
    
    Args:
        df: DataFrame with features and target columns
        expected_features: List of expected feature column names
        
    Returns:
        Dictionary with 'X_scaler', 'y_scaler', 'feature_cols', 'target_cols'
        
    Raises:
        ValueError: If no temperature columns found or DataFrame is empty
    """
    if df.empty:
        raise ValueError("DataFrame is empty, cannot prepare scalers")
    
    temp_cols = [c for c in df.columns if c.startswith('temp_')]
    if not temp_cols:
        raise ValueError("No temperature columns (temp_*) found in DataFrame")

    if expected_features is not None:
        numeric_cols = expected_features
        # Add missing columns with zeros
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0
                logger.debug(f"Added missing column {col} with zeros")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        raise ValueError("No numeric columns found for scaling")

    X = df[numeric_cols].values.astype(np.float32)
    y = df[temp_cols].values.astype(np.float32)

    X_scaler = StandardScaler()
    y_scaler = MinMaxScaler()

    X_scaler.fit(X)
    y_scaler.fit(y)

    logger.debug(f"Prepared scalers for {len(numeric_cols)} features and {len(temp_cols)} targets")
    return {
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'feature_cols': numeric_cols,
        'target_cols': temp_cols
    }


def predict_7day_forecast(
    model: tf.keras.Model,
    recent_data: np.ndarray,
    scalers: Dict[str, any]
) -> np.ndarray:
    """
    Predict 7-day temperature forecast.
    
    Args:
        model: Trained TensorFlow model
        recent_data: Array of recent weather data (shape: [n_samples, n_features])
        scalers: Dictionary with 'X_scaler' and 'y_scaler'
        
    Returns:
        Array of predictions (shape: [forecast_horizon, n_targets])
        
    Raises:
        ValueError: If input data shape is invalid or scalers are missing
    """
    if recent_data.shape[0] < WINDOW_SIZE:
        raise ValueError(
            f"recent_data must have at least {WINDOW_SIZE} samples, "
            f"got {recent_data.shape[0]}"
        )
    
    if 'X_scaler' not in scalers or 'y_scaler' not in scalers:
        raise ValueError("Scalers dictionary must contain 'X_scaler' and 'y_scaler'")

    X_scaler = scalers['X_scaler']
    y_scaler = scalers['y_scaler']

    try:
        recent_scaled = X_scaler.transform(recent_data[-WINDOW_SIZE:])
        X_input = recent_scaled.reshape(1, WINDOW_SIZE, recent_scaled.shape[1])

        y_pred_scaled = model.predict(X_input, verbose=0)[0]
        y_pred = y_scaler.inverse_transform(y_pred_scaled)

        logger.debug(f"Generated forecast with shape {y_pred.shape}")
        return y_pred
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise ValueError(f"Prediction failed: {e}") from e

