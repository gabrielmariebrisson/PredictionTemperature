"""
Model service module
Handles TensorFlow model loading, scaling, and prediction
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import MODELS_BASE_PATH, WINDOW_SIZE

# Configure logger
logger = logging.getLogger(__name__)


def load_model_info(city_key: str) -> Tuple[Optional[tf.keras.Model], Optional[Dict]]:
    """Load TensorFlow model and its feature information from disk.
    
    This function loads a trained Keras model and its associated metadata
    (feature columns, target columns) for a given city. The model file
    must be in .keras format and the info file in .pkl format.
    
    Args:
        city_key: City identifier string (e.g., 'paris', 'silicon_valley').
            Used to construct file paths: {city_key}_model.keras and
            {city_key}_info.pkl.
    
    Returns:
        Tuple containing:
            - model: Loaded TensorFlow Keras model, or None if model file
              doesn't exist.
            - info: Dictionary with model metadata containing:
                - 'feature_cols': List of feature column names used during training.
                - 'target_cols': List of target column names (temperature variables).
              Returns None if info file doesn't exist or cannot be loaded.
    
    Raises:
        ValueError: If the model file exists but cannot be loaded (corrupted
            or incompatible format). The error message includes the city_key
            and original exception details.
    
    Example:
        >>> model, info = load_model_info('paris')
        >>> if model is not None:
        ...     print(f"Model loaded with {len(info['feature_cols'])} features")
    """
    model_path = MODELS_BASE_PATH / f'{city_key}_model.keras'
    info_path = MODELS_BASE_PATH / f'{city_key}_info.pkl'

    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}")
        return None, None

    try:
        model = tf.keras.models.load_model(str(model_path))
        logger.info(f"Successfully loaded model for {city_key}")
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise ValueError(f"Failed to load model for {city_key}: {e}") from e

    # Load feature info if available
    info = None
    if info_path.exists():
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
    """Prepare StandardScaler and MinMaxScaler for features and targets.
    
    This function creates and fits scalers for normalizing input features
    (using StandardScaler) and target variables (using MinMaxScaler).
    Missing expected features are automatically added with zero values.
    
    Args:
        df: DataFrame containing weather data with datetime index. Must include
            columns starting with 'temp_' (temp_avg, temp_min, temp_max).
        expected_features: Optional list of feature column names expected by
            the model. If provided, missing columns are added with zero values.
            If None, all numeric columns are used as features.
    
    Returns:
        Dictionary containing:
            - 'X_scaler': Fitted StandardScaler for input features.
            - 'y_scaler': Fitted MinMaxScaler for target temperatures.
            - 'feature_cols': List of feature column names used for scaling.
            - 'target_cols': List of target column names (temp_avg, temp_min,
              temp_max).
    
    Raises:
        ValueError: If DataFrame is empty, no temperature columns (temp_*)
            are found, or no numeric columns are available for scaling.
    
    Note:
        The function modifies the input DataFrame in-place by adding missing
        columns. Consider passing a copy if the original DataFrame needs to
        remain unchanged.
    
    Example:
        >>> scalers = prepare_scalers(df, expected_features=['temp_avg', 'pressure'])
        >>> scaled_X = scalers['X_scaler'].transform(X)
        >>> scaled_y = scalers['y_scaler'].transform(y)
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
    """Generate 7-day temperature forecast using a trained model.
    
    This function takes recent weather data, scales it using pre-fitted scalers,
    and generates predictions for the next 7 days. Only the last WINDOW_SIZE
    samples from recent_data are used for prediction.
    
    Args:
        model: Trained TensorFlow Keras model that accepts input shape
            [batch_size, WINDOW_SIZE, n_features] and outputs predictions
            of shape [batch_size, FORECAST_HORIZON, n_targets].
        recent_data: NumPy array of recent weather observations with shape
            [n_samples, n_features]. Must have at least WINDOW_SIZE rows.
            Only the last WINDOW_SIZE rows are used for prediction.
        scalers: Dictionary containing:
            - 'X_scaler': Fitted StandardScaler for input features.
            - 'y_scaler': Fitted MinMaxScaler for target temperatures.
    
    Returns:
        NumPy array of shape [FORECAST_HORIZON, n_targets] containing
        temperature predictions. Typically FORECAST_HORIZON=7 and n_targets=3
        (temp_avg, temp_min, temp_max). Values are in original scale (Celsius)
        after inverse transformation.
    
    Raises:
        ValueError: If recent_data has fewer than WINDOW_SIZE samples, or if
            the scalers dictionary is missing 'X_scaler' or 'y_scaler' keys.
            Also raised if model prediction fails (e.g., shape mismatch,
            model error).
    
    Example:
        >>> model, _ = load_model_info('paris')
        >>> recent_data = df[features].tail(40).values  # Last 40 days
        >>> predictions = predict_7day_forecast(model, recent_data, scalers)
        >>> print(f"7-day forecast shape: {predictions.shape}")  # (7, 3)
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

