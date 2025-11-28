"""
Model service module
Handles TensorFlow model loading, scaling, and prediction
"""

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.config import WINDOW_SIZE, MODELS_BASE_PATH


def load_model_info(city_key: str):
    """Load model and its feature information"""
    model_path = os.path.join(MODELS_BASE_PATH, f'{city_key}_model.keras')
    info_path = os.path.join(MODELS_BASE_PATH, f'{city_key}_info.pkl')

    if not os.path.exists(model_path):
        return None, None

    model = tf.keras.models.load_model(model_path)

    # Load feature info if available
    if os.path.exists(info_path):
        with open(info_path, 'rb') as f:
            info = pickle.load(f)
        return model, info
    else:
        # If no info file, return model only
        return model, None


def prepare_scalers(df, expected_features=None):
    """Prépare les scalers, en garantissant les bonnes features."""
    temp_cols = [c for c in df.columns if c.startswith('temp_')]

    if expected_features is not None:
        numeric_cols = expected_features
        # Ajout des colonnes manquantes avec des zéros
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    X = df[numeric_cols].values.astype(np.float32)
    y = df[temp_cols].values.astype(np.float32)

    X_scaler = StandardScaler()
    y_scaler = MinMaxScaler()

    X_scaler.fit(X)
    y_scaler.fit(y)

    return {
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'feature_cols': numeric_cols,
        'target_cols': temp_cols
    }


def predict_7day_forecast(model, recent_data, scalers):
    """Predict 7-day temperature forecast"""
    X_scaler = scalers['X_scaler']
    y_scaler = scalers['y_scaler']

    recent_scaled = X_scaler.transform(recent_data[-WINDOW_SIZE:])
    X_input = recent_scaled.reshape(1, WINDOW_SIZE, recent_scaled.shape[1])

    y_pred_scaled = model.predict(X_input, verbose=0)[0]
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    return y_pred

