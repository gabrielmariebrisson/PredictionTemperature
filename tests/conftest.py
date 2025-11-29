"""
Pytest configuration and fixtures

This module configures pytest to:
1. Add project root to PYTHONPATH for absolute imports
2. Mock external dependencies (streamlit, tensorflow, etc.) before importing src modules
3. Set up test environment variables
4. Provide reusable fixtures for tests
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from datetime import datetime, timedelta

# ============================================================================
# 1. Configuration du PYTHONPATH
# ============================================================================
# Ajouter le répertoire parent au PYTHONPATH pour permettre les imports `from src import ...`
# Cela permet de lancer pytest depuis n'importe quel répertoire
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ============================================================================
# 2. Configuration des variables d'environnement AVANT les imports
# ============================================================================
# Set a dummy API key for testing BEFORE importing config
# This prevents the ValueError from being raised during import
os.environ['OPENWEATHER_API_KEY'] = os.environ.get('OPENWEATHER_API_KEY', 'test_api_key_for_testing')

# ============================================================================
# 3. Mocking des dépendances AVANT l'import des modules src
# ============================================================================
# Mocker les dépendances AVANT l'import des modules src pour éviter ModuleNotFoundError
# Cela permet de tester la logique sans dépendre de ces packages installés

# 3.1. Mocker streamlit
_mock_session_state = MagicMock()
_mock_session_state.language = "fr"
_mock_session_state.translations_cache = {}

_mock_streamlit = MagicMock()
_mock_streamlit.session_state = _mock_session_state
# @st.cache_data devient une fonction identité (pas de cache dans les tests)
_mock_streamlit.cache_data = lambda **kwargs: lambda func: func
_mock_streamlit.cache_resource = lambda **kwargs: lambda func: func
sys.modules["streamlit"] = _mock_streamlit

# 3.2. Mocker tensorflow AVANT l'import des modules src
# On mock TensorFlow pour éviter ModuleNotFoundError si TensorFlow n'est pas installé
# dans l'environnement Python utilisé par pytest
# Les tests utilisent des mocks de modèles, donc on n'a pas besoin de TensorFlow réel

# Vérifier si TensorFlow est déjà dans sys.modules (déjà importé)
if "tensorflow" not in sys.modules:
    # Essayer d'importer TensorFlow
    try:
        import tensorflow as tf
        # Si TensorFlow est disponible, on le laisse tel quel
        # Les tests pourront l'utiliser normalement
        _tensorflow_available = True
    except ImportError:
        # Si TensorFlow n'est pas disponible, on le mock complètement
        _tensorflow_available = False
        _mock_tf = MagicMock()
        _mock_keras = MagicMock()
        _mock_keras_models = MagicMock()
        _mock_keras_models.load_model = MagicMock()
        
        # Créer une structure de mock qui ressemble à TensorFlow
        _mock_tf.keras = _mock_keras
        _mock_keras.Model = MagicMock()
        _mock_keras.models = _mock_keras_models
        
        # Installer le mock dans sys.modules AVANT que les modules src ne soient importés
        sys.modules["tensorflow"] = _mock_tf
        sys.modules["tensorflow.keras"] = _mock_keras
        sys.modules["tensorflow.keras.models"] = _mock_keras_models

# 3.3. Mocker deep_translator (optionnel)
_mock_google_translator = MagicMock()
_mock_google_translator_instance = MagicMock()
_mock_google_translator_instance.translate.return_value = "translated_text"
_mock_google_translator.GoogleTranslator.return_value = _mock_google_translator_instance
sys.modules["deep_translator"] = _mock_google_translator
sys.modules["deep_translator.google"] = MagicMock()
sys.modules["deep_translator.google"].GoogleTranslator = _mock_google_translator.GoogleTranslator

# 3.4. Mocker meteostat (optionnel, pour éviter les appels API réels)
_mock_point = MagicMock()
_mock_daily = MagicMock()
_mock_meteostat = MagicMock()
_mock_meteostat.Point = _mock_point
_mock_meteostat.Daily = _mock_daily
sys.modules["meteostat"] = _mock_meteostat

# ============================================================================
# 4. Import des modules src APRÈS la configuration
# ============================================================================
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

