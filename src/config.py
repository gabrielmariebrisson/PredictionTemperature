"""
Configuration module for Multi-City Weather Prediction System
Contains City dataclass, cities dictionary, languages, and constants
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

# Try to import and load dotenv, but don't fail if it's not available (e.g., in tests)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, skip loading .env file
    # Environment variables should be set directly
    pass

# Configure logger
logger = logging.getLogger(__name__)

# Get base directory (src/ directory)
BASE_DIR = Path(__file__).parent.parent.resolve()

# API Keys - Validate at startup
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
if not OPENWEATHER_API_KEY:
    error_msg = (
        "OPENWEATHER_API_KEY is not set. "
        "Please set it in your .env file or environment variables. "
        "The application requires this key to fetch weather forecasts."
    )
    logger.error(error_msg)
    raise ValueError(error_msg)

logger.info("OPENWEATHER_API_KEY loaded successfully")

# City Configuration
@dataclass
class City:
    name: str
    lat: float
    lon: float
    timezone: str
    emoji: str

CITIES = {
    'paris': City('Paris', 48.8566, 2.3522, 'Europe/Paris', '🗼'),
    'silicon_valley': City('Silicon Valley', 37.3875, -122.0575, 'America/Los_Angeles', '🌉')
}

# Model Configuration
WINDOW_SIZE = 30
FORECAST_HORIZON = 7

# Models base path - Dynamic path relative to project root
MODELS_BASE_PATH = BASE_DIR / 'templates' / 'assets' / 'température' / 'models'

# Languages Configuration
LANGUAGES = {
    "fr": "🇫🇷 Français",
    "en": "🇬🇧 English",
    "es": "🇪🇸 Español",
    "de": "🇩🇪 Deutsch",
    "it": "🇮🇹 Italiano",
    "pt": "🇵🇹 Português",
    "ja": "🇯🇵 日本語",
    "zh-CN": "🇨🇳 中文",
    "ar": "🇸🇦 العربية",
    "ru": "🇷🇺 Русский"
}

