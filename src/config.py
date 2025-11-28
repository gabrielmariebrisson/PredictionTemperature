"""
Configuration module for Multi-City Weather Prediction System
Contains City dataclass, cities dictionary, languages, and constants
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')

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

# Models base path
MODELS_BASE_PATH = 'templates/assets/température/models/'

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

