"""
Utility functions module
Handles translation and caching with robust fallback mechanism
"""

import logging
from typing import Optional

import streamlit as st
from deep_translator import GoogleTranslator

# Configure logger
logger = logging.getLogger(__name__)

# Translation timeout in seconds
TRANSLATION_TIMEOUT = 5
# Maximum retry attempts
MAX_RETRIES = 2


def gettext(text: str, lang: Optional[str] = None) -> str:
    """
    Translate text with caching and robust fallback mechanism.
    
    This function never raises exceptions - it always returns a string,
    falling back to the original text if translation fails.
    
    Args:
        text: Text to translate (assumed to be in French)
        lang: Target language code (e.g., 'en', 'es'). If None, uses session_state
        
    Returns:
        Translated text or original text if translation fails
        
    Note:
        - French (lang='fr') returns text unchanged
        - Uses session_state cache to avoid redundant API calls
        - Falls back to original text on any error (timeout, API failure, etc.)
    """
    # Validate input
    if not text or not isinstance(text, str):
        logger.warning(f"Invalid text input: {type(text)}")
        return str(text) if text else ""
    
    # Initialize cache if necessary
    if 'translations_cache' not in st.session_state:
        st.session_state.translations_cache = {}
    
    # Get language from session_state if not provided
    if lang is None:
        lang = st.session_state.get('language', 'fr')
    
    # No translation needed for French
    if lang == 'fr':
        return text
    
    # Validate language code
    if not isinstance(lang, str) or len(lang) < 2:
        logger.warning(f"Invalid language code: {lang}, returning original text")
        return text
    
    # Check cache first
    cache_key = f"{lang}_{text}"
    if cache_key in st.session_state.translations_cache:
        return st.session_state.translations_cache[cache_key]
    
    # Attempt translation with retries and timeout handling
    for attempt in range(MAX_RETRIES + 1):
        try:
            translator = GoogleTranslator(source='fr', target=lang)
            translated = translator.translate(text)
            
            # Validate translation result
            if translated and isinstance(translated, str) and len(translated) > 0:
                # Cache successful translation
                st.session_state.translations_cache[cache_key] = translated
                logger.debug(f"Successfully translated text to {lang} (attempt {attempt + 1})")
                return translated
            else:
                logger.warning(f"Translation returned empty/invalid result for {lang}")
                # Fall through to return original text
                
        except TimeoutError:
            logger.warning(
                f"Translation timeout for {lang} (attempt {attempt + 1}/{MAX_RETRIES + 1})"
            )
            if attempt < MAX_RETRIES:
                continue  # Retry
            # Fall through to return original text
            
        except Exception as e:
            logger.warning(
                f"Translation error for {lang} (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}"
            )
            if attempt < MAX_RETRIES:
                continue  # Retry
            # Fall through to return original text
    
    # All attempts failed - return original text (robust fallback)
    logger.warning(
        f"Translation failed after {MAX_RETRIES + 1} attempts for {lang}, "
        f"returning original text"
    )
    return text

