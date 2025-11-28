"""
Utility functions module
Handles translation and caching
"""

import streamlit as st
from deep_translator import GoogleTranslator


def gettext(text: str, lang: str = None):
    """Fonction de traduction automatique avec cache"""
    # Initialiser le cache si nécessaire
    if 'translations_cache' not in st.session_state:
        st.session_state.translations_cache = {}
    
    # Utiliser la langue depuis session_state si non fournie
    if lang is None:
        lang = st.session_state.get('language', 'fr')
    
    if lang == 'fr':
        return text

    # Vérifier le cache
    cache_key = f"{lang}_{text}"
    if cache_key in st.session_state.translations_cache:
        return st.session_state.translations_cache[cache_key]

    # Traduire
    try:
        translated = GoogleTranslator(source='fr', target=lang).translate(text)
        st.session_state.translations_cache[cache_key] = translated
        return translated
    except:
        return text

