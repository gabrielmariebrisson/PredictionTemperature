# 🌡️ Multi-City Weather Prediction System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-Pytest-green.svg)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Prédiction météorologique multi-villes avec réseaux de neurones profonds (Conv1D + LSTM)**

Système de prédiction de température utilisant l'apprentissage profond pour prévoir les températures sur 7 jours pour plusieurs villes. Le modèle combine des couches convolutionnelles 1D et des réseaux LSTM pour capturer à la fois les motifs locaux et les dépendances temporelles à long terme dans les données météorologiques historiques.

## 🎯 Fonctionnalités

- **Prédiction multi-villes** : Support pour Paris et Silicon Valley (extensible)
- **Modèle hybride Conv1D + LSTM** : Architecture optimisée pour les séries temporelles
- **Interface Streamlit** : Application web interactive et multilingue
- **Comparaison avec OpenWeatherMap** : Validation des prédictions avec des données externes
- **Visualisations interactives** : Graphiques Plotly pour l'analyse des tendances
- **Architecture modulaire MVC** : Code organisé et maintenable

## 🏗️ Architecture

### Pourquoi Conv1D + LSTM ?

Notre architecture hybride combine le meilleur des deux mondes :

#### **Conv1D (Convolution 1D)**
- **Performance** : Extraction rapide de motifs locaux dans les séquences temporelles
- **Efficacité** : Filtres convolutifs détectent les tendances sur de petites fenêtres de temps
- **Robustesse** : Moins sensible au bruit grâce au partage de paramètres

#### **LSTM (Long Short-Term Memory)**
- **Précision** : Capture des dépendances temporelles à long terme (saisons, cycles annuels)
- **Mémoire** : Maintient l'information sur plusieurs pas de temps
- **Complexité** : Gère les relations non-linéaires complexes dans les données météo

#### **Avantages de la combinaison**
- **Meilleure précision** : MAE de 0.98°C pour Silicon Valley, 2.84°C pour Paris
- **Entraînement rapide** : Moins de 2 secondes pour 10 ans de données par ville
- **Généralisation** : Modèle léger et efficace, facilement déployable

### Structure du Modèle

```
Input (30 jours × n_features)
    ↓
Conv1D (filtres convolutifs)
    ↓
LSTM Layer 1 (return_sequences=True)
    ↓
LSTM Layer 2 (return_sequences=True)
    ↓
LSTM Layer 3 (return_sequences=False)
    ↓
Dense Layers + Dropout
    ↓
Output (7 jours × 3 variables: avg, min, max)
```

## 📦 Installation

### Prérequis

- Python 3.11 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation locale

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/gabrielmariebrisson/PredictionTemperature.git
   cd PredictionTemperature
   ```

2. **Créer un environnement virtuel**
   ```bash
   python -m venv venv
   
   # Sur Linux/Mac
   source venv/bin/activate
   
   # Sur Windows
   venv\Scripts\activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configurer les variables d'environnement**
   
   Créer un fichier `.env` à la racine du projet :
   ```env
   OPENWEATHER_API_KEY=votre_cle_api_ici
   ```
   
   Obtenez une clé API gratuite sur [OpenWeatherMap](https://openweathermap.org/api).

5. **Lancer l'application**
   ```bash
   streamlit run PrédictionTempératuresWeb.py
   ```
   
   L'application sera accessible sur `http://localhost:8501`

### Installation avec Docker

```bash
# Construire l'image (première fois : ~8-13 minutes)
docker build -t prediction-temperature:latest .

# Lancer le conteneur
docker run -d \
  --name prediction-temperature \
  -p 8501:8501 \
  --env-file .env \
  prediction-temperature:latest
```

Ou avec Docker Compose (recommandé) :
```bash
# Construire et lancer en une commande
docker-compose up -d --build

# Ou séparément
docker-compose build
docker-compose up -d
```

**Note** : Le premier build prend du temps car TensorFlow (~500MB) doit être téléchargé. Les builds suivants sont beaucoup plus rapides grâce au cache Docker.

## 📁 Structure du Projet

```
PredictionTemperature/
├── src/                          # Modules source (architecture MVC)
│   ├── __init__.py
│   ├── config.py                 # Configuration (villes, constantes, API keys)
│   ├── data_loader.py            # Chargement et preprocessing des données
│   ├── model_service.py           # Gestion des modèles TensorFlow
│   └── utils.py                  # Utilitaires (traduction, cache)
│
├── tests/                        # Suite de tests unitaires
│   ├── __init__.py
│   ├── conftest.py               # Fixtures pytest
│   ├── test_data_loader.py       # Tests pour data_loader
│   └── test_model_service.py     # Tests pour model_service
│
├── templates/                     # Assets statiques
│   └── assets/
│       └── température/
│           ├── model_architecture.png
│           └── models/           # Modèles entraînés (.keras + .pkl)
│               ├── paris_model.keras
│               ├── paris_info.pkl
│               ├── silicon_valley_model.keras
│               └── silicon_valley_info.pkl
│
├── PrédictionTempératuresWeb.py  # Application Streamlit principale
├── requirements.txt              # Dépendances de production
├── requirements-dev.txt           # Dépendances de développement
├── Dockerfile                     # Configuration Docker
├── docker-compose.yml             # Configuration Docker Compose
├── pytest.ini                     # Configuration pytest
└── README.md                      # Ce fichier
```

### Architecture MVC

Le projet suit une architecture **Model-View-Controller** :

- **Model** (`src/model_service.py`, `src/data_loader.py`) : Logique métier, chargement de données, prédictions
- **View** (`PrédictionTempératuresWeb.py`) : Interface utilisateur Streamlit
- **Controller** (`src/config.py`, `src/utils.py`) : Configuration et utilitaires

Cette séparation facilite la maintenance, les tests et l'évolution du code.

## 🧪 Tests

### Exécuter les tests

**Important** : Assurez-vous d'être dans l'environnement conda `PredictionTemperature` et utilisez `python -m pytest` au lieu de `pytest` directement pour garantir l'utilisation du bon interpréteur Python.

```bash
# Activer l'environnement conda (si nécessaire)
conda activate PredictionTemperature

# Tous les tests (recommandé : utiliser python -m pytest)
python -m pytest

# Ou utiliser le script fourni
./run_tests.sh

# Avec couverture
python -m pytest --cov=src --cov-report=html

# Un fichier spécifique
python -m pytest tests/test_data_loader.py

# Mode verbeux
python -m pytest -v
```

**Note** : Si vous obtenez une erreur `ModuleNotFoundError: No module named 'tensorflow'`, cela signifie que pytest utilise un mauvais interpréteur Python. Utilisez `python -m pytest` au lieu de `pytest` directement.

### Structure des tests

- **Tests unitaires** : Chaque module a sa suite de tests
- **Fixtures** : Données de test réutilisables dans `conftest.py`
- **Mocks** : Simulation des modèles TensorFlow pour éviter les dépendances lourdes

## 🚀 Déploiement

### Streamlit Cloud

1. Connectez votre dépôt GitHub à [Streamlit Cloud](https://streamlit.io/cloud)
2. Configurez la variable d'environnement `OPENWEATHER_API_KEY`
3. Déployez depuis la branche `main`

### Docker

Voir le fichier [DEPLOYMENT.md](DEPLOYMENT.md) pour les instructions détaillées.

### CI/CD

Le workflow GitHub Actions (`.github/workflows/main.yml`) exécute automatiquement :
- Tests unitaires avec pytest
- Linting avec Ruff et Flake8
- Build Docker

## 📊 Résultats

### Performance du Modèle

| Ville | MAE (Mean Absolute Error) | Temps d'entraînement |
|-------|---------------------------|---------------------|
| **Paris** | 2.84°C | < 2 secondes |
| **Silicon Valley** | 0.98°C | < 2 secondes |

### Spécifications Techniques

- **Données d'entraînement** : 10 ans d'historique par ville
- **Fenêtre temporelle** : 30 jours (WINDOW_SIZE)
- **Horizon de prédiction** : 7 jours (FORECAST_HORIZON)
- **Variables prédites** : Température moyenne, minimale, maximale
- **Optimiseur** : Adam (learning rate: 0.001)
- **Loss function** : Huber (robuste aux valeurs aberrantes)

## 🛠️ Technologies Utilisées

- **Deep Learning** : TensorFlow/Keras
- **Web Framework** : Streamlit
- **Data Processing** : Pandas, NumPy
- **Visualization** : Plotly
- **APIs** : Meteostat (données historiques), OpenWeatherMap (prévisions)
- **Testing** : Pytest
- **Containerization** : Docker

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👤 Auteur

**Gabriel Marie-Brisson**

- Portfolio : [gabriel.mariebrisson.fr](https://gabriel.mariebrisson.fr)
- GitHub : [@gabrielmariebrisson](https://github.com/gabrielmariebrisson)

## 🙏 Remerciements

- [Meteostat](https://meteostat.net/) pour les données météorologiques historiques
- [OpenWeatherMap](https://openweathermap.org/) pour les prévisions de référence
- La communauté TensorFlow et Streamlit pour les outils exceptionnels

---

⭐ Si ce projet vous est utile, n'hésitez pas à lui donner une étoile !

