# Guide de Déploiement

Ce document décrit comment déployer l'application de prédiction météorologique.

## Prérequis

- Docker et Docker Compose installés
- Clé API OpenWeatherMap (à définir dans `.env`)

## Déploiement avec Docker

### 1. Configuration

Créez un fichier `.env` à la racine du projet :

```env
OPENWEATHER_API_KEY=votre_cle_api_ici
```

### 2. Construction de l'image

```bash
docker build -t prediction-temperature:latest .
```

### 3. Exécution avec Docker

```bash
docker run -d \
  --name prediction-temperature \
  -p 8501:8501 \
  --env-file .env \
  prediction-temperature:latest
```

### 4. Exécution avec Docker Compose

```bash
docker-compose up -d
```

L'application sera accessible sur `http://localhost:8501`

## Déploiement sur Cloud

### Streamlit Cloud

1. Connectez votre dépôt GitHub à [Streamlit Cloud](https://streamlit.io/cloud)
2. Configurez la variable d'environnement `OPENWEATHER_API_KEY` dans les paramètres
3. Déployez depuis la branche `main`

### AWS/GCP/Azure

Utilisez le Dockerfile fourni avec votre service de conteneurisation préféré :

- **AWS**: ECS, EKS, ou App Runner
- **GCP**: Cloud Run
- **Azure**: Container Instances ou App Service

## CI/CD

Le workflow GitHub Actions (`.github/workflows/main.yml`) exécute automatiquement :

- Tests unitaires avec pytest
- Linting avec Ruff et Flake8
- Build Docker (sans push)

## Vérification des Dépendances

Toutes les dépendances critiques sont présentes dans `requirements.txt` :

- ✅ `streamlit==1.50.0`
- ✅ `tensorflow_cpu==2.20.0`
- ✅ `meteostat==1.7.6`
- ✅ `plotly==6.3.1`
- ✅ `python-dotenv==1.2.1`
- ✅ `deep-translator==1.11.4`
- ✅ `scikit-learn==1.7.2`

## Optimisations Docker

Le Dockerfile utilise :

- **Multi-stage build** : Réduit la taille de l'image finale
- **Cache pip** : Accélère les rebuilds
- **User non-root** : Améliore la sécurité
- **Health check** : Surveillance automatique

## Troubleshooting

### Port déjà utilisé

Changez le port dans `docker-compose.yml` ou la commande Docker :

```bash
docker run -p 8502:8501 ...
```

### Modèles manquants

Assurez-vous que les fichiers de modèles sont présents dans `templates/assets/température/models/`

### Erreur de clé API

Vérifiez que `OPENWEATHER_API_KEY` est bien définie dans `.env` ou les variables d'environnement

