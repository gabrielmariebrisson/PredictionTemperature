# Tests Unitaires

Ce dossier contient la suite de tests unitaires pour le système de prédiction météorologique.

## Structure

- `conftest.py` : Fixtures pytest partagées (City de test, DataFrames mock)
- `test_data_loader.py` : Tests pour le module `data_loader` (preprocessing, features temporelles)
- `test_model_service.py` : Tests pour le module `model_service` (chargement de modèles, prédictions)

## Exécution des tests

### Installer les dépendances de test

```bash
pip install -r requirements.txt
```

### Exécuter tous les tests

```bash
pytest
```

### Exécuter un fichier de test spécifique

```bash
pytest tests/test_data_loader.py
pytest tests/test_model_service.py
```

### Exécuter une classe de test spécifique

```bash
pytest tests/test_data_loader.py::TestCreateTimeFeatures
```

### Exécuter un test spécifique

```bash
pytest tests/test_data_loader.py::TestCreateTimeFeatures::test_creates_time_features
```

### Exécuter avec couverture de code

```bash
pytest --cov=src --cov-report=html
```

## Notes

- Les tests utilisent des mocks pour éviter de charger de vrais modèles TensorFlow
- Les fixtures dans `conftest.py` fournissent des données de test réutilisables
- Les tests sont isolés et ne dépendent pas de services externes (Meteostat, OpenWeatherMap)

