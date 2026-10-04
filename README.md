# Prédiction des recettes communales françaises par méthodes de Statistical Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Contexte et motivation

Ce projet s'inscrit dans le contexte de la suppression progressive de la taxe
d'habitation sur les résidences principales (2018–2023), qui a profondément
recomposé la structure des recettes communales françaises en substituant une
recette locale stable par des fractions de TVA nationale dont le dynamisme
dépend de la conjoncture économique.

Cette recomposition structurelle pose un défi méthodologique direct : les modèles
économétriques linéaires, calibrés sur un régime fiscal désormais révolu, sont
inadaptés à capturer les nouvelles interactions non linéaires entre dotations
d'État, fiscalité résiduelle et caractéristiques socio-démographiques. Ce projet
évalue dans quelle mesure les algorithmes de *machine learning* permettent
d'améliorer la prédiction des recettes communales par rapport aux approches
économétriques à structure imposée.

Ce travail s'inspire directement de Chen et al. (2025),
*Can Machine Learning Algorithms Better Help Predict Fiscal Stress in Local
Governments?*, qui montrent sur des données américaines que les méthodes
ensemblistes surpassent systématiquement les régressions logistiques.

---

## Structure du repository

```
ml-finances-locales/
│
├── README.md                          ← ce fichier
├── requirements                       ← dépendances Python
├── gitignore
│
├── data/
│   ├── raw/                           ← données sources originales
│   │   ├── comptes_communes/          ← OFGL 2017-2023
│   │   ├── dotations/                 ← DGCL 2018-2025
│   │   ├── dvf/                       ← Demandes Valeurs Foncières
│   │   └── observatoire_territoires/  ← ANCT
│   ├── processed/
│   │   └── base_clean.parquet         ← dataset final après construction
│   └── README.md                      ← description des sources et licences
│
├── notebooks/
│   ├── 01_construction_base.ipynb     ← construction du panel et jointures
│   ├── 02_data_cleaning.ipynb         ← nettoyage des données
│   ├── 03_exploratory_analysis.ipynb  ← analyse exploratoire
│   └── 04_ml_models.ipynb             ← pipeline ML complet
│
├── src/                               ← fonctions Python réutilisables
│   ├── __init__.py
│   ├── preprocessing                  ← preprocessing utilities
│   ├── models                         ← modelling utilities
│   └── visualization                  ← visualisation utilities
│
├── outputs/
│   ├── figures/                       ← figures PNG pour le rapport
│   └── results/                       ← tableaux de résultats CSV
```

---

## Données

| Source | Période | Variables | Lien |
|--------|---------|-----------|------|
| OFGL — Comptes des communes | 2017–2023 | 52 variables comptables | [data.ofgl.fr](https://data.ofgl.fr) |
| DGCL — Dotations | 2018–2025 | 261 variables de dotations | [collectivites-locales.gouv.fr](https://www.collectivites-locales.gouv.fr) |
| DGFiP — DVF | 2014–2024 | 8 indicateurs immobiliers | [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/) |
| ANCT — Observatoire des territoires | 2016–2024 | 27 indicateurs socio-démographiques | [observatoire-des-territoires.gouv.fr](https://www.observatoire-des-territoires.gouv.fr) |

Le panel final couvre **N = 34 990 communes** sur **T = 8 années** (2016–2023),
soit **n = 279 920 observations**.

---

## Variables cibles

| Cible | Variable | Description |
|-------|---------|-------------|
| A | `log(1 + RecettesTotales)` | Recettes totales |
| B | `log(1 + RecettesTotales / Population)` | Recettes par habitant |
| C | `log(1 + RecettesHorsEmprunts)` | Recettes totales hors emprunts |
| D | `log(1 + RecettesFonctionnement)` | Recettes de fonctionnement |
| E | `log(1 + RecettesInvestissement)` | Recettes d'investissement hors emprunts |

---

## Modèles comparés

| Modèle | Type | Hyperparamètres |
|--------|------|-----------------|
| OLS | Linéaire | — |
| Ridge | Linéaire régularisé | α = 1.0 |
| Lasso | Linéaire régularisé | α = 0.001 |
| Random Forest | Ensembliste | n=200, max_depth=20 |
| Extra Trees | Ensembliste | n=200, max_depth=20 |
| Gradient Boosting | Ensembliste | n=200, lr=0.1, max_depth=5 |

---

## Résultats principaux

| Cible | Meilleur modèle | R² | RMSE |
|-------|----------------|-----|------|
| A — Recettes totales | Extra Trees | **0.970** | 0.247 |
| B — Recettes/habitant | Gradient Boosting | **0.719** | 0.273 |
| C — Recettes hors emprunts | Extra Trees | **0.982** | 0.191 |
| D — Recettes fonctionnement | Extra Trees | **0.991** | 0.137 |
| E — Recettes investissement | Extra Trees | **0.657** | 1.333 |

*Évaluation hors-échantillon sur le test 2022–2023 (n = 69 864 observations).*

Les modèles linéaires plafonnent à R²≈0.74 sur la cible A contre R²=0.970
pour Extra Trees — soit un gain de +23 points, confirmant la supériorité
des méthodes ensemblistes pour capturer les non-linéarités de la fiscalité
locale française.

---

## Installation

```bash
git clone https://github.com/isaurepillet/Projet-Machine-Learning.git
cd Projet-Machine-Learning
pip install -r requirements
```

### Lancer les notebooks dans l'ordre

```bash
# 1. Construction du panel
jupyter notebook notebooks/01_construction_base.ipynb

# 2. Nettoyage
jupyter notebook notebooks/02_data_cleaning.ipynb

# 3. Analyse exploratoire
jupyter notebook notebooks/03_exploratory_analysis.ipynb

# 4. Modèles ML (pipeline complet)
jupyter notebook notebooks/04_ml_models.ipynb
```

---

## Reproductibilité

Tous les modèles sont initialisés avec `random_state=42`.
Le split temporel (train 2017–2021 / test 2022–2023) est strict et déterministe.
Le preprocessing est estimé exclusivement sur l'ensemble d'entraînement
(*train-only fitting*) pour éviter tout *data leakage*.

---

## Références

- Chen, C., Zha, Y., Ren, C., & Zhao, Y. (2025). *Can Machine Learning
  Algorithms Better Help Predict Fiscal Stress in Local Governments?*
  Working Paper.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely Randomized Trees.
  *Machine Learning*, 63(1), 3–42.
- Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting
  Machine. *The Annals of Statistics*, 29(5), 1189–1232.
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting
  Model Predictions. *NeurIPS*, 30.
- Cour des comptes (2023). *Les finances publiques locales 2023*.
- Cour des comptes (2024). *L'intelligence artificielle dans les politiques
  publiques*.

---

## Licence

Les données sources sont issues de plateformes open data gouvernementales
françaises (licence Etalab 2.0). Le code est distribué sous licence MIT.
