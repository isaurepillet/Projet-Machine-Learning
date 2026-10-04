# Prédiction des recettes communales françaises par méthodes de Statistical Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

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
Projet-Machine-Learning/
│
├── README.md                          ← ce fichier
├── requirements.txt                   ← dépendances Python
├── .gitignore
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
│   ├── preprocessing.py               ← preprocessing utilities
│   ├── models.py                      ← modelling utilities
│   └── visualization.py               ← visualisation utilities
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
| A — Recettes totales | Random Forest | **0.970** | **0.251** |
| B — Recettes/habitant | Gradient Boosting | **≈ 0.719** | — |

*Évaluation hors-échantillon sur 2022–2023. The saved pipeline output contains 69,864 observations for target A and 69,852 for target B.*

For target A, Random Forest reaches an out-of-sample R² of 0.970 in the saved notebook output, compared with about 0.68–0.70 for the linear specifications shown there. For target B, the modelling notebook identifies Gradient Boosting as the best-performing specification (R² ≈ 0.719).

---

## Installation

```bash
git clone https://github.com/isaurepillet/Projet-Machine-Learning.git
cd Projet-Machine-Learning
pip install -r requirements.txt
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

