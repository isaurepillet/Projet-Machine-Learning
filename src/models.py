"""
models.py
---------
Fonctions d'entraînement et d'évaluation des modèles ML.
"""

import pandas as pd
import numpy as np
import time

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform


def evaluate(name, y_true, y_pred):
    """
    Calcule et affiche R², RMSE, MAE pour un modèle.

    Returns
    -------
    dict avec les métriques arrondies.
    """
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f'{name:30s} | R2={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f}')
    return {
        'Modele': name,
        'R2'    : round(r2, 4),
        'RMSE'  : round(rmse, 4),
        'MAE'   : round(mae, 4)
    }


def get_model_list(random_state=42):
    """
    Retourne la liste des modèles à comparer.
    Chaque appel instancie de nouveaux objets pour éviter
    le bug de partage d'état entre appels successifs.
    """
    return [
        ('OLS',
         LinearRegression()),
        ('Ridge',
         Ridge(alpha=1.0)),
        ('Lasso',
         Lasso(alpha=0.001, max_iter=5000)),
        ('Random Forest',
         RandomForestRegressor(
             n_estimators=200, n_jobs=-1,
             random_state=random_state, max_depth=20)),
        ('Extra Trees',
         ExtraTreesRegressor(
             n_estimators=200, n_jobs=-1,
             random_state=random_state, max_depth=20)),
        ('Gradient Boosting',
         GradientBoostingRegressor(
             n_estimators=200, learning_rate=0.1,
             max_depth=5, random_state=random_state,
             subsample=0.8)),
    ]


def run_models(X_tr, X_te, y_tr, y_te, feat, label, random_state=42):
    """
    Entraîne tous les modèles et retourne résultats + objets.

    Parameters
    ----------
    X_tr, X_te : pd.DataFrame
        Features standardisées.
    y_tr, y_te : pd.Series
        Variables cibles.
    feat : list of str
        Noms des features (pour l'importance).
    label : str
        Label pour l'affichage.
    random_state : int

    Returns
    -------
    res_df : pd.DataFrame
        Tableau des métriques trié par R² décroissant.
    models : dict
        Dictionnaire {nom_modele: modele_entraine}.
    """
    print(f'\n{"="*60}')
    print(f'MODELES -- {label}')
    print(f'{"="*60}')

    results = []
    models  = {}

    for name, model in get_model_list(random_state):
        print(f'Training {name}...')
        t0 = time.time()
        model.fit(X_tr, y_tr)
        results.append(evaluate(name, y_te, model.predict(X_te)))
        models[name] = model
        print(f'   temps : {time.time()-t0:.1f}s')

    res_df = pd.DataFrame(results).sort_values('R2', ascending=False)
    print(f'\nTABLEAU {label}')
    print(res_df.to_string(index=False))

    # Overfitting check
    print(f'\nOVERFITTING CHECK {label}')
    for name, m in models.items():
        r2_tr = r2_score(y_tr, m.predict(X_tr))
        r2_te = r2_score(y_te, m.predict(X_te))
        gap   = r2_tr - r2_te
        flag  = 'WARNING' if gap > 0.05 else 'OK'
        print(f'{flag} {name:25s} | '
              f'train={r2_tr:.4f} | test={r2_te:.4f} | gap={gap:.4f}')

    # Feature importance
    print(f'\nTOP 15 FEATURES -- Extra Trees {label}')
    imp_s = pd.Series(
        models['Extra Trees'].feature_importances_,
        index=feat
    ).sort_values(ascending=False)
    print(imp_s.head(15))

    return res_df, models


def tune_model(model, param_dist, X_tr, y_tr, X_te, y_te,
               splits, n_iter=20, random_state=42, label=''):
    """
    Optimise les hyperparamètres par RandomizedSearchCV
    avec validation croisée temporelle.

    Parameters
    ----------
    model : sklearn estimator
    param_dist : dict
        Espace de recherche des hyperparamètres.
    X_tr, y_tr : train
    X_te, y_te : test
    splits : list of (train_idx, val_idx)
        Folds temporels (output de make_temporal_splits).
    n_iter : int
        Nombre de configurations testées.

    Returns
    -------
    search : RandomizedSearchCV fitté
    r2_tuned : float
    """
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=splits,
        scoring='r2',
        random_state=random_state,
        verbose=1,
        refit=True
    )
    t0 = time.time()
    search.fit(X_tr.values, y_tr.values)

    r2_tuned = r2_score(y_te, search.best_estimator_.predict(X_te))

    print(f'\n=== RÉSULTATS TUNING {label} ===')
    print(f'Meilleurs params : {search.best_params_}')
    print(f'R2 CV (val)      : {search.best_score_:.4f}')
    print(f'R2 test (tuned)  : {r2_tuned:.4f}')
    print(f'Temps            : {time.time()-t0:.1f}s')

    return search, r2_tuned


# Espaces de recherche par défaut
PARAM_DIST_ET = {
    'n_estimators'    : [100, 200, 300, 500],
    'max_depth'       : [10, 15, 20, 30, None],
    'min_samples_leaf': [1, 2, 4, 8],
}

PARAM_DIST_GBM = {
    'n_estimators' : [100, 200, 300],
    'learning_rate': loguniform(0.01, 0.3),
    'max_depth'    : [3, 4, 5, 6, 8],
    'subsample'    : uniform(0.6, 0.4),
}
