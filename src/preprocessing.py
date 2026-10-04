"""
preprocessing.py
----------------
Fonctions de preprocessing réutilisables pour le pipeline ML.
Toutes les transformations sont estimées sur le train uniquement
(train-only fitting) pour éviter le data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


def build_pipeline(
    train_df,
    test_df,
    target_col,
    features,
    forced_features,
    var_threshold=0.01,
    corr_threshold=0.95,
    verbose=True
):
    """
    Pipeline complet de preprocessing pour un jeu de données de panel.

    Étapes appliquées dans l'ordre :
    1. Filtre NaN sur la cible
    2. Variance threshold (hors forced_features)
    3. Imputation médiane — fit sur train uniquement
    4. Correlation filter — fit sur train uniquement
    5. Standardisation z-score — fit sur train uniquement

    Parameters
    ----------
    train_df : pd.DataFrame
        Ensemble d'entraînement (2017-2021).
    test_df : pd.DataFrame
        Ensemble de test (2022-2023).
    target_col : str
        Nom de la variable cible (déjà transformée en log).
    features : list of str
        Liste des features candidates.
    forced_features : list of str
        Features protégées des filtres automatiques (ex: lags des cibles).
    var_threshold : float
        Seuil de variance minimum pour conserver une feature.
    corr_threshold : float
        Seuil de corrélation au-delà duquel une feature est supprimée.
    verbose : bool
        Afficher les étapes intermédiaires.

    Returns
    -------
    X_tr, X_te : pd.DataFrame
        Features standardisées.
    y_tr, y_te : pd.Series
        Variables cibles.
    feat_fin : list of str
        Features retenues après filtrage.
    """
    # 1. Filtre NaN sur la cible
    tr = train_df[train_df[target_col].notna()].reset_index(drop=True)
    te = test_df[test_df[target_col].notna()].reset_index(drop=True)
    if verbose:
        print(f'  Train : {len(tr):,} | Test : {len(te):,}')

    X_tr_raw = tr[features].copy()
    X_te_raw = te[features].copy()

    # 2. Variance threshold
    to_filter = [c for c in features if c not in forced_features]
    forced    = [c for c in features if c in forced_features]

    vt = VarianceThreshold(threshold=var_threshold)
    vt.fit(X_tr_raw[to_filter])
    feat_var = [f for f, k in zip(to_filter, vt.get_support()) if k]
    feat_var = list(dict.fromkeys(feat_var + forced))
    if verbose:
        print(f'  Après variance threshold : {len(feat_var)}')

    # 3. Imputation médiane (fit sur train)
    imp = SimpleImputer(strategy='median')
    imp.fit(X_tr_raw[feat_var])
    X_tr_imp = pd.DataFrame(
        imp.transform(X_tr_raw[feat_var]), columns=feat_var)
    X_te_imp = pd.DataFrame(
        imp.transform(X_te_raw[feat_var]), columns=feat_var)

    # 4. Correlation filter (fit sur train)
    corr_m = X_tr_imp.corr().abs()
    upper  = corr_m.where(
        np.triu(np.ones(corr_m.shape), k=1).astype(bool))
    drop = [c for c in upper.columns
            if any(upper[c] > corr_threshold)
            and c not in forced_features]
    feat_fin = list(dict.fromkeys(
        [c for c in feat_var if c not in drop]))
    if verbose:
        print(f'  Après correlation filter : {len(feat_fin)}')

    # 5. Standardisation (fit sur train)
    sc = StandardScaler()
    sc.fit(X_tr_imp[feat_fin])
    X_tr = pd.DataFrame(
        sc.transform(X_tr_imp[feat_fin]), columns=feat_fin)
    X_te = pd.DataFrame(
        sc.transform(X_te_imp[feat_fin]), columns=feat_fin)

    y_tr = tr[target_col]
    y_te = te[target_col]

    if verbose:
        print(f'  NaN X_train : {X_tr.isnull().sum().sum()} | '
              f'NaN y_train : {y_tr.isnull().sum()}')
        print(f'  Lags forcés présents : '
              f'{[c for c in forced_features if c in feat_fin]}')

    return X_tr, X_te, y_tr, y_te, feat_fin


def create_lags(df, cols_to_lag, id_col='codgeo', time_col='an'):
    """
    Crée les lags N-1 pour une liste de colonnes.
    Utilise pd.concat pour éviter le PerformanceWarning.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset trié par [id_col, time_col].
    cols_to_lag : list of str
        Colonnes à lagger.
    id_col : str
        Identifiant de l'entité (commune).
    time_col : str
        Colonne temporelle.

    Returns
    -------
    df_with_lags : pd.DataFrame
    lag_cols : list of str
        Noms des colonnes laggées créées.
    """
    df_sorted  = df.sort_values([id_col, time_col])
    lag_frames = {}
    for col in cols_to_lag:
        lag_frames[col + '_lag1'] = (
            df_sorted.groupby(id_col)[col].shift(1)
        )
    df_with_lags = pd.concat(
        [df_sorted, pd.DataFrame(lag_frames)], axis=1)
    lag_cols = list(lag_frames.keys())
    return df_with_lags, lag_cols


def make_temporal_splits(years, min_train_years=2):
    """
    Crée des folds temporels walk-forward pour la validation croisée.

    Fold k : train = années < année_k, val = année_k.
    Évite le data leakage temporel du KFold classique.

    Parameters
    ----------
    years : np.ndarray
        Années correspondant aux observations du train.
    min_train_years : int
        Nombre minimum d'années dans le premier fold d'entraînement.

    Returns
    -------
    splits : list of (train_idx, val_idx)
    """
    unique_years = sorted(np.unique(years))
    splits = []
    for i in range(min_train_years, len(unique_years)):
        train_idx = np.where(years < unique_years[i])[0]
        val_idx   = np.where(years == unique_years[i])[0]
        splits.append((train_idx, val_idx))
    return splits
