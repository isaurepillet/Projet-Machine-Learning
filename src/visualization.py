"""
visualization.py
----------------
Fonctions de visualisation pour le rapport académique.
Style sobre et lisible, compatible LaTeX (police serif).
Toutes les figures sont sauvegardées en PNG 150 dpi.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

plt.rcParams.update({
    'font.family'      : 'serif',
    'font.size'        : 11,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
})

BLUE  = '#2C5F8A'
GREEN = '#2A8A5A'
RED   = '#C0392B'


def plot_comparison(res_A_df, res_B_df, r2_naive,
                    save_path='outputs/figures/fig4_comparaison.png'):
    """
    Figure 4 — Comparaison R² de tous les modèles (cibles A et B).
    """
    ordre  = ['OLS','Ridge','Lasso',
               'Gradient Boosting','Random Forest','Extra Trees']
    labels = ['OLS','Ridge','Lasso',
               'Gradient\nBoosting','Random\nForest','Extra\nTrees']

    r2_A = [res_A_df[res_A_df['Modele']==m]['R2'].values[0]
            for m in ordre]
    r2_B = [res_B_df[res_B_df['Modele']==m]['R2'].values[0]
            for m in ordre]

    x, w = np.arange(len(ordre)), 0.35
    fig, ax = plt.subplots(figsize=(13, 6))

    bars_A = ax.bar(x - w/2, r2_A, w,
                    label='Cible A — log(recettes totales)',
                    color=BLUE, edgecolor='white', linewidth=0.5)
    bars_B = ax.bar(x + w/2, r2_B, w,
                    label='Cible B — log(recettes/habitant)',
                    color=GREEN, edgecolor='white',
                    linewidth=0.5, alpha=0.85)

    for bar in list(bars_A) + list(bars_B):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                f'{h:.3f}', ha='center', va='bottom', fontsize=8.5)

    ax.axvline(x=2.5, color='grey', linestyle='--',
               linewidth=1, alpha=0.6)
    ax.text(1.0, 1.05, 'Modèles linéaires', ha='center',
            fontsize=9, color='grey', style='italic')
    ax.text(4.5, 1.05, 'Algorithmes ML', ha='center',
            fontsize=9, color='grey', style='italic')
    ax.axhline(y=r2_naive, color=RED, linestyle=':',
               linewidth=1.5,
               label=f'Baseline naïf (R²={r2_naive:.3f})')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('$R^2$ hors-échantillon', fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        'Figure 4 — Comparaison des performances prédictives\n'
        '(ensemble de test 2022–2023)',
        fontsize=12, fontweight='bold', pad=12)
    ax.legend(fontsize=9, framealpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Sauvegardée : {save_path}')


def plot_diagnostic(y_te, y_pred, label='Extra Trees, Cible A',
                    save_path='outputs/figures/fig5_diagnostic.png'):
    """
    Figure 5 — Predicted vs Actual + Distribution des résidus.
    """
    residus = y_pred - np.array(y_te)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_te, y_pred, alpha=0.06,
                    color=BLUE, s=2, rasterized=True)
    lims = [float(np.min(y_te)), float(np.max(y_te))]
    axes[0].plot(lims, lims, 'r--', linewidth=1.5,
                 label='Prédiction parfaite')
    axes[0].set_xlabel('Valeurs réelles $\\tilde{y}_{i,t}$', fontsize=11)
    axes[0].set_ylabel('Valeurs prédites $\\hat{y}_{i,t}$', fontsize=11)
    axes[0].set_title(f'Prédictions vs Réalité\n({label})',
                      fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=9)

    axes[1].hist(residus, bins=80, color=BLUE,
                 edgecolor='white', linewidth=0.3)
    axes[1].axvline(0, color=RED, linestyle='--', linewidth=1.5)
    axes[1].set_xlabel('Résidus ($\\hat{y} - y$)', fontsize=11)
    axes[1].set_ylabel('Fréquence', fontsize=11)
    axes[1].set_title(f'Distribution des résidus\n({label})',
                      fontsize=11, fontweight='bold')
    axes[1].text(
        0.97, 0.95,
        f'$\\bar{{e}}$ = {residus.mean():.4f}\n'
        f'$\\sigma_e$ = {residus.std():.4f}',
        transform=axes[1].transAxes, ha='right', va='top',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle(f'Figure 5 — Diagnostic des prédictions ({label})',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Sauvegardée : {save_path}')


def plot_shap_summary(model, X_sample, title, fig_num,
                      save_path=None):
    """
    SHAP summary plot pour un modèle GBM.
    """
    if save_path is None:
        save_path = f'outputs/figures/fig{fig_num}_shap.png'

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_sample,
                      max_display=20, show=False)
    plt.title(title, fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Sauvegardée : {save_path}')
    return shap_vals


def plot_strate_performance(df_strates,
                             save_path='outputs/figures/fig8_strates.png'):
    """
    Figure 8 — R² et RMSE par strate de taille de commune.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(df_strates['Strate'], df_strates['R2'],
                color=BLUE, edgecolor='white')
    axes[0].set_xlabel('Strate de population', fontsize=11)
    axes[0].set_ylabel('$R^2$', fontsize=11)
    axes[0].set_title('$R^2$ par strate de taille',
                      fontsize=11, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=20)
    for i, row in df_strates.iterrows():
        axes[0].text(i, row['R2'] + 0.005,
                     f"{row['R2']:.3f}", ha='center', fontsize=9)

    axes[1].bar(df_strates['Strate'], df_strates['RMSE'],
                color='#E07B39', edgecolor='white')
    axes[1].set_xlabel('Strate de population', fontsize=11)
    axes[1].set_ylabel('RMSE', fontsize=11)
    axes[1].set_title('RMSE par strate de taille',
                      fontsize=11, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=20)
    for i, row in df_strates.iterrows():
        axes[1].text(i, row['RMSE'] + 0.002,
                     f"{row['RMSE']:.3f}", ha='center', fontsize=9)

    plt.suptitle(
        'Figure 8 — Performances par strate de taille\n'
        '(Extra Trees, Cible A, test 2022–2023)',
        fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Sauvegardée : {save_path}')
