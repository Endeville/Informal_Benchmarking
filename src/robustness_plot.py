"""
Robustness contour plot for OVB sensitivity analysis.

The contour shows all (C_D², C_Y²) pairs where the bias bound B equals |tau_hat|.
Points above (and to the right of) the contour fall in the "danger zone" — a confounder
with those partial-R² values could fully explain the estimated treatment effect.

LOO benchmark points are plotted as reference marks: if all of them lie below the contour,
the conclusion survives confounders at least as strong as any observed covariate.
"""

import numpy as np
import matplotlib.pyplot as plt
from ovb_formula import compute_RV


def plot_robustness_contour(tau_hat, S2, loo_results, ax=None, title="Robustness Contour"):
    """
    Parameters
    ----------
    tau_hat     : float   — short-regression treatment estimate
    S2          : float   — S² from the full short regression (from loo_benchmark)
    loo_results : list of dicts with keys 'covariate', 'C_Y2', 'C_D2', 'B'
    ax          : matplotlib Axes, or None to create a new figure
    title       : str

    Returns
    -------
    ax : the Axes used
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    RV = compute_RV(tau_hat, S2)

    c_d2_vals = [r['C_D2'] for r in loo_results]
    x_max = max(max(c_d2_vals) * 1.6, RV * 2.0, 0.4)
    c_d2_range = np.linspace(1e-4, x_max, 600)

    # Contour: C_Y² = tau_hat² / (S² · C_D²)
    c_y2_contour = tau_hat ** 2 / (S2 * c_d2_range)
    mask = c_y2_contour <= 1.0

    ax.plot(c_d2_range[mask], c_y2_contour[mask],
            color='black', linewidth=2,
            label=r'$B = |\hat{\tau}|$  (danger boundary)')
    ax.fill_between(c_d2_range[mask], c_y2_contour[mask], 1.0,
                    color='crimson', alpha=0.08,
                    label=r'Danger zone  ($B \geq |\hat{\tau}|$)')

    # LOO benchmark points
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(loo_results)))
    for r, col in zip(loo_results, colors):
        ax.scatter(r['C_D2'], r['C_Y2'], color=col, s=90, zorder=5,
                   label=r['covariate'])
        ax.annotate(r['covariate'], (r['C_D2'], r['C_Y2']),
                    textcoords='offset points', xytext=(6, 4), fontsize=9)

    # RV point on the C_Y² = C_D² diagonal
    if RV <= 1.0:
        ax.scatter([RV], [RV], marker='D', color='black', s=110, zorder=6,
                   label=f'RV = {RV:.3f}')
        ax.annotate(f'RV = {RV:.3f}', (RV, RV),
                    textcoords='offset points', xytext=(6, -14), fontsize=9,
                    fontweight='bold')

    ax.set_xlabel(r'$C_D^2$  (treatment sensitivity)', fontsize=12)
    ax.set_ylabel(r'$C_Y^2$  (outcome sensitivity)', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    return ax
