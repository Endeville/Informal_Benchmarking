"""
Core OVB formulas from Corollary 1 (Chernozhukov et al., 2024).

Squared bias bound:  B² = C_Y² · C_D² · S²

  S²   = E(Y - g_s)² · E[α_s²]          -- identifiable from short data
  C_Y² = η²_{Y ~ A | D, X}              -- partial R² of A with outcome
  C_D² = η²_{D ~ A | X} / (1 - η²)     -- precision gain of A in treatment

The bound satisfies |θ_s - θ|² ≤ B².

NOTE on C_Y²: must be computed as (RSS_short - RSS_long) / RSS_short, NOT as
R²(Y_resid ~ A). The two differ when A is correlated with D (the usual case).
"""

import numpy as np
import statsmodels.api as sm


def _rss(Y, X_regressors):
    return np.sum(sm.OLS(Y, sm.add_constant(X_regressors)).fit().resid ** 2)


def _partial_R2(y, X, A):
    """η²_{y ~ A | X} = (RSS_without_A - RSS_with_A) / RSS_without_A."""
    rss_short = _rss(y, X)
    rss_long  = _rss(y, np.column_stack([X, A]))
    return (rss_short - rss_long) / rss_short


def compute_S2(Y, D, X):
    """
    S² = E(Y - g_s)² · E[α_s²]

    Returns (S², Y_res, D_res):
    - Y_res = Y - g_s  (short outcome regression residuals)
    - D_res = D - E[D|X]  (residualised treatment)
    - E[α_s²] = 1 / E[D_res²]   since α_s = D_res / E[D_res²]
    """
    D_res = sm.OLS(D, sm.add_constant(X)).fit().resid
    Y_res = sm.OLS(Y, sm.add_constant(np.column_stack([D, X]))).fit().resid

    S2 = np.mean(Y_res ** 2) / np.mean(D_res ** 2)
    return {
        "S2": S2,
        "Y_res": Y_res,
        "D_res": D_res
    }


def compute_C_Y2(Y, D, X, A):
    """
    C_Y² = η²_{Y ~ A | D, X}
]
    Corollary 1, Equation (7)
    """
    return _partial_R2(Y, np.column_stack([D, X]), A)


def compute_C_D2(D, X, A):
    """
    C_D² = η²_{D ~ A | X} / (1 - η²_{D ~ A | X})

    Corollary 1, Equation (7) - Sub R² for 1-η² (1 - eta²)
    """
    eta2 = _partial_R2(D, X, A)
    return eta2 / (1.0 - eta2) if eta2 < 1.0 else np.inf


def compute_B2(Y, D, X, A):
    """
    Full bias bound: B² = C_Y² · C_D² · S²

    Returns a dict with B, B², C_Y², C_D², S².
    """
    S2 = compute_S2(Y, D, X)["S2"]
    C_Y2 = compute_C_Y2(Y, D, X, A)
    C_D2 = compute_C_D2(D, X, A)
    B2 = C_Y2 * C_D2 * S2
    return {
        "B2":   B2,
        "B":    np.sqrt(B2),
        "C_Y2": C_Y2,
        "C_D2": C_D2,
        "S2":   S2,
    }
