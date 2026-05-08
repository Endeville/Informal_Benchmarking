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

import warnings
import numpy as np
import statsmodels.api as sm


def _rss(Y, X_regressors):
    return np.sum(sm.OLS(Y, sm.add_constant(X_regressors)).fit().resid ** 2)


def _partial_R2(y, X, A):
    """η²_{y ~ A | X} = (RSS_without_A - RSS_with_A) / RSS_without_A."""
    rss_short = _rss(y, X)
    rss_long  = _rss(y, np.column_stack([X, A]))
    return (rss_short - rss_long) / rss_short


def compute_S2(Y, D, X) -> float:
    """
    S² = E(Y - g_s)² · E[α_s²]

    Identifiable from observed data; the scale factor in the bias bound.
      Y_res = Y - g_s         (short outcome regression residuals)
      D_res = D - E[D|X]      (residualised treatment)
      E[α_s²] = 1/E[D_res²]  since α_s = D_res / E[D_res²]
    """
    D_res = sm.OLS(D, sm.add_constant(X)).fit().resid
    Y_res = sm.OLS(Y, sm.add_constant(np.column_stack([D, X]))).fit().resid
    return np.mean(Y_res ** 2) / np.mean(D_res ** 2)


def compute_C_Y2(Y, D, X, A):
    """
    C_Y² = η²_{Y ~ A | D, X}

    Corollary 1, Equation (7)
    """
    return _partial_R2(Y, np.column_stack([D, X]), A)


def compute_C_D2(D, X, A):
    """
    C_D² = η²_{D ~ A | X} / (1 - η²_{D ~ A | X})

    Corollary 1, Equation (7). Returns np.inf when eta² = 1 (perfect collinearity
    between A_mock and X); B_max = inf and coverage trivially holds — flag this.
    """
    eta2 = _partial_R2(D, X, A)
    if eta2 >= 1.0:
        warnings.warn(
            "compute_C_D2: eta² ≈ 1 (near-perfect collinearity between A_mock and X). "
            "C_D² = ∞, B_max = ∞, coverage trivially passes. Check for collinear covariates.",
            RuntimeWarning, stacklevel=2,
        )
        return np.inf
    return eta2 / (1.0 - eta2)


def compute_B2(Y, D, X, A):
    """
    Full bias bound: B² = C_Y² · C_D² · S²

    Returns a dict with B, B², C_Y², C_D², S².
    """
    S2   = compute_S2(Y, D, X)
    C_Y2 = compute_C_Y2(Y, D, X, A)
    C_D2 = compute_C_D2(D, X, A)
    B2   = C_Y2 * C_D2 * S2
    return {
        "B2":   B2,
        "B":    np.sqrt(B2),
        "C_Y2": C_Y2,
        "C_D2": C_D2,
        "S2":   S2,
    }


def compute_B2_oracle(Y, D, X, A_true):
    """
    Bias bound using the true hidden confounder (god-mode access).

    Identical to compute_B2 but explicitly named to distinguish oracle use from
    LOO mock use. In simulations, compare this against B_max from loo_benchmark()
    to see how much the LOO approximation costs.
    """
    return compute_B2(Y, D, X, A_true)


def compute_RV(tau_hat, S2):
    """
    Robustness Value: minimum equal confounding (C_Y² = C_D² = RV) such that B = |tau_hat|.

    Derived from B² = C_Y² · C_D² · S² = RV² · S² = tau_hat²  →  RV = |tau_hat| / √S².

    Interpretation: if the hidden confounder has C_Y² ≥ RV and C_D² ≥ RV simultaneously,
    it can fully account for the observed estimate.  A small RV means the conclusion is fragile.
    """
    return abs(tau_hat) / np.sqrt(max(S2, 1e-15))
