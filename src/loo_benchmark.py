"""
Leave-One-Out (LOO) informal benchmarking.

For each observed covariate X_j, treat it as if it were the "missing" A:
  1. Remove X_j from the short regression  →  X_{-j} are the controls
  2. Compute benchmark C_Y² and C_D² using X_j as the mock confounder
  3. Compute benchmark B = √(C_Y² · C_D² · S²)

Interpretation: B is the OVB bound you would face if the hidden confounder A
were exactly as powerful as X_j in explaining both D and Y.
"""

import numpy as np
from ovb_formula import compute_S2, compute_C_Y2, compute_C_D2


def loo_benchmark(Y, D, X_matrix):
    """
    Parameters
    ----------
    Y, D      : 1-D arrays
    X_matrix  : 2-D array, shape (n, p),  p >= 2

    Returns
    -------
    results   : list of dicts, one per covariate
    S2_full   : S² from the full short regression (all X included)
    """
    n, p = X_matrix.shape
    if p < 2:
        raise ValueError("LOO requires at least 2 covariates in X.")

    S2_full = compute_S2(Y, D, X_matrix)["S2"]

    results = []
    for j in range(p):
        A_mock      = X_matrix[:, j]
        X_remainder = np.delete(X_matrix, j, axis=1)

        C_Y2 = compute_C_Y2(Y, D, X_remainder, A_mock)
        C_D2 = compute_C_D2(D, X_remainder, A_mock)
        B2   = C_Y2 * C_D2 * S2_full

        results.append({
            "covariate": f"X_{j + 1}",
            "C_Y2": C_Y2,
            "C_D2": C_D2,
            "B2":   B2,
            "B":    np.sqrt(B2),
        })

    return results, S2_full
