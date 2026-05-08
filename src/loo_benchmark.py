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


def loo_benchmark(Y, D, X_matrix, *, mock_indices=None, A_true=None):
    """
    Parameters
    ----------
    Y, D          : 1-D arrays
    X_matrix      : 2-D array, shape (n, p),  p >= 2
    mock_indices  : list of int, optional — restrict LOO to these column indices.
                    Default: all p columns.
    A_true        : (n,) or (n, k) array, optional — true hidden confounder (god-mode).
                    When supplied, an extra entry with key 'oracle': True is appended
                    to results so callers can compare the LOO budget to the true bound.

    Returns
    -------
    results  : list of dicts, one per mock covariate (plus optional oracle entry at end)
    S2_full  : float — S² from the full short regression (all X included)
    B_max    : float — max B across LOO mock entries (oracle excluded)
    """
    n, p = X_matrix.shape
    if p < 2:
        raise ValueError("LOO requires at least 2 covariates in X.")

    indices = list(mock_indices) if mock_indices is not None else list(range(p))

    S2_full = compute_S2(Y, D, X_matrix)

    results = []
    for j in indices:
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

    B_max = max(r["B"] for r in results) if results else 0.0

    if A_true is not None:
        A_col = np.reshape(A_true, (n, -1))
        C_Y2_o = compute_C_Y2(Y, D, X_matrix, A_col)
        C_D2_o = compute_C_D2(D, X_matrix, A_col)
        B2_o   = C_Y2_o * C_D2_o * S2_full
        results.append({
            "covariate": "A_true",
            "C_Y2":  C_Y2_o,
            "C_D2":  C_D2_o,
            "B2":    B2_o,
            "B":     np.sqrt(B2_o),
            "oracle": True,
        })

    return results, S2_full, B_max
