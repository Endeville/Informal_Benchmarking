"""
"God" script: simulate the OVB setting where A is known.

Data-generating process (partially linear model):
    D = X @ β_dx + A @ β_da + ε_d
    Y = τ·D  + X @ β_yx + A @ β_ya + ε_y

where X is (n, num_X) and A is (n, num_A); β coefficients are uniform vectors.

We compare:
  - Short regression  OLS(Y ~ D + X)        → biased  τ̂_short
  - Long  regression  OLS(Y ~ D + X + A)    → unbiased τ̂_long
  - Bias bound B from Corollary 1           → verifies |OVB| ≤ B
"""

import numpy as np
import statsmodels.api as sm
from ovb_formula import compute_B2


def generate_data(n=2000, tau=1.0,
                  num_X=5, num_A=1,
                  beta_dx=0.5, beta_da=0.8,
                  beta_yx=0.5, beta_ya=0.8,
                  seed=42):
    """
    Generate synthetic data from a partially linear DGP with OVB.

    Parameters
    ----------
    n       : sample size
    tau     : true treatment effect
    num_X   : number of observed covariates (columns of X)
    num_A   : number of hidden confounders (columns of A)
    beta_dx : uniform coefficient of each X column on D
    beta_da : uniform coefficient of each A column on D
    beta_yx : uniform coefficient of each X column on Y
    beta_ya : uniform coefficient of each A column on Y
    seed    : random seed

    Returns
    -------
    Y : (n,) outcome
    D : (n,) treatment
    X : (n, num_X) observed covariates
    A : (n, num_A) hidden confounders
    """
    def _to_arr(val, size):
        a = np.asarray(val, dtype=float)
        return np.full(size, a) if a.ndim == 0 else a

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, num_X))
    A = rng.standard_normal((n, num_A))

    beta_dx_arr = _to_arr(beta_dx, num_X)
    beta_yx_arr = _to_arr(beta_yx, num_X)
    beta_da_arr = _to_arr(beta_da, num_A)
    beta_ya_arr = _to_arr(beta_ya, num_A)

    D = X @ beta_dx_arr + A @ beta_da_arr + rng.standard_normal(n)
    Y = tau * D + X @ beta_yx_arr + A @ beta_ya_arr + rng.standard_normal(n)

    return Y, D, X, A


def main():
    tau_true = 1.0
    Y, D, X, A = generate_data(tau=tau_true)

    short = sm.OLS(Y, sm.add_constant(np.column_stack([D, X]))).fit()
    long  = sm.OLS(Y, sm.add_constant(np.column_stack([D, X, A]))).fit()

    tau_short = short.params[1]
    tau_long  = long.params[1]
    ovb       = tau_short - tau_long

    bound = compute_B2(Y, D, X, A)

    print("=" * 54)
    print("OVB Simulation — God Mode (A is known)")
    print("=" * 54)
    print(f"  True τ          : {tau_true:.4f}")
    print(f"  Short τ̂  (biased): {tau_short:.4f}   OVB = {ovb:+.4f}")
    print(f"  Long  τ̂  (true)  : {tau_long:.4f}")
    print()
    print("  Bias bound (Corollary 1)")
    print(f"    S²             : {bound['S2']:.4f}")
    print(f"    C_Y²           : {bound['C_Y2']:.4f}")
    print(f"    C_D²           : {bound['C_D2']:.4f}")
    print(f"    B  = √(C_Y²·C_D²·S²) : {bound['B']:.4f}")
    check = "✓" if abs(ovb) <= bound["B"] + 1e-9 else "✗"
    print(f"    |OVB| = {abs(ovb):.4f}  ≤  B = {bound['B']:.4f}  {check}")


if __name__ == "__main__":
    main()
