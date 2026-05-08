"""
DGP configuration and data generation for OVB simulation.

DGPConfig is a frozen dataclass (immutable, hashable experiment spec).
Pass one to generate_data() to sample a dataset.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Union
import numpy as np


class TreatmentType(Enum):
    CONTINUOUS = "continuous"
    BINARY = "binary"           # D = 1(D* > 0), probit threshold


class TreatmentModel(Enum):
    LINEAR = "linear"           # D* = X @ β_dx + A @ β_da + ε
    NONLINEAR = "nonlinear"     # D* = sin(X @ β_dx) + A @ β_da + ε


class OutcomeType(Enum):
    LINEAR = "linear"                       # Y = τD + X @ β_yx + ε
    PARTIALLY_LINEAR = "partially_linear"   # Y = τD + X @ β_yx + A @ β_ya + ε
    NONPARAMETRIC = "nonparametric"         # Y = τ·sin(D) + X @ β_yx + A @ β_ya + ε


class ConfounderCorrelation(Enum):
    INDEPENDENT = "independent"     # A ~ N(0, I)
    CORRELATED = "correlated"       # A = ρ·X[:, :num_A] + √(1−ρ²)·noise  (Var(A) = 1)


class CovariateCorrelation(Enum):
    INDEPENDENT = "independent"         # X ~ N(0, I)
    EQUICORRELATED = "equicorrelated"   # Cov = (1-rho)·I + rho·11ᵀ
    TOEPLITZ = "toeplitz"               # Cov[i,j] = rho^|i-j|  (AR-1)


# --- Config ---

@dataclass(frozen=True)
class DGPConfig:
    n: int = 2000
    tau: float = 1.0

    # Observed covariates
    num_X: int = 5
    covariate_correlation: CovariateCorrelation = CovariateCorrelation.INDEPENDENT
    rho: float = 0.5   # used when covariate_correlation != INDEPENDENT

    # Hidden confounders
    num_A: int = 1
    confounder_correlation: ConfounderCorrelation = ConfounderCorrelation.INDEPENDENT
    # ρ used when confounder_correlation=CORRELATED; Var(A) is preserved regardless of value
    confounder_X_correlation: float = 0.7

    # Treatment
    treatment_type: TreatmentType = TreatmentType.BINARY
    treatment_model: TreatmentModel = TreatmentModel.LINEAR
    # Use float or tuple (not np.ndarray) to preserve hashability of frozen dataclass
    beta_dx: Union[float, tuple, np.ndarray] = 0.5
    beta_da: Union[float, tuple, np.ndarray] = 0.8

    # Outcome
    outcome_type: OutcomeType = OutcomeType.PARTIALLY_LINEAR
    beta_yx: Union[float, tuple, np.ndarray] = 0.5
    beta_ya: Union[float, tuple, np.ndarray] = 0.8

    seed: int = 42

    @classmethod
    def variance_locked(cls, *, num_A: int, total_signal: float = 0.5, **kwargs) -> "DGPConfig":
        """
        Convenience constructor: normalise beta_da/ya so num_A confounders together
        contribute total_signal² variance — identical to the single-confounder baseline.

        Avoids the footgun of setting num_A > 1 without scaling: with k confounders each
        at strength s, total hidden variance is k·s² rather than s².
        """
        beta_per = total_signal / np.sqrt(num_A)
        return cls(num_A=num_A, beta_da=beta_per, beta_ya=beta_per, **kwargs)


# --- Helpers ---

def _to_vec(val, size):
    if isinstance(val, (int, float)):
        return np.full(size, float(val))
    return np.asarray(val, dtype=float)


def _covariance_matrix(num_X, rho, kind):
    if kind == CovariateCorrelation.EQUICORRELATED:
        return (1.0 - rho) * np.eye(num_X) + rho * np.ones((num_X, num_X))
    idx = np.arange(num_X)
    return rho ** np.abs(idx[:, None] - idx[None, :])


# --- Data generation ---

def generate_data(config: DGPConfig):
    """
    Sample (Y, D, X, A) from the DGP specified by config.

    Returns
    -------
    Y : (n,)        outcome
    D : (n,)        treatment; float 0/1 if BINARY, continuous otherwise
    X : (n, num_X)  observed covariates
    A : (n, num_A)  hidden confounders
    """
    rng = np.random.default_rng(config.seed)

    # X
    if config.covariate_correlation == CovariateCorrelation.INDEPENDENT:
        X = rng.standard_normal((config.n, config.num_X))
    else:
        cov = _covariance_matrix(config.num_X, config.rho, config.covariate_correlation)
        X = rng.multivariate_normal(np.zeros(config.num_X), cov, size=config.n)

    # A
    if config.confounder_correlation == ConfounderCorrelation.INDEPENDENT:
        A = rng.standard_normal((config.n, config.num_A))
    else:
        if config.num_A > config.num_X:
            raise ValueError(
                f"CORRELATED confounder requires num_A ({config.num_A}) <= num_X ({config.num_X}). "
                "Use INDEPENDENT or reduce num_A."
            )
        rho_ax = config.confounder_X_correlation
        noise  = rng.standard_normal((config.n, config.num_A))
        # Variance-preserving mixture: Var(A_j) = ρ²·Var(X_j) + (1-ρ²)·Var(noise) = 1
        A = rho_ax * X[:, :config.num_A] + np.sqrt(1.0 - rho_ax ** 2) * noise

    beta_dx = _to_vec(config.beta_dx, config.num_X)
    beta_da = _to_vec(config.beta_da, config.num_A)
    beta_yx = _to_vec(config.beta_yx, config.num_X)
    beta_ya = _to_vec(config.beta_ya, config.num_A)

    # Treatment
    noise_d = rng.standard_normal(config.n)
    if config.treatment_model == TreatmentModel.LINEAR:
        D_latent = X @ beta_dx + A @ beta_da + noise_d
    else:
        D_latent = np.sin(X @ beta_dx) + A @ beta_da + noise_d

    D = (D_latent > 0).astype(float) if config.treatment_type == TreatmentType.BINARY else D_latent

    # Outcome
    noise_y = rng.standard_normal(config.n)
    if config.outcome_type == OutcomeType.LINEAR:
        Y = config.tau * D + X @ beta_yx + noise_y
    elif config.outcome_type == OutcomeType.PARTIALLY_LINEAR:
        Y = config.tau * D + X @ beta_yx + A @ beta_ya + noise_y
    else:
        Y = config.tau * np.sin(D) + X @ beta_yx + A @ beta_ya + noise_y

    return Y, D, X, A
