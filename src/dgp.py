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
    CORRELATED = "correlated"       # A = 0.7·X[:, :num_A] + 0.3·noise


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

    # Treatment
    treatment_type: TreatmentType = TreatmentType.BINARY
    treatment_model: TreatmentModel = TreatmentModel.LINEAR
    beta_dx: Union[float, tuple] = 0.5   # scalar → uniform; tuple → per-column
    beta_da: Union[float, tuple] = 0.8

    # Outcome
    outcome_type: OutcomeType = OutcomeType.PARTIALLY_LINEAR
    beta_yx: Union[float, tuple] = 0.5
    beta_ya: Union[float, tuple] = 0.8

    seed: int = 42


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
        reps = -(-config.num_A // config.num_X)  # ceiling division
        X_tiled = np.tile(X, (1, reps))
        A = 0.7 * X_tiled[:, :config.num_A] + 0.3 * rng.standard_normal((config.n, config.num_A))

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
