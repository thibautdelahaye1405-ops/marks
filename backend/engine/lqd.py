"""
Log-Quantile-Density (LQD) basis functions and reconstruction.

References: paper Section 4 (Definition 4.1, eq.9-11) and Section 10.1 (eq.34-35).

Basis (eq.11):
  φ₀(u) = 1               (constant level, absorbed by integration constant)
  φ₁(u) = -log u           (left-tail singularity)
  φ₂(u) = -log(1-u)        (right-tail singularity)
  φ_m(u) = L_{m-2}(u)      for m >= 3  (shifted Legendre polynomials)

LQD: ψ(u) = log Q'(u),  u ∈ (0,1)
Reconstruction: Q(u) = a + ∫₀ᵘ exp(ψ(s)) ds
"""
import numpy as np
from numpy.polynomial.legendre import legval
from scipy.integrate import cumulative_trapezoid


def quantile_grid(n: int) -> np.ndarray:
    """
    Grid on (0,1) denser near boundaries to capture tail singularities.

    Uses a beta-CDF-like transformation: u = Beta(a,a).ppf(t) for uniform t,
    which clusters points near 0 and 1.
    """
    from scipy.stats import beta as beta_dist
    eps = 1e-6
    t = np.linspace(eps, 1.0 - eps, n)
    # Beta(0.5, 0.5) = arcsine distribution — heavy clustering at boundaries
    # Beta(0.7, 0.7) — moderate clustering, good balance
    return beta_dist.ppf(t, 0.7, 0.7)


def basis_functions(u: np.ndarray, M: int) -> np.ndarray:
    """
    Evaluate M basis functions on quantile grid u.

    Returns shape (M, len(u)).
    φ₀ = 1, φ₁ = -log u, φ₂ = -log(1-u), φ_{m>=3} = shifted Legendre L_{m-2}.
    """
    n = len(u)
    phi = np.zeros((M, n))

    if M >= 1:
        phi[0] = 1.0
    if M >= 2:
        phi[1] = -np.log(u)
    if M >= 3:
        phi[2] = -np.log(1.0 - u)

    # Shifted Legendre polynomials on [0,1]: L_k(2u - 1)
    for m in range(3, M):
        k = m - 2  # Legendre order (1, 2, 3, ...)
        coeffs = np.zeros(k + 1)
        coeffs[k] = 1.0
        phi[m] = legval(2.0 * u - 1.0, coeffs)

    return phi


def evaluate_lqd(psi0: np.ndarray, beta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Updated LQD: ψ_new(u) = ψ⁰(u) + Σ_m β_m φ_m(u).    (eq.34)

    Args:
        psi0: prior LQD evaluated on grid, shape (N,)
        beta: perturbation coefficients, shape (M,)
        phi:  basis functions on grid, shape (M, N)

    Returns:
        psi_new: shape (N,)
    """
    return psi0 + phi.T @ beta


def reconstruct_quantile(psi: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Reconstruct the shape quantile Q̃(u) from LQD ψ(u).     (eq.35)

    Q̃(u) = C + ∫₀ᵘ exp(ψ(s)) ds

    where C is chosen so that ∫₀¹ Q̃(u) du = 0 (zero-mean shape).

    Returns Q_tilde of shape (N,).
    """
    exp_psi = np.exp(psi)
    # Cumulative integral using trapezoidal rule
    Q_raw = np.zeros_like(u)
    Q_raw[1:] = cumulative_trapezoid(exp_psi, u)

    # Enforce zero mean: ∫₀¹ Q̃ du = 0
    mean_Q = np.trapz(Q_raw, u)
    Q_tilde = Q_raw - mean_Q

    return Q_tilde


def renormalise_shape(Q_tilde: np.ndarray, u: np.ndarray) -> tuple:
    """
    Enforce unit-variance normalisation (eq.36-37):
      s_new = s_old · ||Q̃||_L²
      Q̃ ← Q̃ / ||Q̃||_L²

    Returns (Q_tilde_normed, scale_factor) where scale_factor = ||Q̃||_L².
    """
    L2_norm_sq = np.trapz(Q_tilde ** 2, u)
    L2_norm = np.sqrt(L2_norm_sq)
    if L2_norm < 1e-12:
        return Q_tilde, 1.0
    return Q_tilde / L2_norm, L2_norm


def lqd_from_quantile_derivative(Q_prime: np.ndarray) -> np.ndarray:
    """ψ(u) = log Q'(u).  (eq.9)"""
    return np.log(np.maximum(Q_prime, 1e-30))
