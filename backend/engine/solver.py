"""
Solver for the normal equations (eq.29):

  [A^T Sigma^{-1} A  +  lambda H_graph  +  eta (I x Omega)] beta = A^T Sigma^{-1} y

For Phase 1 (single maturity), we also expose the harmonic shortcut (eq.33):
  beta_U = (I - W_UU)^{-1} W_UO beta_O

Reference: paper Section 7.5, Section 9.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, LinearOperator
from typing import List, Optional, Dict


def build_smoothness_matrix(M: int, omega_tail: float, omega_interior: float) -> np.ndarray:
    """
    Build the diagonal smoothness/regularisation matrix Omega (eq.28, Section 14.2).

    Omega = diag(w_1, w_2, w_3, ..., w_M)
    - phi_0 (constant): not penalised (absorbed by integration constant), w_0 ~ small
    - phi_1, phi_2 (tail): omega_tail (strong regularisation)
    - phi_{m>=3} (Legendre): omega_interior (weak, let data speak)
    """
    omega = np.zeros(M)
    if M >= 1:
        omega[0] = omega_interior * 0.1  # constant: light penalty
    if M >= 2:
        omega[1] = omega_tail  # left tail
    if M >= 3:
        omega[2] = omega_tail  # right tail
    for m in range(3, M):
        omega[m] = omega_interior
    return np.diag(omega)


def solve_normal_equations(
    A_blocks: Dict[int, np.ndarray],
    Sigma_inv_blocks: Dict[int, np.ndarray],
    y_blocks: Dict[int, np.ndarray],
    W: np.ndarray,
    N_nodes: int,
    M: int,
    lambda_: float,
    eta: float,
    Omega: np.ndarray,
    beta_max: float = 0.2,
) -> np.ndarray:
    """
    Solve the full normal equations for all nodes simultaneously.

    Uses column-normalisation of the Jacobians as preconditioning to handle
    the wide range of basis function sensitivities (tail vs interior).
    Clamps the output betas to stay within the linearisation regime.
    """
    dim = N_nodes * M

    # Column-normalise each Jacobian for numerical stability
    # Solve in transformed coords: beta_tilde = D * beta, where D_m = ||A[:,m]||
    # This makes all columns of A_normalized have unit norm
    D_blocks: Dict[int, np.ndarray] = {}
    A_norm_blocks: Dict[int, np.ndarray] = {}

    for v, A_v in A_blocks.items():
        col_norms = np.linalg.norm(A_v, axis=0)
        col_norms = np.maximum(col_norms, 1e-10)
        D_blocks[v] = col_norms
        A_norm_blocks[v] = A_v / col_norms[np.newaxis, :]

    # Use average column norms across observed nodes for the global scaling
    all_norms = np.array([D_blocks[v] for v in D_blocks])
    D_global = np.mean(all_norms, axis=0)  # shape (M,)
    D_global = np.maximum(D_global, 1e-10)

    # Build the graph Hessian: H_graph = (L^T L) x I_M  (row-wise penalty)
    L = np.eye(N_nodes) - W  # Laplacian
    LtL = L.T @ L

    H = np.zeros((dim, dim))
    rhs = np.zeros(dim)

    # Data fidelity in normalised coordinates
    for v, A_v in A_blocks.items():
        S_inv = Sigma_inv_blocks[v]
        y_v = y_blocks[v]

        # Normalise: A_tilde = A / D_global
        A_tilde = A_v / D_global[np.newaxis, :]

        i_start = v * M
        i_end = (v + 1) * M

        AtSA = A_tilde.T @ S_inv @ A_tilde
        H[i_start:i_end, i_start:i_end] += AtSA
        rhs[i_start:i_end] += A_tilde.T @ S_inv @ y_v

    # Graph consistency: lambda * (L^T L x I_M) — in normalised coords
    for i in range(N_nodes):
        for j in range(N_nodes):
            if abs(LtL[i, j]) > 1e-15:
                H[i * M:(i + 1) * M, j * M:(j + 1) * M] += lambda_ * LtL[i, j] * np.eye(M)

    # Smoothness: eta * (I_N x Omega) — in normalised coords
    for v in range(N_nodes):
        i_start = v * M
        i_end = (v + 1) * M
        H[i_start:i_end, i_start:i_end] += eta * Omega

    # Solve in normalised coordinates
    try:
        beta_tilde = np.linalg.solve(H, rhs)
    except np.linalg.LinAlgError:
        beta_tilde, _, _, _ = np.linalg.lstsq(H, rhs, rcond=None)

    # Transform back: beta = beta_tilde / D_global
    beta_hat = np.zeros(dim)
    for v in range(N_nodes):
        i_start = v * M
        i_end = (v + 1) * M
        beta_hat[i_start:i_end] = beta_tilde[i_start:i_end] / D_global

    # Clamp to stay within linearisation regime
    beta_hat = np.clip(beta_hat, -beta_max, beta_max)

    return beta_hat


def solve_harmonic_shortcut(
    W: np.ndarray,
    observed_mask: np.ndarray,
    beta_O: np.ndarray,
    M: int,
) -> np.ndarray:
    """
    Harmonic extension shortcut for unobserved nodes (eq.33):
      beta_U = (I - W_UU)^{-1} W_UO beta_O

    Applied per-basis-function (same W for all M coefficients).
    """
    from .graph import partition_observed_unobserved, propagation_matrix

    parts = partition_observed_unobserved(W, observed_mask)
    P = propagation_matrix(parts["W_UU"], parts["W_UO"])

    beta_U = P @ beta_O  # (N_unobs, M)

    N = W.shape[0]
    beta_all = np.zeros((N, M))
    beta_all[parts["obs_idx"]] = beta_O
    beta_all[parts["unobs_idx"]] = beta_U

    return beta_all
