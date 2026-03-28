"""
Jacobian: LQD coefficients beta -> implied volatilities.

Computed via finite differences through the full reconstruction chain
(reconstruct_smile), which includes renormalisation and martingale correction.

Reference: paper Section 7.3, eq.25-27.
"""
import numpy as np
from scipy.stats import norm


def _bs_vega(forward: float, strike: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes vega: dC/dsigma."""
    if sigma < 1e-10 or T < 1e-10:
        return 1e-10
    d1 = (np.log(forward / strike) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    return forward * np.exp(-r * T) * norm.pdf(d1) * np.sqrt(T)


def compute_jacobian(
    psi0: np.ndarray,
    Q_tilde: np.ndarray,
    m_v: float,
    s_v: float,
    forward: float,
    T: float,
    r: float,
    strikes: np.ndarray,
    sigma_prior: float,
    phi: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """
    Compute the Jacobian A_v (d sigma_IV / d beta) for one node via
    central finite differences through the full reconstruction chain.

    Args:
        psi0:       prior LQD on grid, shape (N,)
        Q_tilde:    prior unit-variance shape on grid, shape (N,)
        m_v:        location parameter
        s_v:        scale parameter
        forward:    forward price
        T:          time to maturity
        r:          risk-free rate
        strikes:    observed strike prices, shape (n_v,)
        sigma_prior: prior ATM vol (for vega computation, kept for API compat)
        phi:        basis functions on grid, shape (M, N)
        grid:       quantile grid, shape (N,)

    Returns:
        A_v: Jacobian matrix, shape (n_v, M)
    """
    from .reconstruct import reconstruct_smile

    M = phi.shape[0]
    n_v = len(strikes)

    # Prior Q for the zero-beta reference
    prior_Q = m_v + s_v * Q_tilde

    eps = 1e-5
    A_v = np.zeros((n_v, M))

    for m_idx in range(M):
        beta_plus = np.zeros(M)
        beta_plus[m_idx] = eps
        beta_minus = np.zeros(M)
        beta_minus[m_idx] = -eps

        res_plus = reconstruct_smile(
            psi0=psi0, beta_v=beta_plus, phi=phi, grid=grid,
            s0=s_v, forward=forward, T=T, r=r, strikes=strikes,
            prior_Q=prior_Q, prior_Q_tilde=Q_tilde, prior_m=m_v,
        )
        res_minus = reconstruct_smile(
            psi0=psi0, beta_v=beta_minus, phi=phi, grid=grid,
            s0=s_v, forward=forward, T=T, r=r, strikes=strikes,
            prior_Q=prior_Q, prior_Q_tilde=Q_tilde, prior_m=m_v,
        )

        iv_plus = res_plus["iv_marked"]
        iv_minus = res_minus["iv_marked"]

        # Central difference
        div = (iv_plus - iv_minus) / (2.0 * eps)
        # Replace NaN with 0 (deep OTM strikes may fail IV inversion)
        div = np.where(np.isfinite(div), div, 0.0)
        A_v[:, m_idx] = div

    return A_v
