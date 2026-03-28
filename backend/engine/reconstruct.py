"""
Reconstruction chain: LQD coefficients -> quantile -> call prices -> implied vol.

Reference: paper Section 10 (eq.34-38), Section 4.3 (eq.12-13), Section 5.3.

Uses the LINEARISED reconstruction (Section 5.3) for numerical stability:
  Q_new(u) ~ Q_prior(u) + s_prior * integral_0^u exp(psi0(s)) * delta_psi(s) ds

This avoids re-integrating the full exp(psi) which diverges at boundaries.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid


def quantile_to_call_prices(
    Q: np.ndarray,
    grid: np.ndarray,
    forward: float,
    T: float,
    r: float,
    strikes: np.ndarray,
) -> np.ndarray:
    """
    Compute call prices from quantile function via eq.12:
      C(K,T) = e^{-rT} integral_0^1 (F * e^{Q(u)} - K)^+ du
    """
    discount = np.exp(-r * T)
    S_values = forward * np.exp(Q)

    call_prices = np.zeros(len(strikes))
    for k, K in enumerate(strikes):
        payoff = np.maximum(S_values - K, 0.0)
        call_prices[k] = discount * np.trapz(payoff, grid)

    return call_prices


def _bs_call(forward: float, strike: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price."""
    if sigma < 1e-10 or T < 1e-10:
        return max(forward * np.exp(-r * T) - strike * np.exp(-r * T), 0.0)
    d1 = (np.log(forward / strike) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (forward * norm.cdf(d1) - strike * norm.cdf(d2))


def call_price_to_iv(
    call_prices: np.ndarray,
    forward: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    iv_bounds: tuple = (0.01, 3.0),
) -> np.ndarray:
    """
    Invert Black-Scholes to get implied volatilities from call prices (eq.13).
    """
    iv = np.full(len(strikes), np.nan)
    discount = np.exp(-r * T)
    intrinsic = np.maximum(discount * (forward - strikes), 0.0)

    for k in range(len(strikes)):
        C = call_prices[k]
        K = strikes[k]

        if C <= intrinsic[k] + 1e-10:
            iv[k] = iv_bounds[0]
            continue
        if C >= discount * forward - 1e-10:
            iv[k] = iv_bounds[1]
            continue

        try:
            iv[k] = brentq(
                lambda s: _bs_call(forward, K, T, r, s) - C,
                iv_bounds[0], iv_bounds[1],
                xtol=1e-8, maxiter=100,
            )
        except (ValueError, RuntimeError):
            iv[k] = np.nan

    return iv


def reconstruct_smile(
    psi0: np.ndarray,
    beta_v: np.ndarray,
    phi: np.ndarray,
    grid: np.ndarray,
    s0: float,
    forward: float,
    T: float,
    r: float,
    strikes: np.ndarray,
    prior_Q: np.ndarray = None,
    prior_Q_tilde: np.ndarray = None,
    prior_m: float = None,
) -> dict:
    """
    Full reconstruction for one node: beta_v -> marked IV smile.

    Uses the LINEARISED reconstruction for stability (Section 5.3):
      delta_Q_tilde(u) = integral_0^u exp(psi0(s)) * sum_m beta_m phi_m(s) ds
      Q_tilde_new = Q_tilde_prior + delta_Q_tilde  (then renormalise)

    Args:
        psi0:          prior LQD, shape (N,)
        beta_v:        perturbation coefficients, shape (M,)
        phi:           basis functions, shape (M, N)
        grid:          quantile grid, shape (N,)
        s0:            prior scale parameter
        forward:       forward price
        T:             maturity
        r:             risk-free rate
        strikes:       output strikes, shape (K,)
        prior_Q:       prior full quantile (optional, for bypass)
        prior_Q_tilde: prior unit-variance shape (optional)
        prior_m:       prior location (optional)

    Returns dict with marked smile data.
    """
    N = len(grid)

    # If no perturbation, return prior directly
    if np.all(np.abs(beta_v) < 1e-15) and prior_Q is not None:
        iv_marked = call_price_to_iv(
            quantile_to_call_prices(prior_Q, grid, forward, T, r, strikes),
            forward, strikes, T, r,
        )
        return {
            "iv_marked": iv_marked,
            "Q_new": prior_Q,
            "Q_tilde": prior_Q_tilde if prior_Q_tilde is not None else np.zeros(N),
            "s_new": s0,
            "m_new": prior_m if prior_m is not None else -0.5 * s0 ** 2,
            "psi_new": psi0,
        }

    # Linearised shape perturbation (Section 5.3):
    # delta_Q_tilde(u) ~ integral_0^u exp(psi0_tilde(s)) * delta_psi(s) ds
    # where delta_psi(s) = sum_m beta_m phi_m(s)
    #
    # psi0 = log Q'(u) = log(s0) + log Q_tilde'(u) = log(s0) + psi0_tilde
    # so exp(psi0) = s0 * exp(psi0_tilde).  We need exp(psi0_tilde) = exp(psi0)/s0
    # to stay in shape (Q_tilde) scale.
    delta_psi = phi.T @ beta_v  # shape (N,)
    exp_psi0 = np.exp(np.clip(psi0, -50, 50))  # clip for numerical safety
    # Divide by s0 to convert from full-Q derivative to shape (Q_tilde) derivative
    exp_psi0_tilde = exp_psi0 / max(s0, 1e-12)
    integrand = exp_psi0_tilde * delta_psi

    delta_Q_tilde = np.zeros(N)
    delta_Q_tilde[1:] = cumulative_trapezoid(integrand, grid)

    # Zero-mean the perturbation
    delta_Q_tilde -= np.trapz(delta_Q_tilde, grid)

    # Updated shape quantile
    if prior_Q_tilde is not None:
        Q_tilde_new = prior_Q_tilde + delta_Q_tilde
    else:
        # Fallback: use standard normal as shape
        Q_tilde_new = norm.ppf(grid) + delta_Q_tilde

    # Renormalise to unit variance (eq.36-37)
    Q_tilde_new -= np.trapz(Q_tilde_new, grid)  # zero mean
    L2_sq = np.trapz(Q_tilde_new ** 2, grid)
    L2_norm = np.sqrt(max(L2_sq, 1e-12))
    Q_tilde_normed = Q_tilde_new / L2_norm
    s_new = s0 * L2_norm

    # Martingale fix: m = -log integral_0^1 exp(s * Q_tilde(u)) du   (eq.38)
    exp_sQ = np.exp(np.clip(s_new * Q_tilde_normed, -50, 50))
    integral = np.trapz(exp_sQ, grid)
    m_new = -np.log(max(integral, 1e-30))

    # Full quantile
    Q_new = m_new + s_new * Q_tilde_normed

    # Updated LQD (for reference)
    psi_new = psi0 + delta_psi

    # Convert to IV via call prices
    iv_marked = call_price_to_iv(
        quantile_to_call_prices(Q_new, grid, forward, T, r, strikes),
        forward, strikes, T, r,
    )

    return {
        "iv_marked": iv_marked,
        "Q_new": Q_new,
        "Q_tilde": Q_tilde_normed,
        "s_new": s_new,
        "m_new": m_new,
        "psi_new": psi_new,
    }
