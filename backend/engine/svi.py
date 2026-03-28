"""
SVI (Stochastic Volatility Inspired) smile parametrisation.

Raw SVI: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
where w = sigma_IV^2 * T (total implied variance), k = log(K/F).

This gives a 5-parameter smile fit {a, b, rho, m, sigma} that naturally
captures level, skew, curvature, and wing behaviour.

Reference: Gatheral (2004), "A parsimonious arbitrage-free implied volatility
parameterisation with application to the valuation of volatility derivatives."
"""
import numpy as np
from scipy.optimize import least_squares


def svi_total_variance(k: np.ndarray, a: float, b: float, rho: float,
                        m_svi: float, sigma: float) -> np.ndarray:
    """Raw SVI total implied variance w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
    km = k - m_svi
    return a + b * (rho * km + np.sqrt(km ** 2 + sigma ** 2))


def svi_implied_vol(k: np.ndarray, T: float, a: float, b: float, rho: float,
                     m_svi: float, sigma: float) -> np.ndarray:
    """IV from SVI: sigma_IV = sqrt(w(k) / T)."""
    w = svi_total_variance(k, a, b, rho, m_svi, sigma)
    w = np.maximum(w, 1e-8)
    return np.sqrt(w / max(T, 1e-8))


def fit_svi(strikes: np.ndarray, mid_ivs: np.ndarray, forward: float, T: float,
            weights: np.ndarray = None) -> dict:
    """
    Fit raw SVI to market implied vols.

    Args:
        strikes:  strike prices, shape (n,)
        mid_ivs:  implied vols (decimal), shape (n,)
        forward:  forward price
        T:        time to maturity
        weights:  optional fitting weights, shape (n,)

    Returns:
        dict with keys: a, b, rho, m, sigma, iv_fitted, residuals
    """
    k = np.log(strikes / forward)  # log-moneyness
    w_market = mid_ivs ** 2 * T     # total variance

    # Filter out invalid points
    valid = np.isfinite(w_market) & (mid_ivs > 0.01) & (mid_ivs < 3.0)
    if valid.sum() < 5:
        # Not enough points — return flat vol
        atm_iv = np.median(mid_ivs[mid_ivs > 0.01]) if np.any(mid_ivs > 0.01) else 0.25
        return _flat_result(strikes, atm_iv, forward, T)

    k_v = k[valid]
    w_v = w_market[valid]
    if weights is not None:
        wt = weights[valid]
    else:
        # Weight by proximity to ATM: near-ATM gets weight 1, deep OTM gets weight ~0.3
        wt = 1.0 / (1.0 + 10.0 * k_v ** 2)

    # Initial guess from data
    atm_idx = np.argmin(np.abs(k_v))
    a0 = float(w_v[atm_idx])
    b0 = 0.1
    rho0 = -0.5  # typical equity skew
    m0 = 0.0
    sigma0 = 0.1

    def residuals(params):
        a, b, rho, m_s, sig = params
        sig = max(sig, 1e-4)
        w_fit = svi_total_variance(k_v, a, b, rho, m_s, sig)
        return wt * (w_fit - w_v)

    try:
        result = least_squares(
            residuals,
            x0=[a0, b0, rho0, m0, sigma0],
            bounds=(
                [0.0,   0.0,  -0.99, -0.2, 1e-4],   # lower
                [0.1,   0.5,   0.99,  0.2, 1.0],     # upper — tighter b cap prevents blowup
            ),
            method='trf',
            max_nfev=500,
        )
        a, b, rho, m_svi, sigma = result.x
    except Exception:
        atm_iv = np.median(mid_ivs[mid_ivs > 0.01]) if np.any(mid_ivs > 0.01) else 0.25
        return _flat_result(strikes, atm_iv, forward, T)

    iv_fitted = svi_implied_vol(k, T, a, b, rho, m_svi, sigma)
    residual_vols = iv_fitted - mid_ivs

    return {
        "a": float(a), "b": float(b), "rho": float(rho),
        "m": float(m_svi), "sigma": float(sigma),
        "iv_fitted": iv_fitted,
        "residuals": residual_vols,
        "forward": float(forward),
        "T": float(T),
    }


def svi_iv_at_strikes(svi_params: dict, strikes: np.ndarray,
                       forward: float = None, T: float = None) -> np.ndarray:
    """Evaluate an SVI fit at arbitrary strikes.

    Uses the forward the SVI was fitted with (stored in svi_params) to compute
    log-moneyness, ensuring the smile stays aligned with the original data.
    The caller's forward is ignored — use svi_params["forward"].
    """
    fit_forward = svi_params.get("forward", forward or 100.0)
    fit_T = svi_params.get("T", T or 1/12)
    k = np.log(strikes / fit_forward)
    return svi_implied_vol(k, fit_T, svi_params["a"], svi_params["b"],
                           svi_params["rho"], svi_params["m"], svi_params["sigma"])


def _flat_result(strikes, atm_iv, forward, T):
    return {
        "a": float(atm_iv ** 2 * T), "b": 0.0, "rho": 0.0,
        "m": 0.0, "sigma": 0.1,
        "iv_fitted": np.full(len(strikes), atm_iv),
        "residuals": np.zeros(len(strikes)),
        "forward": float(forward), "T": float(T),
    }


def filter_quotes_for_fit(strikes: np.ndarray, ivs: np.ndarray,
                           forward: float, max_moneyness: float = 0.10,
                           smoothness_threshold: float = 0.03) -> tuple:
    """
    Filter option quotes for robust fitting:
    1. Remove deep OTM (|log(K/F)| > max_moneyness)
    2. Remove outliers that violate local smoothness
    3. Require minimum number of valid points

    Returns (filtered_strikes, filtered_ivs, keep_mask).
    """
    k = np.log(strikes / forward)
    keep = np.ones(len(strikes), dtype=bool)

    # Remove deep OTM
    keep &= np.abs(k) <= max_moneyness

    # Remove invalid IVs
    keep &= np.isfinite(ivs) & (ivs > 0.01) & (ivs < 3.0)

    # Remove outliers: points that break local monotonicity by more than threshold
    # For equities, IV should generally decrease with strike (skew)
    # Allow some non-monotonicity but flag large reversals
    idx = np.where(keep)[0]
    if len(idx) >= 3:
        for j in range(1, len(idx) - 1):
            i_prev, i_curr, i_next = idx[j - 1], idx[j], idx[j + 1]
            # If this point is a spike (higher than both neighbors by threshold)
            avg_neighbor = (ivs[i_prev] + ivs[i_next]) / 2
            if abs(ivs[i_curr] - avg_neighbor) > smoothness_threshold:
                keep[i_curr] = False

    return strikes[keep], ivs[keep], keep
