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


def raw_svi_to_jw_normalized(a: float, b: float, rho: float,
                              m_svi: float, sigma: float, T: float) -> dict:
    """Convert raw SVI {a,b,rho,m,sigma} to normalized SVI-JW parameters.

    Returns:
        v:        ATM implied variance (w(0)/T)
        vt_ratio: min-variance / ATM-variance ratio (in [0,1])
        psi_hat:  normalized ATM skew
        p_hat:    normalized put wing slope (>0)
        c_hat:    normalized call wing slope (>0)
    """
    # ATM total variance: w(0) = a + b*(-rho*m + sqrt(m^2 + sigma^2))
    w_atm = a + b * (-rho * m_svi + np.sqrt(m_svi**2 + sigma**2))
    v = w_atm / max(T, 1e-8)

    # Minimum total variance: w_min = a + b*sigma*sqrt(1 - rho^2)
    w_min = a + b * sigma * np.sqrt(1.0 - rho**2)
    v_tilde = w_min / max(T, 1e-8)
    vt_ratio = np.clip(v_tilde / max(v, 1e-12), 0.0, 1.0)

    # sqrt(v*T) factor for normalization
    sqrt_vT = np.sqrt(max(v * T, 1e-16))

    # Normalized wing slopes
    p_hat = b * (1.0 - rho) / max(sqrt_vT, 1e-12)
    c_hat = b * (1.0 + rho) / max(sqrt_vT, 1e-12)

    # beta = m / sqrt(m^2 + sigma^2)
    denom = np.sqrt(m_svi**2 + sigma**2)
    beta = m_svi / max(denom, 1e-12) if denom > 1e-12 else 0.0

    # Normalized ATM skew: psi_hat = (rho - beta) * b / sqrt(v*T)
    psi_hat = (rho - beta) * b / max(sqrt_vT, 1e-12)

    # Ensure wings are positive (they should be by construction)
    p_hat = max(p_hat, 1e-6)
    c_hat = max(c_hat, 1e-6)

    return {
        "v": float(v),
        "vt_ratio": float(vt_ratio),
        "psi_hat": float(psi_hat),
        "p_hat": float(p_hat),
        "c_hat": float(c_hat),
    }


def jw_normalized_to_raw_svi(v: float, vt_ratio: float, psi_hat: float,
                              p_hat: float, c_hat: float, T: float) -> dict:
    """Convert normalized SVI-JW parameters back to raw SVI {a,b,rho,m,sigma}.

    Based on the algebraic inversion from SVI-JW-normalized.tex.
    """
    eps = 1e-8
    v = max(v, eps)
    T = max(T, eps)
    v_tilde = vt_ratio * v

    # Time/vol independent variables
    wing_sum = p_hat + c_hat
    wing_sum = max(wing_sum, eps)

    rho = (c_hat - p_hat) / wing_sum
    beta = (c_hat - p_hat - 2.0 * psi_hat) / wing_sum
    beta = np.clip(beta, -0.99999, 0.99999)

    # Scale-dependent
    sqrt_vT = np.sqrt(v * T)
    b = 0.5 * wing_sum * sqrt_vT

    # sigma (with singularity guard when v ≈ v_tilde)
    if (v - v_tilde) < eps:
        sigma = 1e-4
    else:
        denom_term1 = (1.0 - rho * beta) / np.sqrt(1.0 - beta**2)
        denom_term2 = np.sqrt(1.0 - rho**2)
        denominator = wing_sum * (denom_term1 - denom_term2)
        if abs(denominator) < eps:
            sigma = 1e-4
        else:
            sigma = np.sqrt(T / v) * (2.0 * (v - v_tilde) / denominator)

    sigma = max(sigma, 1e-4)

    # Translation
    m = beta * sigma / np.sqrt(1.0 - beta**2)
    a = v_tilde * T - b * sigma * np.sqrt(1.0 - rho**2)

    # Clamp to valid raw SVI ranges
    a = float(np.clip(a, -0.05, 0.15))
    b = float(np.clip(b, 0.0, 0.50))
    rho = float(np.clip(rho, -0.999, 0.999))
    m = float(np.clip(m, -0.20, 0.20))
    sigma = float(np.clip(sigma, 1e-4, 1.0))

    return {"a": a, "b": b, "rho": rho, "m": m, "sigma": sigma}


# Dense grid of candidate log-moneyness points for virtual prior data points.
# Covers a wide range including wing extrapolation.  At fit time, only grid
# points far from any observed quote are kept — the prior fills gaps, not
# competes with data.
_PRIOR_K_CANDIDATES = np.linspace(-0.30, 0.30, 15)


def fit_svi(strikes: np.ndarray, mid_ivs: np.ndarray, forward: float, T: float,
            weights: np.ndarray = None, *,
            bid_ask_spread: np.ndarray = None,
            use_bid_ask_fit: bool = True,
            prior_params: np.ndarray = None,
            lambda_prior: float = 0.0) -> dict:
    """
    Fit raw SVI to market implied vols.

    Args:
        strikes:  strike prices, shape (n,)
        mid_ivs:  implied vols (decimal), shape (n,)
        forward:  forward price
        T:        time to maturity
        weights:  optional fitting weights, shape (n,)
        bid_ask_spread: half bid-ask spread in vol points, shape (n,).
                        When provided and use_bid_ask_fit=True, the fit uses a
                        dead-zone loss that is zero inside the spread.
                        If None, treated as zero (bid=ask=mid).
        use_bid_ask_fit: whether to use bid-ask dead-zone (default True).
        prior_params: prior SVI parameters [a, b, rho, m, sigma] to anchor to.
        lambda_prior: strength of prior-anchoring penalty (0 = pure market fit).
                      The penalty adds virtual data points from the prior smile
                      in total variance space, so lambda=1 means each virtual
                      prior point carries the same weight as one market quote.

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

    # Weights: flat when using bid-ask dead-zone (spread already encodes noise),
    # ATM-focused otherwise (de-emphasise noisy deep OTM quotes).
    if weights is not None:
        wt = weights[valid]
    elif use_bid_ask_fit and bid_ask_spread is not None:
        wt = np.ones_like(k_v)
    else:
        wt = 1.0 / (1.0 + 10.0 * k_v ** 2)

    # Bid-ask dead-zone bounds in total variance space
    if use_bid_ask_fit and bid_ask_spread is not None:
        ba = bid_ask_spread[valid]
        bid_ivs = np.maximum(mid_ivs[valid] - ba, 0.001)
        ask_ivs = mid_ivs[valid] + ba
        w_bid = bid_ivs ** 2 * T
        w_ask = ask_ivs ** 2 * T
    else:
        # No spread info → bid=ask=mid (degrades to current behaviour)
        w_bid = w_v
        w_ask = w_v

    # Prior anchoring: evaluate the prior SVI on virtual data points in total
    # variance space.  Only place virtual points where there is no nearby market
    # observation — the prior fills gaps (wings, sparse regions), not competes
    # with actual quotes.
    #
    # The prior is rescaled to match the market ATM level, so only SHAPE
    # (skew, wings, curvature) is anchored — level is always driven by data.
    use_prior = prior_params is not None and lambda_prior > 0.0
    if use_prior:
        a_p, b_p, rho_p, m_p, sig_p = prior_params
        # Keep only candidate grid points far from any observed k
        min_gap = np.median(np.diff(np.sort(k_v))) * 0.5 if len(k_v) > 1 else 0.02
        dists = np.abs(_PRIOR_K_CANDIDATES[:, None] - k_v[None, :]).min(axis=1)
        prior_k_grid = _PRIOR_K_CANDIDATES[dists > min_gap]

        if len(prior_k_grid) == 0:
            # All grid points covered by data — no virtual points needed
            use_prior = False
        else:
            w_prior_raw = svi_total_variance(prior_k_grid, a_p, b_p, rho_p, m_p, sig_p)
            w_prior_atm = svi_total_variance(np.array([0.0]), a_p, b_p, rho_p, m_p, sig_p)[0]
            atm_idx_data = np.argmin(np.abs(k_v))
            w_market_atm = w_v[atm_idx_data]
            level_ratio = w_market_atm / max(w_prior_atm, 1e-8)
            w_prior_grid = w_prior_raw * level_ratio

    # Initial guess: start from prior shape with market level, else from data
    atm_idx = np.argmin(np.abs(k_v))
    if use_prior:
        # Use prior shape params but shift level to match market ATM
        w_prior_atm = svi_total_variance(np.array([0.0]), a_p, b_p, rho_p, m_p, sig_p)[0]
        a_shifted = a_p + (w_v[atm_idx] - w_prior_atm)
        a0 = float(np.clip(a_shifted, 1e-6, 0.099))
        b0 = float(np.clip(b_p, 1e-6, 0.499))
        rho0 = float(np.clip(rho_p, -0.989, 0.989))
        m0 = float(np.clip(m_p, -0.199, 0.199))
        sigma0 = float(np.clip(sig_p, 0.01, 0.999))
    else:
        a0 = float(w_v[atm_idx])
        b0 = 0.1
        rho0 = -0.5  # typical equity skew
        m0 = 0.0
        sigma0 = 0.1

    def residuals(params):
        a, b, rho, m_s, sig = params
        sig = max(sig, 1e-4)
        w_fit = svi_total_variance(k_v, a, b, rho, m_s, sig)

        # Dead-zone residuals: zero inside [w_bid, w_ask]
        data_resid = np.where(
            w_fit < w_bid, wt * (w_fit - w_bid),
            np.where(w_fit > w_ask, wt * (w_fit - w_ask), 0.0)
        )

        if not use_prior:
            return data_resid

        # Prior penalty: virtual data points from prior SVI, same units as data.
        # lambda=1 ⇒ each virtual point has the same weight as one real quote.
        w_fit_grid = svi_total_variance(prior_k_grid, a, b, rho, m_s, sig)
        prior_resid = np.sqrt(lambda_prior) * (w_fit_grid - w_prior_grid)

        # Small parameter-space nudge to break SVI identifiability degeneracy
        # (prevents rho-flip / sigma-collapse to piecewise-linear regime).
        # Scale: ~1% of a typical data residual, so it only matters as tiebreaker.
        param_nudge = 1e-4 * np.sqrt(lambda_prior) * (np.array(params) - prior_params)
        return np.concatenate([data_resid, prior_resid, param_nudge])

    try:
        result = least_squares(
            residuals,
            x0=[a0, b0, rho0, m0, sigma0],
            bounds=(
                [0.0,   0.0,  -0.99, -0.2, 0.01],   # lower — sigma≥0.01 prevents degenerate V-shapes
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


def fit_svi_jw(strikes: np.ndarray, mid_ivs: np.ndarray, forward: float, T: float,
               weights: np.ndarray = None) -> dict:
    """Fit SVI to market data, return both raw SVI and normalized JW parameters."""
    raw = fit_svi(strikes, mid_ivs, forward, T, weights)
    jw = raw_svi_to_jw_normalized(raw["a"], raw["b"], raw["rho"], raw["m"], raw["sigma"], T)
    return {**raw, "_jw": jw}


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
