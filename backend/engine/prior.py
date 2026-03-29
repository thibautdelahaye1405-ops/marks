"""
Black-Scholes flat-vol prior for the LQD.

For a log-normal distribution with vol σ and maturity T:
  X = log(S_T / F) ~ N(-σ²T/2, σ²T)
  Q(u) = -σ²T/2 + σ√T · Φ⁻¹(u)

Quantile derivative: Q'(u) = σ√T / φ(Φ⁻¹(u))
LQD: ψ(u) = log Q'(u) = log(σ√T) - log φ(Φ⁻¹(u))
     = log(σ√T) + ½(Φ⁻¹(u))² + ½ log(2π)

Location-Scale-Shape decomposition (eq.4-7):
  m = E[X] = -σ²T/2  (forward-normalised mean of log-return)
  s = std(X) = σ√T
  Q̃(u) = Φ⁻¹(u)  (standard normal quantile = unit-variance shape)
"""
import numpy as np
from scipy.stats import norm

from .lqd import quantile_grid, basis_functions, lqd_from_quantile_derivative


def bs_prior(sigma_atm: float, T: float, grid: np.ndarray) -> dict:
    """
    Compute the Black-Scholes (log-normal) prior LQD on the given quantile grid.

    Args:
        sigma_atm: ATM implied volatility (annualised, e.g. 0.20 for 20%)
        T:         time to maturity in years
        grid:      quantile grid u ∈ (0,1), shape (N,)

    Returns dict with:
        psi0:     prior LQD ψ⁰(u), shape (N,)
        m:        location (mean of log-forward return)
        s:        scale (std dev of log-forward return)
        Q_tilde:  unit-variance shape quantile Q̃(u) = Φ⁻¹(u)
        Q:        full quantile Q(u) = m + s·Q̃(u)
    """
    s = sigma_atm * np.sqrt(T)
    m = -0.5 * sigma_atm ** 2 * T

    # Standard normal quantile = shape for log-normal
    Q_tilde = norm.ppf(grid)  # Φ⁻¹(u)

    # Full quantile
    Q = m + s * Q_tilde

    # Quantile derivative Q'(u) = s / φ(Φ⁻¹(u))
    Q_prime = s / norm.pdf(Q_tilde)

    # LQD
    psi0 = lqd_from_quantile_derivative(Q_prime)

    # Synthesize an SVI from this flat-vol prior so propagation encoding always
    # has a valid reference.  Flat vol → a = σ²T, b = 0, rho = 0, m = 0, σ_svi = 0.1
    svi_a = float(sigma_atm ** 2 * T)
    svi_params = {
        "a": svi_a, "b": 0.0, "rho": 0.0, "m": 0.0, "sigma": 0.1,
        "forward": 100.0, "T": float(T),
    }

    # Normalized SVI-JW parameters for propagation encoding.
    # Flat vol → degenerate JW: no skew, symmetric default wings.
    jw_params = {
        "v": float(sigma_atm ** 2),
        "vt_ratio": 1.0,
        "psi_hat": 0.0,
        "p_hat": 0.5,
        "c_hat": 0.5,
    }

    return {
        "psi0": psi0,
        "m": m,
        "s": s,
        "Q_tilde": Q_tilde,
        "Q": Q,
        "Q_prime": Q_prime,
        "_svi_params": svi_params,
        "_jw_params": jw_params,
    }


def fit_lqd_prior(
    strikes: np.ndarray,
    mid_ivs: np.ndarray,
    forward: float,
    T: float,
    r: float,
    grid: np.ndarray,
    phi: np.ndarray,
) -> dict:
    """
    Fit a prior smile using SVI parametrisation + BS base for the graph engine.

    Uses SVI for reliable smile fitting (handles skew, wings naturally),
    and keeps the BS base quantile for the LQD graph propagation machinery.
    """
    from .svi import fit_svi, filter_quotes_for_fit

    # Filter quotes for robust fitting
    filt_strikes, filt_ivs, keep = filter_quotes_for_fit(strikes, mid_ivs, forward)

    if len(filt_strikes) < 3:
        # Fallback: use all valid points
        valid = np.isfinite(mid_ivs) & (mid_ivs > 0.01)
        filt_strikes = strikes[valid]
        filt_ivs = mid_ivs[valid]

    # Fit SVI
    svi_params = fit_svi(filt_strikes, filt_ivs, forward, T)

    from .svi import raw_svi_to_jw_normalized
    jw_params = raw_svi_to_jw_normalized(
        svi_params["a"], svi_params["b"], svi_params["rho"],
        svi_params["m"], svi_params["sigma"], T)

    # ATM IV from SVI
    atm_iv = float(svi_params["iv_fitted"][np.argmin(np.abs(filt_strikes - forward))])
    if atm_iv < 0.01:
        atm_iv = float(np.median(mid_ivs[mid_ivs > 0.01])) if np.any(mid_ivs > 0.01) else 0.25

    # BS base for the graph engine
    base = bs_prior(atm_iv, T, grid)

    return {
        **base,
        "beta_fit": np.zeros(phi.shape[0]),  # no LQD beta — SVI handles the smile
        "_bs_base": base,
        "_svi_params": svi_params,           # SVI fit for IV display
        "_jw_params": jw_params,
        "_fit_strikes": strikes,
    }


def apply_beta_overrides(
    prior_dict: dict,
    beta_adj: np.ndarray,
    phi: np.ndarray,
    grid: np.ndarray,
    forward: float,
    T: float,
    r: float,
    strikes: np.ndarray,
) -> dict:
    """
    Apply manual SVI parameter adjustments to a prior.

    beta_adj maps to SVI parameter shifts:
      [0] = da (level shift), [1] = db (wing width), [2] = drho (skew),
      [3] = dm (horizontal shift), [4] = dsigma (curvature)
    Scaled so slider range [-0.05, 0.05] gives ~1-3 vol pt effect.
    """
    from .svi import svi_implied_vol

    svi_params = prior_dict.get("_svi_params")
    if svi_params is None:
        # No SVI — return flat
        atm_iv = prior_dict.get("s", 0.25) / np.sqrt(max(T, 1e-6))
        return {**prior_dict, "iv_marked": np.full(len(strikes), atm_iv)}

    # Scale adjustments: sliders are in [-0.05, 0.05], SVI params have different scales
    k = np.log(strikes / forward)
    a_new = svi_params["a"] + beta_adj[0] * 0.5        # ~level
    b_new = max(svi_params["b"] + beta_adj[1] * 2.0, 0.0)  # wing width
    rho_new = np.clip(svi_params["rho"] + beta_adj[2] * 10.0, -0.999, 0.999)  # skew
    m_new = svi_params["m"] + beta_adj[3] * 1.0         # horizontal shift
    sig_new = max(svi_params["sigma"] + beta_adj[4] * 2.0, 1e-4)  # curvature

    iv_marked = svi_implied_vol(k, T, a_new, b_new, rho_new, m_new, sig_new)
    iv_marked = np.clip(iv_marked, 0.01, 5.0)

    updated_svi = {**svi_params, "a": a_new, "b": b_new, "rho": rho_new,
                   "m": m_new, "sigma": sig_new}

    return {
        **prior_dict,
        "iv_marked": iv_marked,
        "_svi_params": updated_svi,
    }


def _svi_to_cdf_lqd(svi_params: dict, forward: float, T: float, r: float,
                     grid: np.ndarray) -> dict:
    """Derive risk-neutral CDF and LQD from an SVI smile.

    Uses the **analytical** Breeden-Litzenberger formula to avoid numerical
    differentiation noise:

        dC/dK = -e^{-rT} N(d2)  +  vega_K * dσ/dK

    where dσ/dK comes from the SVI closed-form derivative of total variance.
    Then CDF F(K) = 1 + e^{rT} dC/dK, inverted to get quantile Q(u) and
    LQD ψ(u) = log Q'(u).
    """
    from .svi import svi_total_variance
    from scipy.interpolate import PchipInterpolator

    a, b, rho = svi_params["a"], svi_params["b"], svi_params["rho"]
    m_svi, sig_svi = svi_params["m"], svi_params["sigma"]
    fit_fwd = svi_params.get("forward", forward)
    fit_T = svi_params.get("T", T)

    # Use SVI's own forward for self-consistent pricing
    F = fit_fwd

    # Strike grid must cover the full quantile grid [~0, ~1].
    # Iteratively widen until the CDF spans the grid's min/max.
    w_atm = svi_total_variance(np.array([0.0]), a, b, rho, m_svi, sig_svi)[0]
    atm_iv = np.sqrt(max(w_atm, 1e-8) / max(fit_T, 1e-8))
    sqrtT_fit = np.sqrt(max(fit_T, 1e-8))

    # Target: CDF must reach below grid[0] and above grid[-1]
    # Start wide and let the clip handle it
    n_sd = 8  # number of "standard deviations" to cover
    width_lo = atm_iv * sqrtT_fit * n_sd
    width_hi = atm_iv * sqrtT_fit * n_sd
    # For skewed smiles, expand the skew side more
    if rho < 0:
        width_lo *= (1 + abs(rho) * 3)  # left tail needs more room
    else:
        width_hi *= (1 + abs(rho) * 3)
    width_lo = max(width_lo, 0.5)
    width_hi = max(width_hi, 0.5)

    n_cdf = 3000
    log_k = np.linspace(-width_lo, width_hi, n_cdf)  # k = log(K/F)
    K = F * np.exp(log_k)

    # SVI total variance and IV (moneyness relative to SVI's forward = log_k)
    w = svi_total_variance(log_k, a, b, rho, m_svi, sig_svi)
    w = np.maximum(w, 1e-8)
    iv = np.sqrt(w / max(fit_T, 1e-8))
    iv = np.clip(iv, 0.01, 5.0)
    sqrtT = np.sqrt(max(fit_T, 1e-8))

    # BS d1, d2 using SVI's forward
    d1 = (np.log(F / K) + 0.5 * iv ** 2 * fit_T) / (iv * sqrtT + 1e-30)
    d2 = d1 - iv * sqrtT

    # Analytical dw/dk for SVI
    km = log_k - m_svi
    denom = np.sqrt(km ** 2 + sig_svi ** 2)
    dw_dk = b * (rho + km / (denom + 1e-30))

    # dσ/dK = (1/K) * dσ/dk,  where dσ/dk = dw/dk / (2 * σ * T)
    dsigma_dk = dw_dk / (2.0 * iv * max(fit_T, 1e-8))
    dsigma_dK = dsigma_dk / (K + 1e-30)

    # BS vega w.r.t. sigma: F * e^{-rT} * n(d1) * sqrt(T)
    discount = np.exp(-r * fit_T)
    vega = F * discount * norm.pdf(d1) * sqrtT

    # Analytical Breeden-Litzenberger: dC/dK = -e^{-rT} N(d2) + vega * dσ/dK
    dC_dK = -discount * norm.cdf(d2) + vega * dsigma_dK

    # CDF: F(K) = 1 + e^{rT} * dC/dK
    cdf_raw = 1.0 + np.exp(r * fit_T) * dC_dK
    cdf_raw = np.clip(cdf_raw, 1e-8, 1.0 - 1e-8)

    # Ensure strict monotonicity
    cdf_raw = np.maximum.accumulate(cdf_raw)
    # Remove flat regions for inversion
    unique_mask = np.concatenate(([True], np.diff(cdf_raw) > 1e-12))
    if unique_mask.sum() < 10:
        # Fallback to standard normal
        Q = norm.ppf(np.clip(grid, 1e-6, 1 - 1e-6))
        psi = np.log(np.maximum(np.gradient(Q, grid), 1e-30))
        cdf_std = Q / max(np.std(Q), 1e-10)
        return {"cdf_x": cdf_std.tolist(), "cdf_y": grid.tolist(), "Q": Q, "psi": psi}

    # Invert CDF → quantile Q(u) = log(K/F) at each grid point u
    # PchipInterpolator preserves monotonicity (no cubic overshoot)
    cdf_vals = cdf_raw[unique_mask]
    logk_vals = log_k[unique_mask]
    inv_cdf = PchipInterpolator(cdf_vals, logk_vals, extrapolate=False)
    Q = inv_cdf(grid)
    # Clamp any NaN from out-of-range grid points
    Q = np.where(np.isfinite(Q), Q, np.interp(grid, cdf_vals, logk_vals))

    # Q'(u): use the Pchip derivative in the well-covered interior,
    # blend with BS normal Q' in the tails where interpolation is unreliable.
    inv_cdf_deriv = inv_cdf.derivative()
    Q_prime_pchip = inv_cdf_deriv(grid)

    # BS reference: Q_bs'(u) = 1/φ(Φ⁻¹(u)) — normal quantile derivative
    z = norm.ppf(np.clip(grid, 1e-8, 1 - 1e-8))
    Q_prime_bs = 1.0 / (norm.pdf(z) + 1e-30)

    # Blend: use Pchip where valid, fade to BS at extremes
    cdf_lo = float(cdf_vals[0])
    cdf_hi = float(cdf_vals[-1])
    # Weight: 1 in the CDF-covered interior, 0 outside
    w_interior = np.clip((grid - cdf_lo) / max(cdf_lo * 2, 0.01), 0, 1) * \
                 np.clip((cdf_hi - grid) / max((1 - cdf_hi) * 2, 0.01), 0, 1)
    valid = np.isfinite(Q_prime_pchip) & (Q_prime_pchip > 0)
    Q_prime = np.where(valid, Q_prime_pchip, Q_prime_bs)
    # Smooth blend at boundaries
    Q_prime = w_interior * Q_prime + (1 - w_interior) * Q_prime_bs
    Q_prime = np.maximum(Q_prime, 1e-30)

    # Normalisation: centre on median Q(0.5), scale by σ√T = std[log(S_T/F)]
    Q_median = float(np.interp(0.5, grid, Q))
    Q_var = float(np.trapz((Q - float(np.trapz(Q, grid))) ** 2, grid))
    sigma_sqrtT = np.sqrt(max(Q_var, 1e-20))

    # CDF x-axis: x = (Q(u) - median) / σ√T  → CDF(0) = 0.5
    cdf_x = (Q - Q_median) / sigma_sqrtT

    # LQD of the standardised shape: ψ̃(u) = log(Q'(u) / σ√T)
    psi_tilde = np.log(Q_prime) - np.log(sigma_sqrtT)

    return {
        "cdf_x": cdf_x.tolist(),
        "cdf_y": grid.tolist(),
        "psi": psi_tilde,
        "Q": Q,
    }


def compute_distribution_view(
    prior_dict: dict,
    grid: np.ndarray,
    forward: float,
    T: float,
    r: float,
    n_moneyness: int = 50,
    phi: np.ndarray = None,
    market_strikes: np.ndarray = None,
) -> dict:
    """
    Compute the triple distribution view (IV, CDF, LQD) for a given prior/posterior.

    IV comes from SVI (or quantile fallback).
    CDF and LQD are derived from the SVI smile via Breeden-Litzenberger
    when SVI params are available, otherwise from the BS quantile.
    """
    from .reconstruct import quantile_to_call_prices, call_price_to_iv

    Q = prior_dict["Q"]
    s = prior_dict["s"]
    psi = prior_dict["psi0"]
    m = prior_dict["m"]

    # Derive moneyness range from actual strikes if available
    if market_strikes is not None and len(market_strikes) > 2:
        log_m = np.log(market_strikes / forward)
        pad = 0.005
        m_lo = log_m.min() - pad
        m_hi = log_m.max() + pad
    else:
        m_lo, m_hi = -0.06, 0.06

    moneyness = np.linspace(m_lo, m_hi, n_moneyness)
    strikes = forward * np.exp(moneyness)

    lqd_theta = prior_dict.get("_lqd_theta")
    lqd_alpha = prior_dict.get("_lqd_alpha")
    svi_params = prior_dict.get("_svi_params")

    if lqd_theta is not None and lqd_alpha is not None:
        # LQD model: derive IV, CDF, LQD natively from the quantile
        from .lqd import (evaluate_lqd, reconstruct_quantile,
                          basis_functions as lqd_basis, quantile_grid as lqd_grid,
                          lqd_implied_vols)
        theta = np.asarray(lqd_theta, dtype=float)
        alpha = float(lqd_alpha)
        u = lqd_grid(len(grid))
        phi_lqd = lqd_basis(u)
        iv_curve = lqd_implied_vols(theta, alpha, u, strikes, forward, T, r=r, phi=phi_lqd)
        iv_curve = np.where(np.isfinite(iv_curve), iv_curve, 0.25)
        iv_curve = np.clip(iv_curve, 0.01, 5.0)
        # CDF: Q(u) gives the quantile, so CDF_x = Q(u), CDF_y = u
        ell = evaluate_lqd(theta, u, phi_lqd)
        Q_lqd = reconstruct_quantile(ell, u, theta=theta, phi=phi_lqd)
        cdf_x = (alpha * Q_lqd).tolist()  # un-normalise to log-return scale
        cdf_y = u.tolist()
        lqd_psi = ell
    elif svi_params is not None:
        # IV from SVI
        from .svi import svi_iv_at_strikes
        iv_curve = svi_iv_at_strikes(svi_params, strikes, forward, T)
        iv_curve = np.clip(iv_curve, 0.01, 5.0)
        # CDF and LQD from SVI via Breeden-Litzenberger
        svi_dist = _svi_to_cdf_lqd(svi_params, forward, T, r, grid)
        cdf_x = svi_dist["cdf_x"]
        cdf_y = svi_dist["cdf_y"]
        lqd_psi = svi_dist["psi"]
    else:
        # Fallback: quantile-based
        iv_curve = call_price_to_iv(
            quantile_to_call_prices(Q, grid, forward, T, r, strikes),
            forward, strikes, T, r,
        )
        Q_median = float(np.interp(0.5, grid, Q))
        cdf_x = ((Q - Q_median) / max(s, 1e-10)).tolist()
        cdf_y = grid.tolist()
        lqd_psi = psi - np.log(max(s, 1e-10))

    return {
        "moneyness": moneyness.tolist(),
        "iv_curve": [float(v) if np.isfinite(v) else None for v in iv_curve],
        "cdf_x": cdf_x if isinstance(cdf_x, list) else cdf_x.tolist(),
        "cdf_y": cdf_y if isinstance(cdf_y, list) else cdf_y.tolist(),
        "lqd_u": grid.tolist(),
        "lqd_psi": [float(v) if np.isfinite(v) else None for v in lqd_psi],
        "fit_forward": float(svi_params["forward"]) if svi_params and "forward" in svi_params else float(forward),
    }


def bs_implied_vol_from_quantile(Q: np.ndarray, grid: np.ndarray,
                                  forward: float, T: float, r: float,
                                  strikes: np.ndarray) -> np.ndarray:
    """
    Convert a quantile function Q(u) to implied volatilities at given strikes.
    Chain: Q(u) → call prices via eq.12 → BS⁻¹ → σ_IV.

    Args:
        Q:       quantile function values on grid, shape (N,)
        grid:    quantile grid u, shape (N,)
        forward: forward price F
        T:       time to maturity
        r:       risk-free rate
        strikes: strike prices, shape (K,)

    Returns:
        iv: implied volatilities, shape (K,)
    """
    from .reconstruct import quantile_to_call_prices, call_price_to_iv

    call_prices = quantile_to_call_prices(Q, grid, forward, T, r, strikes)
    iv = call_price_to_iv(call_prices, forward, strikes, T, r)
    return iv
