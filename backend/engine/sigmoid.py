"""
Sigmoid-Implied Variance (SIV) smile model.

Convexity V''(z) = K0 * sech²(κ_i(z - z0)/2), piecewise κ_P / κ_C.
Closed-form variance V(z) = V0 + S0(z-z0) + (4K0/κ_i²) ln cosh(κ_i(z-z0)/2).

6 trader params: [σ_ATM, S_ATM, K_ATM, W_P, W_C, σ_min]  (vol space for σ_ATM, σ_min)
6 structural params: V0, S0, K0, z0, κ_P, κ_C

Reference: Sigmoid Model.tex
"""
import numpy as np
from scipy.optimize import least_squares, brentq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_PARAMS = 6
PARAM_NAMES = ["sigma_atm", "s_atm", "k_atm", "w_p", "w_c", "sigma_min"]
TRADER_LABELS = [
    ("sigma_atm", "ATM Vol Level"),
    ("s_atm",     "ATM Skew"),
    ("k_atm",     "ATM Kurtosis"),
    ("w_p",       "Put Wing Slope"),
    ("w_c",       "Call Wing Slope"),
    ("sigma_min", "Min Vol Level"),
]

# ---------------------------------------------------------------------------
# Overflow-safe helpers
# ---------------------------------------------------------------------------

def _safe_logcosh(x):
    """ln(cosh(x)), safe for large |x|."""
    x = np.asarray(x, dtype=float)
    with np.errstate(over='ignore'):
        return np.where(np.abs(x) < 50.0, np.log(np.cosh(x)), np.abs(x) - np.log(2.0))


def _safe_sech2(x):
    """sech²(x) = 1/cosh²(x), safe for large |x|."""
    x = np.asarray(x, dtype=float)
    with np.errstate(over='ignore'):
        return np.where(np.abs(x) < 50.0, 1.0 / np.cosh(x) ** 2, 0.0)


# ---------------------------------------------------------------------------
# Structural functions
# ---------------------------------------------------------------------------

def _kappa_i(z, z0, kappa_P, kappa_C):
    """Piecewise transition speed: κ_P for z < z0, κ_C for z ≥ z0."""
    z = np.asarray(z, dtype=float)
    return np.where(z < z0, kappa_P, kappa_C)


def sigmoid_variance(z, V0, S0, K0, z0, kappa_P, kappa_C):
    """Implied variance V(z) = V0 + S0(z-z0) + (4K0/κ²) ln cosh(κ(z-z0)/2)."""
    z = np.asarray(z, dtype=float)
    ki = _kappa_i(z, z0, kappa_P, kappa_C)
    arg = ki * (z - z0) / 2.0
    return V0 + S0 * (z - z0) + (4.0 * K0 / ki ** 2) * _safe_logcosh(arg)


def sigmoid_slope(z, S0, K0, z0, kappa_P, kappa_C):
    """V'(z) = S0 + (2K0/κ) tanh(κ(z-z0)/2)."""
    z = np.asarray(z, dtype=float)
    ki = _kappa_i(z, z0, kappa_P, kappa_C)
    arg = ki * (z - z0) / 2.0
    return S0 + (2.0 * K0 / ki) * np.tanh(arg)


def sigmoid_convexity(z, K0, z0, kappa_P, kappa_C):
    """V''(z) = K0 * sech²(κ(z-z0)/2)."""
    z = np.asarray(z, dtype=float)
    ki = _kappa_i(z, z0, kappa_P, kappa_C)
    arg = ki * (z - z0) / 2.0
    return K0 * _safe_sech2(arg)


# ---------------------------------------------------------------------------
# Trader ↔ Structural mapping
# ---------------------------------------------------------------------------

def _kappa_from_wings(S0, K0, W_P, W_C):
    """Compute κ_P, κ_C from structural skew and wing slopes.

    κ_P = 2K0 / (S0 + W_P),  κ_C = 2K0 / (W_C - S0)
    Guards: denominators must be positive.
    """
    denom_P = S0 + W_P
    denom_C = W_C - S0
    denom_P = max(denom_P, 1e-6)
    denom_C = max(denom_C, 1e-6)
    kappa_P = 2.0 * K0 / denom_P
    kappa_C = 2.0 * K0 / denom_C
    return max(kappa_P, 1e-4), max(kappa_C, 1e-4)


def _find_vmin_z(S0, K0, z0, kappa_P, kappa_C):
    """Find z* where V'(z*) = 0 (minimum variance location).

    Returns z* or None if no minimum exists in [-20, 20].
    """
    def slope_scalar(z):
        ki = kappa_P if z < z0 else kappa_C
        arg = ki * (z - z0) / 2.0
        return S0 + (2.0 * K0 / ki) * np.tanh(arg)

    # V'(z) = S0 + (2K0/κ) tanh(κ(z-z0)/2)
    # As z→-∞: V'→ S0 - 2K0/κ_P = -W_P  (negative if W_P>0)
    # As z→+∞: V'→ S0 + 2K0/κ_C = W_C    (positive if W_C>0)
    # So if -W_P < 0 and W_C > 0, there must be a root.
    try:
        v_lo = slope_scalar(-20.0)
        v_hi = slope_scalar(20.0)
        if v_lo * v_hi >= 0:
            # No sign change — minimum might be at boundary
            return 0.0
        z_star = brentq(slope_scalar, -20.0, 20.0, xtol=1e-10)
        return z_star
    except (ValueError, RuntimeError):
        return 0.0


def trader_to_structural(sigma_atm, s_atm, k_atm, w_p, w_c, sigma_min):
    """Map 6 trader params → 6 structural params.

    Trader (volatility space): σ_ATM, S_ATM, K_ATM, W_P, W_C, σ_min
    Structural: V0, S0, K0, z0, κ_P, κ_C

    Two-phase approach:
      Phase 1: Estimate z0 from V_min constraint, solve V0/S0/K0 semi-analytically
      Phase 2: Full 4D refinement with good initial guess
    """
    V_ATM = sigma_atm ** 2
    V_min = max(sigma_min ** 2, 1e-8)
    K_ATM = max(k_atm, 1e-6)

    def _eval_constraints(V0, S0, K0, z0):
        """Return residuals for V(0)=V_ATM, V'(0)=S_ATM, V''(0)=K_ATM, V(z*)=V_min."""
        K0 = max(K0, 1e-8)
        kP, kC = _kappa_from_wings(S0, K0, w_p, w_c)
        ki_0 = kP if 0.0 < z0 else kC
        arg_0 = ki_0 * (-z0) / 2.0
        lc = float(_safe_logcosh(np.array([arg_0]))[0])

        v_at_0 = V0 + S0 * (-z0) + (4.0 * K0 / ki_0 ** 2) * lc
        s_at_0 = S0 + (2.0 * K0 / ki_0) * np.tanh(arg_0)
        k_at_0 = K0 * float(_safe_sech2(np.array([arg_0]))[0])

        z_star = _find_vmin_z(S0, K0, z0, kP, kC)
        v_min_actual = float(sigmoid_variance(np.array([z_star]), V0, S0, K0, z0, kP, kC)[0])

        return v_at_0 - V_ATM, s_at_0 - s_atm, k_at_0 - K_ATM, v_min_actual - V_min

    def residuals(x):
        V0, S0, K0, z0 = x
        r = _eval_constraints(V0, S0, K0, z0)
        scale_v = max(V_ATM, 1e-4)
        return [r[0] / scale_v, r[1] / max(abs(s_atm), 0.01),
                r[2] / max(K_ATM, 1e-4), r[3] / scale_v]

    # Phase 1: initial guess assuming z0 ≈ 0
    # At z0=0: V(0)=V0, V'(0)=S0, V''(0)=K0 → direct solution
    V0_init = V_ATM
    S0_init = s_atm
    K0_init = K_ATM
    # z0 guess: for negative skew, minimum is right of ATM → z0 slightly left
    z0_init = -s_atm / max(K_ATM, 0.001) * 0.1  # heuristic
    z0_init = np.clip(z0_init, -3.0, 3.0)

    x0 = np.array([V0_init, S0_init, K0_init, z0_init])

    try:
        result = least_squares(
            residuals, x0,
            bounds=([1e-8, -10.0, 1e-8, -10.0],
                    [10.0,  10.0, 10.0,  10.0]),
            method='trf',
            max_nfev=500,
            ftol=1e-12,
            xtol=1e-12,
        )
        V0, S0, K0, z0 = result.x
        K0 = max(K0, 1e-8)
    except Exception:
        V0, S0, K0, z0 = V_ATM, s_atm, K_ATM, 0.0

    kP, kC = _kappa_from_wings(S0, K0, w_p, w_c)
    return V0, S0, K0, z0, kP, kC


def structural_to_trader(V0, S0, K0, z0, kappa_P, kappa_C):
    """Map 6 structural params → 6 trader params (volatility space).

    Returns: (σ_ATM, S_ATM, K_ATM, W_P, W_C, σ_min)
    """
    # ATM values at z=0
    V_ATM = float(sigmoid_variance(np.array([0.0]), V0, S0, K0, z0, kappa_P, kappa_C)[0])
    S_ATM = float(sigmoid_slope(np.array([0.0]), S0, K0, z0, kappa_P, kappa_C)[0])
    K_ATM = float(sigmoid_convexity(np.array([0.0]), K0, z0, kappa_P, kappa_C)[0])

    # Wing slopes from asymptotes
    W_P = S0 + 2.0 * K0 / kappa_P  # put wing: V'(z→-∞) = S0 - 2K0/κ_P = -(S0 + 2K0/κ_P - 2S0)...
    # Actually: W_P is the absolute value of the put wing slope
    # V'(z→-∞) = S0 - 2K0/κ_P.   From spec: κ_P = 2K0/(S0 + W_P), so S0 - 2K0/κ_P = S0 - (S0+W_P) = -W_P
    # V'(z→+∞) = S0 + 2K0/κ_C.   From spec: κ_C = 2K0/(W_C - S0), so S0 + 2K0/κ_C = S0 + (W_C-S0) = W_C
    W_P = -(S0 - 2.0 * K0 / kappa_P)   # = 2K0/κ_P - S0
    W_C = S0 + 2.0 * K0 / kappa_C

    # Minimum variance
    z_star = _find_vmin_z(S0, K0, z0, kappa_P, kappa_C)
    V_min = float(sigmoid_variance(np.array([z_star]), V0, S0, K0, z0, kappa_P, kappa_C)[0])

    sigma_atm = np.sqrt(max(V_ATM, 1e-8))
    sigma_min = np.sqrt(max(V_min, 1e-8))

    # Feasibility clamps for trader display
    K_ATM = max(K_ATM, 0.0001)
    W_P = max(W_P, abs(S_ATM) + 0.001)
    W_C = max(W_C, abs(S_ATM) + 0.001)
    sigma_min = min(sigma_min, sigma_atm)

    return sigma_atm, S_ATM, K_ATM, W_P, W_C, sigma_min


# ---------------------------------------------------------------------------
# IV evaluation
# ---------------------------------------------------------------------------

def sigmoid_iv_at_strikes(params, strikes, forward, T, sigma_ref):
    """Evaluate Sigmoid model implied vols at given strikes.

    Args:
        params: array-like [σ_ATM, S_ATM, K_ATM, W_P, W_C, σ_min] (trader space, vol)
            OR internal structural tuple from _pack_structural.
        strikes: strike prices
        forward: forward price
        T: time to maturity
        sigma_ref: reference vol for z-normalization (typically ATM IV)

    Returns:
        IV array (annualized vol, not variance)
    """
    params = np.asarray(params, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    sigma_atm, s_atm, k_atm, w_p, w_c, sigma_min = params

    # Enforce feasibility
    sigma_atm = max(sigma_atm, 0.01)
    k_atm = max(k_atm, 1e-6)
    w_p = max(w_p, abs(s_atm) + 0.001)
    w_c = max(w_c, abs(s_atm) + 0.001)
    sigma_min = max(sigma_min, 0.005)
    sigma_min = min(sigma_min, sigma_atm)

    # Map to structural
    V0, S0, K0, z0, kP, kC = trader_to_structural(sigma_atm, s_atm, k_atm, w_p, w_c, sigma_min)
    return _eval_iv_structural(V0, S0, K0, z0, kP, kC, strikes, forward, T, sigma_ref, sigma_atm)


def _eval_iv_structural(V0, S0, K0, z0, kP, kC, strikes, forward, T, sigma_ref, fallback_iv=0.25):
    """Evaluate IV from structural params directly (fast path, no root-find)."""
    sqrtT = np.sqrt(max(T, 1e-8))
    sr = max(sigma_ref, 0.01)
    z = np.log(np.asarray(strikes, dtype=float) / forward) / (sr * sqrtT)
    V = sigmoid_variance(z, V0, S0, K0, z0, kP, kC)
    V = np.maximum(V, 1e-8)
    iv = np.sqrt(V)
    return np.where(np.isfinite(iv), iv, fallback_iv)


# ---------------------------------------------------------------------------
# Fitting (structural-space optimizer — no nested root-find)
# ---------------------------------------------------------------------------

def fit_sigmoid_model(strikes, mid_ivs, forward, T, *,
                      bid_ask_spread=None, use_bid_ask_fit=True,
                      prior_params=None, lambda_prior=0.0,
                      atm_iv=None):
    """
    Fit 6-parameter Sigmoid model to market implied vols.

    Internally optimizes in structural space (V0, S0, K0, z0) with κ_P/κ_C
    derived from wing slopes. The 6 trader params are recovered at the end
    via structural_to_trader.

    Returns:
        dict with params (6 trader in vol space), sigma_ref, iv_fitted, residuals, forward, T
    """
    strikes = np.asarray(strikes, dtype=float)
    mid_ivs = np.asarray(mid_ivs, dtype=float)

    # Filter invalid
    valid = np.isfinite(mid_ivs) & (mid_ivs > 0.01) & (mid_ivs < 3.0)
    if valid.sum() < 5:
        atm = np.median(mid_ivs[mid_ivs > 0.01]) if np.any(mid_ivs > 0.01) else 0.25
        return _flat_result(strikes, atm, forward, T)

    k = np.log(strikes / forward)
    iv_v = mid_ivs[valid]
    k_v = k[valid]
    strikes_v = strikes[valid]

    # σ_ref = ATM IV
    if atm_iv is None:
        atm_idx = np.argmin(np.abs(k_v))
        atm_iv = float(iv_v[atm_idx])
    sigma_ref = max(atm_iv, 0.01)
    sqrtT = np.sqrt(max(T, 1e-8))

    # Weights
    if use_bid_ask_fit and bid_ask_spread is not None:
        wt = np.ones(valid.sum())
    else:
        wt = 1.0 / (1.0 + 10.0 * k_v ** 2)

    # Bid-ask dead-zone bounds
    if use_bid_ask_fit and bid_ask_spread is not None:
        ba = bid_ask_spread[valid]
        iv_lo = np.maximum(iv_v - ba, 0.001)
        iv_hi = iv_v + ba
    else:
        iv_lo = iv_v
        iv_hi = iv_v

    # z coordinates for the fitting data
    z_v = k_v / (sigma_ref * sqrtT)

    # Prior virtual data points
    use_prior = prior_params is not None and lambda_prior > 0.0
    prior_z_grid = None
    iv_prior_grid = None
    if use_prior:
        from .svi import _PRIOR_K_CANDIDATES
        min_gap = np.median(np.diff(np.sort(k_v))) * 0.5 if len(k_v) > 1 else 0.02
        dists = np.abs(_PRIOR_K_CANDIDATES[:, None] - k_v[None, :]).min(axis=1)
        prior_k_mask = dists > min_gap
        if prior_k_mask.any():
            prior_k_grid = _PRIOR_K_CANDIDATES[prior_k_mask]
            prior_z_grid = prior_k_grid / (sigma_ref * sqrtT)
            prior_strikes = forward * np.exp(prior_k_grid)
            iv_prior_grid = sigmoid_iv_at_strikes(prior_params, prior_strikes, forward, T, sigma_ref)
            prior_atm_iv = sigmoid_iv_at_strikes(prior_params, np.array([forward]), forward, T, sigma_ref)[0]
            if np.isfinite(prior_atm_iv) and prior_atm_iv > 0.01:
                iv_prior_grid = iv_prior_grid * (atm_iv / prior_atm_iv)
            if np.any(~np.isfinite(iv_prior_grid)):
                use_prior = False
        else:
            use_prior = False

    # Initial guess in structural space: V0, S0, K0, z0, kP, kC
    V_ATM = atm_iv ** 2
    # Estimate skew from market (variance slope in z-space)
    sort_idx = np.argsort(z_v)
    z_sorted = z_v[sort_idx]
    iv_sorted = iv_v[sort_idx]
    atm_i = np.argmin(np.abs(z_sorted))
    if atm_i > 0 and atm_i < len(z_sorted) - 1:
        S0_init = (iv_sorted[atm_i + 1] ** 2 - iv_sorted[atm_i - 1] ** 2) / \
                  (z_sorted[atm_i + 1] - z_sorted[atm_i - 1])
    else:
        S0_init = 0.0

    K0_init = max(0.005, float(np.std(iv_v ** 2)) * 0.5)
    # Wing slopes from edge data
    if len(z_sorted) >= 4:
        # Put wing slope (left edge variance gradient)
        left_slope = abs((iv_sorted[0] ** 2 - iv_sorted[1] ** 2) / max(abs(z_sorted[0] - z_sorted[1]), 0.1))
        right_slope = abs((iv_sorted[-1] ** 2 - iv_sorted[-2] ** 2) / max(abs(z_sorted[-1] - z_sorted[-2]), 0.1))
    else:
        left_slope = abs(S0_init) + 0.01
        right_slope = abs(S0_init) + 0.01
    kP_init = max(2.0 * K0_init / max(abs(S0_init) + left_slope, 0.001), 0.1)
    kC_init = max(2.0 * K0_init / max(right_slope - abs(S0_init), 0.001), 0.1)
    kP_init = min(kP_init, 20.0)
    kC_init = min(kC_init, 20.0)

    if prior_params is not None:
        pp = np.asarray(prior_params, dtype=float)
        V0_i, S0_i, K0_i, z0_i, kP_i, kC_i = trader_to_structural(*pp)
    else:
        V0_i, S0_i, K0_i, z0_i = V_ATM, S0_init, K0_init, 0.0
        kP_i, kC_i = kP_init, kC_init

    x0 = np.array([V0_i, S0_i, K0_i, z0_i, kP_i, kC_i])

    def residuals(x):
        V0, S0, K0, z0, kP, kC = x
        K0 = max(K0, 1e-8)
        kP = max(kP, 0.01)
        kC = max(kC, 0.01)

        V = sigmoid_variance(z_v, V0, S0, K0, z0, kP, kC)
        V = np.maximum(V, 1e-8)
        iv_fit = np.sqrt(V)
        iv_fit = np.where(np.isfinite(iv_fit), iv_fit, 2.0)

        data_resid = np.where(
            iv_fit < iv_lo, wt * (iv_fit - iv_lo),
            np.where(iv_fit > iv_hi, wt * (iv_fit - iv_hi), 0.0)
        )

        if not use_prior:
            return data_resid

        V_p = sigmoid_variance(prior_z_grid, V0, S0, K0, z0, kP, kC)
        V_p = np.maximum(V_p, 1e-8)
        iv_p = np.sqrt(V_p)
        iv_p = np.where(np.isfinite(iv_p), iv_p, 2.0)
        prior_resid = np.sqrt(lambda_prior) * (iv_p - iv_prior_grid)
        return np.concatenate([data_resid, prior_resid])

    try:
        result = least_squares(
            residuals, x0,
            bounds=([1e-8, -5.0, 1e-8, -10.0, 0.01, 0.01],
                    [10.0,  5.0, 10.0,  10.0, 50.0, 50.0]),
            method='trf',
            max_nfev=500,
        )
        V0, S0, K0, z0, kP, kC = result.x
        K0 = max(K0, 1e-8)
        kP = max(kP, 0.01)
        kC = max(kC, 0.01)
    except Exception:
        return _flat_result(strikes, atm_iv, forward, T)

    # Convert to trader params for storage/display
    trader = structural_to_trader(V0, S0, K0, z0, kP, kC)
    params_opt = np.array(trader)

    # Evaluate at all strikes using structural params (no re-solve)
    iv_fitted = _eval_iv_structural(V0, S0, K0, z0, kP, kC, strikes, forward, T, sigma_ref, atm_iv)
    iv_fitted = np.where(np.isfinite(iv_fitted), iv_fitted, atm_iv)

    iv_fitted = sigmoid_iv_at_strikes(params_opt, strikes, forward, T, sigma_ref)
    iv_fitted = np.where(np.isfinite(iv_fitted), iv_fitted, atm_iv)

    return {
        "params": params_opt,
        "sigma_ref": float(sigma_ref),
        "iv_fitted": iv_fitted,
        "residuals": iv_fitted - mid_ivs,
        "forward": float(forward),
        "T": float(T),
        "atm_iv": float(atm_iv),
    }


def _flat_result(strikes, atm_iv, forward, T):
    """Fallback: flat vol smile."""
    atm_iv = max(atm_iv, 0.01)
    return {
        "params": np.array([atm_iv, 0.0, 0.01, 0.02, 0.02, atm_iv * 0.95]),
        "sigma_ref": float(atm_iv),
        "iv_fitted": np.full(len(strikes), atm_iv),
        "residuals": np.zeros(len(strikes)),
        "forward": float(forward),
        "T": float(T),
        "atm_iv": float(atm_iv),
    }


# ---------------------------------------------------------------------------
# CDF / LQD from Sigmoid smile (Breeden-Litzenberger)
# ---------------------------------------------------------------------------

def sigmoid_to_cdf_lqd(params, sigma_ref, forward, T, r, grid):
    """Derive risk-neutral CDF and LQD from a Sigmoid smile.

    Same approach as _svi_to_cdf_lqd in prior.py: analytical dV/dK → dσ/dK,
    then Breeden-Litzenberger for CDF, invert for quantile Q(u), derive LQD.
    """
    from scipy.stats import norm
    from scipy.interpolate import PchipInterpolator

    params = np.asarray(params, dtype=float)
    sigma_atm = max(params[0], 0.01)

    # Map to structural (needed for analytical derivatives)
    V0, S0, K0, z0, kP, kC = trader_to_structural(*params)

    sqrtT = np.sqrt(max(T, 1e-8))
    sr = max(sigma_ref, 0.01)

    # Strike grid
    n_sd = 8
    width = sigma_atm * sqrtT * n_sd
    width = max(width, 0.5)
    n_cdf = 3000
    log_k = np.linspace(-width, width, n_cdf)
    K = forward * np.exp(log_k)

    # z coordinates
    z = log_k / (sr * sqrtT)

    # Variance and IV
    V = sigmoid_variance(z, V0, S0, K0, z0, kP, kC)
    V = np.maximum(V, 1e-8)
    iv = np.sqrt(V)
    iv = np.clip(iv, 0.01, 5.0)

    # dV/dz = V'(z) (analytical)
    dV_dz = sigmoid_slope(z, S0, K0, z0, kP, kC)

    # dσ/dz = dV/dz / (2σ) where σ = √V
    dsigma_dz = dV_dz / (2.0 * iv)

    # dz/dk = 1/(σ_ref √T),  dk/dK = 1/K
    # dσ/dK = dσ/dz * dz/dk * dk/dK = dσ/dz / (σ_ref √T K)
    dsigma_dK = dsigma_dz / (sr * sqrtT * K)

    # BS d1, d2
    d1 = (np.log(forward / K) + 0.5 * iv ** 2 * T) / (iv * sqrtT + 1e-30)
    d2 = d1 - iv * sqrtT

    # Breeden-Litzenberger
    discount = np.exp(-r * T)
    vega = forward * discount * norm.pdf(d1) * sqrtT
    dC_dK = -discount * norm.cdf(d2) + vega * dsigma_dK
    cdf_raw = 1.0 + np.exp(r * T) * dC_dK
    cdf_raw = np.clip(cdf_raw, 1e-8, 1.0 - 1e-8)

    # Ensure monotonicity
    cdf_raw = np.maximum.accumulate(cdf_raw)
    unique_mask = np.concatenate(([True], np.diff(cdf_raw) > 1e-12))
    if unique_mask.sum() < 10:
        Q = norm.ppf(np.clip(grid, 1e-6, 1 - 1e-6))
        psi = np.log(np.maximum(np.gradient(Q, grid), 1e-30))
        cdf_std = Q / max(np.std(Q), 1e-10)
        return {"cdf_x": cdf_std.tolist(), "cdf_y": grid.tolist(), "Q": Q, "psi": psi}

    # Invert CDF → quantile
    cdf_vals = cdf_raw[unique_mask]
    logk_vals = log_k[unique_mask]
    inv_cdf = PchipInterpolator(cdf_vals, logk_vals, extrapolate=False)
    Q = inv_cdf(grid)
    Q = np.where(np.isfinite(Q), Q, np.interp(grid, cdf_vals, logk_vals))

    # Q'(u) with BS blend in tails
    inv_cdf_deriv = inv_cdf.derivative()
    Q_prime_pchip = inv_cdf_deriv(grid)
    z_bs = norm.ppf(np.clip(grid, 1e-8, 1 - 1e-8))
    Q_prime_bs = 1.0 / (norm.pdf(z_bs) + 1e-30)

    cdf_lo = float(cdf_vals[0])
    cdf_hi = float(cdf_vals[-1])
    w_interior = np.clip((grid - cdf_lo) / max(cdf_lo * 2, 0.01), 0, 1) * \
                 np.clip((cdf_hi - grid) / max((1 - cdf_hi) * 2, 0.01), 0, 1)
    valid_qp = np.isfinite(Q_prime_pchip) & (Q_prime_pchip > 0)
    Q_prime = np.where(valid_qp, Q_prime_pchip, Q_prime_bs)
    Q_prime = w_interior * Q_prime + (1 - w_interior) * Q_prime_bs
    Q_prime = np.maximum(Q_prime, 1e-30)

    # Normalization
    Q_median = float(np.interp(0.5, grid, Q))
    Q_var = float(np.trapz((Q - float(np.trapz(Q, grid))) ** 2, grid))
    sigma_sqrtT = np.sqrt(max(Q_var, 1e-20))
    cdf_x = (Q - Q_median) / sigma_sqrtT
    psi_tilde = np.log(Q_prime) - np.log(sigma_sqrtT)

    return {
        "cdf_x": cdf_x.tolist(),
        "cdf_y": grid.tolist(),
        "psi": psi_tilde,
        "Q": Q,
    }
