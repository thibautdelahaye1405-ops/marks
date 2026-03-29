"""
LQD smile model: fitting, evaluation, and local rotation.

Provides:
  - fit_lqd_model()   — fit 6-param LQD to market IVs
  - lqd_iv_at_strikes()  — evaluate fitted LQD at arbitrary strikes
  - local_rotation()   — build trader-coordinate and orthogonal-coordinate systems
  - Trader ↔ raw θ conversions

Reference: lqd_local_qr_note.tex, Sections 6–10.
"""
import numpy as np
from scipy.optimize import least_squares

from .lqd import (
    N_PARAMS, PARAM_NAMES, quantile_grid, basis_functions,
    evaluate_lqd, reconstruct_quantile, lqd_implied_vols,
)


# ---------------------------------------------------------------------------
# Default numerical parameters (from spec Section 11)
# ---------------------------------------------------------------------------
_Z_GRID_HALF = 3.0       # normalised strike grid: z ∈ [-3, 3]
_N_Z = 50                # grid points for Jacobian / rotation
_FD_STEP = 0.25          # finite-difference step for ATM derivatives
_WING_Z1 = 1.5           # inner wing boundary
_WING_Z2 = 2.5           # outer wing boundary
_SHOULDER_Z = 1.0         # shoulder measurement point
_U_GRID_SIZE = 300        # quantile grid resolution for fitting


# ---------------------------------------------------------------------------
# Forward evaluation helpers
# ---------------------------------------------------------------------------

def _make_context(forward: float, T: float, atm_iv: float = None,
                  u: np.ndarray = None, phi: np.ndarray = None):
    """Build reusable evaluation context."""
    if u is None:
        u = quantile_grid(_U_GRID_SIZE)
    if phi is None:
        phi = basis_functions(u)
    if atm_iv is None:
        atm_iv = 0.25  # fallback
    alpha = atm_iv * np.sqrt(max(T, 1e-8))
    return u, phi, alpha


def lqd_iv_at_strikes(theta: np.ndarray, strikes: np.ndarray,
                       forward: float, T: float, alpha: float,
                       u: np.ndarray = None, phi: np.ndarray = None) -> np.ndarray:
    """Evaluate LQD model at arbitrary strikes.  Returns IV array."""
    if u is None:
        u = quantile_grid(_U_GRID_SIZE)
    if phi is None:
        phi = basis_functions(u)
    return lqd_implied_vols(theta, alpha, u, strikes, forward, T, r=0.0, phi=phi)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_lqd_model(strikes: np.ndarray, mid_ivs: np.ndarray,
                  forward: float, T: float, *,
                  bid_ask_spread: np.ndarray = None,
                  use_bid_ask_fit: bool = True,
                  prior_theta: np.ndarray = None,
                  lambda_prior: float = 0.0,
                  atm_iv: float = None) -> dict:
    """
    Fit 6-parameter LQD model to market implied vols.

    Args:
        strikes:  strike prices, shape (n,)
        mid_ivs:  implied vols (decimal), shape (n,)
        forward:  forward price
        T:        time to maturity
        bid_ask_spread: half bid-ask spread in vol points, shape (n,)
        use_bid_ask_fit: use bid-ask dead-zone loss
        prior_theta: prior LQD parameters for anchoring, shape (6,)
        lambda_prior: prior anchoring strength
        atm_iv: ATM implied vol for scale α.  Estimated from data if None.

    Returns:
        dict with theta, alpha, iv_fitted, residuals, forward, T
    """
    # Filter invalid points
    valid = np.isfinite(mid_ivs) & (mid_ivs > 0.01) & (mid_ivs < 3.0)
    if valid.sum() < 5:
        atm = np.median(mid_ivs[mid_ivs > 0.01]) if np.any(mid_ivs > 0.01) else 0.25
        return _flat_result(strikes, atm, forward, T)

    k = np.log(strikes / forward)
    iv_v = mid_ivs[valid]
    k_v = k[valid]
    strikes_v = strikes[valid]

    # Scale parameter α
    if atm_iv is None:
        atm_idx = np.argmin(np.abs(k_v))
        atm_iv = float(iv_v[atm_idx])
    alpha = atm_iv * np.sqrt(max(T, 1e-8))

    # Evaluation context
    u = quantile_grid(_U_GRID_SIZE)
    phi = basis_functions(u)

    # Weights: flat for bid-ask mode, ATM-focused otherwise
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

    # Prior virtual data points (gap-aware, same approach as SVI)
    use_prior = prior_theta is not None and lambda_prior > 0.0
    prior_k_grid = None
    iv_prior_grid = None
    if use_prior:
        from .svi import _PRIOR_K_CANDIDATES
        min_gap = np.median(np.diff(np.sort(k_v))) * 0.5 if len(k_v) > 1 else 0.02
        dists = np.abs(_PRIOR_K_CANDIDATES[:, None] - k_v[None, :]).min(axis=1)
        prior_k_mask = dists > min_gap
        if prior_k_mask.any():
            prior_k_grid = _PRIOR_K_CANDIDATES[prior_k_mask]
            prior_strikes = forward * np.exp(prior_k_grid)
            iv_prior_grid = lqd_implied_vols(prior_theta, alpha, u, prior_strikes,
                                              forward, T, phi=phi)
            # Level-normalise: scale prior IVs to match market ATM
            prior_atm = lqd_implied_vols(prior_theta, alpha, u,
                                          np.array([forward]), forward, T, phi=phi)[0]
            if np.isfinite(prior_atm) and prior_atm > 0.01:
                iv_prior_grid = iv_prior_grid * (atm_iv / prior_atm)
            if np.any(~np.isfinite(iv_prior_grid)):
                use_prior = False
        else:
            use_prior = False

    # Initial guess: start from prior if available, else zero
    if prior_theta is not None:
        x0 = np.clip(prior_theta, -3.0, 3.0)
    else:
        x0 = np.zeros(N_PARAMS)

    def residuals(theta):
        iv_fit = lqd_implied_vols(theta, alpha, u, strikes_v, forward, T, phi=phi)
        # Replace NaN with large penalty
        iv_fit = np.where(np.isfinite(iv_fit), iv_fit, 2.0)

        # Dead-zone residuals
        data_resid = np.where(
            iv_fit < iv_lo, wt * (iv_fit - iv_lo),
            np.where(iv_fit > iv_hi, wt * (iv_fit - iv_hi), 0.0)
        )

        if not use_prior:
            return data_resid

        # Prior virtual data points
        iv_fit_grid = lqd_implied_vols(theta, alpha, u,
                                        forward * np.exp(prior_k_grid),
                                        forward, T, phi=phi)
        iv_fit_grid = np.where(np.isfinite(iv_fit_grid), iv_fit_grid, 2.0)
        prior_resid = np.sqrt(lambda_prior) * (iv_fit_grid - iv_prior_grid)
        return np.concatenate([data_resid, prior_resid])

    try:
        result = least_squares(
            residuals,
            x0=x0,
            bounds=(np.full(N_PARAMS, -3.0), np.full(N_PARAMS, 3.0)),
            method='trf',
            max_nfev=200,
        )
        theta_opt = result.x
    except Exception:
        return _flat_result(strikes, atm_iv, forward, T)

    iv_fitted = lqd_implied_vols(theta_opt, alpha, u, strikes, forward, T, phi=phi)
    iv_fitted = np.where(np.isfinite(iv_fitted), iv_fitted, atm_iv)
    residual_vols = iv_fitted - mid_ivs

    return {
        "theta": theta_opt,
        "alpha": float(alpha),
        "iv_fitted": iv_fitted,
        "residuals": residual_vols,
        "forward": float(forward),
        "T": float(T),
        "atm_iv": float(atm_iv),
    }


def _flat_result(strikes, atm_iv, forward, T):
    return {
        "theta": np.zeros(N_PARAMS),
        "alpha": float(atm_iv * np.sqrt(max(T, 1e-8))),
        "iv_fitted": np.full(len(strikes), atm_iv),
        "residuals": np.zeros(len(strikes)),
        "forward": float(forward),
        "T": float(T),
        "atm_iv": float(atm_iv),
    }


# ---------------------------------------------------------------------------
# Local rotation  (spec Sections 6–8)
# ---------------------------------------------------------------------------

def local_rotation(theta_star: np.ndarray, alpha: float,
                   forward: float, T: float,
                   weights: np.ndarray = None) -> dict:
    """
    Build the two-stage local rotation at reference θ_*.

    Returns dict with:
        K:     trader-coordinate Jacobian (6×6)
        K_inv: inverse of K
        R:     upper-triangular QR factor (6×6)
        R_inv: inverse of R
        J:     smile Jacobian dσ/dθ at θ_* (N_z × 6)
        z_grid: normalised strike grid (N_z,)
        O:     observable operator (6 × N_z)
        Q_orth: orthonormal columns (N_z × 6)
    """
    u = quantile_grid(_U_GRID_SIZE)
    phi = basis_functions(u)

    # Normalised strike grid
    z_grid = np.linspace(-_Z_GRID_HALF, _Z_GRID_HALF, _N_Z)
    k_grid = z_grid * alpha  # un-normalise to log-moneyness
    strike_grid = forward * np.exp(k_grid)

    # Reference smile
    sigma_star = lqd_implied_vols(theta_star, alpha, u, strike_grid,
                                   forward, T, phi=phi)
    sigma_star = np.where(np.isfinite(sigma_star), sigma_star, 0.25)

    # --- Jacobian J = dσ/dθ (N_z × 6) by finite differences ---
    eps = 1e-3
    J = np.zeros((_N_Z, N_PARAMS))
    for j in range(N_PARAMS):
        theta_up = theta_star.copy()
        theta_up[j] += eps
        sigma_up = lqd_implied_vols(theta_up, alpha, u, strike_grid,
                                     forward, T, phi=phi)
        sigma_up = np.where(np.isfinite(sigma_up), sigma_up, sigma_star)
        J[:, j] = (sigma_up - sigma_star) / eps

    # --- Observable operator O (6 × N_z) ---
    O = _build_observable_operator(z_grid, sigma_star)

    # --- Trader-coordinate Jacobian K = O·J (6×6) ---
    K = O @ J
    try:
        K_inv = np.linalg.inv(K)
    except np.linalg.LinAlgError:
        K_inv = np.linalg.pinv(K)

    # --- Weight matrix ---
    if weights is None:
        weights = 1.0 / (1.0 + z_grid ** 2)
    W_half = np.diag(np.sqrt(weights))

    # --- QR factorisation:  A = W^{1/2} J K^{-1} = Q R ---
    A = W_half @ J @ K_inv
    Q_orth, R = np.linalg.qr(A, mode='reduced')

    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R)

    return {
        "K": K,
        "K_inv": K_inv,
        "R": R,
        "R_inv": R_inv,
        "J": J,
        "z_grid": z_grid,
        "O": O,
        "Q_orth": Q_orth,
        "sigma_star": sigma_star,
        "theta_star": theta_star.copy(),
        "alpha": alpha,
    }


def _build_observable_operator(z: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Build the 6×N observable operator O such that y = O·σ.

    Six observables:
      y₁ = Σ(z_min*)        — minimum-IV level
      y₂ = Σ_z(0)           — ATM skew
      y₃ = Σ_zz(0)          — ATM curvature
      y₄ = S_P              — put wing slope
      y₅ = S_C              — call wing slope
      y₆ = Sh               — shoulder
    """
    N = len(z)
    O = np.zeros((6, N))

    # Helper: find index nearest to target z
    def idx(target):
        return np.argmin(np.abs(z - target))

    # y₁: Σ at minimum-IV strike (frozen at reference)
    i_min = np.argmin(sigma)
    O[0, i_min] = 1.0

    # y₂: ATM skew = Σ_z(0) ≈ (Σ(δ) - Σ(-δ)) / (2δ)
    delta = _FD_STEP
    i_plus = idx(delta)
    i_minus = idx(-delta)
    dz = z[i_plus] - z[i_minus]
    if abs(dz) > 1e-8:
        O[1, i_plus] = 1.0 / dz
        O[1, i_minus] = -1.0 / dz

    # y₃: ATM curvature = Σ_zz(0) ≈ (Σ(δ) - 2Σ(0) + Σ(-δ)) / δ²
    i_0 = idx(0.0)
    d2 = ((z[i_plus] - z[i_0]) + (z[i_0] - z[i_minus])) / 2
    if abs(d2) > 1e-8:
        O[2, i_plus] = 1.0 / d2**2
        O[2, i_0] = -2.0 / d2**2
        O[2, i_minus] = 1.0 / d2**2

    # y₄: put wing slope S_P = (Σ(-z₂) - Σ(-z₁)) / (z₁ - z₂)
    i_pz1 = idx(-_WING_Z1)
    i_pz2 = idx(-_WING_Z2)
    dz_wing = z[i_pz1] - z[i_pz2]
    if abs(dz_wing) > 1e-8:
        O[3, i_pz1] = -1.0 / dz_wing  # note: S_P = (σ(-z2)-σ(-z1))/(z1-z2)
        O[3, i_pz2] = 1.0 / dz_wing

    # y₅: call wing slope S_C = (Σ(z₂) - Σ(z₁)) / (z₂ - z₁)
    i_cz1 = idx(_WING_Z1)
    i_cz2 = idx(_WING_Z2)
    dz_cwing = z[i_cz2] - z[i_cz1]
    if abs(dz_cwing) > 1e-8:
        O[4, i_cz2] = 1.0 / dz_cwing
        O[4, i_cz1] = -1.0 / dz_cwing

    # y₆: shoulder Sh = (Σ(z_s) + Σ(-z_s))/2 - Σ(0) - (z_s²/2)Σ_zz(0)
    i_zs = idx(_SHOULDER_Z)
    i_nzs = idx(-_SHOULDER_Z)
    zs = z[i_zs]
    O[5, i_zs] = 0.5
    O[5, i_nzs] = 0.5
    O[5, i_0] = -1.0
    # Subtract (z_s²/2) × Σ_zz(0) term = (z_s²/2) × row 2 of O
    O[5] -= (zs**2 / 2.0) * O[2]

    return O


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------

def theta_to_trader(theta: np.ndarray, rotation: dict) -> np.ndarray:
    """Convert raw θ to trader coordinates c = K(θ - θ_*)."""
    return rotation["K"] @ (theta - rotation["theta_star"])


def trader_to_theta(c: np.ndarray, rotation: dict) -> np.ndarray:
    """Convert trader coordinates c back to raw θ = θ_* + K⁻¹c."""
    return rotation["theta_star"] + rotation["K_inv"] @ c


def theta_to_ortho(theta: np.ndarray, rotation: dict) -> np.ndarray:
    """Convert raw θ to orthogonal curve coordinates b = R·K(θ - θ_*)."""
    c = theta_to_trader(theta, rotation)
    return rotation["R"] @ c


def ortho_to_theta(b: np.ndarray, rotation: dict) -> np.ndarray:
    """Convert orthogonal coordinates b back to raw θ."""
    c = rotation["R_inv"] @ b
    return trader_to_theta(c, rotation)


# ---------------------------------------------------------------------------
# Trader-coordinate labels (for UI sliders)
# ---------------------------------------------------------------------------

TRADER_LABELS = [
    ("min_iv",    "Min IV Level"),
    ("atm_skew",  "ATM Skew"),
    ("atm_curv",  "ATM Curvature"),
    ("put_slope", "Put Wing Slope"),
    ("call_slope","Call Wing Slope"),
    ("shoulder",  "Shoulder"),
]
