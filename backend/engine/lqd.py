"""
Log-Quantile-Density (LQD) Fourier basis and reconstruction.

Model:  ℓ(u; θ) = -log u - log(1-u) + r(u; θ)
where   r(u; θ) = Σ_{j=1}^{6} θ_j Φ_j(u)

The fixed skeleton -log u - log(1-u) enforces exponential tails (Lee/SVI
consistent).  The 6 Fourier basis functions {Φ_P, Φ_C, Φ_0, Φ_1, Φ_2, Φ_4}
control put wing, call wing, belly level, skew, kurtosis, and shoulder.

Parameter vector θ = (p, c, m, s, κ, h):
  p — put wing (left-tail log-rate, Λ_L = e^{-p})
  c — call wing (right-tail log-rate, Λ_R = e^{-c})
  m — belly level (lifts/lowers smile center)
  s — skew (antisymmetric tilt)
  κ — kurtosis / ATM butterfly (center curvature)
  h — shoulder (wing–belly connection)

Reference: lqd_local_qr_note.tex, equations 154–234.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid


# ---------------------------------------------------------------------------
# Quantile grid
# ---------------------------------------------------------------------------

def quantile_grid(n: int) -> np.ndarray:
    """
    Grid on (0,1) denser near boundaries to capture tail singularities.

    Uses a beta-CDF-like transformation: u = Beta(a,a).ppf(t) for uniform t,
    which clusters points near 0 and 1.
    """
    from scipy.stats import beta as beta_dist
    eps = 1e-6
    t = np.linspace(eps, 1.0 - eps, n)
    return beta_dist.ppf(t, 0.7, 0.7)


# ---------------------------------------------------------------------------
# Fourier basis functions  (equations 162–234)
# ---------------------------------------------------------------------------

_PI = np.pi


def _phi_P(u: np.ndarray) -> np.ndarray:
    """Put-wing mode:  Φ_P(0)=1, Φ_P(1)=0."""
    return (3/16 + (3/8)*np.cos(_PI*u) + (1/4)*np.cos(2*_PI*u)
            + (1/8)*np.cos(3*_PI*u) + (1/16)*np.cos(4*_PI*u))


def _phi_C(u: np.ndarray) -> np.ndarray:
    """Call-wing mode:  Φ_C(0)=0, Φ_C(1)=1."""
    return _phi_P(1.0 - u)


def _phi_0(u: np.ndarray) -> np.ndarray:
    """Belly level:  Φ_0(½)=1, vanishes at endpoints."""
    return ((75/64)*np.sin(_PI*u) + (25/128)*np.sin(3*_PI*u)
            + (3/128)*np.sin(5*_PI*u))


def _phi_1(u: np.ndarray) -> np.ndarray:
    """Skew:  Φ_1(½+t) = t + O(t^5), antisymmetric."""
    return -(8*np.sin(2*_PI*u) + np.sin(4*_PI*u)) / (12*_PI)


def _phi_2(u: np.ndarray) -> np.ndarray:
    """Kurtosis / ATM butterfly:  Φ_2(½+t) = t² + O(t^6)."""
    return (34*np.sin(_PI*u) + 39*np.sin(3*_PI*u)
            + 5*np.sin(5*_PI*u)) / (96*_PI**2)


def _phi_4(u: np.ndarray) -> np.ndarray:
    """Shoulder:  Φ_4(½+t) = t⁴ + O(t^6)."""
    return (2*np.sin(_PI*u) + 3*np.sin(3*_PI*u)
            + np.sin(5*_PI*u)) / (16*_PI**4)


# Ordered as θ = (p, c, m, s, κ, h)
_BASIS_FNS = [_phi_P, _phi_C, _phi_0, _phi_1, _phi_2, _phi_4]
PARAM_NAMES = ["p", "c", "m", "s", "kappa", "h"]
N_PARAMS = 6


def basis_functions(u: np.ndarray, M: int = N_PARAMS) -> np.ndarray:
    """
    Evaluate Fourier LQD basis functions on quantile grid u.

    Returns shape (M, len(u)).  M is clamped to N_PARAMS=6.
    """
    M = min(M, N_PARAMS)
    phi = np.zeros((M, len(u)))
    for j in range(M):
        phi[j] = _BASIS_FNS[j](u)
    return phi


# ---------------------------------------------------------------------------
# LQD evaluation and quantile reconstruction
# ---------------------------------------------------------------------------

def evaluate_lqd(theta: np.ndarray, u: np.ndarray,
                 phi: np.ndarray = None) -> np.ndarray:
    """
    Full LQD:  ℓ(u; θ) = -log u - log(1-u) + Σ θ_j Φ_j(u).

    Args:
        theta: parameter vector, shape (6,)
        u:     quantile grid, shape (N,)
        phi:   pre-computed basis, shape (6, N). Computed if None.

    Returns:
        ℓ(u), shape (N,)
    """
    if phi is None:
        phi = basis_functions(u)
    skeleton = -np.log(u) - np.log(1.0 - u)
    remainder = phi.T @ theta
    return skeleton + remainder


def reconstruct_quantile(ell: np.ndarray, u: np.ndarray,
                         theta: np.ndarray = None,
                         phi: np.ndarray = None) -> np.ndarray:
    """
    Reconstruct the quantile Q(u) from LQD ℓ(u) using logistic coordinates.

    In logistic coordinates x = log(u/(1-u)), the skeleton cancels exactly:
        dQ/dx = exp(r(u(x)))
    where r is the bounded remainder.  This avoids the 1/(u(1-u)) singularity.

    If theta and phi are provided, uses the stable logistic-coordinate
    integration.  Otherwise falls back to direct integration of exp(ℓ).

    Returns Q of shape (N,), centred so Q(½) = 0.
    """
    if theta is not None and phi is not None:
        # Stable path: integrate exp(r) in logistic coordinates
        r = phi.T @ theta  # bounded remainder, shape (N,)
        x = np.log(u / (1.0 - u))  # logistic coordinate
        exp_r = np.exp(np.clip(r, -50, 50))
        Q_raw = np.zeros_like(u)
        Q_raw[1:] = cumulative_trapezoid(exp_r, x)
    else:
        # Fallback: direct integration (less stable near endpoints)
        exp_psi = np.exp(np.clip(ell, -50, 50))
        Q_raw = np.zeros_like(u)
        Q_raw[1:] = cumulative_trapezoid(exp_psi, u)

    mid_idx = np.argmin(np.abs(u - 0.5))
    Q_raw -= Q_raw[mid_idx]
    return Q_raw


def renormalise_shape(Q_tilde: np.ndarray, u: np.ndarray) -> tuple:
    """
    Enforce unit-variance normalisation:
      s_new = s_old · ||Q̃||_L²
      Q̃ ← Q̃ / ||Q̃||_L²

    Returns (Q_tilde_normed, scale_factor).
    """
    L2_norm_sq = np.trapz(Q_tilde ** 2, u)
    L2_norm = np.sqrt(L2_norm_sq)
    if L2_norm < 1e-12:
        return Q_tilde, 1.0
    return Q_tilde / L2_norm, L2_norm


def lqd_from_quantile_derivative(Q_prime: np.ndarray) -> np.ndarray:
    """ψ(u) = log Q'(u)."""
    return np.log(np.maximum(Q_prime, 1e-30))


# ---------------------------------------------------------------------------
# Risk-neutral pricing:  θ → call prices → implied vols
# ---------------------------------------------------------------------------

def lqd_call_prices(theta: np.ndarray, alpha: float, u: np.ndarray,
                    strikes: np.ndarray, forward: float, T: float,
                    r: float = 0.0, phi: np.ndarray = None) -> np.ndarray:
    """
    Compute undiscounted forward call prices from LQD parameters.

    X = μ + α·Z  where Z has quantile Q(u; θ).
    μ = -log E[e^{αZ}]  (martingale constraint).
    C(K) = ∫₀¹ (F·e^{μ+αQ(u)} - K)⁺ du.

    Args:
        theta:   LQD parameters, shape (6,)
        alpha:   normalisation scale = σ_ref · √T
        u:       quantile grid, shape (N,)
        strikes: strike prices, shape (n_K,)
        forward: forward price F
        T:       time to maturity
        r:       risk-free rate (for discounting)
        phi:     pre-computed basis, shape (6, N)

    Returns:
        call_prices, shape (n_K,)
    """
    if phi is None:
        phi = basis_functions(u)
    ell = evaluate_lqd(theta, u, phi)
    Q = reconstruct_quantile(ell, u, theta=theta, phi=phi)

    # Martingale constraint: μ = -log E[e^{αZ}]
    alpha_Q = alpha * Q
    alpha_Q_safe = alpha_Q - np.max(alpha_Q)  # numerical stability
    mu = -np.max(alpha_Q) - np.log(np.trapz(np.exp(alpha_Q_safe), u))

    # S_T values: F · e^{μ + αQ(u)}
    log_S = np.log(forward) + mu + alpha_Q
    S_values = np.exp(log_S)

    discount = np.exp(-r * T)
    prices = np.zeros(len(strikes))
    for k, K in enumerate(strikes):
        payoff = np.maximum(S_values - K, 0.0)
        prices[k] = discount * np.trapz(payoff, u)
    return prices


def lqd_implied_vols(theta: np.ndarray, alpha: float, u: np.ndarray,
                     strikes: np.ndarray, forward: float, T: float,
                     r: float = 0.0, phi: np.ndarray = None) -> np.ndarray:
    """
    Compute implied volatilities from LQD parameters.

    θ → call prices → Black-Scholes inversion → IVs.
    """
    from .reconstruct import call_price_to_iv

    prices = lqd_call_prices(theta, alpha, u, strikes, forward, T, r, phi)
    return call_price_to_iv(prices, forward, strikes, T, r)
