"""
Influence graph construction: the W matrix.

Reference: paper Section 6 (eq.19-22).

Graph construction heuristics:
1. Cross-asset at same maturity: W_{(i,T),(j,T)} ∝ |ρ_ij| · √(ℓ_j / (ℓ_i + ℓ_j))
2. Index-constituent link:       W_{(i,T),(I,T)} ∝ w_i^I · ρ_iI
3. Row normalisation:            W_vw ← (1 - α_v) · W_vw / Σ_{w'} W_{vw'}

Properties (W1)-(W3):
  W_vv = 0 (no self-influence)
  W_vw ≥ 0 (non-negative)
  Σ_w W_vw ≤ 1 (sub-stochastic, residual α_v = self-trust)
"""
import numpy as np
from scipy import sparse
from typing import List, Dict, Optional, Tuple

from ..config import AssetDef, get_correlation


def build_influence_matrix(
    assets: List[AssetDef],
    correlations: Optional[Dict[Tuple[str, str], float]] = None,
    alpha_overrides: Optional[Dict[str, float]] = None,
    alpha_min: float = 0.10,
    alpha_max: float = 0.90,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the influence matrix W for the asset universe (single maturity).

    Args:
        assets:           list of AssetDef
        correlations:     optional override for pairwise correlations
        alpha_overrides:  optional per-ticker self-trust overrides
        alpha_min:        self-trust floor (least liquid nodes)
        alpha_max:        self-trust cap   (most liquid nodes)

    Returns:
        W:      influence matrix, shape (N, N), dense (small enough for Phase 1)
        alphas: self-trust vector, shape (N,)
    """
    N = len(assets)
    W_raw = np.zeros((N, N))

    tickers = [a.ticker for a in assets]
    liq = np.array([a.liquidity_score for a in assets])
    liq_max = liq.max() if N > 0 else 1.0

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            ai, aj = assets[i], assets[j]

            # Correlation
            if correlations and (ai.ticker, aj.ticker) in correlations:
                rho = correlations[(ai.ticker, aj.ticker)]
            elif correlations and (aj.ticker, ai.ticker) in correlations:
                rho = correlations[(aj.ticker, ai.ticker)]
            else:
                rho = get_correlation(ai.ticker, aj.ticker)

            # Base weight: |ρ| · liquidity asymmetry (eq.21)
            # W[i,j] = how much j pulls i.
            # Liquid j → illiquid i: high.  Illiquid j → liquid i: low.
            # Ratio (ℓ_j/ℓ_i) without sqrt for stronger asymmetry.
            liq_ratio = liq[j] / (liq[i] + liq[j])
            w = abs(rho) * liq_ratio

            # Index-constituent boost (eq.22)
            # Only boost constituent influenced BY index, not the reverse.
            if aj.is_index and ai.index_weight > 0:
                w *= (1.0 + ai.index_weight * 10.0)

            # Sector affinity boost
            if ai.sector == aj.sector and ai.sector != "Index":
                w *= 1.2

            W_raw[i, j] = w

    # Self-trust α_v: continuous, proportional to liquidity
    # Liquid nodes are mostly self-trusting (high α → small row sum).
    # Illiquid nodes are heavily influenced by neighbours (low α → large row sum).
    alphas = np.zeros(N)
    for i, a in enumerate(assets):
        if alpha_overrides and a.ticker in alpha_overrides:
            alphas[i] = alpha_overrides[a.ticker]
        else:
            # Linear in liquidity: α_min at ℓ=0, α_max at ℓ=ℓ_max
            alphas[i] = alpha_min + (alpha_max - alpha_min) * liq[i] / liq_max

    # Row normalisation: W_vw ← (1 - α_v) · W_vw / Σ_{w'} W_{vw'}  (eq.23 area)
    W = np.zeros_like(W_raw)
    for i in range(N):
        row_sum = W_raw[i].sum()
        if row_sum > 0:
            W[i] = (1.0 - alphas[i]) * W_raw[i] / row_sum

    return W, alphas


def get_laplacian(W: np.ndarray) -> np.ndarray:
    """L = I - W  (eq.19)."""
    return np.eye(W.shape[0]) - W


def partition_observed_unobserved(
    W: np.ndarray, observed_mask: np.ndarray
) -> dict:
    """
    Partition W into blocks for the harmonic extension (Section 9).

    Args:
        W:             full influence matrix (N, N)
        observed_mask: boolean array, True = observed node

    Returns dict with:
        W_UU: influence among unobserved nodes
        W_UO: influence from observed → unobserved
        obs_idx:   indices of observed nodes
        unobs_idx: indices of unobserved nodes
    """
    obs_idx = np.where(observed_mask)[0]
    unobs_idx = np.where(~observed_mask)[0]

    W_UU = W[np.ix_(unobs_idx, unobs_idx)]
    W_UO = W[np.ix_(unobs_idx, obs_idx)]

    return {
        "W_UU": W_UU,
        "W_UO": W_UO,
        "obs_idx": obs_idx,
        "unobs_idx": unobs_idx,
    }


def compute_influence_scores(W: np.ndarray) -> np.ndarray:
    """
    Compute total downstream influence for each node.

    F = (I - W)^{-1} is the full resolvent: F[i,j] = total response at node i
    when node j receives a +1 unit shock (including multi-hop propagation).

    influence_score[j] = sum_i F[i,j] = total impact of node j on the universe.

    Since W is sub-stochastic (spectral radius < 1), (I - W) is invertible.
    """
    N = W.shape[0]
    F = np.linalg.inv(np.eye(N) - W)
    return F.sum(axis=0)


def propagation_matrix(W_UU: np.ndarray, W_UO: np.ndarray) -> np.ndarray:
    """
    Compute the propagation matrix P = (I - W_UU)⁻¹ W_UO   (eq.33).

    This tells you exactly how many vol points each unobserved stock moves
    per vol point of each observed stock.
    """
    N_U = W_UU.shape[0]
    I_U = np.eye(N_U)
    L_UU_inv = np.linalg.inv(I_U - W_UU)
    return L_UU_inv @ W_UO


def neumann_series_terms(W_UU: np.ndarray, W_UO: np.ndarray,
                          beta_O: np.ndarray, n_terms: int = 5) -> list:
    """
    Compute Neumann series terms for propagation animation.

    (I - W_UU)⁻¹ = Σ_{k=0}^∞ W_UU^k

    Returns list of β_U contributions at each hop:
      term_k = W_UU^k · W_UO · β_O

    The sum converges to β_U = P · β_O.
    """
    terms = []
    current = W_UO @ beta_O  # k=0: direct influence from observed
    terms.append(current.copy())

    W_power = W_UU.copy()
    for k in range(1, n_terms):
        current = W_power @ W_UO @ beta_O
        terms.append(current.copy())
        W_power = W_power @ W_UU

    return terms


# ---------------------------------------------------------------------------
# Multi-expiry: time kernel and Kronecker product tensor
# ---------------------------------------------------------------------------

def build_time_kernel(T_values: np.ndarray, lambda_T: float) -> np.ndarray:
    """Build the cross-maturity influence kernel.

    K[i,j] = exp(-lambda_T * |sqrt(T_i) - sqrt(T_j)|)

    The kernel is row-normalized to be sub-stochastic (like W_asset):
    K[i,j] /= sum_j K[i,j], then scaled so diagonal = 0 and row sums ≤ 1.

    Args:
        T_values: maturities in years, shape (M,)
        lambda_T: decay speed (higher = less cross-maturity influence)

    Returns:
        K: (M, M) row-normalized time kernel
    """
    T_values = np.asarray(T_values, dtype=float)
    M = len(T_values)
    if M <= 1:
        return np.array([[1.0]])

    sqrt_T = np.sqrt(np.maximum(T_values, 1e-8))
    # Raw kernel: exponential decay in sqrt(T) distance
    K_raw = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            K_raw[i, j] = np.exp(-lambda_T * abs(sqrt_T[i] - sqrt_T[j]))

    # Row-normalize: each row sums to 1 (all influence goes to neighbours)
    K = np.zeros_like(K_raw)
    for i in range(M):
        row_sum = K_raw[i].sum()
        if row_sum > 0:
            K[i] = K_raw[i] / row_sum

    return K


def build_full_tensor(
    W_asset: np.ndarray,
    K_time: np.ndarray,
    alphas_asset: np.ndarray,
    alpha_time: float = 0.5,
) -> tuple:
    """Build the full (N*M, N*M) influence tensor via Kronecker product.

    W_full = W_asset ⊗ K_time

    Node ordering is ticker-major:
      [(ticker_0, T_0), (ticker_0, T_1), ..., (ticker_0, T_{M-1}),
       (ticker_1, T_0), ...]

    The Kronecker product preserves sub-stochastic structure:
    if W_asset and K_time both have row sums ≤ 1, so does W_full.

    Actually, we blend asset influence and time influence:
    - Same ticker, different maturity: influenced by time kernel
    - Different ticker, same maturity: influenced by W_asset
    - Different ticker, different maturity: influenced by both (product)

    We construct: W_full[iM+j, kM+l] = W_asset[i,k] * K_time[j,l]
    where i,k are ticker indices and j,l are maturity indices.

    For same-ticker cross-maturity influence, we add alpha_time * K_time
    on the block diagonal (ticker i with itself across maturities).

    Args:
        W_asset: (N, N) cross-asset influence matrix
        K_time: (M, M) cross-maturity kernel
        alphas_asset: (N,) self-trust per asset
        alpha_time: fraction of residual self-trust allocated to cross-maturity

    Returns:
        W_full: (N*M, N*M) full tensor
        alphas_full: (N*M,) self-trust per node
    """
    N = W_asset.shape[0]
    M = K_time.shape[0]
    NM = N * M

    W_full = np.zeros((NM, NM))

    for i in range(N):
        for k in range(N):
            if i == k:
                # Same ticker, different maturities: use time kernel
                # scaled by the asset's residual self-trust portion
                for j in range(M):
                    for l in range(M):
                        if j != l:
                            W_full[i * M + j, k * M + l] = alphas_asset[i] * alpha_time * K_time[j, l]
            else:
                # Different tickers: use W_asset, spread across maturity pairs
                for j in range(M):
                    for l in range(M):
                        # Cross-asset influence: W_asset[i,k] at same maturity,
                        # attenuated by K_time[j,l] at different maturities
                        if j == l:
                            W_full[i * M + j, k * M + l] = W_asset[i, k]
                        else:
                            W_full[i * M + j, k * M + l] = W_asset[i, k] * K_time[j, l]

    # Alphas: residual self-trust = 1 - row_sum
    alphas_full = 1.0 - W_full.sum(axis=1)
    alphas_full = np.clip(alphas_full, 0.01, 0.99)

    return W_full, alphas_full
