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
