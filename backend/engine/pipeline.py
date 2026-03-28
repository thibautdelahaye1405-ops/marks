"""
Full single-maturity marking pipeline.

Orchestrates: prior → shocks → Jacobian → solve → reconstruct → output.

Reference: paper Section 11 (The Full Algorithm), Section 9 (single-maturity simplification).
"""
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from ..config import AssetDef, EngineConfig
from .lqd import quantile_grid, basis_functions
from .prior import bs_prior
from .jacobian import compute_jacobian
from .graph import build_influence_matrix, partition_observed_unobserved, propagation_matrix, neumann_series_terms, compute_influence_scores
from .solver import build_smoothness_matrix, solve_normal_equations, solve_harmonic_shortcut
from .reconstruct import reconstruct_smile, quantile_to_call_prices, call_price_to_iv


@dataclass
class NodeQuotes:
    """Market quotes for one (asset, maturity) node."""
    ticker: str
    strikes: np.ndarray         # shape (n_v,)
    mid_ivs: np.ndarray         # observed mid implied vols, shape (n_v,)
    bid_ask_spread: np.ndarray  # half bid-ask spread in vol points, shape (n_v,)
    forward: float              # forward price
    spot: float                 # spot price
    T: float                    # time to maturity in years
    atm_iv: float               # ATM implied vol (for prior)


@dataclass
class NodeResult:
    """Marking result for one node."""
    ticker: str
    strikes: np.ndarray
    iv_prior: np.ndarray
    iv_marked: np.ndarray
    beta: np.ndarray
    is_observed: bool
    shock_vector: Optional[np.ndarray] = None
    wasserstein_dist: float = 0.0
    Q_prior: Optional[np.ndarray] = None
    Q_marked: Optional[np.ndarray] = None


@dataclass
class MarkingResult:
    """Full marking output."""
    nodes: Dict[str, NodeResult]
    W: np.ndarray
    alphas: np.ndarray
    propagation_matrix: Optional[np.ndarray] = None
    neumann_terms: Optional[list] = None
    tickers: List[str] = field(default_factory=list)
    influence_scores: Optional[np.ndarray] = None
    wasserstein_distances: Optional[Dict[str, float]] = None


def run_marking(
    assets: List[AssetDef],
    quotes: Dict[str, NodeQuotes],
    config: EngineConfig,
    W_override: Optional[np.ndarray] = None,
    alpha_overrides: Optional[Dict[str, float]] = None,
    shock_nudges: Optional[Dict[str, float]] = None,
    calibrated_priors: Optional[Dict[str, dict]] = None,
    full_chains: Optional[Dict] = None,
) -> MarkingResult:
    """
    Run the full single-maturity marking pipeline.

    Args:
        assets:          universe definition
        quotes:          market quotes per observed ticker
        config:          engine hyperparameters
        W_override:      optional manual W matrix
        alpha_overrides: optional per-ticker self-trust
        shock_nudges:    optional per-ticker ATM vol nudge (added to market shock)

    Returns:
        MarkingResult with per-node marked smiles and propagation data.
    """
    N = len(assets)
    M = config.M
    tickers = [a.ticker for a in assets]

    # Build quantile grid and basis functions
    grid = quantile_grid(config.quantile_grid_size)
    phi = basis_functions(grid, M)

    # Build influence matrix
    if W_override is not None:
        W = W_override
        alphas = 1.0 - W.sum(axis=1)
    else:
        W, alphas = build_influence_matrix(
            assets,
            alpha_overrides=alpha_overrides,
            alpha_liquid=config.alpha_liquid,
            alpha_illiquid=config.alpha_illiquid,
        )

    # Smoothness matrix Ω
    Omega = build_smoothness_matrix(M, config.tail_reg_weight, config.interior_reg_weight)

    # Determine observed vs unobserved
    observed_mask = np.array([t in quotes for t in tickers])

    # Compute priors for all nodes
    priors = {}
    for i, a in enumerate(assets):
        # Use calibrated prior if available (from previous close data)
        if calibrated_priors and a.ticker in calibrated_priors:
            priors[a.ticker] = calibrated_priors[a.ticker]
            continue

        if a.ticker in quotes:
            q = quotes[a.ticker]
            atm_iv = q.atm_iv
            T = q.T
        else:
            avg_iv = np.mean([q.atm_iv for q in quotes.values()])
            T = list(quotes.values())[0].T
            atm_iv = avg_iv

        priors[a.ticker] = bs_prior(atm_iv, T, grid)

    # For each observed node: compute shocks y_v and Jacobian A_v
    A_blocks = {}
    Sigma_inv_blocks = {}
    y_blocks = {}

    for v_idx, ticker in enumerate(tickers):
        if ticker not in quotes:
            continue

        q = quotes[ticker]
        p = priors[ticker]
        bs_base = p.get("_bs_base", p)
        beta_fit = np.array(p.get("beta_fit", np.zeros(M)))

        # Jacobian from BS base at observed strikes
        A_v = compute_jacobian(
            psi0=bs_base["psi0"],
            Q_tilde=bs_base["Q_tilde"],
            m_v=bs_base["m"],
            s_v=bs_base["s"],
            forward=q.forward,
            T=q.T,
            r=config.risk_free_rate or 0.045,
            strikes=q.strikes,
            sigma_prior=bs_base["s"] / np.sqrt(max(q.T, 1e-6)),
            phi=phi,
            grid=grid,
        )

        # Calibrated prior IVs at observed strikes
        svi_params = p.get("_svi_params")
        if svi_params is not None:
            from .svi import svi_iv_at_strikes
            iv_cal_prior = svi_iv_at_strikes(svi_params, q.strikes, q.forward, q.T)
        else:
            iv_bs = call_price_to_iv(
                quantile_to_call_prices(bs_base["Q"], grid, q.forward, q.T, config.risk_free_rate or 0.045, q.strikes),
                q.forward, q.strikes, q.T, config.risk_free_rate or 0.045,
            )
            iv_cal_prior = iv_bs + A_v @ beta_fit

        # Shock: market - calibrated prior
        y_v = q.mid_ivs - iv_cal_prior

        # Apply nudge if provided
        if shock_nudges and ticker in shock_nudges:
            y_v = y_v + shock_nudges[ticker]

        # Replace NaN shocks with 0
        y_v = np.where(np.isfinite(y_v), y_v, 0.0)

        # Noise covariance
        sigma_noise = np.maximum(q.bid_ask_spread, 0.001)
        Sigma_inv = np.diag(1.0 / sigma_noise ** 2)

        A_blocks[v_idx] = A_v
        Sigma_inv_blocks[v_idx] = Sigma_inv
        y_blocks[v_idx] = y_v


    # Solve the normal equations (eq.29)
    beta_flat = solve_normal_equations(
        A_blocks=A_blocks,
        Sigma_inv_blocks=Sigma_inv_blocks,
        y_blocks=y_blocks,
        W=W,
        N_nodes=N,
        M=M,
        lambda_=config.lambda_,
        eta=config.eta,
        Omega=Omega,
    )

    beta_all = beta_flat.reshape(N, M)

    # Compute propagation matrix for visualization
    parts = partition_observed_unobserved(W, observed_mask)
    P = None
    neumann = None
    if len(parts["unobs_idx"]) > 0 and len(parts["obs_idx"]) > 0:
        P = propagation_matrix(parts["W_UU"], parts["W_UO"])
        beta_O = beta_all[parts["obs_idx"]]
        neumann = neumann_series_terms(parts["W_UU"], parts["W_UO"], beta_O, n_terms=5)

    # Reconstruct marked smiles
    # Observed nodes: direct SVI fit to market quotes
    # Unobserved nodes: prior SVI + propagated SVI deltas from observed nodes
    from .svi import svi_iv_at_strikes, svi_implied_vol, fit_svi, filter_quotes_for_fit

    SVI_KEYS = ["a", "b", "rho", "m", "sigma"]
    # Propagation mode per param: "prop" = proportional (ratio), "abs" = absolute (diff)
    SVI_MODES = ["prop", "prop", "abs", "abs", "prop"]

    def _svi_encode(market: dict, prior: dict) -> np.ndarray:
        """Encode the market-vs-prior change in propagation space.

        Proportional params: g = market/prior - 1  (fractional change)
        Absolute params:     g = market - prior     (level shift)
        """
        g = np.zeros(5)
        for j, (k, mode) in enumerate(zip(SVI_KEYS, SVI_MODES)):
            mv = market.get(k, 0.0)
            pv = prior.get(k, 0.0)
            if mode == "prop":
                g[j] = (mv / pv - 1.0) if abs(pv) > 1e-12 else 0.0
            else:
                g[j] = mv - pv
        return g

    def _svi_decode(prior: dict, g: np.ndarray) -> dict:
        """Apply propagated change vector to a prior SVI → new SVI params.

        Proportional params: new = prior * (1 + g)
        Absolute params:     new = prior + g
        Then clamp to valid SVI ranges.
        """
        out = {**prior}
        for j, (k, mode) in enumerate(zip(SVI_KEYS, SVI_MODES)):
            pv = prior.get(k, 0.0)
            if mode == "prop":
                out[k] = pv * (1.0 + g[j])
            else:
                out[k] = pv + g[j]
        # Clamp to valid SVI ranges
        out["a"] = max(out["a"], 0.0)
        out["b"] = max(out["b"], 0.0)
        out["rho"] = float(np.clip(out["rho"], -0.999, 0.999))
        out["sigma"] = max(out["sigma"], 1e-4)
        return out

    # --- Phase A: fit SVI for observed nodes, encode changes ---
    obs_svi_encoded = {}  # ticker -> np.array (5,) in propagation space
    obs_svi_fitted = {}   # ticker -> dict (full SVI params)

    for v_idx, ticker in enumerate(tickers):
        if ticker not in quotes:
            continue
        q = quotes[ticker]
        p = priors[ticker]
        svi_prior = p.get("_svi_params")

        filt_k, filt_iv, _ = filter_quotes_for_fit(q.strikes, q.mid_ivs, q.forward)
        if len(filt_k) >= 5:
            market_svi = fit_svi(filt_k, filt_iv, q.forward, q.T)
            obs_svi_fitted[ticker] = market_svi

            if svi_prior:
                obs_svi_encoded[ticker] = _svi_encode(market_svi, svi_prior)
            else:
                obs_svi_encoded[ticker] = np.zeros(5)

    # --- Phase B: propagate encoded changes to unobserved nodes ---
    # P = (I - W_UU)^{-1} W_UO applied to the encoded change vectors
    unobs_svi_propagated = {}  # ticker -> np.array (5,) in propagation space
    if P is not None and len(parts["unobs_idx"]) > 0 and len(parts["obs_idx"]) > 0:
        obs_tickers_ordered = [tickers[i] for i in parts["obs_idx"]]
        unobs_tickers_ordered = [tickers[i] for i in parts["unobs_idx"]]

        # Stack encoded changes matching P's column order
        g_O = np.zeros((len(obs_tickers_ordered), 5))
        for j, t in enumerate(obs_tickers_ordered):
            if t in obs_svi_encoded:
                g_O[j] = obs_svi_encoded[t]

        # Propagate: g_U = P @ g_O  (shape: N_unobs x 5)
        g_U = P @ g_O

        for i, t in enumerate(unobs_tickers_ordered):
            unobs_svi_propagated[t] = g_U[i]

    # --- Phase C: build display smiles ---
    nodes = {}
    for v_idx, ticker in enumerate(tickers):
        q = quotes.get(ticker)
        p = priors[ticker]
        bs_base = p.get("_bs_base", p)
        svi_params = p.get("_svi_params")

        # Display strike range from full chain
        chain = full_chains.get(ticker) if full_chains else None
        if chain is not None:
            strikes = chain.strikes
            forward = chain.forward
            T = chain.T
        elif q is not None:
            strikes = q.strikes
            forward = q.forward
            T = q.T
        elif svi_params and svi_params.get("forward"):
            forward = svi_params["forward"]
            T = svi_params.get("T", 30 / 365)
            strikes = forward * np.exp(np.linspace(-0.08, 0.08, 30))
        else:
            ref_q = list(quotes.values())[0] if quotes else None
            forward = ref_q.spot if ref_q else 100.0
            T = ref_q.T if ref_q else 30 / 365
            strikes = forward * np.exp(np.linspace(-0.06, 0.06, 25))

        # Prior IVs
        if svi_params is not None:
            iv_prior_fitted = svi_iv_at_strikes(svi_params, strikes, forward, T)
        else:
            iv_prior_fitted = call_price_to_iv(
                quantile_to_call_prices(bs_base["Q"], grid, forward, T, config.risk_free_rate or 0.045, strikes),
                forward, strikes, T, config.risk_free_rate or 0.045,
            )
        iv_prior_fitted = np.clip(iv_prior_fitted, 0.01, 5.0)

        if ticker in quotes and ticker in obs_svi_fitted:
            # OBSERVED: direct SVI fit
            market_svi = obs_svi_fitted[ticker]
            iv_marked = svi_iv_at_strikes(market_svi, strikes, q.forward, q.T)
            iv_marked = np.clip(iv_marked, 0.01, 5.0)
            iv_marked = np.where(np.isfinite(iv_marked), iv_marked, np.nan)
            node_beta = np.array([market_svi.get(k, 0) for k in SVI_KEYS])
        elif ticker in unobs_svi_propagated and svi_params is not None:
            # UNOBSERVED: decode propagated change onto this asset's prior SVI
            prop_svi = _svi_decode(svi_params, unobs_svi_propagated[ticker])

            iv_marked = svi_iv_at_strikes(prop_svi, strikes, forward, T)
            iv_marked = np.clip(iv_marked, 0.01, 5.0)
            iv_marked = np.where(np.isfinite(iv_marked), iv_marked, np.nan)
            node_beta = np.array([prop_svi.get(k, 0) for k in SVI_KEYS])
        else:
            # No propagation data: show prior
            iv_marked = iv_prior_fitted.copy()
            node_beta = np.array([svi_params.get(k, 0) for k in SVI_KEYS]) if svi_params else np.zeros(5)

        # Distance
        valid = np.isfinite(iv_marked) & np.isfinite(iv_prior_fitted)
        w2_dist = float(np.sqrt(np.mean((iv_marked[valid] - iv_prior_fitted[valid]) ** 2))) if valid.sum() > 0 else 0.0

        nodes[ticker] = NodeResult(
            ticker=ticker,
            strikes=strikes,
            iv_prior=iv_prior_fitted,
            iv_marked=iv_marked,
            beta=node_beta,
            is_observed=(ticker in quotes),
            shock_vector=y_blocks.get(v_idx),
            wasserstein_dist=w2_dist,
            Q_prior=p["Q"],
            Q_marked=p["Q"],
        )

    # Compute influence scores
    inf_scores = compute_influence_scores(W)
    w2_dists = {t: nodes[t].wasserstein_dist for t in tickers}

    return MarkingResult(
        nodes=nodes,
        W=W,
        alphas=alphas,
        propagation_matrix=P,
        neumann_terms=neumann,
        tickers=tickers,
        influence_scores=inf_scores,
        wasserstein_distances=w2_dists,
    )
