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
            alpha_min=config.alpha_min,
            alpha_max=config.alpha_max,
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
    # --- SVI-JW Normalized Propagation ---
    # Encoding scheme:
    #   v (ATM var):     ln(v_mkt / v_prior)           — log-ratio (level channel)
    #   psi_hat (skew):  (psi_mkt - psi_prior) / ref   — clamped-reference (signed param)
    #   p_hat (put wing): ln(p_mkt / p_prior)           — log-ratio
    #   c_hat (call wing): ln(c_mkt / c_prior)          — log-ratio
    #   vt_ratio (min-var ratio): ln(vtr_mkt / vtr_prior) — log-ratio
    #
    # Two propagation channels through same P matrix:
    #   Channel 0: level (v only)
    #   Channel 1-4: shape (psi_hat, p_hat, c_hat, vt_ratio)

    from .svi import (svi_iv_at_strikes, svi_implied_vol, fit_svi, filter_quotes_for_fit,
                      raw_svi_to_jw_normalized, jw_normalized_to_raw_svi)

    JW_KEYS = ["v", "psi_hat", "p_hat", "c_hat", "vt_ratio"]
    # Skew threshold: below this |psi_hat|, encoding switches from relative to absolute
    SKEW_EPS = 0.1
    # Log-ratio cap (prevents extreme propagation)
    LOG_CAP = 2.0

    def _jw_encode(market_jw: dict, prior_jw: dict) -> np.ndarray:
        """Encode market-vs-prior change in normalized JW propagation space.

        Log-ratio for positive params (v, p_hat, c_hat, vt_ratio).
        Clamped-reference for signed params (psi_hat).
        """
        g = np.zeros(5)
        # v: log-ratio
        g[0] = np.clip(np.log(max(market_jw["v"], 1e-10) / max(prior_jw["v"], 1e-10)),
                        -LOG_CAP, LOG_CAP)
        # psi_hat: clamped-reference encoding
        ref = max(abs(prior_jw["psi_hat"]), SKEW_EPS)
        g[1] = np.clip((market_jw["psi_hat"] - prior_jw["psi_hat"]) / ref,
                        -LOG_CAP, LOG_CAP)
        # p_hat: log-ratio
        g[2] = np.clip(np.log(max(market_jw["p_hat"], 1e-10) / max(prior_jw["p_hat"], 1e-10)),
                        -LOG_CAP, LOG_CAP)
        # c_hat: log-ratio
        g[3] = np.clip(np.log(max(market_jw["c_hat"], 1e-10) / max(prior_jw["c_hat"], 1e-10)),
                        -LOG_CAP, LOG_CAP)
        # vt_ratio: log-ratio
        g[4] = np.clip(np.log(max(market_jw["vt_ratio"], 1e-10) / max(prior_jw["vt_ratio"], 1e-10)),
                        -LOG_CAP, LOG_CAP)
        return g

    def _jw_decode(prior_jw: dict, g: np.ndarray, T: float) -> dict:
        """Apply propagated JW change vector to a prior → new raw SVI params.

        Decodes from propagation space, converts back to raw SVI.
        """
        g = np.clip(g, -LOG_CAP, LOG_CAP)

        # v: exp decode
        v_new = prior_jw["v"] * np.exp(g[0])
        v_new = max(v_new, 1e-8)

        # psi_hat: clamped-reference decode
        ref = max(abs(prior_jw["psi_hat"]), SKEW_EPS)
        psi_new = prior_jw["psi_hat"] + ref * g[1]

        # p_hat: exp decode
        p_new = prior_jw["p_hat"] * np.exp(g[2])
        p_new = max(p_new, 1e-6)

        # c_hat: exp decode
        c_new = prior_jw["c_hat"] * np.exp(g[3])
        c_new = max(c_new, 1e-6)

        # vt_ratio: exp decode, clamp to [0, 1]
        vtr_new = prior_jw["vt_ratio"] * np.exp(g[4])
        vtr_new = float(np.clip(vtr_new, 0.0, 1.0))

        # Convert back to raw SVI
        raw = jw_normalized_to_raw_svi(v_new, vtr_new, psi_new, p_new, c_new, T)
        return raw

    # --- Phase A: fit SVI for observed nodes, convert to JW, encode changes ---
    obs_jw_encoded = {}   # ticker -> np.array(5,) in JW propagation space
    obs_svi_fitted = {}   # ticker -> dict (full SVI params)
    obs_jw_fitted = {}    # ticker -> dict (JW params)

    for v_idx, ticker in enumerate(tickers):
        if ticker not in quotes:
            continue
        q = quotes[ticker]
        p = priors[ticker]
        svi_prior = p.get("_svi_params")

        filt_k, filt_iv, filt_mask = filter_quotes_for_fit(q.strikes, q.mid_ivs, q.forward)
        if len(filt_k) >= 5:
            # Build fit kwargs from config
            fit_kw = dict(
                bid_ask_spread=q.bid_ask_spread[filt_mask] if q.bid_ask_spread is not None else None,
                use_bid_ask_fit=config.use_bid_ask_fit,
                lambda_prior=config.lambda_prior,
            )
            if svi_prior and config.lambda_prior > 0:
                fit_kw["prior_params"] = np.array([svi_prior["a"], svi_prior["b"],
                                                    svi_prior["rho"], svi_prior["m"],
                                                    svi_prior["sigma"]])
            market_svi = fit_svi(filt_k, filt_iv, q.forward, q.T, **fit_kw)
            obs_svi_fitted[ticker] = market_svi

            # Convert both market and prior to normalized JW
            market_jw = raw_svi_to_jw_normalized(
                market_svi["a"], market_svi["b"], market_svi["rho"],
                market_svi["m"], market_svi["sigma"], q.T)
            obs_jw_fitted[ticker] = market_jw

            if svi_prior:
                prior_jw = raw_svi_to_jw_normalized(
                    svi_prior["a"], svi_prior["b"], svi_prior["rho"],
                    svi_prior["m"], svi_prior["sigma"],
                    svi_prior.get("T", q.T))
                # Store JW on prior for decode phase
                p["_jw_params"] = prior_jw
                obs_jw_encoded[ticker] = _jw_encode(market_jw, prior_jw)
            else:
                # No prior SVI — encode against flat prior
                flat_jw = {"v": market_jw["v"], "vt_ratio": 1.0,
                           "psi_hat": 0.0, "p_hat": 0.5, "c_hat": 0.5}
                p["_jw_params"] = flat_jw
                obs_jw_encoded[ticker] = _jw_encode(market_jw, flat_jw)

    # --- Phase B: propagate JW-encoded changes to unobserved nodes ---
    unobs_jw_propagated = {}  # ticker -> np.array(5,) in JW propagation space
    if P is not None and len(parts["unobs_idx"]) > 0 and len(parts["obs_idx"]) > 0:
        obs_tickers_ordered = [tickers[i] for i in parts["obs_idx"]]
        unobs_tickers_ordered = [tickers[i] for i in parts["unobs_idx"]]

        # Stack encoded changes matching P's column order
        g_O = np.zeros((len(obs_tickers_ordered), 5))
        for j, t in enumerate(obs_tickers_ordered):
            if t in obs_jw_encoded:
                g_O[j] = obs_jw_encoded[t]

        # Propagate: g_U = P @ g_O  (shape: N_unobs x 5)
        g_U = P @ g_O

        for i, t in enumerate(unobs_tickers_ordered):
            unobs_jw_propagated[t] = g_U[i]

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
            market_jw = obs_jw_fitted.get(ticker)
            if market_jw is None:
                market_jw = raw_svi_to_jw_normalized(
                    market_svi["a"], market_svi["b"], market_svi["rho"],
                    market_svi["m"], market_svi["sigma"], q.T)
            node_beta = np.array([market_jw["v"], market_jw["psi_hat"], market_jw["p_hat"], market_jw["c_hat"], market_jw["vt_ratio"]])
        elif ticker in unobs_jw_propagated and svi_params is not None:
            # UNOBSERVED: decode propagated JW change onto this asset's prior
            prior_jw = priors[ticker].get("_jw_params")
            if prior_jw is None:
                # Compute JW from raw prior SVI
                prior_jw = raw_svi_to_jw_normalized(
                    svi_params["a"], svi_params["b"], svi_params["rho"],
                    svi_params["m"], svi_params["sigma"],
                    svi_params.get("T", T))

            prop_raw = _jw_decode(prior_jw, unobs_jw_propagated[ticker], T)
            prop_svi = {**prop_raw, "forward": forward, "T": T}

            iv_marked = svi_iv_at_strikes(prop_svi, strikes, forward, T)
            iv_marked = np.clip(iv_marked, 0.01, 5.0)
            iv_marked = np.where(np.isfinite(iv_marked), iv_marked, np.nan)
            # Compute JW from the propagated raw SVI
            prop_jw = raw_svi_to_jw_normalized(
                prop_raw["a"], prop_raw["b"], prop_raw["rho"],
                prop_raw["m"], prop_raw["sigma"], T)
            node_beta = np.array([prop_jw["v"], prop_jw["psi_hat"], prop_jw["p_hat"], prop_jw["c_hat"], prop_jw["vt_ratio"]])
        else:
            # No propagation data: show prior
            iv_marked = iv_prior_fitted.copy()
            jw = priors[ticker].get("_jw_params")
            if jw:
                node_beta = np.array([jw["v"], jw["psi_hat"], jw["p_hat"], jw["c_hat"], jw["vt_ratio"]])
            elif svi_params:
                fb_jw = raw_svi_to_jw_normalized(svi_params["a"], svi_params["b"], svi_params["rho"], svi_params["m"], svi_params["sigma"], svi_params.get("T", T))
                node_beta = np.array([fb_jw["v"], fb_jw["psi_hat"], fb_jw["p_hat"], fb_jw["c_hat"], fb_jw["vt_ratio"]])
            else:
                node_beta = np.array([0.04, 0.0, 0.5, 0.5, 1.0])

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
