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

    # ===================================================================
    # Model-dispatched propagation
    # ===================================================================
    # Each model provides encode/decode/fit/eval for its param space.
    # The propagation matrix P is model-independent (same graph structure).
    # ===================================================================

    smile_model = config.smile_model

    # --- Shared constants ---
    SKEW_EPS = 0.1
    LOG_CAP = 2.0

    def _log_encode(mkt, pri):
        return np.clip(np.log(max(mkt, 1e-10) / max(pri, 1e-10)), -LOG_CAP, LOG_CAP)

    def _log_decode(pri, g):
        return max(pri * np.exp(np.clip(g, -LOG_CAP, LOG_CAP)), 1e-10)

    def _ref_encode(mkt, pri, eps=SKEW_EPS):
        ref = max(abs(pri), eps)
        return np.clip((mkt - pri) / ref, -LOG_CAP, LOG_CAP)

    def _ref_decode(pri, g, eps=SKEW_EPS):
        ref = max(abs(pri), eps)
        return pri + ref * np.clip(g, -LOG_CAP, LOG_CAP)

    # --- Helper: display strike range ---
    def _display_strikes(ticker):
        chain = full_chains.get(ticker) if full_chains else None
        q = quotes.get(ticker)
        p = priors[ticker]
        svi_params = p.get("_svi_params")
        if chain is not None:
            return chain.strikes, chain.forward, chain.T
        elif q is not None:
            return q.strikes, q.forward, q.T
        elif svi_params and svi_params.get("forward"):
            fwd = svi_params["forward"]
            t = svi_params.get("T", 30 / 365)
            return fwd * np.exp(np.linspace(-0.08, 0.08, 30)), fwd, t
        else:
            ref_q = list(quotes.values())[0] if quotes else None
            fwd = ref_q.spot if ref_q else 100.0
            t = ref_q.T if ref_q else 30 / 365
            return fwd * np.exp(np.linspace(-0.06, 0.06, 25)), fwd, t

    # ===================================================================
    # SVI propagation
    # ===================================================================
    if smile_model == "svi":
        from .svi import (svi_iv_at_strikes, fit_svi, filter_quotes_for_fit,
                          raw_svi_to_jw_normalized, jw_normalized_to_raw_svi)

        def _jw_encode(market_jw, prior_jw):
            g = np.zeros(5)
            g[0] = _log_encode(market_jw["v"], prior_jw["v"])
            g[1] = _ref_encode(market_jw["psi_hat"], prior_jw["psi_hat"])
            g[2] = _log_encode(market_jw["p_hat"], prior_jw["p_hat"])
            g[3] = _log_encode(market_jw["c_hat"], prior_jw["c_hat"])
            g[4] = _log_encode(market_jw["vt_ratio"], prior_jw["vt_ratio"])
            return g

        def _jw_decode(prior_jw, g, T):
            v_new = _log_decode(prior_jw["v"], g[0])
            psi_new = _ref_decode(prior_jw["psi_hat"], g[1])
            p_new = _log_decode(prior_jw["p_hat"], g[2])
            c_new = _log_decode(prior_jw["c_hat"], g[3])
            vtr_new = float(np.clip(_log_decode(prior_jw["vt_ratio"], g[4]), 0.0, 1.0))
            return jw_normalized_to_raw_svi(v_new, vtr_new, psi_new, p_new, c_new, T)

        N_CH = 5
        obs_encoded = {}
        obs_fitted = {}
        obs_params = {}

        for ticker in tickers:
            if ticker not in quotes:
                continue
            q = quotes[ticker]
            p = priors[ticker]
            svi_prior = p.get("_svi_params")
            filt_k, filt_iv, filt_mask = filter_quotes_for_fit(q.strikes, q.mid_ivs, q.forward)
            if len(filt_k) >= 5:
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
                obs_fitted[ticker] = market_svi
                market_jw = raw_svi_to_jw_normalized(
                    market_svi["a"], market_svi["b"], market_svi["rho"],
                    market_svi["m"], market_svi["sigma"], q.T)
                obs_params[ticker] = market_jw
                if svi_prior:
                    prior_jw = raw_svi_to_jw_normalized(
                        svi_prior["a"], svi_prior["b"], svi_prior["rho"],
                        svi_prior["m"], svi_prior["sigma"], svi_prior.get("T", q.T))
                    p["_jw_params"] = prior_jw
                    obs_encoded[ticker] = _jw_encode(market_jw, prior_jw)
                else:
                    flat_jw = {"v": market_jw["v"], "vt_ratio": 1.0,
                               "psi_hat": 0.0, "p_hat": 0.5, "c_hat": 0.5}
                    p["_jw_params"] = flat_jw
                    obs_encoded[ticker] = _jw_encode(market_jw, flat_jw)

        # Propagate
        unobs_propagated = {}
        if P is not None and len(parts["unobs_idx"]) > 0 and len(parts["obs_idx"]) > 0:
            obs_ord = [tickers[i] for i in parts["obs_idx"]]
            unobs_ord = [tickers[i] for i in parts["unobs_idx"]]
            g_O = np.zeros((len(obs_ord), N_CH))
            for j, t in enumerate(obs_ord):
                if t in obs_encoded:
                    g_O[j] = obs_encoded[t]
            g_U = P @ g_O
            for i, t in enumerate(unobs_ord):
                unobs_propagated[t] = g_U[i]

        # Build display smiles
        nodes = {}
        for v_idx, ticker in enumerate(tickers):
            q = quotes.get(ticker)
            p = priors[ticker]
            bs_base = p.get("_bs_base", p)
            svi_params = p.get("_svi_params")
            strikes, forward, T = _display_strikes(ticker)

            if svi_params is not None:
                iv_prior_fitted = svi_iv_at_strikes(svi_params, strikes, forward, T)
            else:
                iv_prior_fitted = call_price_to_iv(
                    quantile_to_call_prices(bs_base["Q"], grid, forward, T, config.risk_free_rate or 0.045, strikes),
                    forward, strikes, T, config.risk_free_rate or 0.045)
            iv_prior_fitted = np.clip(iv_prior_fitted, 0.01, 5.0)

            if ticker in quotes and ticker in obs_fitted:
                market_svi = obs_fitted[ticker]
                iv_marked = svi_iv_at_strikes(market_svi, strikes, q.forward, q.T)
                iv_marked = np.clip(iv_marked, 0.01, 5.0)
                iv_marked = np.where(np.isfinite(iv_marked), iv_marked, np.nan)
                mj = obs_params.get(ticker)
                if mj is None:
                    mj = raw_svi_to_jw_normalized(market_svi["a"], market_svi["b"], market_svi["rho"],
                                                   market_svi["m"], market_svi["sigma"], q.T)
                node_beta = np.array([mj["v"], mj["psi_hat"], mj["p_hat"], mj["c_hat"], mj["vt_ratio"]])
            elif ticker in unobs_propagated and svi_params is not None:
                prior_jw = priors[ticker].get("_jw_params")
                if prior_jw is None:
                    prior_jw = raw_svi_to_jw_normalized(
                        svi_params["a"], svi_params["b"], svi_params["rho"],
                        svi_params["m"], svi_params["sigma"], svi_params.get("T", T))
                prop_raw = _jw_decode(prior_jw, unobs_propagated[ticker], T)
                prop_svi = {**prop_raw, "forward": forward, "T": T}
                iv_marked = svi_iv_at_strikes(prop_svi, strikes, forward, T)
                iv_marked = np.clip(iv_marked, 0.01, 5.0)
                iv_marked = np.where(np.isfinite(iv_marked), iv_marked, np.nan)
                pj = raw_svi_to_jw_normalized(prop_raw["a"], prop_raw["b"], prop_raw["rho"],
                                               prop_raw["m"], prop_raw["sigma"], T)
                node_beta = np.array([pj["v"], pj["psi_hat"], pj["p_hat"], pj["c_hat"], pj["vt_ratio"]])
            else:
                iv_marked = iv_prior_fitted.copy()
                jw = priors[ticker].get("_jw_params")
                if jw:
                    node_beta = np.array([jw["v"], jw["psi_hat"], jw["p_hat"], jw["c_hat"], jw["vt_ratio"]])
                elif svi_params:
                    fj = raw_svi_to_jw_normalized(svi_params["a"], svi_params["b"], svi_params["rho"],
                                                   svi_params["m"], svi_params["sigma"], svi_params.get("T", T))
                    node_beta = np.array([fj["v"], fj["psi_hat"], fj["p_hat"], fj["c_hat"], fj["vt_ratio"]])
                else:
                    node_beta = np.array([0.04, 0.0, 0.5, 0.5, 1.0])

            valid = np.isfinite(iv_marked) & np.isfinite(iv_prior_fitted)
            w2_dist = float(np.sqrt(np.mean((iv_marked[valid] - iv_prior_fitted[valid]) ** 2))) if valid.sum() > 0 else 0.0
            nodes[ticker] = NodeResult(
                ticker=ticker, strikes=strikes, iv_prior=iv_prior_fitted,
                iv_marked=iv_marked, beta=node_beta, is_observed=(ticker in quotes),
                shock_vector=y_blocks.get(v_idx), wasserstein_dist=w2_dist,
                Q_prior=p["Q"], Q_marked=p["Q"])

    # ===================================================================
    # LQD propagation (native)
    # ===================================================================
    elif smile_model == "lqd":
        from .lqd_model import fit_lqd_model, lqd_iv_at_strikes
        from .svi import fit_svi, filter_quotes_for_fit, svi_iv_at_strikes

        N_CH = 6
        obs_encoded = {}
        obs_fitted = {}   # ticker -> lqd result dict
        obs_params = {}   # ticker -> theta array

        for ticker in tickers:
            if ticker not in quotes:
                continue
            q = quotes[ticker]
            p = priors[ticker]
            prior_theta = p.get("_lqd_theta")

            filt_k, filt_iv, filt_mask = filter_quotes_for_fit(q.strikes, q.mid_ivs, q.forward)
            if len(filt_k) >= 5:
                pt = np.array(prior_theta) if prior_theta is not None and config.lambda_prior > 0 else None
                filt_strikes_lqd = q.forward * np.exp(filt_k)
                lqd_res = fit_lqd_model(filt_strikes_lqd, filt_iv, q.forward, q.T,
                                         bid_ask_spread=q.bid_ask_spread[filt_mask] if q.bid_ask_spread is not None else None,
                                         use_bid_ask_fit=config.use_bid_ask_fit,
                                         prior_theta=pt,
                                         lambda_prior=config.lambda_prior,
                                         atm_iv=q.atm_iv)
                obs_fitted[ticker] = lqd_res
                mkt_theta = lqd_res["theta"]
                obs_params[ticker] = mkt_theta

                if prior_theta is not None:
                    pri_arr = np.asarray(prior_theta, dtype=float)
                else:
                    pri_arr = np.zeros(6)
                # LQD encoding: additive delta (linear basis space)
                obs_encoded[ticker] = np.clip(mkt_theta - pri_arr, -LOG_CAP, LOG_CAP)

        # Propagate
        unobs_propagated = {}
        if P is not None and len(parts["unobs_idx"]) > 0 and len(parts["obs_idx"]) > 0:
            obs_ord = [tickers[i] for i in parts["obs_idx"]]
            unobs_ord = [tickers[i] for i in parts["unobs_idx"]]
            g_O = np.zeros((len(obs_ord), N_CH))
            for j, t in enumerate(obs_ord):
                if t in obs_encoded:
                    g_O[j] = obs_encoded[t]
            g_U = P @ g_O
            for i, t in enumerate(unobs_ord):
                unobs_propagated[t] = g_U[i]

        # Build display smiles
        nodes = {}
        for v_idx, ticker in enumerate(tickers):
            q = quotes.get(ticker)
            p = priors[ticker]
            bs_base = p.get("_bs_base", p)
            svi_params = p.get("_svi_params")
            prior_theta = p.get("_lqd_theta")
            prior_alpha = p.get("_lqd_alpha")
            strikes, forward, T = _display_strikes(ticker)

            # Prior IVs from LQD if available, else SVI fallback
            if prior_theta is not None and prior_alpha is not None:
                iv_prior_fitted = lqd_iv_at_strikes(np.asarray(prior_theta), strikes, forward, T, prior_alpha)
            elif svi_params is not None:
                iv_prior_fitted = svi_iv_at_strikes(svi_params, strikes, forward, T)
            else:
                iv_prior_fitted = call_price_to_iv(
                    quantile_to_call_prices(bs_base["Q"], grid, forward, T, config.risk_free_rate or 0.045, strikes),
                    forward, strikes, T, config.risk_free_rate or 0.045)
            iv_prior_fitted = np.clip(iv_prior_fitted, 0.01, 5.0)

            if ticker in quotes and ticker in obs_fitted:
                lqd_res = obs_fitted[ticker]
                iv_marked = lqd_iv_at_strikes(lqd_res["theta"], strikes, forward, T, lqd_res["alpha"])
                iv_marked = np.clip(np.where(np.isfinite(iv_marked), iv_marked, q.atm_iv), 0.01, 5.0)
                node_beta = lqd_res["theta"].copy()
            elif ticker in unobs_propagated:
                pri_arr = np.asarray(prior_theta if prior_theta is not None else np.zeros(6), dtype=float)
                prop_theta = pri_arr + np.clip(unobs_propagated[ticker], -LOG_CAP, LOG_CAP)
                alpha = prior_alpha if prior_alpha else (q.atm_iv if q else 0.25) * np.sqrt(max(T, 1e-8))
                iv_marked = lqd_iv_at_strikes(prop_theta, strikes, forward, T, alpha)
                iv_marked = np.clip(np.where(np.isfinite(iv_marked), iv_marked, 0.25), 0.01, 5.0)
                node_beta = prop_theta
            else:
                iv_marked = iv_prior_fitted.copy()
                node_beta = np.asarray(prior_theta if prior_theta is not None else np.zeros(6), dtype=float)

            valid = np.isfinite(iv_marked) & np.isfinite(iv_prior_fitted)
            w2_dist = float(np.sqrt(np.mean((iv_marked[valid] - iv_prior_fitted[valid]) ** 2))) if valid.sum() > 0 else 0.0
            nodes[ticker] = NodeResult(
                ticker=ticker, strikes=strikes, iv_prior=iv_prior_fitted,
                iv_marked=iv_marked, beta=node_beta, is_observed=(ticker in quotes),
                shock_vector=y_blocks.get(v_idx), wasserstein_dist=w2_dist,
                Q_prior=p["Q"], Q_marked=p["Q"])

    # ===================================================================
    # Sigmoid propagation (native)
    # ===================================================================
    elif smile_model == "sigmoid":
        from .sigmoid import fit_sigmoid_model, sigmoid_iv_at_strikes
        from .svi import filter_quotes_for_fit, svi_iv_at_strikes

        N_CH = 6
        obs_encoded = {}
        obs_fitted = {}
        obs_params = {}

        for ticker in tickers:
            if ticker not in quotes:
                continue
            q = quotes[ticker]
            p = priors[ticker]
            prior_sig = p.get("_sigmoid_params")

            filt_k, filt_iv, filt_mask = filter_quotes_for_fit(q.strikes, q.mid_ivs, q.forward)
            if len(filt_k) >= 5:
                pp = np.array(prior_sig) if prior_sig is not None and config.lambda_prior > 0 else None
                filt_strikes_sig = q.forward * np.exp(filt_k)
                sig_res = fit_sigmoid_model(filt_strikes_sig, filt_iv, q.forward, q.T,
                                             bid_ask_spread=q.bid_ask_spread[filt_mask] if q.bid_ask_spread is not None else None,
                                             use_bid_ask_fit=config.use_bid_ask_fit,
                                             prior_params=pp,
                                             lambda_prior=config.lambda_prior,
                                             atm_iv=q.atm_iv)
                obs_fitted[ticker] = sig_res
                mkt_p = sig_res["params"]
                obs_params[ticker] = mkt_p

                if prior_sig is not None:
                    pri = np.asarray(prior_sig, dtype=float)
                else:
                    pri = mkt_p.copy()

                # Sigmoid encoding: 6 channels
                # [0] σ_ATM:  log-ratio (level)
                # [1] S_ATM:  clamped-reference (signed skew)
                # [2] K_ATM:  log-ratio (positive)
                # [3] W_P:    log-ratio (positive)
                # [4] W_C:    log-ratio (positive)
                # [5] σ_min:  log-ratio (positive)
                g = np.zeros(6)
                g[0] = _log_encode(mkt_p[0], pri[0])
                g[1] = _ref_encode(mkt_p[1], pri[1])
                g[2] = _log_encode(mkt_p[2], pri[2])
                g[3] = _log_encode(mkt_p[3], pri[3])
                g[4] = _log_encode(mkt_p[4], pri[4])
                g[5] = _log_encode(mkt_p[5], pri[5])
                obs_encoded[ticker] = g

        # Propagate
        unobs_propagated = {}
        if P is not None and len(parts["unobs_idx"]) > 0 and len(parts["obs_idx"]) > 0:
            obs_ord = [tickers[i] for i in parts["obs_idx"]]
            unobs_ord = [tickers[i] for i in parts["unobs_idx"]]
            g_O = np.zeros((len(obs_ord), N_CH))
            for j, t in enumerate(obs_ord):
                if t in obs_encoded:
                    g_O[j] = obs_encoded[t]
            g_U = P @ g_O
            for i, t in enumerate(unobs_ord):
                unobs_propagated[t] = g_U[i]

        # Build display smiles
        nodes = {}
        for v_idx, ticker in enumerate(tickers):
            q = quotes.get(ticker)
            p = priors[ticker]
            bs_base = p.get("_bs_base", p)
            svi_params = p.get("_svi_params")
            prior_sig = p.get("_sigmoid_params")
            prior_sr = p.get("_sigmoid_sigma_ref")
            strikes, forward, T = _display_strikes(ticker)

            # Prior IVs from Sigmoid if available, else SVI fallback
            if prior_sig is not None and prior_sr is not None:
                iv_prior_fitted = sigmoid_iv_at_strikes(np.asarray(prior_sig), strikes, forward, T, prior_sr)
            elif svi_params is not None:
                iv_prior_fitted = svi_iv_at_strikes(svi_params, strikes, forward, T)
            else:
                iv_prior_fitted = call_price_to_iv(
                    quantile_to_call_prices(bs_base["Q"], grid, forward, T, config.risk_free_rate or 0.045, strikes),
                    forward, strikes, T, config.risk_free_rate or 0.045)
            iv_prior_fitted = np.clip(iv_prior_fitted, 0.01, 5.0)

            if ticker in quotes and ticker in obs_fitted:
                sig_res = obs_fitted[ticker]
                sr = sig_res["sigma_ref"]
                iv_marked = sigmoid_iv_at_strikes(sig_res["params"], strikes, forward, T, sr)
                iv_marked = np.clip(np.where(np.isfinite(iv_marked), iv_marked, q.atm_iv), 0.01, 5.0)
                node_beta = sig_res["params"].copy()
            elif ticker in unobs_propagated:
                pri = np.asarray(prior_sig if prior_sig is not None else [0.25, 0.0, 0.01, 0.02, 0.02, 0.20], dtype=float)
                g = unobs_propagated[ticker]
                # Sigmoid decoding
                prop = np.zeros(6)
                prop[0] = _log_decode(pri[0], g[0])            # σ_ATM
                prop[1] = _ref_decode(pri[1], g[1])             # S_ATM
                prop[2] = _log_decode(pri[2], g[2])             # K_ATM
                prop[3] = _log_decode(pri[3], g[3])             # W_P
                prop[4] = _log_decode(pri[4], g[4])             # W_C
                prop[5] = _log_decode(pri[5], g[5])             # σ_min
                # Feasibility clamps
                prop[0] = max(prop[0], 0.01)
                prop[2] = max(prop[2], 1e-6)
                prop[3] = max(prop[3], abs(prop[1]) + 1e-4)
                prop[4] = max(prop[4], abs(prop[1]) + 1e-4)
                prop[5] = max(prop[5], 0.005)
                prop[5] = min(prop[5], prop[0])
                sr = prior_sr if prior_sr else max(pri[0], 0.01)
                iv_marked = sigmoid_iv_at_strikes(prop, strikes, forward, T, sr)
                iv_marked = np.clip(np.where(np.isfinite(iv_marked), iv_marked, 0.25), 0.01, 5.0)
                node_beta = prop
            else:
                iv_marked = iv_prior_fitted.copy()
                node_beta = np.asarray(prior_sig if prior_sig is not None else [0.25, 0.0, 0.01, 0.02, 0.02, 0.20], dtype=float)

            valid = np.isfinite(iv_marked) & np.isfinite(iv_prior_fitted)
            w2_dist = float(np.sqrt(np.mean((iv_marked[valid] - iv_prior_fitted[valid]) ** 2))) if valid.sum() > 0 else 0.0
            nodes[ticker] = NodeResult(
                ticker=ticker, strikes=strikes, iv_prior=iv_prior_fitted,
                iv_marked=iv_marked, beta=node_beta, is_observed=(ticker in quotes),
                shock_vector=y_blocks.get(v_idx), wasserstein_dist=w2_dist,
                Q_prior=p["Q"], Q_marked=p["Q"])

    else:
        raise ValueError(f"Unknown smile_model: {smile_model}")

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
