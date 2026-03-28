"""
FastAPI routes for the vol marking application.
"""
import numpy as np
from fastapi import APIRouter, HTTPException
from typing import Dict, Optional

from .schemas import (
    Asset, QuoteSnapshot, SolveRequest, SolveResponse,
    SmileData, GraphData, WMatrixUpdate,
    DistributionView, PriorOverrideRequest, BetaOverrideRequest,
    NodeDistributionResponse, PriorRefitRequest,
    SavedPriorInfo, SavePriorRequest, SviOverrideRequest, FitRequest,
)
from ..data.prior_store import (
    list_saved_priors as _list_saved_priors,
    save_prior as _save_prior,
    load_prior as _load_prior,
)
from ..config import EngineConfig, DEFAULT_UNIVERSE
from ..data.referential import get_universe
from ..data.quotes import fetch_option_chain, OptionChainData
from ..data.store import save_snapshot, get_latest_snapshots
from ..engine.lqd import quantile_grid, basis_functions
from ..engine.prior import bs_prior, fit_lqd_prior, apply_beta_overrides, compute_distribution_view
from ..engine.graph import build_influence_matrix
from ..engine.pipeline import run_marking, NodeQuotes

router = APIRouter(prefix="/api")


def _chain_to_snapshot(c) -> QuoteSnapshot:
    """Convert an OptionChainData to a QuoteSnapshot response."""
    return QuoteSnapshot(
        ticker=c.ticker, expiry=c.expiry, T=c.T,
        spot=c.spot, forward=c.forward, atm_iv=c.atm_iv,
        strikes=c.strikes.tolist(), mid_ivs=c.mid_ivs.tolist(),
        bid_ask_spread=c.bid_ask_spread.tolist(),
        open_interest=c.open_interest.tolist(),
        prev_close_ivs=c.prev_close_ivs.tolist() if c.prev_close_ivs is not None else None,
        prev_close_atm_iv=c.prev_close_atm_iv,
        prev_spot=c.prev_spot,
    )

# In-memory state for the current session
_state = {
    "quotes": {},       # ticker -> OptionChainData
    "W": None,          # current influence matrix
    "alphas": None,     # current self-trust vector
    "config": EngineConfig(),
    "priors_base": {},  # ticker -> immutable base prior (from calibrate-priors or fit)
    "priors": {},       # ticker -> active prior (base + overrides, used by solve)
    "solve_result": None,  # last MarkingResult for distribution views
}


@router.get("/universe", response_model=list[Asset])
def get_universe_endpoint():
    universe = get_universe()
    return [
        Asset(
            ticker=a.ticker, name=a.name, sector=a.sector,
            is_index=a.is_index, index_weight=a.index_weight,
            liquidity_score=a.liquidity_score,
        )
        for a in universe
    ]


@router.post("/fetch-quotes", response_model=Dict[str, QuoteSnapshot])
def fetch_quotes_endpoint():
    """Fetch fresh option chains from Yahoo Finance for the entire universe."""
    universe = get_universe()
    config = _state["config"]
    results = {}

    for asset in universe:
        chain = fetch_option_chain(
            asset.ticker,
            target_maturity_days=config.target_maturity_days,
            r=config.risk_free_rate,
        )
        if chain is not None:
            _state["quotes"][asset.ticker] = chain
            save_snapshot(chain)
            results[asset.ticker] = _chain_to_snapshot(chain)

    if not results:
        raise HTTPException(status_code=503, detail="Could not fetch any quotes")

    return results


@router.get("/quotes/latest", response_model=Dict[str, QuoteSnapshot])
def get_latest_quotes():
    """Return the latest stored quotes."""
    if _state["quotes"]:
        return {t: _chain_to_snapshot(c) for t, c in _state["quotes"].items()}

    # Try loading from DB
    tickers = [a.ticker for a in get_universe()]
    stored = get_latest_snapshots(tickers)
    if not stored:
        raise HTTPException(status_code=404, detail="No quotes available. Fetch first.")

    results = {}
    for t, data in stored.items():
        chain = OptionChainData(
            ticker=t, expiry=data["expiry"], T=data["T"],
            spot=data["spot"], forward=data["forward"],
            strikes=data["strikes"], mid_ivs=data["mid_ivs"],
            bid_ask_spread=data["bid_ask_spread"],
            atm_iv=data["atm_iv"],
            open_interest=data["open_interest"],
            call_mids=data.get("call_mids", data["strikes"] * 0),
        )
        _state["quotes"][t] = chain
        results[t] = _chain_to_snapshot(chain)
    return results


@router.post("/fit")
def fit_endpoint(req: FitRequest):
    """Lightweight fit: SVI for observed assets, prior for unobserved.

    No graph propagation — just direct SVI fits from market quotes for
    observed assets, and prior curves for unobserved. Used for pre-propagation
    display.
    """
    if not _state["quotes"]:
        raise HTTPException(status_code=400, detail="No quotes loaded.")

    from ..engine.svi import fit_svi, svi_iv_at_strikes, filter_quotes_for_fit

    universe = get_universe()
    tickers = [a.ticker for a in universe]
    observed_set = set(req.observed_tickers) if req.observed_tickers else set(_state["quotes"].keys())

    nodes_resp = {}
    for ticker in tickers:
        chain = _state["quotes"].get(ticker)
        if chain is None:
            continue

        prior = _state["priors"].get(ticker) if _state.get("priors") else None
        svi_prior = prior.get("_svi_params") if prior else None
        strikes = chain.strikes
        forward = chain.forward
        T = chain.T

        # Prior IV
        if svi_prior:
            iv_prior = svi_iv_at_strikes(svi_prior, strikes, forward, T)
            iv_prior = np.clip(iv_prior, 0.01, 5.0)
        else:
            iv_prior = np.full(len(strikes), chain.atm_iv)

        if ticker in observed_set:
            # Observed: SVI fit to current market quotes (with exclusions/additions)
            fit_strikes = chain.strikes.copy()
            fit_ivs = chain.mid_ivs.copy()

            # Apply exclusions
            if req.excluded_quotes and ticker in req.excluded_quotes:
                keep = np.ones(len(fit_strikes), dtype=bool)
                for idx in req.excluded_quotes[ticker]:
                    if 0 <= idx < len(keep):
                        keep[idx] = False
                fit_strikes = fit_strikes[keep]
                fit_ivs = fit_ivs[keep]

            # Apply additions
            if req.added_quotes and ticker in req.added_quotes:
                for pt in req.added_quotes[ticker]:
                    if len(pt) >= 2:
                        fit_strikes = np.append(fit_strikes, pt[0])
                        fit_ivs = np.append(fit_ivs, pt[1])
                order = np.argsort(fit_strikes)
                fit_strikes = fit_strikes[order]
                fit_ivs = fit_ivs[order]

            filt_k, filt_iv, _ = filter_quotes_for_fit(fit_strikes, fit_ivs, forward)
            if len(filt_k) >= 5:
                market_svi = fit_svi(filt_k, filt_iv, forward, T)
                iv_marked = svi_iv_at_strikes(market_svi, strikes, forward, T)
                iv_marked = np.clip(iv_marked, 0.01, 5.0)
                svi_beta = [market_svi.get("a", 0), market_svi.get("b", 0),
                            market_svi.get("rho", 0), market_svi.get("m", 0),
                            market_svi.get("sigma", 0.1)]
            else:
                iv_marked = iv_prior.copy()
                svi_beta = [0.0] * 5
        else:
            # Unobserved: show prior as current smile
            iv_marked = iv_prior.copy()
            svi_beta = [svi_prior.get("a", 0), svi_prior.get("b", 0),
                        svi_prior.get("rho", 0), svi_prior.get("m", 0),
                        svi_prior.get("sigma", 0.1)] if svi_prior else [0.0] * 5

        iv_prior_list = [float(v) if np.isfinite(v) else None for v in iv_prior]
        iv_marked_list = [float(v) if np.isfinite(v) else None for v in iv_marked]

        nodes_resp[ticker] = SmileData(
            ticker=ticker,
            strikes=strikes.tolist(),
            iv_prior=iv_prior_list,
            iv_marked=iv_marked_list,
            beta=svi_beta,
            is_observed=(ticker in observed_set),
        )

    return {
        "nodes": {t: n.dict() for t, n in nodes_resp.items()},
        "tickers": list(nodes_resp.keys()),
    }


@router.post("/solve", response_model=SolveResponse)
def solve_endpoint(req: SolveRequest):
    """Run the marking engine with current quotes and user overrides."""
    if not _state["quotes"]:
        raise HTTPException(status_code=400, detail="No quotes loaded. Fetch first.")

    universe = get_universe()
    config = _state["config"]
    config.lambda_ = req.lambda_
    config.eta = req.eta

    # Auto-calibrate priors if not yet done
    if not _state["priors"]:
        grid = quantile_grid(config.quantile_grid_size)
        phi = basis_functions(grid, config.M)
        import copy
        priors = {}
        for ticker, chain in _state["quotes"].items():
            try:
                if chain.prev_close_ivs is not None and chain.prev_spot is not None:
                    prev_forward = chain.prev_spot * np.exp(config.risk_free_rate * chain.T)
                    priors[ticker] = fit_lqd_prior(
                        chain.strikes, chain.prev_close_ivs, prev_forward,
                        chain.T, config.risk_free_rate, grid, phi,
                    )
                else:
                    priors[ticker] = fit_lqd_prior(
                        chain.strikes, chain.mid_ivs, chain.forward,
                        chain.T, config.risk_free_rate, grid, phi,
                    )
            except Exception:
                priors[ticker] = bs_prior(chain.atm_iv, chain.T, grid)
        _state["priors_base"] = copy.deepcopy(priors)
        _state["priors"] = priors

    # Determine which tickers are observed
    observed_set = set(req.observed_tickers) if req.observed_tickers else set(_state["quotes"].keys())

    # Convert OptionChainData → NodeQuotes for observed tickers only
    node_quotes = {}
    for ticker, chain in _state["quotes"].items():
        if ticker not in observed_set:
            continue

        strikes = chain.strikes.copy()
        mid_ivs = chain.mid_ivs.copy()
        spread = chain.bid_ask_spread.copy()

        # Exclude specific quote points if requested
        if req.excluded_quotes and ticker in req.excluded_quotes:
            keep = np.ones(len(strikes), dtype=bool)
            for idx in req.excluded_quotes[ticker]:
                if 0 <= idx < len(keep):
                    keep[idx] = False
            n_before = len(strikes)
            strikes = strikes[keep]
            mid_ivs = mid_ivs[keep]
            spread = spread[keep]


        # Append user-added synthetic quotes
        if req.added_quotes and ticker in req.added_quotes:
            for pt in req.added_quotes[ticker]:
                if len(pt) >= 2:
                    strikes = np.append(strikes, pt[0])
                    mid_ivs = np.append(mid_ivs, pt[1])
                    # Use median spread as uncertainty for added points
                    spread = np.append(spread, np.median(spread) if len(spread) > 0 else 0.01)
            # Re-sort by strike
            order = np.argsort(strikes)
            strikes = strikes[order]
            mid_ivs = mid_ivs[order]
            spread = spread[order]

        if len(strikes) < 3:
            continue

        node_quotes[ticker] = NodeQuotes(
            ticker=ticker,
            strikes=strikes,
            mid_ivs=mid_ivs,
            bid_ask_spread=spread,
            forward=chain.forward,
            spot=chain.spot,
            T=chain.T,
            atm_iv=chain.atm_iv,
        )

    W_override = None
    if req.W_override is not None:
        W_override = np.array(req.W_override)

    # Full (unfiltered) chain data for display strike ranges
    # Full chain data for ALL tickers (not just observed) so unobserved nodes
    # get their own strike ranges in the display
    full_chains = dict(_state["quotes"])

    result = run_marking(
        assets=universe,
        quotes=node_quotes,
        config=config,
        W_override=W_override,
        alpha_overrides=req.alpha_overrides,
        shock_nudges=req.shock_nudges,
        calibrated_priors=_state["priors"] if _state["priors"] else None,
        full_chains=full_chains,
    )

    # Cache results
    _state["W"] = result.W
    _state["alphas"] = result.alphas
    _state["solve_result"] = result

    # Build response
    nodes_resp = {}
    for ticker, nr in result.nodes.items():
        iv_prior = nr.iv_prior.tolist() if nr.iv_prior is not None else []
        iv_marked = nr.iv_marked.tolist() if nr.iv_marked is not None else []
        # Replace NaN with None for JSON
        iv_prior = [v if np.isfinite(v) else None for v in iv_prior]
        iv_marked = [v if np.isfinite(v) else None for v in iv_marked]

        nodes_resp[ticker] = SmileData(
            ticker=ticker,
            strikes=nr.strikes.tolist(),
            iv_prior=iv_prior,
            iv_marked=iv_marked,
            beta=nr.beta.tolist(),
            is_observed=nr.is_observed,
        )

    neumann = None
    if result.neumann_terms is not None:
        neumann = [t.tolist() for t in result.neumann_terms]

    return SolveResponse(
        nodes=nodes_resp,
        W=result.W.tolist(),
        alphas=result.alphas.tolist(),
        tickers=result.tickers,
        propagation_matrix=result.propagation_matrix.tolist() if result.propagation_matrix is not None else None,
        neumann_terms=neumann,
        influence_scores=result.influence_scores.tolist() if result.influence_scores is not None else None,
        wasserstein_distances=result.wasserstein_distances,
    )


@router.get("/graph", response_model=GraphData)
def get_graph():
    """Return the current influence graph."""
    universe = get_universe()
    tickers = [a.ticker for a in universe]

    if _state["W"] is None:
        W, alphas = build_influence_matrix(universe)
        _state["W"] = W
        _state["alphas"] = alphas

    return GraphData(
        tickers=tickers,
        W=_state["W"].tolist(),
        alphas=_state["alphas"].tolist(),
        assets=[
            Asset(
                ticker=a.ticker, name=a.name, sector=a.sector,
                is_index=a.is_index, index_weight=a.index_weight,
                liquidity_score=a.liquidity_score,
            )
            for a in universe
        ],
    )


@router.put("/graph")
def update_graph(update: WMatrixUpdate):
    """Update the influence matrix W."""
    W = np.array(update.W)
    N = W.shape[0]

    # Validate
    if W.shape != (N, N):
        raise HTTPException(status_code=400, detail="W must be square")
    if np.any(np.diag(W) != 0):
        raise HTTPException(status_code=400, detail="Diagonal must be zero (no self-influence)")
    if np.any(W < 0):
        raise HTTPException(status_code=400, detail="W must be non-negative")
    row_sums = W.sum(axis=1)
    if np.any(row_sums > 1.0 + 1e-6):
        raise HTTPException(status_code=400, detail=f"Row sums must be <= 1. Got max {row_sums.max():.4f}")

    _state["W"] = W
    if update.alphas is not None:
        _state["alphas"] = np.array(update.alphas)
    else:
        _state["alphas"] = 1.0 - row_sums

    return {"status": "ok", "row_sums": row_sums.tolist()}


# --- Phase 3: Prior Calibration ---

@router.post("/calibrate-priors")
def calibrate_priors():
    """Fit LQD priors from previous close option prices.

    Uses lastPrice - change from the option chain to recover yesterday's
    closing prices, then computes IVs and fits the LQD basis.
    """
    if not _state["quotes"]:
        raise HTTPException(status_code=400, detail="No quotes loaded.")

    config = _state["config"]
    grid = quantile_grid(config.quantile_grid_size)
    phi = basis_functions(grid, config.M)

    priors = {}
    for ticker, chain in _state["quotes"].items():
        try:
            # Use previous close IVs if available
            if chain.prev_close_ivs is not None and chain.prev_spot is not None:
                prev_forward = chain.prev_spot * np.exp(config.risk_free_rate * chain.T)
                prior = fit_lqd_prior(
                    strikes=chain.strikes,
                    mid_ivs=chain.prev_close_ivs,
                    forward=prev_forward,
                    T=chain.T,
                    r=config.risk_free_rate,
                    grid=grid, phi=phi,
                )
            else:
                # Fallback: use current IVs
                prior = fit_lqd_prior(
                    strikes=chain.strikes, mid_ivs=chain.mid_ivs,
                    forward=chain.forward, T=chain.T, r=config.risk_free_rate,
                    grid=grid, phi=phi,
                )
            priors[ticker] = prior
        except Exception:
            atm = chain.prev_close_atm_iv if chain.prev_close_atm_iv else chain.atm_iv
            priors[ticker] = bs_prior(atm, chain.T, grid)

    import copy
    _state["priors_base"] = copy.deepcopy(priors)
    _state["priors"] = priors
    return {"status": "ok", "calibrated": list(priors.keys())}


@router.get("/prior/{ticker}", response_model=DistributionView)
def get_prior(ticker: str):
    """Get distribution view for a prior."""
    config = _state["config"]
    grid = quantile_grid(config.quantile_grid_size)
    phi = basis_functions(grid, config.M)

    prior = _state["priors"].get(ticker)
    if prior is None:
        # Fallback to BS prior from quotes
        chain = _state["quotes"].get(ticker)
        if chain is None:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")
        prior = bs_prior(chain.atm_iv, chain.T, grid)

    chain = _state["quotes"].get(ticker)
    forward = chain.forward if chain else 100.0
    T = chain.T if chain else 30 / 365

    view = compute_distribution_view(prior, grid, forward, T, config.risk_free_rate, phi=phi, market_strikes=chain.strikes if chain else None)

    # Return SVI params as "beta" for the slider UI
    svi = prior.get("_svi_params")
    if svi:
        beta = [svi.get("a", 0), svi.get("b", 0), svi.get("rho", 0),
                svi.get("m", 0), svi.get("sigma", 0.1)]
    else:
        beta = [0.0] * 5

    return DistributionView(
        moneyness=view["moneyness"],
        iv_curve=view["iv_curve"],
        cdf_x=view["cdf_x"],
        cdf_y=view["cdf_y"],
        lqd_u=view["lqd_u"],
        lqd_psi=view["lqd_psi"], fit_forward=view.get("fit_forward"),
        beta=beta,
    )


@router.post("/prior/{ticker}/override", response_model=DistributionView)
def override_prior(ticker: str, req: PriorOverrideRequest):
    """Apply beta overrides to a prior and return updated distribution view.

    Always applies relative to the BASE prior (from calibration), not the
    previously overridden one. This prevents compounding.
    """
    config = _state["config"]
    grid = quantile_grid(config.quantile_grid_size)
    phi = basis_functions(grid, config.M)

    # Always start from the immutable base prior
    base_prior = _state["priors_base"].get(ticker)
    if base_prior is None:
        base_prior = _state["priors"].get(ticker)
    if base_prior is None:
        chain = _state["quotes"].get(ticker)
        if chain is None:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")
        base_prior = bs_prior(chain.atm_iv, chain.T, grid)

    chain = _state["quotes"].get(ticker)
    forward = chain.forward if chain else 100.0
    T = chain.T if chain else 30 / 365

    beta_adj = np.array(req.beta[:config.M])

    # If all zeros, just restore the base prior
    if np.all(np.abs(beta_adj) < 1e-12):
        _state["priors"][ticker] = base_prior
        view = compute_distribution_view(base_prior, grid, forward, T, config.risk_free_rate, phi=phi, market_strikes=chain.strikes if chain else None)
        base_beta = base_prior.get("beta_fit", [0.0] * config.M)
        if hasattr(base_beta, "tolist"):
            base_beta = base_beta.tolist()
        return DistributionView(
            moneyness=view["moneyness"], iv_curve=view["iv_curve"],
            cdf_x=view["cdf_x"], cdf_y=view["cdf_y"],
            lqd_u=view["lqd_u"], lqd_psi=view["lqd_psi"], fit_forward=view.get("fit_forward"),
            beta=base_beta,
        )

    updated = apply_beta_overrides(
        base_prior, beta_adj, phi, grid, forward, T, config.risk_free_rate,
        forward * np.exp(np.linspace(-0.3, 0.3, 50)),
    )

    # Store the overridden prior for solve (but never touch priors_base)
    _state["priors"][ticker] = updated

    view = compute_distribution_view(updated, grid, forward, T, config.risk_free_rate, phi=phi, market_strikes=chain.strikes if chain else None)

    return DistributionView(
        moneyness=view["moneyness"], iv_curve=view["iv_curve"],
        cdf_x=view["cdf_x"], cdf_y=view["cdf_y"],
        lqd_u=view["lqd_u"], lqd_psi=view["lqd_psi"], fit_forward=view.get("fit_forward"),
        beta=beta_adj.tolist(),
    )


@router.post("/prior/{ticker}/svi-override", response_model=DistributionView)
def svi_override_prior(ticker: str, req: SviOverrideRequest):
    """Override the prior SVI params directly and return updated distribution view."""
    import copy
    config = _state["config"]
    grid = quantile_grid(config.quantile_grid_size)
    phi = basis_functions(grid, config.M)

    base_prior = _state["priors_base"].get(ticker) or _state["priors"].get(ticker)
    if base_prior is None:
        raise HTTPException(status_code=404, detail=f"No prior for {ticker}")

    chain = _state["quotes"].get(ticker)
    forward = chain.forward if chain else 100.0
    T = chain.T if chain else 30 / 365

    # Build new SVI params
    old_svi = base_prior.get("_svi_params", {})
    new_svi = {
        **old_svi,
        "a": req.a, "b": req.b, "rho": req.rho,
        "m": req.m, "sigma": req.sigma,
    }

    # Rebuild the prior with new SVI
    atm_iv_new = np.sqrt(max(req.a + req.b * (req.rho * (-req.m) + np.sqrt(req.m**2 + req.sigma**2)), 1e-8) / max(T, 1e-8))
    base = bs_prior(atm_iv_new, T, grid)
    updated = {
        **base,
        "beta_fit": np.zeros(config.M),
        "_bs_base": base,
        "_svi_params": new_svi,
        "_fit_strikes": base_prior.get("_fit_strikes", np.array([])),
    }

    _state["priors"][ticker] = updated

    view = compute_distribution_view(updated, grid, forward, T, config.risk_free_rate, phi=phi,
                                     market_strikes=chain.strikes if chain else None)
    return DistributionView(
        moneyness=view["moneyness"], iv_curve=view["iv_curve"],
        cdf_x=view["cdf_x"], cdf_y=view["cdf_y"],
        lqd_u=view["lqd_u"], lqd_psi=view["lqd_psi"],
        fit_forward=view.get("fit_forward"),
        beta=[req.a, req.b, req.rho, req.m, req.sigma],
    )


@router.post("/smile/{ticker}/svi-override")
def svi_override_smile(ticker: str, req: SviOverrideRequest):
    """Evaluate an SVI smile with given params and return IV curve at display strikes."""
    chain = _state["quotes"].get(ticker)
    if chain is None:
        raise HTTPException(status_code=404, detail=f"No quotes for {ticker}")

    from ..engine.svi import svi_implied_vol

    forward = chain.forward
    T = chain.T
    fit_fwd = req.a  # Not used — we use the SVI's stored forward or chain forward
    # Evaluate at full chain strikes
    k = np.log(chain.strikes / forward)
    iv_marked = svi_implied_vol(k, T, req.a, req.b, req.rho, req.m, req.sigma)
    iv_marked = np.clip(iv_marked, 0.01, 5.0)

    # Also get the prior for comparison
    prior = _state["priors"].get(ticker)
    svi_prior = prior.get("_svi_params") if prior else None
    if svi_prior:
        from ..engine.svi import svi_iv_at_strikes
        iv_prior = svi_iv_at_strikes(svi_prior, chain.strikes, forward, T)
        iv_prior = np.clip(iv_prior, 0.01, 5.0)
    else:
        iv_prior = iv_marked.copy()

    return {
        "ticker": ticker,
        "strikes": chain.strikes.tolist(),
        "iv_prior": [float(v) if np.isfinite(v) else None for v in iv_prior],
        "iv_marked": [float(v) if np.isfinite(v) else None for v in iv_marked],
        "beta": [req.a, req.b, req.rho, req.m, req.sigma],
        "is_observed": True,
    }


@router.post("/prior/{ticker}/refit", response_model=DistributionView)
def refit_prior(ticker: str, req: PriorRefitRequest):
    """Refit the prior for one asset, excluding specific prev-close quote points."""
    config = _state["config"]
    grid = quantile_grid(config.quantile_grid_size)
    phi = basis_functions(grid, config.M)

    chain = _state["quotes"].get(ticker)
    if chain is None:
        raise HTTPException(status_code=404, detail=f"No quotes for {ticker}")
    if chain.prev_close_ivs is None:
        raise HTTPException(status_code=400, detail=f"No prev close data for {ticker}")

    # Filter out excluded indices
    keep = np.ones(len(chain.strikes), dtype=bool)
    for idx in req.excluded_indices:
        if 0 <= idx < len(keep):
            keep[idx] = False

    strikes = chain.strikes[keep]
    prev_ivs = chain.prev_close_ivs[keep]

    # Append user-added synthetic prior quotes
    if req.added_quotes:
        for pt in req.added_quotes:
            if len(pt) >= 2:
                strikes = np.append(strikes, pt[0])
                prev_ivs = np.append(prev_ivs, pt[1])
        order = np.argsort(strikes)
        strikes = strikes[order]
        prev_ivs = prev_ivs[order]

    if len(strikes) < 3:
        raise HTTPException(status_code=400, detail="Too few quotes remaining after exclusion")

    prev_forward = (chain.prev_spot or chain.spot) * np.exp(config.risk_free_rate * chain.T)

    try:
        prior = fit_lqd_prior(
            strikes=strikes, mid_ivs=prev_ivs,
            forward=prev_forward, T=chain.T, r=config.risk_free_rate,
            grid=grid, phi=phi,
        )
    except Exception:
        atm = chain.prev_close_atm_iv or chain.atm_iv
        prior = bs_prior(atm, chain.T, grid)

    _state["priors_base"][ticker] = prior
    _state["priors"][ticker] = prior

    forward = chain.forward
    view = compute_distribution_view(prior, grid, forward, chain.T, config.risk_free_rate, phi=phi, market_strikes=chain.strikes if chain else None)
    beta_fit = prior.get("beta_fit", [0.0] * config.M)
    if hasattr(beta_fit, "tolist"):
        beta_fit = beta_fit.tolist()

    return DistributionView(
        moneyness=view["moneyness"],
        iv_curve=view["iv_curve"],
        cdf_x=view["cdf_x"],
        cdf_y=view["cdf_y"],
        lqd_u=view["lqd_u"],
        lqd_psi=view["lqd_psi"], fit_forward=view.get("fit_forward"),
        beta=beta_fit,
    )


# --- Phase 4: Node Distribution Views ---

@router.get("/node/{ticker}/distribution", response_model=NodeDistributionResponse)
def get_node_distribution(ticker: str):
    """Get prior and marked distribution views for a node."""
    config = _state["config"]
    grid = quantile_grid(config.quantile_grid_size)
    phi = basis_functions(grid, config.M)
    result = _state.get("solve_result")

    # Prior view
    prior = _state["priors"].get(ticker)
    chain = _state["quotes"].get(ticker)
    if prior is None and chain is not None:
        prior = bs_prior(chain.atm_iv, chain.T, grid)
    if prior is None:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")

    forward = chain.forward if chain else 100.0
    T = chain.T if chain else 30 / 365

    prior_view = compute_distribution_view(prior, grid, forward, T, config.risk_free_rate, phi=phi, market_strikes=chain.strikes if chain else None)
    prior_beta = prior.get("beta_fit", [0.0] * config.M)
    if hasattr(prior_beta, "tolist"):
        prior_beta = prior_beta.tolist()

    prior_dv = DistributionView(
        moneyness=prior_view["moneyness"],
        iv_curve=prior_view["iv_curve"],
        cdf_x=prior_view["cdf_x"],
        cdf_y=prior_view["cdf_y"],
        lqd_u=prior_view["lqd_u"],
        lqd_psi=prior_view["lqd_psi"],
        beta=prior_beta,
    )

    # Marked view (if solve has been run)
    marked_dv = None
    is_observed = True
    w2_dist = 0.0

    if result and ticker in result.nodes:
        nr = result.nodes[ticker]
        is_observed = nr.is_observed
        w2_dist = nr.wasserstein_dist

        if nr.is_observed:
            # Observed nodes: build a marked prior dict from the market SVI fit
            # so that compute_distribution_view uses SVI for IV/CDF/LQD
            from ..engine.svi import fit_svi, filter_quotes_for_fit
            mkt_strikes = chain.strikes if chain else nr.strikes
            mkt_ivs = chain.mid_ivs if chain else nr.iv_marked
            mkt_fwd = chain.forward if chain else forward

            filt_k, filt_iv, _ = filter_quotes_for_fit(mkt_strikes, mkt_ivs, mkt_fwd)
            if len(filt_k) >= 5:
                market_svi = fit_svi(filt_k, filt_iv, mkt_fwd, T)
                # Build a pseudo-prior dict with the market SVI for distribution view
                marked_prior = {
                    **prior,
                    "_svi_params": market_svi,
                }
                marked_view = compute_distribution_view(
                    marked_prior, grid, forward, T, config.risk_free_rate,
                    phi=phi, market_strikes=chain.strikes if chain else None,
                )
                marked_dv = DistributionView(
                    moneyness=marked_view["moneyness"],
                    iv_curve=marked_view["iv_curve"],
                    cdf_x=marked_view["cdf_x"],
                    cdf_y=marked_view["cdf_y"],
                    lqd_u=marked_view["lqd_u"],
                    lqd_psi=marked_view["lqd_psi"],
                    beta=nr.beta.tolist(),
                )
        elif nr.Q_marked is not None:
            # Unobserved nodes: use quantile-based reconstruction
            marked_dict = {
                "psi0": nr.Q_marked * 0 + prior["psi0"],
                "m": float(np.trapz(nr.Q_marked, grid)),
                "s": float(np.sqrt(np.trapz((nr.Q_marked - np.trapz(nr.Q_marked, grid))**2, grid))),
                "Q": nr.Q_marked,
                "Q_tilde": prior.get("Q_tilde", grid),
            }
            Q_diff = np.diff(nr.Q_marked) / np.diff(grid)
            psi_marked = np.log(np.maximum(Q_diff, 1e-30))
            psi_marked = np.append(psi_marked, psi_marked[-1])

            marked_view = compute_distribution_view(
                {**marked_dict, "psi0": psi_marked},
                grid, forward, T, config.risk_free_rate, phi=phi,
                market_strikes=chain.strikes if chain else None,
            )
            marked_dv = DistributionView(
                moneyness=marked_view["moneyness"],
                iv_curve=marked_view["iv_curve"],
                cdf_x=marked_view["cdf_x"],
                cdf_y=marked_view["cdf_y"],
                lqd_u=marked_view["lqd_u"],
                lqd_psi=marked_view["lqd_psi"],
                beta=nr.beta.tolist(),
            )

    return NodeDistributionResponse(
        prior=prior_dv,
        marked=marked_dv,
        ticker=ticker,
        is_observed=is_observed,
        wasserstein_dist=w2_dist,
    )


# --- Phase 5: Prior Save / Load ---

@router.get("/priors/saved")
def list_saved_priors_endpoint():
    """List all saved prior files."""
    return _list_saved_priors()


@router.post("/priors/save/{ticker}")
def save_prior_endpoint(ticker: str, req: SavePriorRequest = SavePriorRequest()):
    """Save the current prior for a ticker (including any modifications)."""
    prior = _state["priors"].get(ticker)
    if prior is None:
        raise HTTPException(status_code=404, detail=f"No active prior for {ticker}")

    # Get the original chain data (prev close IVs) for reconstruction on load
    chain = _state["quotes"].get(ticker)

    filename = _save_prior(
        ticker, prior,
        excluded_indices=req.excluded_indices,
        added_quotes=req.added_quotes or [],
        chain=chain,
    )
    return {"status": "ok", "filename": filename, "ticker": ticker}


@router.post("/priors/load/{ticker}")
def load_prior_endpoint(ticker: str):
    """Load a saved prior from file and set it as the active prior.

    Replaces the chain's prev_close data with the saved strikes/IVs so the
    Prior tab shows the correct quote dots (matching the saved exclusions
    and additions).
    """
    import copy, json as _json
    from pathlib import Path as _Path

    try:
        prior = _load_prior(ticker)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No saved prior for {ticker}")

    svi = prior.get("_svi_params", {})


    # Read the saved file for raw data
    filepath = _Path(__file__).parent.parent.parent / "priors" / f"{ticker}_prior.json"
    with open(filepath, "r") as f:
        saved_data = _json.load(f)
    excluded_indices = saved_data.get("excluded_indices", [])
    added_quotes = saved_data.get("added_quotes", [])
    saved_strikes = np.array(saved_data.get("strikes", []), dtype=float)
    saved_ivs = np.array(saved_data.get("ivs", []), dtype=float)
    saved_forward = saved_data.get("forward", 100.0)
    saved_T = saved_data.get("T", 30 / 365)

    # Replace the chain's prev_close data with saved data so the Prior tab
    # shows the correct dots.  This makes exclusion indices valid again.
    chain = _state["quotes"].get(ticker)
    if chain is not None and len(saved_strikes) > 0 and len(saved_ivs) == len(saved_strikes):
        chain.strikes = saved_strikes
        chain.prev_close_ivs = saved_ivs
        chain.mid_ivs = saved_ivs  # also update mid so Smile tab is consistent
        chain.prev_spot = saved_forward / np.exp(0.045 * saved_T)
        chain.prev_close_atm_iv = float(saved_ivs[np.argmin(np.abs(saved_strikes - saved_forward))])
        # Update bid-ask spreads for the new strike count
        chain.bid_ask_spread = np.full(len(saved_strikes), 0.02)
        chain.open_interest = np.full(len(saved_strikes), 100.0)
        chain.call_mids = np.zeros(len(saved_strikes))

    # Verify priors dict exists
    if not _state.get("priors"):
        _state["priors"] = {}
    if not _state.get("priors_base"):
        _state["priors_base"] = {}

    _state["priors"][ticker] = prior
    _state["priors_base"][ticker] = copy.deepcopy(prior)

    # Return the chain snapshot so frontend can update its quotes state
    chain_snapshot = None
    if chain is not None:
        chain_snapshot = _chain_to_snapshot(chain)

    return {
        "status": "ok",
        "ticker": ticker,
        "excluded_indices": excluded_indices,
        "added_quotes": added_quotes,
        "chain_snapshot": chain_snapshot.dict() if chain_snapshot else None,
    }
