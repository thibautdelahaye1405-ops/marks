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
    UniverseSelectRequest, AddTickerRequest, CatalogResponse,
    AvailableExpiriesResponse, ExpirySelectionRequest,
)
from ..data.prior_store import (
    list_saved_priors as _list_saved_priors,
    save_prior as _save_prior,
    load_prior as _load_prior,
)
from ..config import EngineConfig, DEFAULT_UNIVERSE
from ..data.referential import (
    get_universe, get_catalog, get_active_tickers,
    set_active_tickers, add_ticker as _add_ticker,
    confirm_ticker, save_selection,
)
from ..data.quotes import fetch_option_chain, fetch_available_expiries, fetch_option_chains, OptionChainData, validate_ticker
from ..data.store import save_snapshot, get_latest_snapshots
from ..utils.node_key import make_node_key, split_node_key, is_compound_key, ticker_of, group_by_ticker, unique_tickers
from ..engine.lqd import quantile_grid, basis_functions
from ..engine.prior import bs_prior, fit_lqd_prior, apply_beta_overrides, compute_distribution_view
from ..engine.graph import build_influence_matrix
from ..engine.pipeline import run_marking, NodeQuotes
from ..data.rates import fetch_treasury_curve
from ..data.dividends import fetch_dividend_info

router = APIRouter(prefix="/api")


def _chain_to_snapshot(c, node_key: str = None) -> QuoteSnapshot:
    """Convert an OptionChainData to a QuoteSnapshot response."""
    return QuoteSnapshot(
        node_key=node_key or make_node_key(c.ticker, c.expiry),
        ticker=c.ticker, expiry=c.expiry, T=c.T,
        spot=c.spot, forward=c.forward, atm_iv=c.atm_iv,
        strikes=c.strikes.tolist(), mid_ivs=c.mid_ivs.tolist(),
        bid_ask_spread=c.bid_ask_spread.tolist(),
        open_interest=c.open_interest.tolist(),
        prev_close_ivs=c.prev_close_ivs.tolist() if c.prev_close_ivs is not None else None,
        prev_close_atm_iv=c.prev_close_atm_iv,
        prev_spot=c.prev_spot,
        forward_parity=getattr(c, 'forward_parity', None),
        forward_model=getattr(c, 'forward_model', None),
        rate_used=getattr(c, 'rate_used', None),
        div_yield_used=getattr(c, 'div_yield_used', None),
        repo_rate_used=getattr(c, 'repo_rate_used', None),
    )

# In-memory state for the current session
# All quote/prior/solve dicts are keyed by node key (plain ticker or "TICKER:EXPIRY")
_state = {
    "quotes": {},       # node_key -> OptionChainData
    "W": None,          # current influence matrix (may be full tensor in multi-expiry)
    "alphas": None,     # current self-trust vector
    "config": EngineConfig(),
    "priors_base": {},  # node_key -> immutable base prior
    "priors": {},       # node_key -> active prior (base + overrides)
    "solve_result": None,  # last MarkingResult for distribution views
    "treasury_curve": None,  # TreasuryCurve from FRED
    "dividends": {},         # ticker -> DividendInfo
    "forward_overrides": {},       # node_key -> absolute forward override (current smile)
    "prev_forward_overrides": {},   # node_key -> absolute forward override (prior)
    # Multi-expiry state
    "selected_expiries": {},   # ticker -> [expiry_dates] (user selections)
    "available_expiries": {},  # ticker -> [expiry_dates] (cached from Yahoo)
    "node_keys": [],           # ordered list of active node keys
}


def _get_rate_func():
    """Return a rate function: either Treasury curve interpolation or flat override."""
    config = _state["config"]
    if config.risk_free_rate is not None:
        flat_r = config.risk_free_rate
        return lambda T: flat_r
    curve = _state.get("treasury_curve")
    if curve is None:
        curve = fetch_treasury_curve()
        _state["treasury_curve"] = curve
    return curve.rate_at


def _get_rate(T: float) -> float:
    """Get rate for a specific maturity."""
    return _get_rate_func()(T)


def _compute_forward(spot, T, ticker):
    """Compute forward with full formula: S*exp((r-q-repo)*T) - PV(divs).

    ticker can be a plain ticker or a node key ("TICKER:EXPIRY").
    """
    import math
    config = _state["config"]
    r = _get_rate(T)
    tk = ticker_of(ticker)  # extract plain ticker from node key
    div_info = _state.get("dividends", {}).get(tk)
    q = div_info.continuous_yield if div_info else 0.0
    repo = config.repo_rate_for(tk)
    pv_divs = 0.0
    if div_info and div_info.discrete_dividends and not div_info.is_index:
        from ..data.dividends import pv_discrete_dividends
        from datetime import datetime
        pv_divs = pv_discrete_dividends(
            div_info.discrete_dividends, _get_rate_func(), datetime.now()
        )
    return spot * math.exp((r - q - repo) * T) - pv_divs


def _effective_forward(ticker: str) -> float:
    """Return the forward to use: user override if set, else chain.forward."""
    ovr = _state["forward_overrides"].get(ticker)
    if ovr is not None:
        return ovr
    chain = _state["quotes"].get(ticker)
    return chain.forward if chain else 100.0


def _effective_prev_forward(ticker: str) -> float:
    """Return the prior forward: user override if set, else model from prev_spot."""
    ovr = _state["prev_forward_overrides"].get(ticker)
    if ovr is not None:
        return ovr
    chain = _state["quotes"].get(ticker)
    if chain is None:
        return 100.0
    return _compute_forward(chain.prev_spot or chain.spot, chain.T, ticker)


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


@router.get("/catalog", response_model=CatalogResponse)
def get_catalog_endpoint():
    """Return the full asset catalog and currently active tickers."""
    catalog = get_catalog()
    return CatalogResponse(
        assets=[
            Asset(
                ticker=a.ticker, name=a.name, sector=a.sector,
                is_index=a.is_index, index_weight=a.index_weight,
                liquidity_score=a.liquidity_score,
            )
            for a in catalog
        ],
        active_tickers=get_active_tickers(),
    )


@router.post("/universe/select")
def select_universe(req: UniverseSelectRequest):
    """
    Set the active universe to the given tickers.
    Rebuilds the influence matrix W for the new universe.
    """
    universe = set_active_tickers(req.tickers)
    # Rebuild W for new universe
    W, alphas = build_influence_matrix(universe)
    _state["W"] = W
    _state["alphas"] = alphas
    # Clear stale quotes for tickers no longer in universe
    active = set(t.ticker for t in universe)
    _state["quotes"] = {t: c for t, c in _state["quotes"].items() if t in active}
    tickers = [a.ticker for a in universe]
    return {
        "status": "ok",
        "tickers": tickers,
        "graph": GraphData(
            tickers=tickers,
            W=W.tolist(),
            alphas=alphas.tolist(),
            assets=[
                Asset(
                    ticker=a.ticker, name=a.name, sector=a.sector,
                    is_index=a.is_index, index_weight=a.index_weight,
                    liquidity_score=a.liquidity_score,
                )
                for a in universe
            ],
        ),
    }


@router.post("/universe/add")
def add_ticker_endpoint(req: AddTickerRequest):
    """
    Add an arbitrary ticker to the catalog and active universe.
    Validates against Yahoo Finance first — rejects if not found.
    Rebuilds the influence matrix W on success.
    """
    ticker = req.ticker.upper().strip()

    # Already in catalog — just add to active set
    from ..config import CATALOG_MAP
    if ticker in CATALOG_MAP:
        asset = _add_ticker(ticker=ticker)
    else:
        # Validate on Yahoo Finance before committing
        yahoo = validate_ticker(ticker)
        if yahoo is None:
            raise HTTPException(
                status_code=404,
                detail=f"Ticker '{ticker}' not found on Yahoo Finance",
            )
        name = req.name or yahoo["name"]
        asset = _add_ticker(
            ticker=ticker, name=name, sector=req.sector,
            is_index=req.is_index, liquidity_score=req.liquidity_score,
        )

    # Rebuild W with updated universe
    universe = get_universe()
    W, alphas = build_influence_matrix(universe)
    _state["W"] = W
    _state["alphas"] = alphas
    tickers = [a.ticker for a in universe]
    return {
        "status": "ok",
        "asset": Asset(
            ticker=asset.ticker, name=asset.name, sector=asset.sector,
            is_index=asset.is_index, index_weight=asset.index_weight,
            liquidity_score=asset.liquidity_score,
        ),
        "tickers": tickers,
    }


@router.post("/universe/save")
def save_universe_selection():
    """Persist the current active universe selection to disk."""
    tickers = get_active_tickers()
    save_selection(tickers)
    return {"status": "ok", "tickers": tickers}


@router.get("/expiries/{ticker}", response_model=AvailableExpiriesResponse)
def get_expiries(ticker: str):
    """Get all available option expiry dates for a ticker."""
    from datetime import datetime
    expiries = fetch_available_expiries(ticker)
    T_values = [max((datetime.strptime(e, "%Y-%m-%d") - datetime.now()).days / 365.0, 1/365) for e in expiries]
    _state["available_expiries"][ticker] = expiries
    return AvailableExpiriesResponse(ticker=ticker, expiries=expiries, T_values=T_values)


@router.post("/expiry-selection")
def set_expiry_selection(req: ExpirySelectionRequest):
    """Set which expiries to fetch for each ticker."""
    _state["selected_expiries"] = req.selections
    return {"status": "ok", "selections": req.selections}


@router.get("/expiry-selection")
def get_expiry_selection():
    """Get current expiry selections."""
    return {"selections": _state["selected_expiries"]}


@router.post("/fetch-quotes", response_model=Dict[str, QuoteSnapshot])
def fetch_quotes_endpoint():
    """Fetch fresh option chains from Yahoo Finance for the entire universe."""
    universe = get_universe()
    config = _state["config"]

    # Fetch Treasury curve
    curve = fetch_treasury_curve()
    _state["treasury_curve"] = curve
    rate_func = _get_rate_func()

    # Fetch dividend info for all tickers
    asset_map = {a.ticker: a for a in universe}
    dividends = {}
    for asset in universe:
        dividends[asset.ticker] = fetch_dividend_info(asset.ticker, is_index=asset.is_index)
    _state["dividends"] = dividends

    selected_expiries = _state.get("selected_expiries", {})
    multi_expiry = len(selected_expiries) > 0

    # Clear old quotes to avoid key format mixing
    _state["quotes"] = {}
    _state["priors"] = {}
    _state["priors_base"] = {}

    results = {}
    for asset in universe:
        if multi_expiry and asset.ticker in selected_expiries:
            # Multi-expiry: fetch each selected expiry
            chains = fetch_option_chains(
                asset.ticker, selected_expiries[asset.ticker],
                rate_func=rate_func,
                dividend_info=dividends.get(asset.ticker),
                repo_rate=config.repo_rate_for(asset.ticker),
            )
            for nk, chain in chains.items():
                _state["quotes"][nk] = chain
                save_snapshot(chain)
                results[nk] = _chain_to_snapshot(chain, node_key=nk)
                confirm_ticker(asset.ticker)
        else:
            # Single-expiry (default)
            chain = fetch_option_chain(
                asset.ticker,
                target_maturity_days=config.target_maturity_days,
                rate_func=rate_func,
                dividend_info=dividends.get(asset.ticker),
                repo_rate=config.repo_rate_for(asset.ticker),
            )
            if chain is not None:
                nk = make_node_key(asset.ticker, chain.expiry) if multi_expiry else asset.ticker
                _state["quotes"][nk] = chain
                save_snapshot(chain)
                results[nk] = _chain_to_snapshot(chain, node_key=nk)
                confirm_ticker(asset.ticker)

    if not results:
        raise HTTPException(status_code=503, detail="Could not fetch any quotes")

    # Update ordered node keys
    _state["node_keys"] = list(results.keys())

    return results


@router.get("/quotes/latest", response_model=Dict[str, QuoteSnapshot])
def get_latest_quotes():
    """Return the latest stored quotes."""
    if _state["quotes"]:
        return {nk: _chain_to_snapshot(c, node_key=nk) for nk, c in _state["quotes"].items()}

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


def _fit_single_ticker(ticker: str, chain, prior, req: FitRequest, is_observed: bool) -> SmileData:
    """Fit a single ticker's smile. Shared logic for /fit and /fit/{ticker}."""
    from ..engine.svi import fit_svi, svi_iv_at_strikes, filter_quotes_for_fit

    smile_model = req.smile_model if hasattr(req, 'smile_model') else "svi"

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

    if is_observed:
        # Observed: SVI fit to current market quotes (with exclusions/additions)
        fit_strikes = chain.strikes.copy()
        fit_ivs = chain.mid_ivs.copy()
        fit_spread = chain.bid_ask_spread.copy()

        # Apply exclusions
        if req.excluded_quotes and ticker in req.excluded_quotes:
            keep = np.ones(len(fit_strikes), dtype=bool)
            for idx in req.excluded_quotes[ticker]:
                if 0 <= idx < len(keep):
                    keep[idx] = False
            fit_strikes = fit_strikes[keep]
            fit_ivs = fit_ivs[keep]
            fit_spread = fit_spread[keep]

        # Apply additions
        if req.added_quotes and ticker in req.added_quotes:
            med_spread = np.median(fit_spread) if len(fit_spread) > 0 else 0.01
            for pt in req.added_quotes[ticker]:
                if len(pt) >= 2:
                    fit_strikes = np.append(fit_strikes, pt[0])
                    fit_ivs = np.append(fit_ivs, pt[1])
                    fit_spread = np.append(fit_spread, med_spread)
            order = np.argsort(fit_strikes)
            fit_strikes = fit_strikes[order]
            fit_ivs = fit_ivs[order]
            fit_spread = fit_spread[order]

        # Prior params for anchoring
        prior_params = None
        if svi_prior and req.lambda_prior > 0:
            prior_params = np.array([svi_prior["a"], svi_prior["b"], svi_prior["rho"],
                                     svi_prior["m"], svi_prior["sigma"]])

        filt_k, filt_iv, filt_mask = filter_quotes_for_fit(fit_strikes, fit_ivs, forward)
        filt_spread = fit_spread[filt_mask] if len(fit_spread) == len(fit_strikes) else None
        if len(filt_k) >= 5:
            if smile_model == "svi":
                market_svi = fit_svi(filt_k, filt_iv, forward, T,
                                     bid_ask_spread=filt_spread,
                                     use_bid_ask_fit=req.use_bid_ask_fit,
                                     prior_params=prior_params,
                                     lambda_prior=req.lambda_prior)
                iv_marked = svi_iv_at_strikes(market_svi, strikes, forward, T)
                iv_marked = np.clip(iv_marked, 0.01, 5.0)
                from ..engine.svi import raw_svi_to_jw_normalized
                jw = raw_svi_to_jw_normalized(market_svi["a"], market_svi["b"], market_svi["rho"],
                                              market_svi["m"], market_svi["sigma"], T)
                svi_beta = [jw["v"], jw["psi_hat"], jw["p_hat"], jw["c_hat"], jw["vt_ratio"]]
            elif smile_model == "lqd":
                from ..engine.lqd_model import fit_lqd_model, lqd_iv_at_strikes
                prior_theta = None
                if prior and req.lambda_prior > 0:
                    prior_theta = prior.get("_lqd_theta")
                    if prior_theta is not None:
                        prior_theta = np.array(prior_theta)
                # filt_k from filter_quotes_for_fit is actually filtered strikes
                lqd_result = fit_lqd_model(filt_k, filt_iv, forward, T,
                                           bid_ask_spread=filt_spread,
                                           use_bid_ask_fit=req.use_bid_ask_fit,
                                           prior_theta=prior_theta,
                                           lambda_prior=req.lambda_prior,
                                           atm_iv=chain.atm_iv)
                iv_marked = lqd_iv_at_strikes(lqd_result["theta"], strikes, forward, T, lqd_result["alpha"])
                iv_marked = np.clip(np.where(np.isfinite(iv_marked), iv_marked, chain.atm_iv), 0.01, 5.0)
                svi_beta = lqd_result["theta"].tolist()
                # pad to match expected length if needed
                if len(svi_beta) < 6:
                    svi_beta.extend([0.0] * (6 - len(svi_beta)))
            elif smile_model == "sigmoid":
                from ..engine.sigmoid import fit_sigmoid_model, sigmoid_iv_at_strikes
                prior_sig = None
                if prior and req.lambda_prior > 0:
                    prior_sig = prior.get("_sigmoid_params")
                    if prior_sig is not None:
                        prior_sig = np.array(prior_sig)
                # filt_k from filter_quotes_for_fit is actually filtered strikes
                sig_result = fit_sigmoid_model(filt_k, filt_iv, forward, T,
                                               bid_ask_spread=filt_spread,
                                               use_bid_ask_fit=req.use_bid_ask_fit,
                                               prior_params=prior_sig,
                                               lambda_prior=req.lambda_prior,
                                               atm_iv=chain.atm_iv)
                sig_ref = sig_result["sigma_ref"]
                iv_marked = sigmoid_iv_at_strikes(sig_result["params"], strikes, forward, T, sig_ref)
                iv_marked = np.clip(np.where(np.isfinite(iv_marked), iv_marked, chain.atm_iv), 0.01, 5.0)
                svi_beta = sig_result["params"].tolist()
        else:
            iv_marked = iv_prior.copy()
            svi_beta = [0.04, 0.0, 0.5, 0.5, 1.0]
    else:
        # Unobserved: show prior as current smile
        iv_marked = iv_prior.copy()
        jw_prior = prior.get("_jw_params") if prior else None
        if jw_prior:
            svi_beta = [jw_prior["v"], jw_prior["psi_hat"], jw_prior["p_hat"], jw_prior["c_hat"], jw_prior["vt_ratio"]]
        elif svi_prior:
            from ..engine.svi import raw_svi_to_jw_normalized
            jw = raw_svi_to_jw_normalized(svi_prior["a"], svi_prior["b"], svi_prior["rho"],
                                          svi_prior["m"], svi_prior["sigma"], svi_prior.get("T", T))
            svi_beta = [jw["v"], jw["psi_hat"], jw["p_hat"], jw["c_hat"], jw["vt_ratio"]]
        else:
            svi_beta = [0.04, 0.0, 0.5, 0.5, 1.0]

    iv_prior_list = [float(v) if np.isfinite(v) else None for v in iv_prior]
    iv_marked_list = [float(v) if np.isfinite(v) else None for v in iv_marked]

    return SmileData(
        ticker=ticker,
        strikes=strikes.tolist(),
        iv_prior=iv_prior_list,
        iv_marked=iv_marked_list,
        beta=svi_beta,
        is_observed=is_observed,
    )


@router.post("/fit")
def fit_endpoint(req: FitRequest):
    """Lightweight fit: SVI for observed assets, prior for unobserved.

    No graph propagation — just direct SVI fits from market quotes for
    observed assets, and prior curves for unobserved. Used for pre-propagation
    display.
    """
    if not _state["quotes"]:
        raise HTTPException(status_code=400, detail="No quotes loaded.")

    universe = get_universe()
    tickers = [a.ticker for a in universe]
    observed_set = set(req.observed_tickers) if req.observed_tickers else set(_state["quotes"].keys())

    nodes_resp = {}
    for ticker in tickers:
        chain = _state["quotes"].get(ticker)
        if chain is None:
            continue

        prior = _state["priors"].get(ticker) if _state.get("priors") else None
        is_observed = ticker in observed_set
        nodes_resp[ticker] = _fit_single_ticker(ticker, chain, prior, req, is_observed)

    return {
        "nodes": {t: n.dict() for t, n in nodes_resp.items()},
        "tickers": list(nodes_resp.keys()),
    }


@router.post("/fit/{ticker}")
def fit_single(ticker: str, req: FitRequest):
    """Fit a single asset's smile and return SmileData."""
    chain = _state["quotes"].get(ticker)
    if chain is None:
        raise HTTPException(status_code=404, detail=f"No quotes loaded for {ticker}")

    prior = _state["priors"].get(ticker) if _state.get("priors") else None
    observed_set = set(req.observed_tickers) if req.observed_tickers else set(_state["quotes"].keys())
    is_observed = ticker in observed_set

    smile_data = _fit_single_ticker(ticker, chain, prior, req, is_observed)

    # Update solve result for this ticker if it exists
    if _state["solve_result"] is not None and hasattr(_state["solve_result"], "nodes"):
        solve_nodes = _state["solve_result"].nodes
        if ticker in solve_nodes:
            solve_nodes[ticker] = smile_data

    return smile_data.dict()


@router.post("/solve", response_model=SolveResponse)
def solve_endpoint(req: SolveRequest):
    """Run the marking engine with current quotes and user overrides."""
    if not _state["quotes"]:
        raise HTTPException(status_code=400, detail="No quotes loaded. Fetch first.")

    universe = get_universe()
    config = _state["config"]
    config.lambda_ = req.lambda_
    config.eta = req.eta
    config.lambda_prior = req.lambda_prior
    config.use_bid_ask_fit = req.use_bid_ask_fit
    config.smile_model = req.smile_model

    # Auto-calibrate priors if not yet done
    if not _state["priors"]:
        grid = quantile_grid(config.quantile_grid_size)
        phi = basis_functions(grid, config.M)
        import copy
        priors = {}
        for ticker, chain in _state["quotes"].items():
            try:
                if chain.prev_close_ivs is not None and chain.prev_spot is not None:
                    prev_forward = _effective_prev_forward(ticker)
                    priors[ticker] = fit_lqd_prior(
                        chain.strikes, chain.prev_close_ivs, prev_forward,
                        chain.T, _get_rate(chain.T), grid, phi,
                    )
                else:
                    priors[ticker] = fit_lqd_prior(
                        chain.strikes, chain.mid_ivs, chain.forward,
                        chain.T, _get_rate(chain.T), grid, phi,
                    )
            except Exception:
                priors[ticker] = bs_prior(chain.atm_iv, chain.T, grid)
        _state["priors_base"] = copy.deepcopy(priors)
        _state["priors"] = priors

    # Determine node keys and observed status
    # observed_tickers from frontend may be plain tickers or compound keys
    req_observed = set(req.observed_tickers) if req.observed_tickers else None
    all_quote_keys = list(_state["quotes"].keys())
    multi_expiry = any(is_compound_key(k) for k in all_quote_keys)

    # Resolve observed set: if frontend sends plain tickers, expand to all
    # compound keys for that ticker
    if req_observed is not None:
        observed_set = set()
        for key in all_quote_keys:
            tk = ticker_of(key)
            if key in req_observed or tk in req_observed:
                observed_set.add(key)
    else:
        observed_set = set(all_quote_keys)

    # Convert OptionChainData → NodeQuotes for observed nodes
    node_quotes = {}
    for key, chain in _state["quotes"].items():
        if key not in observed_set:
            continue

        strikes = chain.strikes.copy()
        mid_ivs = chain.mid_ivs.copy()
        spread = chain.bid_ask_spread.copy()

        # Exclusions/additions: check both compound key and plain ticker
        tk = ticker_of(key)
        excl_key = key if (req.excluded_quotes and key in req.excluded_quotes) else tk
        if req.excluded_quotes and excl_key in req.excluded_quotes:
            keep = np.ones(len(strikes), dtype=bool)
            for idx in req.excluded_quotes[excl_key]:
                if 0 <= idx < len(keep):
                    keep[idx] = False
            strikes = strikes[keep]
            mid_ivs = mid_ivs[keep]
            spread = spread[keep]

        add_key = key if (req.added_quotes and key in req.added_quotes) else tk
        if req.added_quotes and add_key in req.added_quotes:
            for pt in req.added_quotes[add_key]:
                if len(pt) >= 2:
                    strikes = np.append(strikes, pt[0])
                    mid_ivs = np.append(mid_ivs, pt[1])
                    spread = np.append(spread, np.median(spread) if len(spread) > 0 else 0.01)
            order = np.argsort(strikes)
            strikes, mid_ivs, spread = strikes[order], mid_ivs[order], spread[order]

        if len(strikes) < 3:
            continue

        node_quotes[key] = NodeQuotes(
            ticker=key,
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

    # Resolve priors: ensure keys match quote keys
    priors_resolved = {}
    if _state["priors"]:
        for key in all_quote_keys:
            tk = ticker_of(key)
            if key in _state["priors"]:
                priors_resolved[key] = _state["priors"][key]
            elif tk in _state["priors"]:
                priors_resolved[key] = _state["priors"][tk]

    if multi_expiry:
        # Build node_keys in ticker-major order with uniform maturity grid
        from ..engine.graph import build_time_kernel, build_full_tensor
        lambda_T = getattr(req, 'lambda_T', config.lambda_T)
        config.lambda_T = lambda_T

        # Collect all unique T values and build a uniform grid
        T_set = {}
        for key in all_quote_keys:
            chain = _state["quotes"][key]
            T_set[chain.expiry] = chain.T
        unique_expiries = sorted(T_set.keys(), key=lambda e: T_set[e])
        T_values = np.array([T_set[e] for e in unique_expiries])

        # Build node_keys: ticker-major order (all expiries for ticker 0, then ticker 1, ...)
        tickers_ordered = []
        for a in universe:
            if any(ticker_of(k) == a.ticker for k in all_quote_keys):
                tickers_ordered.append(a.ticker)

        # For the Kronecker product, every ticker needs every expiry.
        # Create node keys for the full grid, using actual data where available.
        all_node_keys = []
        for tk in tickers_ordered:
            for exp in unique_expiries:
                from ..utils.node_key import make_node_key
                nk = make_node_key(tk, exp)
                all_node_keys.append(nk)
                # If this node doesn't have a quote, clone from the closest available
                if nk not in _state["quotes"]:
                    # Find closest available expiry for this ticker
                    tk_keys = [k for k in all_quote_keys if ticker_of(k) == tk]
                    if tk_keys:
                        closest = tk_keys[0]  # use first available as placeholder
                        node_quotes.pop(nk, None)  # ensure not in observed
                        if nk not in priors_resolved and closest in priors_resolved:
                            priors_resolved[nk] = priors_resolved[closest]

        N_tickers = len(tickers_ordered)
        N_mats = len(unique_expiries)

        if W_override is None and N_mats > 1:
            W_asset, alphas_asset = build_influence_matrix(universe)
            # W_asset is for full universe, but we only have some tickers with quotes
            # Build a sub-matrix for just the tickers that have quotes
            asset_indices = [i for i, a in enumerate(universe) if a.ticker in tickers_ordered]
            W_sub = W_asset[np.ix_(asset_indices, asset_indices)]
            alphas_sub = np.array([1.0 - W_sub[i].sum() for i in range(len(asset_indices))])

            K_time = build_time_kernel(T_values, lambda_T)
            W_override, _ = build_full_tensor(W_sub, K_time, alphas_sub)
        elif W_override is None:
            W_asset, _ = build_influence_matrix(universe)
            asset_indices = [i for i, a in enumerate(universe) if a.ticker in tickers_ordered]
            W_override = W_asset[np.ix_(asset_indices, asset_indices)]
    else:
        all_node_keys = []

    # Full chain data for display
    full_chains = dict(_state["quotes"])

    try:
        result = run_marking(
            assets=universe,
            quotes=node_quotes,
            config=config,
            W_override=W_override,
            alpha_overrides=req.alpha_overrides,
            shock_nudges=req.shock_nudges,
            calibrated_priors=priors_resolved if priors_resolved else None,
            full_chains=full_chains,
            node_keys=all_node_keys if multi_expiry else None,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Marking engine error: {e}")

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


@router.get("/time-kernel")
def get_time_kernel(lambda_T: float = None):
    """Return the cross-maturity influence kernel K_time.

    Uses T values from loaded quotes. Returns the kernel matrix,
    maturity labels, and T values.
    """
    from ..engine.graph import build_time_kernel

    config = _state["config"]
    lt = lambda_T if lambda_T is not None else config.lambda_T

    # Collect unique T values from quotes
    T_map = {}  # expiry -> T
    for key, chain in _state["quotes"].items():
        T_map[chain.expiry] = chain.T

    if len(T_map) <= 1:
        # Single maturity: return trivial 1x1 kernel
        expiries = list(T_map.keys()) or [""]
        T_values = list(T_map.values()) or [0.08]
        return {
            "K": [[1.0]],
            "expiries": expiries,
            "T_values": T_values,
            "labels": [f"{int(t*365)}d" for t in T_values],
            "lambda_T": lt,
        }

    # Sort by T
    sorted_items = sorted(T_map.items(), key=lambda x: x[1])
    expiries = [e for e, _ in sorted_items]
    T_values = [t for _, t in sorted_items]
    T_arr = np.array(T_values)

    K = build_time_kernel(T_arr, lt)

    return {
        "K": K.tolist(),
        "expiries": expiries,
        "T_values": T_values,
        "labels": [f"{int(t*365)}d" for t in T_values],
        "lambda_T": lt,
    }


@router.get("/graph", response_model=GraphData)
def get_graph():
    """Return the current influence graph.

    In multi-expiry mode, returns the full tensor with node keys.
    In single-maturity mode, returns the asset-only graph with ticker keys.
    """
    universe = get_universe()
    all_quote_keys = list(_state.get("quotes", {}).keys())
    multi_expiry = any(is_compound_key(k) for k in all_quote_keys)

    if _state["W"] is None:
        W, alphas = build_influence_matrix(universe)
        _state["W"] = W
        _state["alphas"] = alphas

    graph_tickers = all_quote_keys if multi_expiry else [a.ticker for a in universe]

    return GraphData(
        tickers=graph_tickers,
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

def _calibrate_single_prior(ticker: str, chain, config, grid, phi, smile_model: str = "svi") -> dict:
    """Calibrate a prior for a single ticker. Returns the prior dict."""
    from ..engine.svi import fit_svi, filter_quotes_for_fit

    try:
        # Use previous close IVs if available
        if chain.prev_close_ivs is not None and chain.prev_spot is not None:
            prev_forward = _effective_prev_forward(ticker)
            prior = fit_lqd_prior(
                strikes=chain.strikes,
                mid_ivs=chain.prev_close_ivs,
                forward=prev_forward,
                T=chain.T,
                r=_get_rate(chain.T),
                grid=grid, phi=phi,
            )
        else:
            # Fallback: use current IVs
            prior = fit_lqd_prior(
                strikes=chain.strikes, mid_ivs=chain.mid_ivs,
                forward=chain.forward, T=chain.T, r=_get_rate(chain.T),
                grid=grid, phi=phi,
            )
    except Exception:
        atm = chain.prev_close_atm_iv if chain.prev_close_atm_iv else chain.atm_iv
        prior = bs_prior(atm, chain.T, grid)

    # Overlay model-specific params when LQD model is selected
    if smile_model == "lqd":
        try:
            from ..engine.lqd_model import fit_lqd_model

            ivs = chain.prev_close_ivs if chain.prev_close_ivs is not None else chain.mid_ivs
            fwd = _effective_prev_forward(ticker) if chain.prev_close_ivs is not None else chain.forward

            # Fit LQD model
            lqd_result = fit_lqd_model(
                chain.strikes, ivs, fwd, chain.T,
                atm_iv=float(ivs[np.argmin(np.abs(np.log(chain.strikes / fwd)))]),
            )
            prior["_lqd_theta"] = lqd_result["theta"].tolist()
            prior["_lqd_alpha"] = lqd_result["alpha"]

            # Also fit SVI for prior IV curve evaluation
            filt_k, filt_iv, _ = filter_quotes_for_fit(chain.strikes, ivs, fwd)
            if len(filt_k) >= 5:
                svi_result = fit_svi(filt_k, filt_iv, fwd, chain.T)
                prior["_svi_params"] = svi_result
        except Exception:
            pass  # LQD overlay failed; base prior remains valid

    elif smile_model == "sigmoid":
        try:
            from ..engine.sigmoid import fit_sigmoid_model

            ivs = chain.prev_close_ivs if chain.prev_close_ivs is not None else chain.mid_ivs
            fwd = _effective_prev_forward(ticker) if chain.prev_close_ivs is not None else chain.forward
            atm_idx = np.argmin(np.abs(np.log(chain.strikes / fwd)))

            sig_result = fit_sigmoid_model(
                chain.strikes, ivs, fwd, chain.T,
                atm_iv=float(ivs[atm_idx]),
            )
            prior["_sigmoid_params"] = sig_result["params"].tolist()
            prior["_sigmoid_sigma_ref"] = sig_result["sigma_ref"]

            # Also fit SVI for fallback IV evaluation
            filt_k, filt_iv, _ = filter_quotes_for_fit(chain.strikes, ivs, fwd)
            if len(filt_k) >= 5:
                svi_result = fit_svi(filt_k, filt_iv, fwd, chain.T)
                prior["_svi_params"] = svi_result
        except Exception:
            pass

    return prior


@router.post("/calibrate-priors")
def calibrate_priors(smile_model: str = "svi"):
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
        priors[ticker] = _calibrate_single_prior(ticker, chain, config, grid, phi, smile_model)

    import copy
    _state["priors_base"] = copy.deepcopy(priors)
    _state["priors"] = priors
    return {"status": "ok", "calibrated": list(priors.keys())}


@router.post("/calibrate-prior/{ticker}")
def calibrate_single_prior(ticker: str, smile_model: str = "svi"):
    """Calibrate the prior for a single ticker."""
    chain = _state["quotes"].get(ticker)
    if chain is None:
        raise HTTPException(status_code=404, detail=f"No quotes loaded for {ticker}")

    config = _state["config"]
    grid = quantile_grid(config.quantile_grid_size)
    phi = basis_functions(grid, config.M)

    import copy
    prior = _calibrate_single_prior(ticker, chain, config, grid, phi, smile_model)
    _state["priors"][ticker] = prior
    _state["priors_base"][ticker] = copy.deepcopy(prior)

    return {"status": "ok", "ticker": ticker}


@router.get("/prior/{ticker}", response_model=DistributionView)
def get_prior(ticker: str, smile_model: str = None):
    """Get distribution view for a prior."""
    config = _state["config"]
    effective_model = smile_model or config.smile_model
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

    view = compute_distribution_view(prior, grid, forward, T, _get_rate(T), phi=phi, market_strikes=chain.strikes if chain else None)

    # Return model params as "beta" for the slider UI
    sigmoid_params = prior.get("_sigmoid_params")
    lqd_theta = prior.get("_lqd_theta")
    if sigmoid_params is not None and effective_model == "sigmoid":
        beta = list(sigmoid_params)
    elif lqd_theta is not None and effective_model == "lqd":
        beta = list(lqd_theta)
    else:
        jw = prior.get("_jw_params")
        if jw:
            beta = [jw["v"], jw["psi_hat"], jw["p_hat"], jw["c_hat"], jw["vt_ratio"]]
        else:
            svi = prior.get("_svi_params")
            if svi:
                from ..engine.svi import raw_svi_to_jw_normalized
                chain_T = chain.T if chain else 30 / 365
                jw = raw_svi_to_jw_normalized(svi["a"], svi["b"], svi["rho"], svi["m"], svi["sigma"], chain_T)
                beta = [jw["v"], jw["psi_hat"], jw["p_hat"], jw["c_hat"], jw["vt_ratio"]]
            else:
                beta = [0.04, 0.0, 0.5, 0.5, 1.0]

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
        view = compute_distribution_view(base_prior, grid, forward, T, _get_rate(T), phi=phi, market_strikes=chain.strikes if chain else None)
        jw = base_prior.get("_jw_params")
        if jw:
            base_beta = [jw["v"], jw["psi_hat"], jw["p_hat"], jw["c_hat"], jw["vt_ratio"]]
        else:
            base_beta = [0.04, 0.0, 0.5, 0.5, 1.0]
        return DistributionView(
            moneyness=view["moneyness"], iv_curve=view["iv_curve"],
            cdf_x=view["cdf_x"], cdf_y=view["cdf_y"],
            lqd_u=view["lqd_u"], lqd_psi=view["lqd_psi"], fit_forward=view.get("fit_forward"),
            beta=base_beta,
        )

    updated = apply_beta_overrides(
        base_prior, beta_adj, phi, grid, forward, T, _get_rate(T),
        forward * np.exp(np.linspace(-0.3, 0.3, 50)),
    )

    # Store the overridden prior for solve (but never touch priors_base)
    _state["priors"][ticker] = updated

    view = compute_distribution_view(updated, grid, forward, T, _get_rate(T), phi=phi, market_strikes=chain.strikes if chain else None)

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

    from ..engine.svi import jw_normalized_to_raw_svi, raw_svi_to_jw_normalized

    # Convert JW → raw SVI for evaluation
    raw = jw_normalized_to_raw_svi(req.v, req.vt_ratio, req.psi_hat, req.p_hat, req.c_hat, T)

    old_svi = base_prior.get("_svi_params", {})
    new_svi = {**old_svi, **raw}

    # Rebuild the prior with new SVI
    from ..engine.svi import svi_total_variance
    w_atm = svi_total_variance(np.array([0.0]), raw["a"], raw["b"], raw["rho"], raw["m"], raw["sigma"])[0]
    atm_iv_new = np.sqrt(max(w_atm, 1e-8) / max(T, 1e-8))
    base = bs_prior(atm_iv_new, T, grid)

    jw_params = {"v": req.v, "psi_hat": req.psi_hat, "p_hat": req.p_hat, "c_hat": req.c_hat, "vt_ratio": req.vt_ratio}

    updated = {
        **base,
        "beta_fit": np.zeros(config.M),
        "_bs_base": base,
        "_svi_params": new_svi,
        "_jw_params": jw_params,
        "_fit_strikes": base_prior.get("_fit_strikes", np.array([])),
    }

    _state["priors"][ticker] = updated

    view = compute_distribution_view(updated, grid, forward, T, _get_rate(T), phi=phi,
                                     market_strikes=chain.strikes if chain else None)
    return DistributionView(
        moneyness=view["moneyness"], iv_curve=view["iv_curve"],
        cdf_x=view["cdf_x"], cdf_y=view["cdf_y"],
        lqd_u=view["lqd_u"], lqd_psi=view["lqd_psi"],
        fit_forward=view.get("fit_forward"),
        beta=[req.v, req.psi_hat, req.p_hat, req.c_hat, req.vt_ratio],
    )


@router.post("/prior/{ticker}/lqd-override", response_model=DistributionView)
def lqd_override_prior(ticker: str, req: dict):
    """Override the prior LQD theta and return updated distribution view."""
    config = _state["config"]
    grid = quantile_grid(config.quantile_grid_size)
    phi = basis_functions(grid, config.M)

    base_prior = _state["priors_base"].get(ticker) or _state["priors"].get(ticker)
    if base_prior is None:
        raise HTTPException(status_code=404, detail=f"No prior for {ticker}")

    chain = _state["quotes"].get(ticker)
    forward = chain.forward if chain else 100.0
    T = chain.T if chain else 30 / 365

    theta = np.array(req.get("theta", [0.0] * 6), dtype=float)
    alpha = base_prior.get("_lqd_alpha", chain.atm_iv * np.sqrt(max(T, 1e-8)) if chain else 0.125)

    # Update the prior with new LQD params
    updated = {**base_prior, "_lqd_theta": theta.tolist(), "_lqd_alpha": float(alpha)}
    _state["priors"][ticker] = updated

    view = compute_distribution_view(updated, grid, forward, T, _get_rate(T), phi=phi,
                                     market_strikes=chain.strikes if chain else None)
    return DistributionView(
        moneyness=view["moneyness"], iv_curve=view["iv_curve"],
        cdf_x=view["cdf_x"], cdf_y=view["cdf_y"],
        lqd_u=view["lqd_u"], lqd_psi=view["lqd_psi"],
        fit_forward=view.get("fit_forward"),
        beta=theta.tolist(),
    )


@router.post("/smile/{ticker}/svi-override")
def svi_override_smile(ticker: str, req: SviOverrideRequest):
    """Evaluate a smile with given JW-normalized params at display strikes."""
    chain = _state["quotes"].get(ticker)
    if chain is None:
        raise HTTPException(status_code=404, detail=f"No quotes for {ticker}")

    from ..engine.svi import jw_normalized_to_raw_svi, svi_implied_vol, svi_iv_at_strikes

    forward = chain.forward
    T = chain.T

    # Convert JW → raw SVI
    raw = jw_normalized_to_raw_svi(req.v, req.vt_ratio, req.psi_hat, req.p_hat, req.c_hat, T)

    k = np.log(chain.strikes / forward)
    iv_marked = svi_implied_vol(k, T, raw["a"], raw["b"], raw["rho"], raw["m"], raw["sigma"])
    iv_marked = np.clip(iv_marked, 0.01, 5.0)

    # Prior for comparison
    prior = _state["priors"].get(ticker)
    svi_prior = prior.get("_svi_params") if prior else None
    if svi_prior:
        iv_prior = svi_iv_at_strikes(svi_prior, chain.strikes, forward, T)
        iv_prior = np.clip(iv_prior, 0.01, 5.0)
    else:
        iv_prior = iv_marked.copy()

    return {
        "ticker": ticker,
        "strikes": chain.strikes.tolist(),
        "iv_prior": [float(v) if np.isfinite(v) else None for v in iv_prior],
        "iv_marked": [float(v) if np.isfinite(v) else None for v in iv_marked],
        "beta": [req.v, req.psi_hat, req.p_hat, req.c_hat, req.vt_ratio],
        "is_observed": True,
    }


@router.post("/smile/{ticker}/lqd-override")
def lqd_override_smile(ticker: str, req: dict):
    """Evaluate a smile with given LQD theta parameters at display strikes."""
    chain = _state["quotes"].get(ticker)
    if chain is None:
        raise HTTPException(status_code=404, detail=f"No quotes for {ticker}")

    from ..engine.lqd_model import lqd_iv_at_strikes

    theta = np.array(req.get("theta", [0.0] * 6), dtype=float)
    forward = chain.forward
    T = chain.T
    alpha = chain.atm_iv * np.sqrt(max(T, 1e-8))

    iv_marked = lqd_iv_at_strikes(theta, chain.strikes, forward, T, alpha)
    iv_marked = np.where(np.isfinite(iv_marked), iv_marked, chain.atm_iv)
    iv_marked = np.clip(iv_marked, 0.01, 5.0)

    # Prior for comparison
    prior = _state["priors"].get(ticker)
    svi_prior = prior.get("_svi_params") if prior else None
    if svi_prior:
        from ..engine.svi import svi_iv_at_strikes as svi_eval
        iv_prior = svi_eval(svi_prior, chain.strikes, forward, T)
        iv_prior = np.clip(iv_prior, 0.01, 5.0)
    else:
        iv_prior = iv_marked.copy()

    return {
        "ticker": ticker,
        "strikes": chain.strikes.tolist(),
        "iv_prior": [float(v) if np.isfinite(v) else None for v in iv_prior],
        "iv_marked": [float(v) if np.isfinite(v) else None for v in iv_marked],
        "beta": theta.tolist(),
        "is_observed": True,
    }


@router.post("/prior/{ticker}/sigmoid-override", response_model=DistributionView)
def sigmoid_override_prior(ticker: str, req: dict):
    """Override the prior Sigmoid params and return updated distribution view."""
    config = _state["config"]
    grid = quantile_grid(config.quantile_grid_size)
    phi = basis_functions(grid, config.M)

    base_prior = _state["priors_base"].get(ticker) or _state["priors"].get(ticker)
    if base_prior is None:
        raise HTTPException(status_code=404, detail=f"No prior for {ticker}")

    chain = _state["quotes"].get(ticker)
    forward = chain.forward if chain else 100.0
    T = chain.T if chain else 30 / 365

    params = np.array(req.get("params", [0.25, 0.0, 0.01, 0.02, 0.02, 0.20]), dtype=float)
    sigma_ref = base_prior.get("_sigmoid_sigma_ref", chain.atm_iv if chain else 0.25)

    updated = {**base_prior, "_sigmoid_params": params.tolist(), "_sigmoid_sigma_ref": float(sigma_ref)}
    _state["priors"][ticker] = updated

    view = compute_distribution_view(updated, grid, forward, T, _get_rate(T), phi=phi,
                                     market_strikes=chain.strikes if chain else None)
    return DistributionView(
        moneyness=view["moneyness"], iv_curve=view["iv_curve"],
        cdf_x=view["cdf_x"], cdf_y=view["cdf_y"],
        lqd_u=view["lqd_u"], lqd_psi=view["lqd_psi"],
        fit_forward=view.get("fit_forward"),
        beta=params.tolist(),
    )


@router.post("/smile/{ticker}/sigmoid-override")
def sigmoid_override_smile(ticker: str, req: dict):
    """Evaluate a smile with given Sigmoid trader params at display strikes."""
    chain = _state["quotes"].get(ticker)
    if chain is None:
        raise HTTPException(status_code=404, detail=f"No quotes for {ticker}")

    from ..engine.sigmoid import sigmoid_iv_at_strikes

    params = np.array(req.get("params", [0.25, 0.0, 0.01, 0.02, 0.02, 0.20]), dtype=float)
    forward = chain.forward
    T = chain.T
    sigma_ref = chain.atm_iv

    iv_marked = sigmoid_iv_at_strikes(params, chain.strikes, forward, T, sigma_ref)
    iv_marked = np.where(np.isfinite(iv_marked), iv_marked, chain.atm_iv)
    iv_marked = np.clip(iv_marked, 0.01, 5.0)

    # Prior for comparison
    prior = _state["priors"].get(ticker)
    svi_prior = prior.get("_svi_params") if prior else None
    if svi_prior:
        from ..engine.svi import svi_iv_at_strikes as svi_eval
        iv_prior = svi_eval(svi_prior, chain.strikes, forward, T)
        iv_prior = np.clip(iv_prior, 0.01, 5.0)
    else:
        iv_prior = iv_marked.copy()

    return {
        "ticker": ticker,
        "strikes": chain.strikes.tolist(),
        "iv_prior": [float(v) if np.isfinite(v) else None for v in iv_prior],
        "iv_marked": [float(v) if np.isfinite(v) else None for v in iv_marked],
        "beta": params.tolist(),
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

    prev_forward = _effective_prev_forward(ticker)

    try:
        prior = fit_lqd_prior(
            strikes=strikes, mid_ivs=prev_ivs,
            forward=prev_forward, T=chain.T, r=_get_rate(chain.T),
            grid=grid, phi=phi,
        )
    except Exception:
        atm = chain.prev_close_atm_iv or chain.atm_iv
        prior = bs_prior(atm, chain.T, grid)

    _state["priors_base"][ticker] = prior
    _state["priors"][ticker] = prior

    forward = chain.forward
    view = compute_distribution_view(prior, grid, forward, chain.T, _get_rate(chain.T), phi=phi, market_strikes=chain.strikes if chain else None)

    # Return JW-normalized params so the frontend sliders work
    jw = prior.get("_jw_params")
    if jw:
        beta = [jw["v"], jw["psi_hat"], jw["p_hat"], jw["c_hat"], jw["vt_ratio"]]
    else:
        svi = prior.get("_svi_params")
        if svi:
            from ..engine.svi import raw_svi_to_jw_normalized
            jw = raw_svi_to_jw_normalized(svi["a"], svi["b"], svi["rho"], svi["m"], svi["sigma"], chain.T)
            beta = [jw["v"], jw["psi_hat"], jw["p_hat"], jw["c_hat"], jw["vt_ratio"]]
        else:
            beta = [0.04, 0.0, 0.5, 0.5, 1.0]

    return DistributionView(
        moneyness=view["moneyness"],
        iv_curve=view["iv_curve"],
        cdf_x=view["cdf_x"],
        cdf_y=view["cdf_y"],
        lqd_u=view["lqd_u"],
        lqd_psi=view["lqd_psi"], fit_forward=view.get("fit_forward"),
        beta=beta,
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

    prior_view = compute_distribution_view(prior, grid, forward, T, _get_rate(T), phi=phi, market_strikes=chain.strikes if chain else None)
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

    # Marked view (from solve result, or fallback to direct SVI fit from quotes)
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

            filt_k, filt_iv, filt_mask = filter_quotes_for_fit(mkt_strikes, mkt_ivs, mkt_fwd)
            cfg = _state["config"]
            filt_spread = chain.bid_ask_spread[filt_mask] if chain is not None else None
            svi_prior_p = prior.get("_svi_params") if prior else None
            prior_p = (np.array([svi_prior_p["a"], svi_prior_p["b"], svi_prior_p["rho"],
                                 svi_prior_p["m"], svi_prior_p["sigma"]])
                       if svi_prior_p and cfg.lambda_prior > 0 else None)
            if len(filt_k) >= 5:
                market_svi = fit_svi(filt_k, filt_iv, mkt_fwd, T,
                                     bid_ask_spread=filt_spread,
                                     use_bid_ask_fit=cfg.use_bid_ask_fit,
                                     prior_params=prior_p,
                                     lambda_prior=cfg.lambda_prior)
                # Build a pseudo-prior dict with the market SVI for distribution view
                marked_prior = {
                    **prior,
                    "_svi_params": market_svi,
                }
                marked_view = compute_distribution_view(
                    marked_prior, grid, forward, T, _get_rate(T),
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
                grid, forward, T, _get_rate(T), phi=phi,
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

    # Fallback: if no solve result yet but we have quotes, build marked view
    # from a direct SVI fit to current market data
    if marked_dv is None and chain is not None:
        from ..engine.svi import fit_svi, filter_quotes_for_fit
        filt_k, filt_iv, filt_mask2 = filter_quotes_for_fit(chain.strikes, chain.mid_ivs, chain.forward)
        if len(filt_k) >= 5:
            try:
                cfg2 = _state["config"]
                filt_spread2 = chain.bid_ask_spread[filt_mask2]
                svi_prior_fb = prior.get("_svi_params") if prior else None
                prior_p2 = (np.array([svi_prior_fb["a"], svi_prior_fb["b"], svi_prior_fb["rho"],
                                      svi_prior_fb["m"], svi_prior_fb["sigma"]])
                            if svi_prior_fb and cfg2.lambda_prior > 0 else None)
                market_svi = fit_svi(filt_k, filt_iv, chain.forward, T,
                                     bid_ask_spread=filt_spread2,
                                     use_bid_ask_fit=cfg2.use_bid_ask_fit,
                                     prior_params=prior_p2,
                                     lambda_prior=cfg2.lambda_prior)
                marked_prior = {**prior, "_svi_params": market_svi}
                marked_view = compute_distribution_view(
                    marked_prior, grid, forward, T, _get_rate(T),
                    phi=phi, market_strikes=chain.strikes,
                )
                from ..engine.svi import raw_svi_to_jw_normalized
                jw = raw_svi_to_jw_normalized(market_svi["a"], market_svi["b"], market_svi["rho"],
                                              market_svi["m"], market_svi["sigma"], T)
                svi_beta = [jw["v"], jw["psi_hat"], jw["p_hat"], jw["c_hat"], jw["vt_ratio"]]
                marked_dv = DistributionView(
                    moneyness=marked_view["moneyness"],
                    iv_curve=marked_view["iv_curve"],
                    cdf_x=marked_view["cdf_x"],
                    cdf_y=marked_view["cdf_y"],
                    lqd_u=marked_view["lqd_u"],
                    lqd_psi=marked_view["lqd_psi"],
                    beta=svi_beta,
                )
            except Exception:
                pass

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
        chain.prev_spot = saved_forward / np.exp(_get_rate(saved_T) * saved_T)
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


# --- Rates & Forwards ---

@router.get("/rates/treasury")
def get_treasury_curve():
    curve = _state.get("treasury_curve")
    if curve is None:
        curve = fetch_treasury_curve()
        _state["treasury_curve"] = curve
    return {
        "date": curve.date,
        "tenors": curve.tenors.tolist(),
        "rates": curve.rates.tolist(),
    }


@router.get("/rates/config")
def get_rates_config():
    config = _state["config"]
    return {
        "repo_rate_gc": config.repo_rate_gc,
        "repo_overrides": config.repo_overrides,
    }


@router.put("/rates/config")
def update_rates_config(req: dict):
    config = _state["config"]
    if "repo_rate_gc" in req:
        config.repo_rate_gc = float(req["repo_rate_gc"])
    if "repo_overrides" in req:
        config.repo_overrides = {k: float(v) for k, v in req["repo_overrides"].items()}
    return {"status": "ok"}


@router.put("/forward/{ticker}")
def set_forward_override(ticker: str, req: dict):
    """Set or clear a forward override for current smile.

    Body: {"forward": 635.5} to override, or {"forward": null} to clear.
    """
    chain = _state["quotes"].get(ticker)
    if chain is None:
        raise HTTPException(status_code=404, detail=f"No quotes for {ticker}")

    fwd_val = req.get("forward")
    if fwd_val is None:
        _state["forward_overrides"].pop(ticker, None)
        chain.forward = chain.forward_model or chain.forward
    else:
        fwd_val = float(fwd_val)
        _state["forward_overrides"][ticker] = fwd_val
        chain.forward = fwd_val

    return {
        "status": "ok",
        "ticker": ticker,
        "forward": chain.forward,
        "forward_model": chain.forward_model,
        "forward_parity": chain.forward_parity,
    }


@router.put("/forward/{ticker}/prior")
def set_prior_forward_override(ticker: str, req: dict):
    """Set or clear a forward override for the prior (prev close).

    Body: {"forward": 630.0} to override, or {"forward": null} to clear.
    """
    chain = _state["quotes"].get(ticker)
    if chain is None:
        raise HTTPException(status_code=404, detail=f"No quotes for {ticker}")

    fwd_val = req.get("forward")
    if fwd_val is None:
        _state["prev_forward_overrides"].pop(ticker, None)
    else:
        _state["prev_forward_overrides"][ticker] = float(fwd_val)

    return {"status": "ok", "ticker": ticker}
